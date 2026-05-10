from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data.catalog import DatasetRecord, build_dataset_catalog, filter_catalog
from src.data.loader import load_flow_csv
from src.data.windowing import add_window_id, derive_burst_windows, derive_instance_labels
from src.evaluation.evaluator import (
    binary_metrics,
    dataset_metrics,
    detection_latency,
    window_metrics,
)
from src.features.pipeline import compute_instance_features
from src.training.trainer import SplitData, build_split_data
from src.utils.config import configure_logging, ensure_output_dirs, load_config, resolve_device
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DDoS detection baselines.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    return parser.parse_args()


def _evaluate_split(
    split_data: SplitData,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = split_data.metadata["y_instance"].to_numpy(dtype=int)
    return {
        "instance": binary_metrics(labels, probabilities, threshold),
        "window": window_metrics(
            split_data.metadata, probabilities, threshold, attack_only=True
        ),
        "dataset": dataset_metrics(split_data.metadata, probabilities, threshold),
        "detection_latency": detection_latency(
            split_data.metadata, split_data.burst_windows, probabilities, threshold
        ),
    }


def _save_benchmark_results(
    output_root: Path, family: str, benchmark_name: str, payload: dict[str, Any]
) -> None:
    path = output_root / f"results_{family}.json"
    existing: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    existing.setdefault("benchmarks", {})[benchmark_name] = payload
    with path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)
    logger.info("Updated %s with %s", path, benchmark_name)


def _classical_rbf_pipeline(config: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rbf",
                RBFSampler(
                    gamma=float(config["rbf_gamma"]),
                    n_components=int(config["rbf_components"]),
                    random_state=int(config["seed"]),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=int(config["logistic_regression_max_iter"]),
                    random_state=int(config["seed"]),
                ),
            ),
        ]
    )


def _run_majority(config: dict, family: str, splits: dict[str, SplitData]) -> dict[str, Any]:
    threshold = float(config["inference_threshold"])
    payload = {"family": family, "threshold": threshold, "splits": {}}
    for split_name, split_data in splits.items():
        probabilities = np.zeros(len(split_data.metadata), dtype="float64")
        payload["splits"][split_name] = _evaluate_split(split_data, probabilities, threshold)
    return payload


def _run_classical_rbf(
    config: dict,
    family: str,
    train_data: SplitData,
    splits: dict[str, SplitData],
) -> dict[str, Any]:
    threshold = float(config["inference_threshold"])
    model = _classical_rbf_pipeline(config)
    model.fit(
        train_data.features.to_numpy(dtype="float64"),
        train_data.metadata["y_instance"].to_numpy(dtype=int),
    )
    payload = {"family": family, "threshold": threshold, "splits": {}}
    for split_name, split_data in splits.items():
        probabilities = model.predict_proba(split_data.features.to_numpy(dtype="float64"))[:, 1]
        payload["splits"][split_name] = _evaluate_split(split_data, probabilities, threshold)
    return payload


def _linear_epoch_metrics(
    model: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(features).squeeze(-1)
        loss = float(criterion(logits, labels).detach().cpu().item())
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    predictions = (probabilities >= 0.5).astype(int)
    f1 = binary_metrics(labels.detach().cpu().numpy().astype(int), probabilities, 0.5)["f1"]
    return loss, float(f1)


def _run_pca_linear(
    config: dict,
    family: str,
    train_data: SplitData,
    splits: dict[str, SplitData],
) -> dict[str, Any]:
    device = resolve_device(config)
    scaler = StandardScaler()
    pca = PCA(n_components=int(config["pca_components"]), random_state=int(config["seed"]))
    x_train = scaler.fit_transform(train_data.features.to_numpy(dtype="float64"))
    z_train = pca.fit_transform(x_train).astype("float32")
    train_labels_np = train_data.metadata["y_instance"].to_numpy(dtype="float32")
    train_features = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    train_labels = torch.as_tensor(train_labels_np, dtype=torch.float32, device=device)

    model = torch.nn.Linear(int(config["pca_components"]), 1, bias=True).to(device)
    positives = float((train_labels_np == 1).sum())
    negatives = float((train_labels_np == 0).sum())
    pos_weight = torch.tensor(
        negatives / positives if positives > 0.0 else 1.0,
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    dataset = TensorDataset(train_features, train_labels)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
    )

    best_state = model.state_dict()
    best_f1 = -1.0
    patience = 0
    progress = tqdm(
        range(1, int(config["max_epochs"]) + 1),
        desc=f"baseline pca-linear {family}",
        file=sys.stdout,
    )
    for _epoch in progress:
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features).squeeze(-1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
        loss_value, f1_value = _linear_epoch_metrics(
            model, train_features, train_labels, criterion
        )
        progress.set_postfix({"loss": f"{loss_value:.4f}", "f1": f"{f1_value:.3f}"})
        if f1_value > best_f1:
            best_f1 = f1_value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= int(config["early_stopping_patience"]):
            break
    model.load_state_dict(best_state)

    payload = {"family": family, "threshold": 0.5, "splits": {}}
    for split_name, split_data in splits.items():
        x_split = scaler.transform(split_data.features.to_numpy(dtype="float64"))
        z_split = pca.transform(x_split).astype("float32")
        split_features = torch.as_tensor(z_split, dtype=torch.float32, device=device)
        with torch.no_grad():
            probabilities = (
                torch.sigmoid(model(split_features).squeeze(-1)).detach().cpu().numpy()
            )
        payload["splits"][split_name] = _evaluate_split(split_data, probabilities, 0.5)
    return payload


def _build_bag_level_split(config: dict, records: list[DatasetRecord]) -> SplitData:
    feature_frames: list[pd.DataFrame] = []
    metadata_frames: list[pd.DataFrame] = []
    burst_frames: list[pd.DataFrame] = []
    for record in records:
        flows = load_flow_csv(record.path).copy()
        flows["window_id"] = 0
        features = compute_instance_features(flows, config)
        labels = derive_instance_labels(flows)
        table = features.merge(labels, on=["src_ip", "window_id"], how="left")
        table["family"] = record.family
        table["split"] = record.split
        table["scenario"] = record.scenario
        table["dataset_name"] = record.dataset_name
        table["y_instance"] = table["y_instance"].fillna(0).astype("int64")
        table["y_window"] = int(record.scenario == "attack")
        table["y_dataset"] = int(record.scenario == "attack")
        feature_frames.append(table.drop(columns=[
            "family",
            "split",
            "scenario",
            "dataset_name",
            "src_ip",
            "window_id",
            "y_instance",
            "y_window",
            "y_dataset",
        ]))
        metadata_frames.append(
            table[
                [
                    "family",
                    "split",
                    "scenario",
                    "dataset_name",
                    "src_ip",
                    "window_id",
                    "y_instance",
                    "y_window",
                    "y_dataset",
                ]
            ]
        )
        bursts = derive_burst_windows(add_window_id(load_flow_csv(record.path), int(config["window_width"])))
        if not bursts.empty:
            bursts["family"] = record.family
            bursts["split"] = record.split
            bursts["scenario"] = record.scenario
            bursts["dataset_name"] = record.dataset_name
            burst_frames.append(bursts)
    return SplitData(
        features=pd.concat(feature_frames, ignore_index=True),
        metadata=pd.concat(metadata_frames, ignore_index=True),
        burst_windows=pd.concat(burst_frames, ignore_index=True)
        if burst_frames
        else pd.DataFrame(),
    )


def _run_bag_level(
    config: dict,
    family: str,
    catalog_records: list[DatasetRecord],
) -> dict[str, Any]:
    train_data = _build_bag_level_split(
        config, filter_catalog(catalog_records, family=family, split="train")
    )
    splits = {
        "validation": _build_bag_level_split(
            config, filter_catalog(catalog_records, family=family, split="validation")
        ),
        "test": _build_bag_level_split(
            config, filter_catalog(catalog_records, family=family, split="test")
        ),
    }
    return _run_classical_rbf(config, family, train_data, splits)


def run_family_benchmarks(config: dict, family: str) -> None:
    output_root, _ = ensure_output_dirs(config)
    catalog = build_dataset_catalog(config)
    train_data = build_split_data(config, filter_catalog(catalog, family=family, split="train"))
    splits = {
        "validation": build_split_data(
            config, filter_catalog(catalog, family=family, split="validation")
        ),
        "test": build_split_data(config, filter_catalog(catalog, family=family, split="test")),
    }
    benchmarks = {
        "majority_class": _run_majority(config, family, splits),
        "classical_window_rbf": _run_classical_rbf(config, family, train_data, splits),
        "pca_linear_ablation": _run_pca_linear(config, family, train_data, splits),
        "classical_bag_level": _run_bag_level(config, family, catalog),
    }
    for name, payload in benchmarks.items():
        _save_benchmark_results(output_root, family, name, payload)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config)
    ensure_output_dirs(config)
    set_global_seed(int(config["seed"]))
    logger.info("Using device: %s", resolve_device(config))
    for family in config["families"]:
        run_family_benchmarks(config, family)


if __name__ == "__main__":
    main()
