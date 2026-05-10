from __future__ import annotations

import copy
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data.catalog import DatasetRecord, build_dataset_catalog, filter_catalog
from src.data.loader import load_flow_csv
from src.data.windowing import (
    add_window_id,
    derive_burst_windows,
    derive_instance_labels,
    derive_window_labels,
)
from src.features.pipeline import compute_instance_features
from src.features.schema import FEATURE_COLUMNS, METADATA_COLUMNS
from src.model.scorer import (
    QuantumScorer,
    build_pos_weight,
    calibrate_threshold,
    save_model,
    save_threshold,
)
from src.preprocessing.preprocessor import Preprocessor
from src.quantum.embedder import QuantumEmbedder
from src.utils.config import ensure_output_dirs, resolve_device

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    features: pd.DataFrame
    metadata: pd.DataFrame
    burst_windows: pd.DataFrame


def build_split_data(config: dict, records: Iterable[DatasetRecord]) -> SplitData:
    feature_frames: list[pd.DataFrame] = []
    metadata_frames: list[pd.DataFrame] = []
    burst_frames: list[pd.DataFrame] = []

    for record in records:
        logger.info("Loading %s", record.path)
        flows = add_window_id(load_flow_csv(record.path), int(config["window_width"]))
        features = compute_instance_features(flows, config)
        instance_labels = derive_instance_labels(flows)
        window_labels = derive_window_labels(flows)

        table = (
            features.merge(instance_labels, on=["src_ip", "window_id"], how="left")
            .merge(window_labels, on="window_id", how="left")
            .sort_values(["window_id", "src_ip"])
            .reset_index(drop=True)
        )
        table["family"] = record.family
        table["split"] = record.split
        table["scenario"] = record.scenario
        table["dataset_name"] = record.dataset_name
        table["y_instance"] = table["y_instance"].fillna(0).astype("int64")
        table["y_window"] = table["y_window"].fillna(0).astype("int64")
        table["y_dataset"] = int(record.scenario == "attack")

        feature_frames.append(table[FEATURE_COLUMNS].copy())
        metadata_frames.append(
            table[METADATA_COLUMNS + ["y_instance", "y_window", "y_dataset"]].copy()
        )

        bursts = derive_burst_windows(flows)
        if not bursts.empty:
            bursts["family"] = record.family
            bursts["split"] = record.split
            bursts["scenario"] = record.scenario
            bursts["dataset_name"] = record.dataset_name
            burst_frames.append(bursts)

    if not feature_frames:
        raise ValueError("No datasets matched the requested split/family.")

    features_all = pd.concat(feature_frames, ignore_index=True)
    metadata_all = pd.concat(metadata_frames, ignore_index=True)
    if burst_frames:
        bursts_all = pd.concat(burst_frames, ignore_index=True)
    else:
        bursts_all = pd.DataFrame(
            columns=[
                "burst_id",
                "window_id",
                "has_ramp_up",
                "has_seeded_ddos",
                "family",
                "split",
                "scenario",
                "dataset_name",
            ]
        )
    return SplitData(features=features_all, metadata=metadata_all, burst_windows=bursts_all)


def compute_or_load_quantum_features(
    config: dict,
    family: str,
    split: str,
    features: pd.DataFrame,
    preprocessor: Preprocessor,
    embedder: QuantumEmbedder,
    force_recompute: bool,
) -> torch.Tensor:
    output_root, _ = ensure_output_dirs(config)
    cache_path = output_root / f"quantum_features_{family}_{split}.pt"
    device = resolve_device(config)
    if cache_path.exists() and not force_recompute:
        payload = torch.load(cache_path, map_location=device)
        tensor = payload["quantum_features"] if isinstance(payload, dict) else payload
        return tensor.to(device=device, dtype=torch.float32)

    angles = preprocessor.transform(features)
    quantum_features = embedder.embed(angles, show_progress=True)
    torch.save(
        {
            "family": family,
            "split": split,
            "quantum_features": quantum_features.detach().cpu(),
            "shape": list(quantum_features.shape),
        },
        cache_path,
    )
    logger.info("Saved quantum feature cache to %s", cache_path)
    return quantum_features


def _safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(labels, probabilities))
    except ValueError:
        return None


def _metrics_from_probs(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | None]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc_roc": _safe_auc(labels, probabilities),
    }


def _evaluate_tensor(
    model: QuantumScorer,
    features: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
) -> tuple[float, dict[str, float | None], np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = float(criterion(logits, labels).detach().cpu().item())
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    metrics = _metrics_from_probs(labels.detach().cpu().numpy().astype(int), probabilities, 0.5)
    return loss, metrics, probabilities


def _plot_training_curves(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_f1"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_f1"] for row in history], label="validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("F1")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_family(config: dict, family: str) -> dict:
    output_root, plots_dir = ensure_output_dirs(config)
    device = resolve_device(config)
    catalog = build_dataset_catalog(config)
    train_records = filter_catalog(catalog, family=family, split="train")
    val_records = filter_catalog(catalog, family=family, split="validation")
    train_data = build_split_data(config, train_records)
    val_data = build_split_data(config, val_records)

    preprocessor = Preprocessor(
        n_components=int(config["pca_components"]),
        variance_warning_threshold=float(config["pca_variance_warning_threshold"]),
    ).fit(train_data.features)
    preprocessor.save(output_root / f"preprocessor_{family}.pkl")

    embedder = QuantumEmbedder(
        n_qubits=int(config["pca_components"]),
        reps=int(config["circuit_reps"]),
        device=device,
        backend=config["quantum_backend"],
        batch_size=int(config["quantum_batch_size"]),
    )
    force_recompute = not bool(config["reuse_quantum_cache"])
    train_quantum = compute_or_load_quantum_features(
        config, family, "train", train_data.features, preprocessor, embedder, force_recompute
    )
    val_quantum = compute_or_load_quantum_features(
        config,
        family,
        "validation",
        val_data.features,
        preprocessor,
        embedder,
        force_recompute,
    )

    train_labels = torch.as_tensor(
        train_data.metadata["y_instance"].to_numpy(dtype="float32"),
        dtype=torch.float32,
        device=device,
    )
    val_labels = torch.as_tensor(
        val_data.metadata["y_instance"].to_numpy(dtype="float32"),
        dtype=torch.float32,
        device=device,
    )

    model = QuantumScorer(int(config["pca_components"])).to(device)
    pos_weight = build_pos_weight(train_labels, device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=int(config["scheduler_patience"]),
        factor=0.5,
    )

    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    dataset = TensorDataset(train_quantum, train_labels)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=generator,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    progress = tqdm(
        range(1, int(config["max_epochs"]) + 1),
        desc=f"train {family}",
        file=sys.stdout,
    )
    for epoch in progress:
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            batch_size = int(batch_labels.shape[0])
            running_loss += float(loss.detach().cpu().item()) * batch_size
            seen += batch_size

        train_loss_eval, train_metrics, _ = _evaluate_tensor(
            model, train_quantum, train_labels, criterion
        )
        val_loss, val_metrics, val_probabilities = _evaluate_tensor(
            model, val_quantum, val_labels, criterion
        )
        scheduler.step(float(val_metrics["f1"] or 0.0))
        train_loss = running_loss / max(seen, 1)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss_eval if np.isfinite(train_loss_eval) else train_loss),
            "val_loss": float(val_loss),
            "train_accuracy": float(train_metrics["accuracy"] or 0.0),
            "train_precision": float(train_metrics["precision"] or 0.0),
            "train_recall": float(train_metrics["recall"] or 0.0),
            "train_f1": float(train_metrics["f1"] or 0.0),
            "train_auc_roc": float(train_metrics["auc_roc"] or 0.0),
            "val_accuracy": float(val_metrics["accuracy"] or 0.0),
            "val_precision": float(val_metrics["precision"] or 0.0),
            "val_recall": float(val_metrics["recall"] or 0.0),
            "val_f1": float(val_metrics["f1"] or 0.0),
            "val_auc_roc": float(val_metrics["auc_roc"] or 0.0),
        }
        history.append(row)
        progress.set_postfix(
            {
                "loss": f"{row['train_loss']:.4f}",
                "acc": f"{row['train_accuracy']:.3f}",
                "prec": f"{row['train_precision']:.3f}",
                "rec": f"{row['train_recall']:.3f}",
                "f1": f"{row['train_f1']:.3f}",
                "auc": f"{row['train_auc_roc']:.3f}",
                "val_loss": f"{row['val_loss']:.4f}",
                "val_acc": f"{row['val_accuracy']:.3f}",
                "val_prec": f"{row['val_precision']:.3f}",
                "val_rec": f"{row['val_recall']:.3f}",
                "val_f1": f"{row['val_f1']:.3f}",
                "val_auc": f"{row['val_auc_roc']:.3f}",
            }
        )

        current_val_f1 = float(val_metrics["f1"] or 0.0)
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            logger.info("Early stopping %s at epoch %s", family, epoch)
            break

    model.load_state_dict(best_state)
    save_model(output_root / f"best_model_{family}.pt", model, family, config)
    _, _, best_val_probabilities = _evaluate_tensor(model, val_quantum, val_labels, criterion)
    threshold, threshold_f1 = calibrate_threshold(
        best_val_probabilities,
        val_data.metadata["y_instance"].to_numpy(dtype=int),
        float(config["threshold_sweep_min"]),
        float(config["threshold_sweep_max"]),
        float(config["threshold_sweep_step"]),
    )
    save_threshold(output_root / f"threshold_{family}.json", family, threshold, threshold_f1)
    _plot_training_curves(history, plots_dir / f"training_curves_{family}.png")
    logger.info("Saved best model and threshold for %s", family)
    return {
        "family": family,
        "best_val_f1": best_val_f1,
        "threshold": threshold,
        "threshold_validation_f1": threshold_f1,
        "epochs": len(history),
    }
