from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.catalog import build_dataset_catalog, filter_catalog
from src.model.scorer import load_model, load_threshold
from src.preprocessing.preprocessor import Preprocessor
from src.quantum.embedder import QuantumEmbedder
from src.training.trainer import build_split_data, compute_or_load_quantum_features
from src.utils.config import ensure_output_dirs, resolve_device

logger = logging.getLogger(__name__)


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities).astype(float)
    predictions = (probabilities >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc_roc": auc,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def aggregate_window_scores(
    metadata: pd.DataFrame, probabilities: np.ndarray, threshold: float
) -> pd.DataFrame:
    table = metadata.copy()
    table["probability"] = probabilities
    grouped = (
        table.groupby(["family", "split", "scenario", "dataset_name", "window_id"], as_index=False)
        .agg(score=("probability", "max"), y_window=("y_window", "max"))
        .sort_values(["dataset_name", "window_id"])
    )
    grouped["prediction"] = (grouped["score"] >= threshold).astype(int)
    return grouped


def aggregate_dataset_scores(
    metadata: pd.DataFrame, probabilities: np.ndarray, threshold: float
) -> pd.DataFrame:
    table = metadata.copy()
    table["probability"] = probabilities
    grouped = (
        table.groupby(["family", "split", "scenario", "dataset_name"], as_index=False)
        .agg(score=("probability", "max"), y_dataset=("y_dataset", "max"))
        .sort_values(["dataset_name"])
    )
    grouped["prediction"] = (grouped["score"] >= threshold).astype(int)
    return grouped


def window_metrics(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    attack_only: bool,
) -> dict[str, Any]:
    windows = aggregate_window_scores(metadata, probabilities, threshold)
    if attack_only:
        windows = windows[windows["scenario"] == "attack"]
    if windows.empty:
        return {"error": "no windows available"}
    return binary_metrics(
        windows["y_window"].to_numpy(), windows["score"].to_numpy(), threshold
    )


def dataset_metrics(
    metadata: pd.DataFrame, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    datasets = aggregate_dataset_scores(metadata, probabilities, threshold)
    if datasets.empty:
        return {"error": "no datasets available"}
    return binary_metrics(
        datasets["y_dataset"].to_numpy(), datasets["score"].to_numpy(), threshold
    )


def detection_latency(
    metadata: pd.DataFrame,
    burst_windows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    if burst_windows.empty:
        return {"mean": None, "std": None, "count": 0, "missed": 0, "latencies": []}
    windows = aggregate_window_scores(metadata, probabilities, threshold)
    latencies: list[int] = []
    missed = 0
    for (dataset_name, burst_id), group in burst_windows.groupby(["dataset_name", "burst_id"]):
        ramp_rows = group[group["has_ramp_up"]]
        if ramp_rows.empty:
            ramp_start = int(group["window_id"].min())
        else:
            ramp_start = int(ramp_rows["window_id"].min())
        flagged = windows[
            (windows["dataset_name"] == dataset_name)
            & (windows["window_id"] >= ramp_start)
            & (windows["prediction"] == 1)
        ]
        if flagged.empty:
            missed += 1
            continue
        latencies.append(int(flagged["window_id"].min()) - ramp_start)
    if not latencies:
        return {"mean": None, "std": None, "count": 0, "missed": missed, "latencies": []}
    return {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "count": len(latencies),
        "missed": missed,
        "latencies": latencies,
    }


def plot_training_curves(history: list[dict[str, float]], path: str | Path) -> None:
    if not history:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(target, dpi=160)
    plt.close(fig)


def plot_temporal_detection(
    window_scores: pd.DataFrame,
    threshold: float,
    path: str | Path,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(window_scores["window_id"], window_scores["score"], marker="o", label="max score")
    true_windows = window_scores[window_scores["y_window"] == 1]["window_id"].astype(int).tolist()
    for window_id in true_windows:
        ax.axvspan(window_id - 0.5, window_id + 0.5, color="tab:red", alpha=0.18)
    ax.axhline(threshold, linestyle="--", color="black", label="threshold")
    ax.set_xlabel("window index")
    ax.set_ylabel("max instance score")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def evaluate_family(config: dict, family: str) -> dict[str, Any]:
    output_root, plots_dir = ensure_output_dirs(config)
    device = resolve_device(config)
    catalog = build_dataset_catalog(config)
    preprocessor = Preprocessor.load(output_root / f"preprocessor_{family}.pkl")
    model = load_model(output_root / f"best_model_{family}.pt", device)
    configured_threshold = float(config["inference_threshold"])
    if config["use_calibrated_threshold"]:
        threshold = load_threshold(
            output_root / f"threshold_{family}.json", configured_threshold
        )
    else:
        threshold = configured_threshold

    embedder = QuantumEmbedder(
        n_qubits=int(config["pca_components"]),
        reps=int(config["circuit_reps"]),
        device=device,
        backend=config["quantum_backend"],
        batch_size=int(config["quantum_batch_size"]),
    )

    results: dict[str, Any] = {"family": family, "threshold": threshold, "splits": {}}
    for split in ("validation", "test"):
        records = filter_catalog(catalog, family=family, split=split)
        split_data = build_split_data(config, records)
        quantum_features = compute_or_load_quantum_features(
            config,
            family,
            split,
            split_data.features,
            preprocessor,
            embedder,
            force_recompute=not bool(config["reuse_quantum_cache"]),
        )
        with torch.no_grad():
            probabilities = torch.sigmoid(model(quantum_features)).detach().cpu().numpy()
        labels = split_data.metadata["y_instance"].to_numpy(dtype=int)
        split_results = {
            "instance": binary_metrics(labels, probabilities, threshold),
            "window": window_metrics(
                split_data.metadata, probabilities, threshold, attack_only=True
            ),
            "dataset": dataset_metrics(split_data.metadata, probabilities, threshold),
            "detection_latency": detection_latency(
                split_data.metadata, split_data.burst_windows, probabilities, threshold
            ),
        }
        results["splits"][split] = split_results
        if split == "test":
            windows = aggregate_window_scores(split_data.metadata, probabilities, threshold)
            for dataset_name, dataset_windows in windows[
                windows["scenario"] == "attack"
            ].groupby("dataset_name"):
                plot_temporal_detection(
                    dataset_windows,
                    threshold,
                    plots_dir / f"temporal_{family}_{dataset_name}.png",
                )

    results_path = output_root / f"results_{family}.json"
    existing: dict[str, Any] = {}
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    existing["quantum_pipeline"] = results
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)
    logger.info("Saved evaluation results to %s", results_path)
    return results
