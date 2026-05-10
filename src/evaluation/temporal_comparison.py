from __future__ import annotations

import copy
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
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

from src.data.catalog import build_dataset_catalog, filter_catalog
from src.evaluation.evaluator import aggregate_window_scores
from src.evaluation.evaluator import predict_architecture_probabilities
from src.model.scorer import build_pos_weight, load_model, load_threshold
from src.preprocessing.preprocessor import Preprocessor
from src.quantum.embedder import QuantumEmbedder
from src.training.trainer import (
    SplitData,
    build_split_data,
)
from src.utils.config import ensure_output_dirs, resolve_device
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)

ONLINE_MODEL_LABELS = {
    "quantum_pipeline": "Quantum ZZ",
    "classical_window_rbf": "Classical RBF",
    "pca_linear_ablation": "PCA Linear",
}


@dataclass(frozen=True)
class BestOnlineModel:
    key: str
    label: str
    test_window_f1: float
    threshold: float


def generate_temporal_comparison_plots(
    config: dict,
    clean_existing: bool = False,
    write_summary_table: bool = True,
) -> list[Path]:
    output_root, plots_dir = ensure_output_dirs(config)
    catalog = build_dataset_catalog(config)
    written: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    architecture = config.get("quantum_architecture", "zz_linear")

    for family in config["families"]:
        best = select_best_online_model(config, family)
        quantum_label = _configured_quantum_label(config)
        logger.info(
            "Best online model for %s by test window F1: %s (F1=%.4f)",
            family,
            best.label,
            best.test_window_f1,
        )
        train_data = build_split_data(
            config, filter_catalog(catalog, family=family, split="train")
        )
        test_data = build_split_data(
            config, filter_catalog(catalog, family=family, split="test")
        )

        quantum_probabilities, quantum_threshold = predict_quantum_architecture(
            config, family, test_data
        )
        if best.key == "quantum_pipeline":
            best_probabilities = quantum_probabilities
        else:
            best_probabilities = predict_online_baseline(config, best.key, train_data, test_data)

        quantum_windows = aggregate_window_scores(
            test_data.metadata, quantum_probabilities, quantum_threshold
        )
        best_windows = aggregate_window_scores(
            test_data.metadata, best_probabilities, best.threshold
        )

        for dataset_name, quantum_dataset_windows in quantum_windows[
            quantum_windows["scenario"] == "attack"
        ].groupby("dataset_name"):
            best_dataset_windows = best_windows[best_windows["dataset_name"] == dataset_name]
            output_path = (
                plots_dir
                / f"temporal_compare_{architecture}_{family}_{dataset_name}.png"
            )
            plot_temporal_comparison(
                quantum_dataset_windows,
                best_dataset_windows,
                quantum_threshold,
                best.threshold,
                quantum_label,
                best.label,
                output_path,
            )
            written.append(output_path)
            summary_rows.append(
                {
                    "quantum_architecture": architecture,
                    "family": family,
                    "dataset_name": dataset_name,
                    "best_online_model": best.label,
                    "best_online_test_window_f1": best.test_window_f1,
                    **_window_summary("quantum", quantum_dataset_windows),
                    **_window_summary("best", best_dataset_windows),
                }
            )

    if summary_rows and write_summary_table:
        tables_dir = output_root / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(summary_rows)
        csv_path = tables_dir / "temporal_comparison_summary.csv"
        md_path = tables_dir / "temporal_comparison_summary.md"
        summary.to_csv(csv_path, index=False)
        md_path.write_text(_to_markdown(summary), encoding="utf-8")
        written.extend([csv_path, md_path])
    if clean_existing:
        _remove_stale_temporal_compare_plots(plots_dir, written)
    return written


def select_best_online_model(config: dict, family: str) -> BestOnlineModel:
    output_root = Path(config["output_root"])
    results_path = output_root / f"results_{family}.json"
    with results_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)

    candidates: list[BestOnlineModel] = []
    quantum = results.get("quantum_pipeline")
    if quantum is not None:
        configured_architecture = config.get("quantum_architecture", "zz_linear")
        result_architecture = quantum.get("architecture", "zz_linear")
        if result_architecture == configured_architecture:
            candidates.append(_model_from_payload("quantum_pipeline", quantum))
        else:
            logger.warning(
                "Skipping stale quantum result for %s: results were produced with "
                "architecture=%s but config selects architecture=%s.",
                family,
                result_architecture,
                configured_architecture,
            )
    benchmarks = results.get("benchmarks", {})
    for key in ("classical_window_rbf", "pca_linear_ablation"):
        if key in benchmarks:
            candidates.append(_model_from_payload(key, benchmarks[key]))
    if not candidates:
        raise ValueError(f"No online model results found for {family}.")
    return max(candidates, key=lambda item: item.test_window_f1)


def predict_quantum_architecture(
    config: dict,
    family: str,
    test_data: SplitData,
) -> tuple[np.ndarray, float]:
    output_root, _ = ensure_output_dirs(config)
    device = resolve_device(config)
    architecture = config.get("quantum_architecture", "zz_linear")
    threshold = (
        load_threshold(output_root / f"threshold_{family}.json", float(config["inference_threshold"]))
        if config["use_calibrated_threshold"]
        else float(config["inference_threshold"])
    )
    preprocessor = Preprocessor.load(output_root / f"preprocessor_{family}.pkl")
    model = None
    kernel_payload = None
    if architecture == "quantum_kernel":
        kernel_payload = joblib.load(output_root / f"best_model_{family}.joblib")
    else:
        model = load_model(
            output_root / f"best_model_{family}.pt",
            device,
            expected_architecture=architecture,
        )
    embedder = None
    if architecture in {"zz_linear", "quantum_kernel"}:
        embedder = QuantumEmbedder(
            n_qubits=int(config["pca_components"]),
            reps=int(config["circuit_reps"]),
            device=device,
            backend=config["quantum_backend"],
            batch_size=int(config["quantum_batch_size"]),
        )
    probabilities = predict_architecture_probabilities(
        config,
        family,
        "test",
        test_data,
        preprocessor,
        model,
        embedder,
        kernel_payload,
        architecture,
        device,
    )
    return probabilities, threshold


def predict_online_baseline(
    config: dict,
    model_key: str,
    train_data: SplitData,
    test_data: SplitData,
) -> np.ndarray:
    if model_key == "classical_window_rbf":
        model = _classical_rbf_pipeline(config)
        model.fit(
            train_data.features.to_numpy(dtype="float64"),
            train_data.metadata["y_instance"].to_numpy(dtype=int),
        )
        return model.predict_proba(test_data.features.to_numpy(dtype="float64"))[:, 1]
    if model_key == "pca_linear_ablation":
        return _fit_predict_pca_linear(config, train_data, test_data)
    raise ValueError(f"Unsupported online baseline for temporal comparison: {model_key}")


def plot_temporal_comparison(
    quantum_windows: pd.DataFrame,
    best_windows: pd.DataFrame,
    quantum_threshold: float,
    best_threshold: float,
    quantum_label: str,
    best_label: str,
    path: str | Path,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    quantum = quantum_windows.sort_values("window_id").copy()
    best = best_windows.sort_values("window_id").copy()
    quantum["prediction"] = (quantum["score"] >= quantum_threshold).astype(int)
    best["prediction"] = (best["score"] >= best_threshold).astype(int)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    true_windows = quantum[quantum["y_window"] == 1]["window_id"].astype(int).tolist()
    for index, window_id in enumerate(true_windows):
        ax.axvspan(
            window_id - 0.5,
            window_id + 0.5,
            color="tab:red",
            alpha=0.14,
            label="true burst window" if index == 0 else None,
        )

    ax.plot(
        quantum["window_id"],
        quantum["score"],
        marker="o",
        linewidth=2.0,
        color="#1f77b4",
        label=f"{quantum_label} max score",
    )
    ax.plot(
        best["window_id"],
        best["score"],
        marker="s",
        linewidth=2.0,
        color="#2ca02c",
        label=f"{best_label} max score",
    )
    _scatter_predictions(ax, quantum, marker="^", color="#ff7f0e", label=f"{quantum_label} predicted")
    _scatter_predictions(ax, best, marker="D", color="#9467bd", label=f"{best_label} predicted")
    ax.axhline(
        quantum_threshold,
        linestyle="--",
        color="#1f77b4",
        linewidth=1.8,
        label=f"{quantum_label} threshold={quantum_threshold:.2f}",
    )
    ax.axhline(
        best_threshold,
        linestyle=":",
        color="#2ca02c",
        linewidth=2.0,
        label=f"{best_label} threshold={best_threshold:.2f}",
    )

    quantum_summary = _window_summary_text(quantum_label, quantum)
    best_summary = _window_summary_text(best_label, best)
    ax.text(
        0.99,
        0.03,
        f"{quantum_summary}\n{best_summary}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.86},
    )
    family = str(quantum["family"].iloc[0])
    dataset_name = str(quantum["dataset_name"].iloc[0])
    ax.set_title(
        f"{family} {dataset_name}: {quantum_label} vs best online model",
        fontsize=15,
        weight="bold",
    )
    ax.set_xlabel("window index")
    ax.set_ylabel("max instance score")
    ax.set_ylim(0.0, 1.04)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _model_from_payload(key: str, payload: dict[str, Any]) -> BestOnlineModel:
    window_payload = payload["splits"]["test"]["window"]
    threshold = float(payload.get("threshold", 0.5))
    label = payload.get("model_label") if key == "quantum_pipeline" else ONLINE_MODEL_LABELS[key]
    return BestOnlineModel(
        key=key,
        label=label,
        test_window_f1=float(window_payload.get("f1") or 0.0),
        threshold=threshold,
    )


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


def _fit_predict_pca_linear(
    config: dict,
    train_data: SplitData,
    test_data: SplitData,
) -> np.ndarray:
    set_global_seed(int(config["seed"]))
    device = resolve_device(config)
    scaler = StandardScaler()
    pca = PCA(n_components=int(config["pca_components"]), random_state=int(config["seed"]))
    z_train = pca.fit_transform(
        scaler.fit_transform(train_data.features.to_numpy(dtype="float64"))
    ).astype("float32")
    train_labels_np = train_data.metadata["y_instance"].to_numpy(dtype="float32")
    train_features = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    train_labels = torch.as_tensor(train_labels_np, dtype=torch.float32, device=device)

    model = torch.nn.Linear(int(config["pca_components"]), 1, bias=True).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=build_pos_weight(train_labels_np, device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=generator,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    epochs_without_improvement = 0
    progress = tqdm(
        range(1, int(config["max_epochs"]) + 1),
        desc="fit temporal PCA Linear",
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

        f1_value = _train_f1(model, train_features, train_labels)
        progress.set_postfix({"train_f1": f"{f1_value:.3f}"})
        if f1_value > best_f1:
            best_f1 = f1_value
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            break

    model.load_state_dict(best_state)
    z_test = pca.transform(
        scaler.transform(test_data.features.to_numpy(dtype="float64"))
    ).astype("float32")
    test_features = torch.as_tensor(z_test, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(test_features).squeeze(-1)).detach().cpu().numpy()


def _train_f1(
    model: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    from sklearn.metrics import f1_score

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(features).squeeze(-1)).detach().cpu().numpy()
    predictions = (probabilities >= 0.5).astype(int)
    return float(f1_score(labels.detach().cpu().numpy().astype(int), predictions, zero_division=0))


def _scatter_predictions(
    ax: plt.Axes,
    windows: pd.DataFrame,
    marker: str,
    color: str,
    label: str,
) -> None:
    predicted = windows[windows["prediction"] == 1]
    if predicted.empty:
        return
    ax.scatter(
        predicted["window_id"],
        predicted["score"],
        marker=marker,
        s=88,
        color=color,
        edgecolor="black",
        linewidth=0.45,
        zorder=5,
        label=label,
    )


def _window_summary(prefix: str, windows: pd.DataFrame) -> dict[str, int]:
    predictions = windows["prediction"].astype(int)
    labels = windows["y_window"].astype(int)
    return {
        f"{prefix}_predicted_windows": int(predictions.sum()),
        f"{prefix}_false_positive_windows": int(((predictions == 1) & (labels == 0)).sum()),
        f"{prefix}_missed_windows": int(((predictions == 0) & (labels == 1)).sum()),
    }


def _window_summary_text(label: str, windows: pd.DataFrame) -> str:
    summary = _window_summary(label.lower().replace(" ", "_"), windows)
    prefix = label.lower().replace(" ", "_")
    return (
        f"{label}: predicted={summary[f'{prefix}_predicted_windows']}  "
        f"FP={summary[f'{prefix}_false_positive_windows']}  "
        f"missed={summary[f'{prefix}_missed_windows']}"
    )


def _configured_quantum_label(config: dict) -> str:
    labels = {
        "zz_linear": "Quantum ZZ",
        "vqc": "VQC",
        "quantum_kernel": "Pure Quantum Kernel",
    }
    return labels.get(config.get("quantum_architecture", "zz_linear"), "Quantum")


def _remove_stale_temporal_compare_plots(plots_dir: Path, keep_paths: list[Path]) -> None:
    keep = {path.resolve() for path in keep_paths if path.suffix == ".png"}
    for path in plots_dir.glob("temporal_compare*.png"):
        if path.resolve() not in keep:
            path.unlink()


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._\n"
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
        else:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else str(value))
    rows = [list(formatted.columns)] + formatted.astype(str).values.tolist()
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    header = "| " + " | ".join(rows[0][col].ljust(widths[col]) for col in range(len(widths))) + " |"
    separator = "| " + " | ".join("-" * widths[col] for col in range(len(widths))) + " |"
    body = [
        "| " + " | ".join(row[col].ljust(widths[col]) for col in range(len(widths))) + " |"
        for row in rows[1:]
    ]
    return "\n".join([header, separator] + body) + "\n"
