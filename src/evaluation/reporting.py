from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METHOD_LABELS = {
    "quantum_pipeline": "Quantum ZZ",
    "majority_class": "Majority",
    "classical_window_rbf": "Classical RBF",
    "pca_linear_ablation": "PCA Linear",
    "classical_bag_level": "Bag RBF",
}

LEVEL_LABELS = {
    "instance": "IP instance",
    "window": "Window",
    "dataset": "Dataset",
}

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "auc_roc"]
PLOT_METRICS = ["precision", "recall", "f1", "auc_roc"]
PLOT_COLORS = ["#2F6FBB", "#E07A2D", "#2E9D6F", "#A43D78", "#6D5BD0"]


def generate_reports(config: dict) -> dict[str, list[Path]]:
    output_root = Path(config["output_root"])
    plots_dir = output_root / "plots"
    tables_dir = output_root / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    write_tables_enabled = bool(config.get("write_report_tables", True))
    if write_tables_enabled:
        tables_dir.mkdir(parents=True, exist_ok=True)
    elif tables_dir.exists():
        shutil.rmtree(tables_dir)

    metrics, latencies, thresholds = load_report_frames(config)
    written: dict[str, list[Path]] = {"tables": [], "plots": []}

    if write_tables_enabled:
        written["tables"].extend(write_tables(metrics, latencies, thresholds, tables_dir))
    written["plots"].extend(write_plots(metrics, latencies, thresholds, plots_dir, config))

    logger.info(
        "Generated %s tables and %s plots.",
        len(written["tables"]),
        len(written["plots"]),
    )
    return written


def load_report_frames(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_root = Path(config["output_root"])
    metric_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for family in config["families"]:
        results_path = output_root / f"results_{family}.json"
        if not results_path.exists():
            logger.warning("Skipping missing result file: %s", results_path)
            continue
        with results_path.open("r", encoding="utf-8") as handle:
            results = json.load(handle)

        method_payloads: list[tuple[str, dict[str, Any]]] = []
        if "quantum_pipeline" in results:
            method_payloads.append(("quantum_pipeline", results["quantum_pipeline"]))
        for benchmark_name, benchmark_payload in results.get("benchmarks", {}).items():
            method_payloads.append((benchmark_name, benchmark_payload))

        for method_key, payload in method_payloads:
            method_label = (
                payload.get("model_label")
                if method_key == "quantum_pipeline"
                else METHOD_LABELS.get(method_key, method_key.replace("_", " ").title())
            )
            for split, split_payload in payload.get("splits", {}).items():
                for level in ("instance", "window", "dataset"):
                    metrics = split_payload.get(level)
                    if not isinstance(metrics, dict):
                        continue
                    row = {
                        "family": family,
                        "method": method_key,
                        "method_label": method_label,
                        "split": split,
                        "level": level,
                        "level_label": LEVEL_LABELS[level],
                        "threshold": payload.get("threshold"),
                        "confusion_matrix": metrics.get("confusion_matrix"),
                    }
                    for metric in METRIC_COLUMNS:
                        row[metric] = metrics.get(metric)
                    metric_rows.append(row)

                latency = split_payload.get("detection_latency", {})
                if isinstance(latency, dict):
                    latency_rows.append(
                        {
                            "family": family,
                            "method": method_key,
                            "method_label": method_label,
                            "split": split,
                            "mean_windows": latency.get("mean"),
                            "std_windows": latency.get("std"),
                            "detected_bursts": latency.get("count"),
                            "missed_bursts": latency.get("missed"),
                        }
                    )

        threshold_path = output_root / f"threshold_{family}.json"
        if threshold_path.exists():
            with threshold_path.open("r", encoding="utf-8") as handle:
                threshold_payload = json.load(handle)
            threshold_rows.append(
                {
                    "family": family,
                    "threshold": threshold_payload.get("threshold"),
                    "validation_f1": threshold_payload.get("validation_f1"),
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    latencies_df = pd.DataFrame(latency_rows)
    thresholds_df = pd.DataFrame(threshold_rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(["family", "split", "level", "method_label"])
    return metrics_df, latencies_df, thresholds_df


def write_tables(
    metrics: pd.DataFrame,
    latencies: pd.DataFrame,
    thresholds: pd.DataFrame,
    tables_dir: Path,
) -> list[Path]:
    written: list[Path] = []
    if not metrics.empty:
        metrics_export = metrics.drop(columns=["confusion_matrix"])
        written.extend(_write_table_pair(metrics_export, tables_dir / "metrics_summary"))

        test_summary = _presentation_metric_table(metrics, split="test")
        written.extend(_write_table_pair(test_summary, tables_dir / "test_metrics_presentation"))
        written.extend(
            _write_table_pair(
                _method_rankings(metrics, split="test"), tables_dir / "test_method_rankings"
            )
        )
        written.extend(
            _write_table_pair(
                _quantum_executive_summary(metrics, latencies, thresholds),
                tables_dir / "quantum_executive_summary",
            )
        )

        confusion_rows = []
        for row in metrics.itertuples(index=False):
            matrix = getattr(row, "confusion_matrix")
            if matrix is None:
                continue
            tn, fp, fn, tp = _confusion_counts(matrix)
            confusion_rows.append(
                {
                    "family": row.family,
                    "method": row.method_label,
                    "split": row.split,
                    "level": row.level_label,
                    "true_negative": tn,
                    "false_positive": fp,
                    "false_negative": fn,
                    "true_positive": tp,
                }
            )
        if confusion_rows:
            written.extend(
                _write_table_pair(
                    pd.DataFrame(confusion_rows),
                    tables_dir / "confusion_matrices_summary",
                )
            )

    if not latencies.empty:
        written.extend(_write_table_pair(latencies, tables_dir / "detection_latency_summary"))
    if not thresholds.empty:
        written.extend(_write_table_pair(thresholds, tables_dir / "threshold_summary"))
    return written


def write_plots(
    metrics: pd.DataFrame,
    latencies: pd.DataFrame,
    thresholds: pd.DataFrame,
    plots_dir: Path,
    config: dict,
) -> list[Path]:
    written: list[Path] = []
    if not metrics.empty:
        path = plots_dir / "presentation_test_f1_comparison.png"
        plot_test_f1_comparison(metrics, path)
        written.append(path)

        path = plots_dir / "presentation_test_metrics_table.png"
        plot_metrics_table(_presentation_metric_table(metrics, split="test"), path)
        written.append(path)

        path = plots_dir / "presentation_quantum_executive_summary.png"
        plot_metrics_table(
            _quantum_executive_summary(metrics, latencies, thresholds),
            path,
            title="Quantum Pipeline Executive Summary",
        )
        written.append(path)

        for family in sorted(metrics["family"].unique()):
            path = plots_dir / f"presentation_{family}_test_metric_heatmap.png"
            plot_family_metric_heatmap(metrics, family, "test", path)
            written.append(path)

            family_table = _presentation_metric_table(
                metrics[metrics["family"] == family], split="test"
            )
            path = plots_dir / f"presentation_{family}_test_metrics_table.png"
            plot_metrics_table(family_table, path, title=f"{family} Test Metrics")
            written.append(path)

            path = plots_dir / f"presentation_{family}_quantum_confusion_matrices.png"
            plot_quantum_confusion_grid(metrics, family, "test", path)
            written.append(path)

        include_individual = bool(config.get("include_individual_confusion_plots", False))
        if config.get("report_detail_level", "presentation") == "full":
            include_individual = True
        if include_individual:
            for row in metrics.itertuples(index=False):
                matrix = getattr(row, "confusion_matrix")
                if matrix is None:
                    continue
                path = (
                    plots_dir
                    / "confusion_matrices"
                    / f"{row.family}_{row.method}_{row.split}_{row.level}.png"
                )
                plot_single_confusion_matrix(
                    matrix,
                    f"{row.method_label} | {row.family} | {row.split} | {row.level_label}",
                    path,
                )
                written.append(path)
        else:
            stale_detail_dir = plots_dir / "confusion_matrices"
            if stale_detail_dir.exists():
                shutil.rmtree(stale_detail_dir)

    if not latencies.empty:
        path = plots_dir / "presentation_detection_latency.png"
        plot_detection_latency(latencies, path)
        written.append(path)
    if not thresholds.empty:
        path = plots_dir / "presentation_thresholds.png"
        plot_thresholds(thresholds, path)
        written.append(path)
    return written


def plot_test_f1_comparison(metrics: pd.DataFrame, path: Path) -> None:
    data = metrics[(metrics["split"] == "test") & (metrics["level"].isin(["instance", "window", "dataset"]))]
    families = sorted(data["family"].unique())
    fig, axes = plt.subplots(
        1,
        len(families),
        figsize=(7.2 * max(len(families), 1), 5.2),
        sharey=True,
    )
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        family_data = data[data["family"] == family]
        pivot = (
            family_data.pivot_table(
                index="method_label", columns="level_label", values="f1", aggfunc="first"
            )
            .reindex(columns=["IP instance", "Window", "Dataset"])
            .sort_index()
        )
        x = np.arange(len(pivot.index))
        width = 0.24
        for offset, level in enumerate(pivot.columns):
            values = pivot[level].fillna(0.0).to_numpy(dtype=float)
            bars = ax.bar(
                x + (offset - 1) * width,
                values,
                width=width,
                label=level,
                color=PLOT_COLORS[offset],
            )
            _annotate_bars(ax, bars)
        ax.set_title(f"{family} test F1", fontsize=16, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("F1 score")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_family_metric_heatmap(
    metrics: pd.DataFrame, family: str, split: str, path: Path
) -> None:
    data = metrics[(metrics["family"] == family) & (metrics["split"] == split)].copy()
    data["row_label"] = data["method_label"] + " - " + data["level_label"]
    table = data.pivot_table(index="row_label", values=PLOT_METRICS, aggfunc="first")
    table = table.rename(columns={"auc_roc": "AUC"})
    table = table[["precision", "recall", "f1", "AUC"]]

    fig, ax = plt.subplots(figsize=(8.8, max(5.0, 0.42 * len(table.index))))
    values = table.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title(f"{family} {split} metrics", fontsize=16, weight="bold")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels(["Precision", "Recall", "F1", "AUC"])
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(table.index)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "n/a" if np.isnan(value) else f"{value:.2f}"
            ax.text(col, row, label, ha="center", va="center", color="#111111", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_quantum_confusion_grid(
    metrics: pd.DataFrame, family: str, split: str, path: Path
) -> None:
    data = metrics[
        (metrics["family"] == family)
        & (metrics["split"] == split)
        & (metrics["method"] == "quantum_pipeline")
    ]
    levels = ["instance", "window", "dataset"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, level in zip(axes, levels):
        row = data[data["level"] == level]
        if row.empty:
            ax.axis("off")
            continue
        _draw_confusion_matrix(
            ax,
            row.iloc[0]["confusion_matrix"],
            LEVEL_LABELS[level],
        )
    fig.suptitle(f"{family} quantum pipeline test confusion matrices", fontsize=17, weight="bold")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_single_confusion_matrix(matrix: list[list[int]], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    _draw_confusion_matrix(ax, matrix, title)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_metrics_table(
    table: pd.DataFrame, path: Path, title: str = "Test Metrics Summary"
) -> None:
    display = table.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )
    fig_height = max(3.5, 0.34 * len(display) + 1.6)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=18, weight="bold", pad=16)
    rendered = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9)
    rendered.scale(1.0, 1.35)
    for (row, _col), cell in rendered.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2F3A4A")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F6FA")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_detection_latency(latencies: pd.DataFrame, path: Path) -> None:
    data = latencies[latencies["split"] == "test"].copy()
    data["mean_windows"] = data["mean_windows"].fillna(np.nan)
    labels = data["family"] + "\n" + data["method_label"]
    values = data["mean_windows"].fillna(0.0).to_numpy(dtype=float)
    colors = ["#2E9D6F" if not pd.isna(value) else "#AAB2BD" for value in data["mean_windows"]]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(np.arange(len(data)), values, color=colors)
    for bar, missed in zip(bars, data["missed_bursts"].fillna(0).astype(int)):
        label = f"{bar.get_height():.1f}" if bar.get_height() > 0 else "0"
        if missed:
            label += f"\nmissed {missed}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.03,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("Test Detection Latency", fontsize=16, weight="bold")
    ax.set_ylabel("Mean windows after ramp-up")
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_thresholds(thresholds: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(len(thresholds))
    width = 0.34
    bars_a = ax.bar(
        x - width / 2,
        thresholds["threshold"].to_numpy(dtype=float),
        width=width,
        label="Threshold",
        color="#2F6FBB",
    )
    bars_b = ax.bar(
        x + width / 2,
        thresholds["validation_f1"].to_numpy(dtype=float),
        width=width,
        label="Validation F1",
        color="#E07A2D",
    )
    _annotate_bars(ax, bars_a)
    _annotate_bars(ax, bars_b)
    ax.set_title("Calibrated Thresholds", fontsize=16, weight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds["family"])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, path)


def _presentation_metric_table(metrics: pd.DataFrame, split: str) -> pd.DataFrame:
    data = metrics[metrics["split"] == split].copy()
    data = data[
        ["family", "method_label", "level_label", "precision", "recall", "f1", "auc_roc"]
    ].rename(
        columns={
            "family": "Family",
            "method_label": "Method",
            "level_label": "Level",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "auc_roc": "AUC",
        }
    )
    order = ["Quantum ZZ", "Classical RBF", "PCA Linear", "Bag RBF", "Majority"]
    data["method_order"] = data["Method"].map({name: index for index, name in enumerate(order)})
    level_order = {"IP instance": 0, "Window": 1, "Dataset": 2}
    data["level_order"] = data["Level"].map(level_order)
    data = data.sort_values(["Family", "method_order", "level_order", "Method"])
    return data.drop(columns=["method_order", "level_order"]).reset_index(drop=True)


def _method_rankings(metrics: pd.DataFrame, split: str) -> pd.DataFrame:
    data = metrics[metrics["split"] == split].copy()
    data = data[
        ["family", "level_label", "method_label", "precision", "recall", "f1", "auc_roc"]
    ].rename(
        columns={
            "family": "Family",
            "level_label": "Level",
            "method_label": "Method",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "auc_roc": "AUC",
        }
    )
    data["Rank"] = (
        data.groupby(["Family", "Level"])["F1"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return data.sort_values(["Family", "Level", "Rank", "Method"]).reset_index(drop=True)


def _quantum_executive_summary(
    metrics: pd.DataFrame, latencies: pd.DataFrame, thresholds: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    quantum = metrics[(metrics["method"] == "quantum_pipeline") & (metrics["split"] == "test")]
    for family in sorted(quantum["family"].unique()):
        family_quantum = quantum[quantum["family"] == family]
        instance = _level_row(family_quantum, "instance")
        window = _level_row(family_quantum, "window")
        dataset = _level_row(family_quantum, "dataset")
        family_threshold = thresholds[thresholds["family"] == family]
        family_latency = latencies[
            (latencies["family"] == family)
            & (latencies["method"] == "quantum_pipeline")
            & (latencies["split"] == "test")
        ]
        rows.append(
            {
                "Family": family,
                "Thr": _first_or_nan(family_threshold, "threshold"),
                "Val F1": _first_or_nan(family_threshold, "validation_f1"),
                "Inst F1": instance.get("f1"),
                "Inst AUC": instance.get("auc_roc"),
                "Win F1": window.get("f1"),
                "Win AUC": window.get("auc_roc"),
                "Data F1": dataset.get("f1"),
                "Latency": _first_or_nan(family_latency, "mean_windows"),
                "Missed": _first_or_nan(family_latency, "missed_bursts"),
            }
        )
    return pd.DataFrame(rows)


def _level_row(frame: pd.DataFrame, level: str) -> dict[str, Any]:
    row = frame[frame["level"] == level]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def _first_or_nan(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return frame.iloc[0][column]


def _write_table_pair(frame: pd.DataFrame, stem: Path) -> list[Path]:
    csv_path = stem.with_suffix(".csv")
    md_path = stem.with_suffix(".md")
    frame.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown(frame), encoding="utf-8")
    return [csv_path, md_path]


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
            formatted[column] = formatted[column].map(
                lambda value: "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
            )
    rows = [list(formatted.columns)] + formatted.astype(str).values.tolist()
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    header = "| " + " | ".join(rows[0][col].ljust(widths[col]) for col in range(len(widths))) + " |"
    separator = "| " + " | ".join("-" * widths[col] for col in range(len(widths))) + " |"
    body = [
        "| " + " | ".join(row[col].ljust(widths[col]) for col in range(len(widths))) + " |"
        for row in rows[1:]
    ]
    return "\n".join([header, separator] + body) + "\n"


def _draw_confusion_matrix(ax: plt.Axes, matrix: list[list[int]], title: str) -> None:
    values = np.asarray(matrix, dtype=float)
    image = ax.imshow(values, cmap="Blues")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred benign", "Pred attack"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True benign", "True attack"])
    total = values.sum()
    for row in range(2):
        row_total = values[row].sum()
        for col in range(2):
            count = int(values[row, col])
            row_pct = 0.0 if row_total == 0 else 100.0 * values[row, col] / row_total
            label = f"{count:,}\n{row_pct:.1f}% row"
            color = "white" if values[row, col] > values.max() * 0.55 else "#111111"
            ax.text(col, row, label, ha="center", va="center", color=color, fontsize=10)
    ax.set_xlabel(f"Total: {int(total):,}")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _confusion_counts(matrix: list[list[int]]) -> tuple[int, int, int, int]:
    values = np.asarray(matrix, dtype=int)
    return int(values[0, 0]), int(values[0, 1]), int(values[1, 0]), int(values[1, 1])


def _annotate_bars(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(height + 0.025, 1.02),
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
