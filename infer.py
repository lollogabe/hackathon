from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data.loader import MODEL_INPUT_COLUMNS
from src.features.pipeline import compute_instance_features
from src.model.scorer import load_model, load_threshold
from src.preprocessing.preprocessor import Preprocessor
from src.quantum.embedder import QuantumEmbedder
from src.utils.config import configure_logging, ensure_output_dirs, load_config, resolve_device
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run online DDoS inference on a CSV file.")
    parser.add_argument("--family", choices=["family_a", "family_b"], required=True)
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    return parser.parse_args()


def _validate_inference_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in MODEL_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Inference CSV missing column(s): " + ", ".join(missing))


def _plot_inference(results: pd.DataFrame, threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(results["window_id"], results["max_score"], marker="o")
    ax.axhline(threshold, linestyle="--", color="black", label="threshold")
    ax.set_xlabel("window index")
    ax.set_ylabel("max score")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config)
    output_root, plots_dir = ensure_output_dirs(config)
    set_global_seed(int(config["seed"]))
    logger = logging.getLogger(__name__)
    device = resolve_device(config)
    logger.info("Using device: %s", device)

    preprocessor = Preprocessor.load(output_root / f"preprocessor_{args.family}.pkl")
    model = load_model(output_root / f"best_model_{args.family}.pt", device)
    configured_threshold = float(config["inference_threshold"])
    threshold = (
        load_threshold(output_root / f"threshold_{args.family}.json", configured_threshold)
        if config["use_calibrated_threshold"]
        else configured_threshold
    )
    embedder = QuantumEmbedder(
        n_qubits=int(config["pca_components"]),
        reps=int(config["circuit_reps"]),
        device=device,
        backend=config["quantum_backend"],
        batch_size=int(config["quantum_batch_size"]),
    )

    rows = []
    input_path = Path(args.input)
    for window_id, chunk in enumerate(
        pd.read_csv(input_path, chunksize=int(config["window_width"]))
    ):
        _validate_inference_columns(chunk)
        chunk = chunk.copy()
        chunk["window_id"] = window_id
        features = compute_instance_features(chunk, config)
        if features.empty:
            max_score = 0.0
            malicious_ips: list[str] = []
        else:
            angles = preprocessor.transform(features)
            quantum_features = embedder.embed(angles, show_progress=False)
            with torch.no_grad():
                probabilities = torch.sigmoid(model(quantum_features)).detach().cpu().numpy()
            features = features.copy()
            features["score"] = probabilities
            flagged = features[features["score"] >= threshold].sort_values(
                "score", ascending=False
            )
            max_score = float(features["score"].max())
            malicious_ips = flagged["src_ip"].astype(str).tolist()
        label = "ATTACK" if max_score >= threshold else "NORMAL"
        print(
            f"window={window_id}  score={max_score:.3f}  "
            f"label={label}  ips=[{', '.join(malicious_ips)}]"
        )
        rows.append(
            {
                "window_id": window_id,
                "max_score": max_score,
                "predicted_label": label,
                "malicious_ips": ";".join(malicious_ips),
            }
        )

    results = pd.DataFrame(rows)
    summary_path = output_root / "infer_results.csv"
    results.to_csv(summary_path, index=False)
    _plot_inference(
        results,
        threshold,
        plots_dir / f"infer_scores_{args.family}_{input_path.stem}.png",
    )
    logger.info("Saved inference summary to %s", summary_path)


if __name__ == "__main__":
    main()
