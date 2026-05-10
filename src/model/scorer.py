from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score


class QuantumScorer(torch.nn.Module):
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.linear = torch.nn.Linear(2 * n_qubits - 1, 1, bias=True)

    def forward(self, quantum_features: torch.Tensor) -> torch.Tensor:
        return self.linear(quantum_features).squeeze(-1)


def build_pos_weight(labels: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)
    positives = float((labels_np == 1).sum())
    negatives = float((labels_np == 0).sum())
    weight = negatives / positives if positives > 0.0 else 1.0
    return torch.tensor(weight, dtype=torch.float32, device=device)


def calibrate_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sweep_min: float,
    sweep_max: float,
    sweep_step: float,
) -> tuple[float, float]:
    thresholds = np.arange(sweep_min, sweep_max + sweep_step / 2.0, sweep_step)
    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        f1 = f1_score(labels, predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def save_threshold(path: str | Path, family: str, threshold: float, validation_f1: float) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "family": family,
                "threshold": float(threshold),
                "validation_f1": float(validation_f1),
            },
            handle,
            indent=2,
        )


def load_threshold(path: str | Path, fallback: float) -> float:
    threshold_path = Path(path)
    if not threshold_path.exists():
        return float(fallback)
    with threshold_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload["threshold"])


def save_model(path: str | Path, model: QuantumScorer, family: str, config: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "family": family,
            "n_qubits": model.n_qubits,
            "state_dict": model.state_dict(),
            "config": {
                "pca_components": config["pca_components"],
                "circuit_reps": config["circuit_reps"],
            },
        },
        target,
    )


def load_model(path: str | Path, device: torch.device) -> QuantumScorer:
    payload = torch.load(path, map_location=device)
    model = QuantumScorer(int(payload["n_qubits"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
