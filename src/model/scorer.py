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


class VQCScorer(torch.nn.Module):
    """Differentiable statevector VQC with trainable rotations and linear readout."""

    def __init__(self, n_qubits: int, layers: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.dimension = 2**n_qubits
        self.ry_angles = torch.nn.Parameter(torch.zeros(layers, n_qubits))
        self.rzz_angles = torch.nn.Parameter(torch.zeros(layers, max(n_qubits - 1, 1)))
        self.linear = torch.nn.Linear(2 * n_qubits - 1, 1, bias=True)
        z_eigs = self._build_z_eigenvalues()
        zz_eigs = z_eigs[:, :-1] * z_eigs[:, 1:] if n_qubits > 1 else z_eigs[:, :0]
        self.register_buffer("z_eigs", z_eigs)
        self.register_buffer("zz_eigs", zz_eigs)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if angles.ndim != 2 or angles.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected angle tensor with shape (n, {self.n_qubits}), got {tuple(angles.shape)}."
            )
        state = torch.zeros(
            (angles.shape[0], self.dimension),
            dtype=torch.complex64,
            device=angles.device,
        )
        state[:, 0] = 1.0 + 0.0j
        for layer in range(self.layers):
            state = self._apply_h_all(state)
            state = self._apply_data_phase(state, angles)
            state = self._apply_trainable_rzz(state, layer)
            for qubit in range(self.n_qubits):
                state = self._apply_ry(state, qubit, self.ry_angles[layer, qubit])
        probabilities = torch.abs(state) ** 2
        z_expectations = probabilities @ self.z_eigs
        if self.n_qubits > 1:
            zz_expectations = probabilities @ self.zz_eigs
            readout = torch.cat([z_expectations, zz_expectations], dim=1)
        else:
            readout = z_expectations
        return self.linear(readout).squeeze(-1)

    def _apply_data_phase(self, state: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        phase = angles @ self.z_eigs.T
        if self.n_qubits > 1:
            pair_angles = angles[:, :-1] * angles[:, 1:]
            phase = phase + pair_angles @ self.zz_eigs.T
        return state * torch.exp((-0.5j) * phase.to(torch.complex64))

    def _apply_trainable_rzz(self, state: torch.Tensor, layer: int) -> torch.Tensor:
        if self.n_qubits <= 1:
            return state
        phase = self.rzz_angles[layer, : self.n_qubits - 1] @ self.zz_eigs.T
        return state * torch.exp((-0.5j) * phase.to(torch.complex64))

    def _apply_h_all(self, state: torch.Tensor) -> torch.Tensor:
        output = state
        inv_sqrt2 = torch.tensor(2.0**-0.5, dtype=output.real.dtype, device=output.device)
        for qubit in range(self.n_qubits):
            idx0, idx1 = self._basis_pair_indices(qubit, output.device)
            a0 = output[:, idx0]
            a1 = output[:, idx1]
            updated = output.clone()
            updated[:, idx0] = (a0 + a1) * inv_sqrt2
            updated[:, idx1] = (a0 - a1) * inv_sqrt2
            output = updated
        return output

    def _apply_ry(self, state: torch.Tensor, qubit: int, angle: torch.Tensor) -> torch.Tensor:
        idx0, idx1 = self._basis_pair_indices(qubit, state.device)
        a0 = state[:, idx0]
        a1 = state[:, idx1]
        cos = torch.cos(angle / 2.0).to(state.dtype)
        sin = torch.sin(angle / 2.0).to(state.dtype)
        output = state.clone()
        output[:, idx0] = cos * a0 - sin * a1
        output[:, idx1] = sin * a0 + cos * a1
        return output

    def _basis_pair_indices(
        self, qubit: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        basis = torch.arange(self.dimension, device=device, dtype=torch.long)
        idx0 = basis[(basis & (1 << qubit)) == 0]
        idx1 = idx0 | (1 << qubit)
        return idx0, idx1

    def _build_z_eigenvalues(self) -> torch.Tensor:
        basis = torch.arange(self.dimension, dtype=torch.long).unsqueeze(1)
        bit_positions = torch.arange(self.n_qubits, dtype=torch.long).unsqueeze(0)
        bits = (basis >> bit_positions) & 1
        return (1.0 - 2.0 * bits.to(torch.float32))


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


def save_model(path: str | Path, model: torch.nn.Module, family: str, config: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    architecture = config.get("quantum_architecture", "zz_linear")
    torch.save(
        {
            "family": family,
            "architecture": architecture,
            "n_qubits": int(getattr(model, "n_qubits")),
            "vqc_layers": int(getattr(model, "layers", config.get("vqc_layers", 0))),
            "state_dict": model.state_dict(),
            "config": {
                "pca_components": config["pca_components"],
                "circuit_reps": config["circuit_reps"],
                "quantum_architecture": architecture,
                "vqc_layers": config.get("vqc_layers"),
            },
        },
        target,
    )


def load_model(
    path: str | Path,
    device: torch.device,
    expected_architecture: str | None = None,
) -> torch.nn.Module:
    payload = torch.load(path, map_location=device)
    architecture = payload.get("architecture", "zz_linear")
    if expected_architecture is not None and architecture != expected_architecture:
        raise ValueError(
            f"Model checkpoint at {path} was trained as '{architecture}', "
            f"but config requests '{expected_architecture}'. Run train.py after "
            "changing quantum_architecture."
        )
    if architecture == "vqc":
        model = VQCScorer(int(payload["n_qubits"]), int(payload["vqc_layers"])).to(device)
    else:
        model = QuantumScorer(int(payload["n_qubits"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
