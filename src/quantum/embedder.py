from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class _QiskitObjects:
    simulator: object
    circuit: object
    parameters: object


class QuantumEmbedder:
    """Fixed ZZFeatureMap embedder with exact statevector expectation values."""

    def __init__(
        self,
        n_qubits: int,
        reps: int,
        device: torch.device,
        backend: str = "auto",
        batch_size: int = 128,
    ) -> None:
        self.n_qubits = n_qubits
        self.reps = reps
        self.device = device
        self.backend = backend
        self.batch_size = batch_size
        self.dimension = 2**n_qubits
        self._z_eigs = self._build_z_eigenvalues()
        self._zz_eigs = self._z_eigs[:, :-1] * self._z_eigs[:, 1:]
        self._hadamard_all = self._build_hadamard_all()
        self._qiskit: _QiskitObjects | None = None
        self._active_backend = self._select_backend(backend)
        logger.info("Quantum embedder using %s backend.", self._active_backend)

    @property
    def output_dim(self) -> int:
        return 2 * self.n_qubits - 1

    def embed(self, angles: np.ndarray, show_progress: bool = False) -> torch.Tensor:
        theta = np.asarray(angles, dtype="float64")
        if theta.ndim != 2 or theta.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected angle matrix with shape (n, {self.n_qubits}), got {theta.shape}."
            )
        batches = range(0, theta.shape[0], self.batch_size)
        if show_progress:
            batches = tqdm(list(batches), desc="Quantum embedding")
        outputs: list[np.ndarray] = []
        for start in batches:
            batch = theta[start : start + self.batch_size]
            if self._active_backend == "qiskit":
                outputs.append(self._embed_qiskit_batch(batch))
            else:
                outputs.append(self._embed_numpy_batch(batch))
        if outputs:
            matrix = np.vstack(outputs).astype("float32")
        else:
            matrix = np.empty((0, self.output_dim), dtype="float32")
        return torch.as_tensor(matrix, dtype=torch.float32, device=self.device)

    def statevectors(self, angles: np.ndarray, show_progress: bool = False) -> np.ndarray:
        theta = np.asarray(angles, dtype="float64")
        if theta.ndim != 2 or theta.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected angle matrix with shape (n, {self.n_qubits}), got {theta.shape}."
            )
        batches = range(0, theta.shape[0], self.batch_size)
        if show_progress:
            batches = tqdm(list(batches), desc="Quantum statevectors")
        outputs: list[np.ndarray] = []
        for start in batches:
            batch = theta[start : start + self.batch_size]
            if self._active_backend == "qiskit":
                outputs.append(self._statevectors_qiskit_batch(batch))
            else:
                outputs.append(self._statevectors_numpy_batch(batch))
        if outputs:
            return np.vstack(outputs).astype("complex64")
        return np.empty((0, self.dimension), dtype="complex64")

    def _select_backend(self, requested: str) -> str:
        if requested == "numpy":
            return "numpy"
        try:
            self._qiskit = self._build_qiskit_objects()
            return "qiskit"
        except Exception as exc:
            if requested == "qiskit":
                raise RuntimeError("Qiskit backend requested but unavailable.") from exc
            logger.warning("Qiskit backend unavailable; using NumPy statevector fallback.")
            return "numpy"

    def _build_z_eigenvalues(self) -> np.ndarray:
        basis = np.arange(self.dimension, dtype=np.int64)[:, None]
        bit_positions = np.arange(self.n_qubits, dtype=np.int64)[None, :]
        bits = (basis >> bit_positions) & 1
        return (1.0 - 2.0 * bits).astype("float64")

    def _build_hadamard_all(self) -> np.ndarray:
        single = np.array([[1.0, 1.0], [1.0, -1.0]], dtype="complex128") / np.sqrt(2.0)
        matrix = np.array([[1.0 + 0.0j]], dtype="complex128")
        for _ in range(self.n_qubits):
            matrix = np.kron(matrix, single)
        return matrix

    def _statevectors_numpy_batch(self, theta: np.ndarray) -> np.ndarray:
        state = np.zeros((theta.shape[0], self.dimension), dtype="complex128")
        state[:, 0] = 1.0 + 0.0j
        hadamard_t = self._hadamard_all.T
        z_t = self._z_eigs.T
        zz_t = self._zz_eigs.T
        for _ in range(self.reps):
            state = state @ hadamard_t
            single_phase = theta @ z_t
            if self.n_qubits > 1:
                pair_angles = theta[:, :-1] * theta[:, 1:]
                pair_phase = pair_angles @ zz_t
            else:
                pair_phase = 0.0
            phase = 0.5 * (single_phase + pair_phase)
            state *= np.exp(-1j * phase)
        return state

    def _embed_numpy_batch(self, theta: np.ndarray) -> np.ndarray:
        state = self._statevectors_numpy_batch(theta)
        probabilities = np.abs(state) ** 2
        z_expectations = probabilities @ self._z_eigs
        if self.n_qubits > 1:
            zz_expectations = probabilities @ self._zz_eigs
            return np.concatenate([z_expectations, zz_expectations], axis=1)
        return z_expectations

    def _build_qiskit_objects(self) -> _QiskitObjects:
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import ParameterVector
        from qiskit_aer import AerSimulator

        parameters = ParameterVector("theta", self.n_qubits)
        circuit = QuantumCircuit(self.n_qubits)
        for _ in range(self.reps):
            circuit.h(range(self.n_qubits))
            for qubit in range(self.n_qubits):
                circuit.rz(parameters[qubit], qubit)
            for qubit in range(self.n_qubits - 1):
                circuit.rzz(parameters[qubit] * parameters[qubit + 1], qubit, qubit + 1)
        circuit.save_statevector()
        simulator = AerSimulator(method="statevector")
        circuit = transpile(circuit, simulator)
        return _QiskitObjects(simulator=simulator, circuit=circuit, parameters=parameters)

    def _embed_qiskit_batch(self, theta: np.ndarray) -> np.ndarray:
        statevectors = self._statevectors_qiskit_batch(theta)
        probabilities = np.abs(statevectors) ** 2
        z_expectations = probabilities @ self._z_eigs
        if self.n_qubits > 1:
            zz_expectations = probabilities @ self._zz_eigs
            return np.concatenate([z_expectations, zz_expectations], axis=1)
        return z_expectations

    def _statevectors_qiskit_batch(self, theta: np.ndarray) -> np.ndarray:
        if self._qiskit is None:
            raise RuntimeError("Qiskit backend was not initialised.")
        outputs = []
        for row in theta:
            bind = {
                self._qiskit.parameters[index]: float(value)
                for index, value in enumerate(row)
            }
            bound = self._qiskit.circuit.assign_parameters(bind, inplace=False)
            result = self._qiskit.simulator.run(bound).result()
            statevector = np.asarray(result.get_statevector(bound), dtype="complex128")
            outputs.append(statevector)
        return np.vstack(outputs).astype("complex128")
