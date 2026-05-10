# Configuration Guide

All runtime settings live in `config.json` at the project root. The loader validates every field at startup and creates `outputs/` plus `outputs/plots/` automatically.

## Required Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `seed` | int | Seed for `random`, NumPy, PyTorch, and CUDA. |
| `data_root` | string | Root directory containing `Option 1/` and `Option 2/`. |
| `output_root` | string | Directory for models, caches, JSON results, and plots. |
| `families` | list | Families to run, drawn from `family_a` and `family_b`. |
| `family_paths` | object | Relative reduced-schema paths for each family under `data_root`. |
| `window_width` | int | Number of rows per tumbling window. Recommended: `6667`. |
| `pca_components` | int | PCA output dimension and number of qubits `k`. |
| `circuit_reps` | int | Repetitions of the ZZ feature-map circuit. |
| `batch_size` | int | PyTorch training batch size for cached quantum features. |
| `quantum_batch_size` | int | Number of angle vectors embedded per quantum batch. |
| `learning_rate` | number | Adam learning rate for the quantum scorer and PCA-linear baseline. |
| `weight_decay` | number | Adam weight decay. |
| `max_epochs` | int | Maximum training epochs. |
| `early_stopping_patience` | int | Epochs without validation F1 improvement before stopping. |
| `scheduler_patience` | int | Patience for `ReduceLROnPlateau`. |
| `inference_threshold` | number | Fallback probability threshold in `[0, 1]`. |
| `use_calibrated_threshold` | bool | If true, load `threshold_{family}.json` after training/evaluation. |
| `threshold_sweep_min` | number | Minimum validation threshold for F1 calibration. |
| `threshold_sweep_max` | number | Maximum validation threshold for F1 calibration. |
| `threshold_sweep_step` | number | Step size for threshold calibration. |
| `log_level` | string | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |
| `device` | string | `auto`, `cuda`, or `cpu`. |
| `quantum_backend` | string | `auto`, `qiskit`, or `numpy`. |
| `pca_variance_warning_threshold` | number | Warning threshold for cumulative PCA explained variance. |
| `small_packet_threshold` | number | Packet-size cutoff for `small_packet_share`. |
| `high_pps_quantile` | number | Per-window quantile for `high_pps_share`. |
| `low_outbound_threshold` | number | Outbound-ratio cutoff for `low_outbound_share`. |
| `entropy_epsilon` | number | Numerical epsilon in entropy terms. |
| `rbf_gamma` | number | Gamma for the classical RBF baseline. |
| `rbf_components` | int | Number of random Fourier features in the RBF baseline. |
| `logistic_regression_max_iter` | int | Max iterations for sklearn LogisticRegression baselines. |
| `num_workers` | int | PyTorch DataLoader worker count. Use `0` for CUDA tensors. |
| `reuse_quantum_cache` | bool | If true, reuse existing `quantum_features_{family}_{split}.pt` files. |

## Device Selection

- `auto`: uses CUDA when available, otherwise CPU.
- `cuda`: requires CUDA and raises an error if unavailable.
- `cpu`: always uses CPU.

All PyTorch model parameters, labels, cached training tensors, and scoring tensors are moved to the resolved device.

## Quantum Backend

- `auto`: tries Qiskit Aer first, then falls back to the exact NumPy statevector backend.
- `qiskit`: requires `qiskit-aer` and raises an error if unavailable.
- `numpy`: uses the built-in exact statevector implementation.

The NumPy backend is useful for CPU-only tests and Intel Mac development. It computes the same Z and adjacent-ZZ expectation values and returns PyTorch tensors on the resolved device.

## Recommended Local Mac Setup

```bash
conda env create -f environment_cpu.yml
conda activate ddos-quantum-cpu
python train.py --dry-run
pytest -q
```

`environment_cpu.yml` uses conda-forge and installs `pytorch` by package name.
