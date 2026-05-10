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
| `quantum_architecture` | string | Quantum model architecture: `zz_linear`, `vqc`, or `quantum_kernel`. |
| `vqc_layers` | int | Number of trainable VQC layers when `quantum_architecture` is `vqc`. |
| `quantum_kernel_c` | number | SVC regularization strength when using `quantum_kernel`. |
| `quantum_kernel_max_train_instances` | int | Maximum training instances for the pure quantum-kernel Gram matrix; `0` means use all. |
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
| `report_detail_level` | string | `presentation` creates compact presentation plots; `full` also emits detailed diagnostic plots. |
| `include_individual_confusion_plots` | bool | If true, emit one confusion-matrix PNG per method/split/level. |

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

## Quantum Architecture

`quantum_architecture` controls the model trained by `train.py`, evaluated by `evaluate.py`, and used by `infer.py`.

| Value | Architecture | Saved model |
| --- | --- | --- |
| `zz_linear` | Fixed ZZ feature map -> Z/ZZ expectation values -> CUDA `nn.Linear` scorer. This is the original pipeline. | `outputs/best_model_{family}.pt` |
| `vqc` | Differentiable trainable statevector circuit in PyTorch. It uses data re-uploading, trainable `RY` rotations, trainable adjacent `RZZ` phases, Z/ZZ readout, and a CUDA linear scorer. | `outputs/best_model_{family}.pt` |
| `quantum_kernel` | Pure fixed quantum kernel: exact statevectors from the ZZ map, kernel `K(x, y) = |<phi(x)|phi(y)>|^2`, and a balanced precomputed-kernel SVC. | `outputs/best_model_{family}.joblib` |

Notes:

- `zz_linear` and `vqc` keep all PyTorch tensors and model parameters on the resolved device.
- `quantum_kernel` is a scikit-learn SVC over exact statevectors and therefore runs on CPU. It has no PyTorch model parameters.
- `quantum_kernel_max_train_instances` prevents the Gram matrix from becoming too large. Set it to `0` for the full training set if memory allows.

## Reporting

`report_detail_level: "presentation"` keeps the plot set compact: comparison charts, model-summary tables, heatmaps, threshold/latency plots, and summary confusion grids. It does not generate the large family of individual confusion matrices unless `include_individual_confusion_plots` is true.

For side-by-side presentation outputs for every quantum architecture, run:

```bash
python run_all_architectures.py
```

The runner does not edit `config.json`; it creates per-architecture configs in memory and writes artifacts under:

```text
outputs/architectures/zz_linear/
outputs/architectures/vqc/
outputs/architectures/quantum_kernel/
```

By default, those directories contain presentation plots only and no `tables/` directory. Add `--keep-tables` if CSV/Markdown exports are needed for a run.

Use `"full"` only when you want all diagnostic confusion-matrix PNGs.

## Recommended Local Mac Setup

```bash
conda env create -f environment_cpu.yml
conda activate ddos-quantum-cpu
python train.py --dry-run
pytest -q
```

`environment_cpu.yml` uses conda-forge and installs `pytorch` by package name.
