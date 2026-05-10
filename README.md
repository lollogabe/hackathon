# DDoS Detection with Quantum-Inspired Machine Learning

This project detects DDoS bursts in synthetic 15-minute network-flow captures using a fixed quantum-inspired ZZ feature map followed by a small CUDA-aware linear scoring layer. It trains only on `(src_ip, window_id)` instance labels derived from `is_seeded_ddos`, then derives window-level alerts and malicious IP shortlists from the maximum instance score in each temporal window.

## Requirements

Use Python 3.11 or newer. For an Intel Mac or CPU-only environment, create the conda environment with conda-forge packages:

```bash
conda env create -f environment_cpu.yml
conda activate ddos-quantum-cpu
```

The conda environment installs the `pytorch` package from conda-forge, not a package named `torch`. For pip-based CUDA environments, use:

```bash
pip install -r requirements.txt
```

CUDA is used automatically when `torch.cuda.is_available()` is true and `config.json` has `"device": "auto"`. The Qiskit Aer backend is supported through `qiskit-aer`; when it is unavailable and `quantum_backend` is `"auto"`, the project uses an exact NumPy statevector fallback with the same observables.

## Dataset Setup

Place the datasets under `Datasets/` with this reduced-schema structure:

```text
Datasets/Option 1/option1_nf_unsw_dos_as_ddos_reduced_schema/{attack,normal}/{train,validation,test}/*.csv
Datasets/Option 2/option2_nf_unsw_base_cse_native_ddos_reduced_schema/{attack,normal}/{train,validation,test}/*.csv
```

Verify the catalog before training:

```bash
python train.py --dry-run
```

The dry run scans both families and prints counts by family, split, and scenario.

## Quick Start

```bash
python train.py          # trains both families, saves models and quantum feature caches
python evaluate.py       # evaluates on val and test, saves results JSON and plots
python benchmarks.py     # runs all four baselines, appends to results JSON
```

## Config Reference

All runtime paths and hyperparameters live in `config.json`. See `CONFIG_GUIDE.md` for every field. The most important knobs are `families`, `window_width`, `pca_components`, `circuit_reps`, `batch_size`, `learning_rate`, `device`, and `quantum_backend`.

## Pipeline Overview

| Stage | Input | Output |
| --- | --- | --- |
| 0 | Raw 100,000-row CSV | Pandas DataFrame with audit columns retained |
| 1 | `row_in_window`, `window_width` | `window_id = row_in_window // W` |
| 2 | `(src_ip, window_id, is_seeded_ddos)` | Instance and window labels for training/evaluation |
| 3 | Flow rows grouped by `(src_ip, window_id)` | 33 numeric engineered features |
| 4 | Feature matrix | Log-scaled heavy-tail columns |
| 5 | Log-scaled matrix | Standard-scaled matrix |
| 6 | Scaled matrix | PCA matrix with `k = pca_components` |
| 7 | PCA matrix | Angles in `(-pi, pi)` |
| 8 | Angles | Quantum expectation vector of size `2k - 1` |
| 9 | Quantum features | Linear logits and probabilities |
| 10 | Probabilities and threshold | IP alerts, window alerts, dataset labels |

For the full shape and semantic contract, read `PIPELINE.md`.

## Outputs

The project creates `outputs/` automatically. Important files:

- `preprocessor_{family}.pkl`: fitted log scaling, StandardScaler, PCA, and angle pipeline.
- `best_model_{family}.pt`: best linear scoring layer weights.
- `quantum_features_{family}_{split}.pt`: cached quantum feature tensors.
- `threshold_{family}.json`: validation-calibrated threshold and validation F1.
- `results_{family}.json`: quantum metrics and appended baseline metrics.
- `plots/training_curves_{family}.png`: train/validation loss and F1 curves.
- `plots/temporal_{family}_{dataset_name}.png`: test attack temporal detection plots.
- `infer_results.csv`: latest online inference summary.
- `plots/infer_scores_{family}_{input_name}.png`: human-readable inference score plot.

For presentation-ready result artifacts, run:

```bash
python report_results.py
```

This writes slide-friendly comparison charts, confusion-matrix plots, metric heatmaps, threshold charts, and CSV/Markdown tables to `outputs/plots/` and `outputs/tables/`.

For speaker notes and artifact-by-artifact interpretation, see `PRESENTATION_EXPLANATIONS.md`.

## Benchmarking

`benchmarks.py` appends four baselines under the `benchmarks` key in `outputs/results_{family}.json`:

- Majority class: predicts all instances as benign.
- Classical window RBF: StandardScaler, RBFSampler, balanced LogisticRegression on the same instance features.
- PCA + Linear ablation: StandardScaler, PCA, CUDA linear layer trained with Adam and weighted BCE.
- Classical bag-level reference: whole-dataset per-IP aggregation with RBF LogisticRegression.

Each baseline reports instance, window, dataset, and detection-latency metrics where applicable.

## Online Inference

Run the streaming-style detector on a new CSV:

```bash
python infer.py --family family_a --input path/to/new.csv --config config.json
```

The script reads chunks of `window_width` rows, computes per-IP features once each window closes, scores every active source IP, prints one line per window, and writes `outputs/infer_results.csv` with `window_id`, `max_score`, `predicted_label`, and `malicious_ips`.

Example output:

```text
window=0  score=0.023  label=NORMAL  ips=[]
window=7  score=0.891  label=ATTACK  ips=[10.1.2.3, 10.1.2.4]
```

## Limitations and Future Work

The current `k = 6` to `8` regime is constrained by exact statevector simulation. On a fault-tolerant quantum computer with `k = d = 33` qubits, the ZZ kernel would not be efficiently classically simulable in the same way. Detection performance on stealthier, lower-rate, or more distributed attacks may require threshold recalibration and possibly additional causal features.
