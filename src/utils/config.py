from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch


REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "seed": int,
    "data_root": str,
    "output_root": str,
    "families": list,
    "family_paths": dict,
    "window_width": int,
    "pca_components": int,
    "circuit_reps": int,
    "batch_size": int,
    "quantum_batch_size": int,
    "learning_rate": (int, float),
    "weight_decay": (int, float),
    "max_epochs": int,
    "early_stopping_patience": int,
    "scheduler_patience": int,
    "inference_threshold": (int, float),
    "use_calibrated_threshold": bool,
    "threshold_sweep_min": (int, float),
    "threshold_sweep_max": (int, float),
    "threshold_sweep_step": (int, float),
    "log_level": str,
    "device": str,
    "quantum_backend": str,
    "pca_variance_warning_threshold": (int, float),
    "small_packet_threshold": (int, float),
    "high_pps_quantile": (int, float),
    "low_outbound_threshold": (int, float),
    "entropy_epsilon": (int, float),
    "rbf_gamma": (int, float),
    "rbf_components": int,
    "logistic_regression_max_iter": int,
    "num_workers": int,
    "reuse_quantum_cache": bool,
}

ALLOWED_FAMILIES = {"family_a", "family_b"}
ALLOWED_DEVICE_MODES = {"auto", "cuda", "cpu"}
ALLOWED_QUANTUM_BACKENDS = {"auto", "qiskit", "numpy"}
ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def load_config(path: str | Path = "config.json") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Missing required config field(s): {', '.join(missing)}")

    for field, expected_type in REQUIRED_FIELDS.items():
        if not isinstance(config[field], expected_type):
            raise TypeError(
                f"Config field '{field}' must be {expected_type}, got "
                f"{type(config[field]).__name__}"
            )

    families = config["families"]
    if not families:
        raise ValueError("Config field 'families' must contain at least one family.")
    unknown_families = sorted(set(families) - ALLOWED_FAMILIES)
    if unknown_families:
        raise ValueError(f"Unknown family name(s): {', '.join(unknown_families)}")
    missing_paths = [family for family in families if family not in config["family_paths"]]
    if missing_paths:
        raise ValueError(
            "Config field 'family_paths' is missing path(s) for: "
            + ", ".join(missing_paths)
        )

    if config["device"] not in ALLOWED_DEVICE_MODES:
        raise ValueError(f"device must be one of {sorted(ALLOWED_DEVICE_MODES)}")
    if config["quantum_backend"] not in ALLOWED_QUANTUM_BACKENDS:
        raise ValueError(
            f"quantum_backend must be one of {sorted(ALLOWED_QUANTUM_BACKENDS)}"
        )
    if config["log_level"].upper() not in ALLOWED_LOG_LEVELS:
        raise ValueError(f"log_level must be one of {sorted(ALLOWED_LOG_LEVELS)}")

    positive_integer_fields = [
        "window_width",
        "pca_components",
        "circuit_reps",
        "batch_size",
        "quantum_batch_size",
        "max_epochs",
        "early_stopping_patience",
        "scheduler_patience",
        "rbf_components",
        "logistic_regression_max_iter",
    ]
    for field in positive_integer_fields:
        if config[field] <= 0:
            raise ValueError(f"Config field '{field}' must be positive.")

    if config["num_workers"] < 0:
        raise ValueError("Config field 'num_workers' cannot be negative.")
    if not 0.0 <= float(config["inference_threshold"]) <= 1.0:
        raise ValueError("inference_threshold must be in [0, 1].")
    if not 0.0 < float(config["high_pps_quantile"]) < 1.0:
        raise ValueError("high_pps_quantile must be in (0, 1).")
    if float(config["threshold_sweep_min"]) <= 0.0:
        raise ValueError("threshold_sweep_min must be positive.")
    if float(config["threshold_sweep_max"]) >= 1.0:
        raise ValueError("threshold_sweep_max must be less than 1.")
    if float(config["threshold_sweep_min"]) >= float(config["threshold_sweep_max"]):
        raise ValueError("threshold_sweep_min must be less than threshold_sweep_max.")
    if float(config["threshold_sweep_step"]) <= 0.0:
        raise ValueError("threshold_sweep_step must be positive.")
    if float(config["entropy_epsilon"]) <= 0.0:
        raise ValueError("entropy_epsilon must be positive.")


def configure_logging(config: dict[str, Any]) -> None:
    logging.basicConfig(
        level=getattr(logging, config["log_level"].upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = config["device"]
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is false.")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if cuda_available else "cpu")


def ensure_output_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    output_root = Path(config["output_root"])
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return output_root, plots_dir
