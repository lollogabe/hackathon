from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.windowing import add_window_id, derive_instance_labels, derive_window_labels
from src.features.pipeline import compute_instance_features
from src.features.schema import FEATURE_COLUMNS
from src.model.scorer import calibrate_threshold
from src.model.scorer import VQCScorer
from src.preprocessing.preprocessor import Preprocessor
from src.quantum.embedder import QuantumEmbedder
from src.utils.config import validate_config


def test_config_validation_accepts_project_config():
    config = _config()
    validate_config(config)


def test_windowing_labels_and_feature_contract():
    config = _config()
    frame = add_window_id(_flow_frame(), config["window_width"])
    instance_labels = derive_instance_labels(frame)
    window_labels = derive_window_labels(frame)
    features = compute_instance_features(frame, config)

    assert set(instance_labels.columns) == {"src_ip", "window_id", "y_instance"}
    assert set(window_labels.columns) == {"window_id", "y_window"}
    assert features[FEATURE_COLUMNS].shape[1] == 33
    assert list(features[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
    assert not features[FEATURE_COLUMNS].isna().any().any()

    ip_a_w0 = features[(features["src_ip"] == "10.0.0.1") & (features["window_id"] == 0)]
    assert float(ip_a_w0["flow_count"].iloc[0]) == 2.0
    assert float(ip_a_w0["unique_dst_ip"].iloc[0]) == 1.0
    assert float(ip_a_w0["tcp_share"].iloc[0]) == 0.5
    assert float(ip_a_w0["udp_share"].iloc[0]) == 0.5

    label_a_w0 = instance_labels[
        (instance_labels["src_ip"] == "10.0.0.1") & (instance_labels["window_id"] == 0)
    ]["y_instance"].iloc[0]
    assert int(label_a_w0) == 1
    assert int(window_labels[window_labels["window_id"] == 0]["y_window"].iloc[0]) == 1


def test_preprocessor_save_load_and_angle_shape(tmp_path):
    config = _config()
    features = _feature_matrix(12)
    preprocessor = Preprocessor(
        n_components=config["pca_components"],
        variance_warning_threshold=config["pca_variance_warning_threshold"],
    ).fit(features)
    angles = preprocessor.transform(features)
    assert angles.shape == (12, config["pca_components"])
    assert np.all(angles < np.pi)
    assert np.all(angles > -np.pi)

    path = tmp_path / "preprocessor.pkl"
    preprocessor.save(path)
    loaded = Preprocessor.load(path)
    np.testing.assert_allclose(angles, loaded.transform(features))


def test_numpy_quantum_embedder_returns_device_tensor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    angles = np.array(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.5, -0.6],
        ],
        dtype="float64",
    )
    embedder = QuantumEmbedder(
        n_qubits=3,
        reps=2,
        device=device,
        backend="numpy",
        batch_size=2,
    )
    quantum_features = embedder.embed(angles)
    assert quantum_features.shape == (2, 5)
    assert quantum_features.device.type == device.type
    assert torch.all(quantum_features <= 1.00001)
    assert torch.all(quantum_features >= -1.00001)


def test_quantum_embedder_statevectors_are_normalized():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    angles = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype="float64")
    embedder = QuantumEmbedder(
        n_qubits=3,
        reps=2,
        device=device,
        backend="numpy",
        batch_size=2,
    )
    states = embedder.statevectors(angles)
    assert states.shape == (2, 8)
    np.testing.assert_allclose(np.sum(np.abs(states) ** 2, axis=1), np.ones(2), atol=1e-5)


def test_vqc_scorer_forward_backward_on_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQCScorer(n_qubits=3, layers=2).to(device)
    angles = torch.randn(4, 3, dtype=torch.float32, device=device)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32, device=device)
    logits = model(angles)
    assert logits.shape == (4,)
    loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_threshold_calibration_selects_useful_threshold():
    probabilities = np.array([0.05, 0.1, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1])
    threshold, f1 = calibrate_threshold(probabilities, labels, 0.01, 0.99, 0.01)
    assert 0.1 < threshold <= 0.7
    assert f1 == 1.0


def _config() -> dict:
    return {
        "seed": 42,
        "data_root": "Datasets",
        "output_root": "outputs",
        "families": ["family_a"],
        "family_paths": {
            "family_a": "Option 1/option1_nf_unsw_dos_as_ddos_reduced_schema"
        },
        "window_width": 3,
        "pca_components": 3,
        "circuit_reps": 2,
        "batch_size": 2,
        "quantum_batch_size": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 2,
        "early_stopping_patience": 2,
        "scheduler_patience": 1,
        "inference_threshold": 0.5,
        "use_calibrated_threshold": True,
        "threshold_sweep_min": 0.01,
        "threshold_sweep_max": 0.99,
        "threshold_sweep_step": 0.01,
        "log_level": "INFO",
        "device": "auto",
        "quantum_backend": "numpy",
        "quantum_architecture": "zz_linear",
        "vqc_layers": 2,
        "quantum_kernel_c": 1.0,
        "quantum_kernel_max_train_instances": 0,
        "pca_variance_warning_threshold": 0.85,
        "small_packet_threshold": 200.0,
        "high_pps_quantile": 0.95,
        "low_outbound_threshold": 0.2,
        "entropy_epsilon": 1e-12,
        "rbf_gamma": 0.5,
        "rbf_components": 16,
        "logistic_regression_max_iter": 100,
        "num_workers": 0,
        "reuse_quantum_cache": False,
        "report_detail_level": "presentation",
        "include_individual_confusion_plots": False,
    }


def _flow_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "src_ip": ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.2", "10.0.0.3"],
            "dst_ip": ["10.1.0.1", "10.1.0.1", "10.1.0.1", "10.1.0.2", "10.1.0.2"],
            "src_port": [1000, 1001, 1002, 1003, 1004],
            "dst_port": [80, 80, 80, 443, 443],
            "protocol": ["TCP", "UDP", "TCP", "ICMP", "UDP"],
            "duration": [1.0, 2.0, 1.5, 3.0, 0.5],
            "packets_per_second": [10.0, 200.0, 30.0, 40.0, 50.0],
            "bytes_per_second": [1000.0, 2000.0, 1500.0, 500.0, 900.0],
            "inter_packet_arrival_mean": [0.1, 0.2, 0.3, 0.4, 0.5],
            "inter_packet_arrival_std": [0.01, 0.02, 0.03, 0.04, 0.05],
            "total_packets": [10, 20, 30, 40, 50],
            "total_bytes": [1000, 2000, 3000, 4000, 5000],
            "packet_size_avg": [100.0, 250.0, 120.0, 130.0, 140.0],
            "packet_size_std": [5.0, 6.0, 7.0, 8.0, 9.0],
            "outbound_byte_ratio": [0.1, 0.3, 0.2, 0.9, 0.8],
            "Label": [0, 1, 0, 0, 0],
            "Attack": [0, 1, 0, 0, 0],
            "scenario": ["attack"] * 5,
            "split": ["train"] * 5,
            "dataset_id": ["toy"] * 5,
            "row_in_window": [0, 1, 2, 3, 4],
            "is_seeded_ddos": [0, 1, 0, 0, 0],
            "burst_id": [np.nan, "b1", np.nan, np.nan, np.nan],
            "burst_phase": [np.nan, "ramp-up", np.nan, np.nan, np.nan],
            "source_dataset": ["toy"] * 5,
        }
    )


def _feature_matrix(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    values = rng.uniform(0.0, 10.0, size=(rows, len(FEATURE_COLUMNS)))
    frame = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    for column in [
        "dst_concentration",
        "dst_port_concentration",
        "tcp_share",
        "udp_share",
        "icmp_share",
        "small_packet_share",
        "high_pps_share",
        "low_outbound_share",
        "outbound_ratio_mean",
        "outbound_ratio_min",
        "dst_ip_global_flow_share",
        "dst_ip_global_source_share",
        "co_targeting_score",
    ]:
        frame[column] = rng.uniform(0.0, 1.0, size=rows)
    return frame
