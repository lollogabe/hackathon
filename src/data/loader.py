from __future__ import annotations

from pathlib import Path

import pandas as pd


MODEL_INPUT_COLUMNS = [
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "duration",
    "packets_per_second",
    "bytes_per_second",
    "inter_packet_arrival_mean",
    "inter_packet_arrival_std",
    "total_packets",
    "total_bytes",
    "packet_size_avg",
    "packet_size_std",
    "outbound_byte_ratio",
]

AUDIT_COLUMNS = [
    "row_in_window",
    "is_seeded_ddos",
    "burst_id",
    "burst_phase",
    "scenario",
    "split",
    "dataset_id",
    "source_dataset",
]

OPTIONAL_AUDIT_COLUMNS = ["Label", "Attack"]
REQUIRED_COLUMNS = MODEL_INPUT_COLUMNS + AUDIT_COLUMNS


def load_flow_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    frame = pd.read_csv(csv_path, low_memory=False)
    validate_flow_schema(frame, csv_path)
    return frame


def validate_flow_schema(frame: pd.DataFrame, path: str | Path | None = None) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        location = f" in {path}" if path is not None else ""
        raise ValueError(f"Missing required column(s){location}: {', '.join(missing)}")
