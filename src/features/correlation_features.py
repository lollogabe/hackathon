from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.schema import CORRELATION_FEATURE_COLUMNS


def _entropy(counts: pd.Series, epsilon: float) -> float:
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probabilities = counts.astype("float64") / total
    return float(-(probabilities * np.log(probabilities + epsilon)).sum())


def add_correlation_features(
    flow_frame: pd.DataFrame, base_features: pd.DataFrame, config: dict
) -> pd.DataFrame:
    required_flows = {"window_id", "src_ip", "dst_ip", "dst_port"}
    missing_flows = required_flows - set(flow_frame.columns)
    if missing_flows:
        raise ValueError(
            "Cannot compute correlation features; missing flow columns: "
            + ", ".join(sorted(missing_flows))
        )
    required_base = {"src_ip", "window_id", "top_dst_ip", "top_dst_port"}
    missing_base = required_base - set(base_features.columns)
    if missing_base:
        raise ValueError(
            "Cannot compute correlation features; missing base columns: "
            + ", ".join(sorted(missing_base))
        )

    epsilon = float(config["entropy_epsilon"])
    correlation_frames: list[pd.DataFrame] = []
    for window_id, window_instances in base_features.groupby("window_id", sort=False):
        window_flows = flow_frame[flow_frame["window_id"] == window_id]
        instance_count = float(window_instances["src_ip"].nunique())
        flow_count = float(len(window_flows))

        top_dst_counts = window_instances["top_dst_ip"].value_counts(dropna=False)
        top_port_counts = window_instances["top_dst_port"].value_counts(dropna=False)
        dst_flow_counts = window_flows["dst_ip"].value_counts(dropna=False)
        dst_source_counts = window_flows.groupby("dst_ip")["src_ip"].nunique()

        dst_entropy = _entropy(dst_flow_counts, epsilon)
        port_entropy = _entropy(window_flows["dst_port"].value_counts(dropna=False), epsilon)

        records = []
        for row in window_instances.itertuples(index=False):
            top_dst = row.top_dst_ip
            top_port = row.top_dst_port
            dst_flow_share = (
                float(dst_flow_counts.get(top_dst, 0.0)) / flow_count
                if flow_count > 0.0
                else 0.0
            )
            dst_source_share = (
                float(dst_source_counts.get(top_dst, 0.0)) / instance_count
                if instance_count > 0.0
                else 0.0
            )
            records.append(
                {
                    "src_ip": row.src_ip,
                    "window_id": int(window_id),
                    "same_dst_ip_source_count": float(top_dst_counts.get(top_dst, 0.0)),
                    "same_dst_port_source_count": float(top_port_counts.get(top_port, 0.0)),
                    "dst_ip_global_flow_share": dst_flow_share,
                    "dst_ip_global_source_share": dst_source_share,
                    "co_targeting_score": dst_flow_share * dst_source_share,
                    "window_dst_entropy": dst_entropy,
                    "window_port_entropy": port_entropy,
                }
            )
        correlation_frames.append(pd.DataFrame.from_records(records))

    if correlation_frames:
        correlations = pd.concat(correlation_frames, ignore_index=True)
    else:
        correlations = pd.DataFrame(
            columns=["src_ip", "window_id"] + CORRELATION_FEATURE_COLUMNS
        )
    output = base_features.merge(correlations, on=["src_ip", "window_id"], how="left")
    for column in CORRELATION_FEATURE_COLUMNS:
        output[column] = output[column].fillna(0.0).astype("float64")
    return output
