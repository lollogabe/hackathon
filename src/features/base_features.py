from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.schema import BASE_FEATURE_COLUMNS


def _modal_value(series: pd.Series):
    counts = series.value_counts(dropna=False)
    if counts.empty:
        return np.nan
    return counts.index[0]


def _modal_share(series: pd.Series) -> float:
    counts = series.value_counts(dropna=False)
    if counts.empty:
        return 0.0
    return float(counts.iloc[0] / len(series))


def compute_base_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    required = {
        "src_ip",
        "dst_ip",
        "dst_port",
        "protocol",
        "window_id",
        "duration",
        "packets_per_second",
        "bytes_per_second",
        "inter_packet_arrival_mean",
        "inter_packet_arrival_std",
        "total_packets",
        "total_bytes",
        "packet_size_avg",
        "outbound_byte_ratio",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Cannot compute base features; missing columns: " + ", ".join(sorted(missing))
        )

    work = frame.copy()
    work["protocol_norm"] = work["protocol"].astype(str).str.upper()
    work["is_tcp"] = work["protocol_norm"].eq("TCP").astype(float)
    work["is_udp"] = work["protocol_norm"].eq("UDP").astype(float)
    work["is_icmp"] = work["protocol_norm"].eq("ICMP").astype(float)
    work["small_packet"] = (
        work["packet_size_avg"].astype(float) <= float(config["small_packet_threshold"])
    ).astype(float)
    pps_thresholds = work.groupby("window_id")["packets_per_second"].transform(
        lambda values: values.quantile(float(config["high_pps_quantile"]))
    )
    work["high_pps"] = (
        work["packets_per_second"].astype(float) >= pps_thresholds.astype(float)
    ).astype(float)
    work["low_outbound"] = (
        work["outbound_byte_ratio"].astype(float) <= float(config["low_outbound_threshold"])
    ).astype(float)

    grouped = work.groupby(["src_ip", "window_id"], sort=False, dropna=False)
    features = grouped.agg(
        flow_count=("src_ip", "size"),
        total_packets_sum=("total_packets", "sum"),
        total_bytes_sum=("total_bytes", "sum"),
        pps_mean=("packets_per_second", "mean"),
        pps_max=("packets_per_second", "max"),
        bps_mean=("bytes_per_second", "mean"),
        bps_max=("bytes_per_second", "max"),
        duration_mean=("duration", "mean"),
        duration_max=("duration", "max"),
        packet_size_avg_mean=("packet_size_avg", "mean"),
        packet_size_avg_std=("packet_size_avg", lambda values: values.std(ddof=0)),
        inter_arrival_mean_mean=("inter_packet_arrival_mean", "mean"),
        inter_arrival_std_mean=("inter_packet_arrival_std", "mean"),
        outbound_ratio_mean=("outbound_byte_ratio", "mean"),
        outbound_ratio_min=("outbound_byte_ratio", "min"),
        unique_dst_ip=("dst_ip", "nunique"),
        unique_dst_port=("dst_port", "nunique"),
        unique_protocols=("protocol_norm", "nunique"),
        dst_concentration=("dst_ip", _modal_share),
        dst_port_concentration=("dst_port", _modal_share),
        tcp_share=("is_tcp", "mean"),
        udp_share=("is_udp", "mean"),
        icmp_share=("is_icmp", "mean"),
        small_packet_share=("small_packet", "mean"),
        high_pps_share=("high_pps", "mean"),
        low_outbound_share=("low_outbound", "mean"),
        top_dst_ip=("dst_ip", _modal_value),
        top_dst_port=("dst_port", _modal_value),
        top_protocol=("protocol_norm", _modal_value),
    )
    features = features.reset_index()
    features["packet_size_avg_std"] = features["packet_size_avg_std"].fillna(0.0)
    for column in BASE_FEATURE_COLUMNS:
        features[column] = features[column].astype("float64")
    return features
