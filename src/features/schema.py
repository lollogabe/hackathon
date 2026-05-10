from __future__ import annotations


BASE_FEATURE_COLUMNS = [
    "flow_count",
    "total_packets_sum",
    "total_bytes_sum",
    "pps_mean",
    "pps_max",
    "bps_mean",
    "bps_max",
    "duration_mean",
    "duration_max",
    "packet_size_avg_mean",
    "packet_size_avg_std",
    "inter_arrival_mean_mean",
    "inter_arrival_std_mean",
    "outbound_ratio_mean",
    "outbound_ratio_min",
    "unique_dst_ip",
    "unique_dst_port",
    "unique_protocols",
    "dst_concentration",
    "dst_port_concentration",
    "tcp_share",
    "udp_share",
    "icmp_share",
    "small_packet_share",
    "high_pps_share",
    "low_outbound_share",
]

CORRELATION_FEATURE_COLUMNS = [
    "same_dst_ip_source_count",
    "same_dst_port_source_count",
    "dst_ip_global_flow_share",
    "dst_ip_global_source_share",
    "co_targeting_score",
    "window_dst_entropy",
    "window_port_entropy",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + CORRELATION_FEATURE_COLUMNS
LOG_SCALE_COLUMNS = [
    "flow_count",
    "total_packets_sum",
    "total_bytes_sum",
    "pps_max",
    "bps_max",
    "duration_max",
]

METADATA_COLUMNS = [
    "family",
    "split",
    "scenario",
    "dataset_name",
    "src_ip",
    "window_id",
]
