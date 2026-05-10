from __future__ import annotations

import pandas as pd


def add_window_id(frame: pd.DataFrame, window_width: int) -> pd.DataFrame:
    if "row_in_window" not in frame.columns:
        raise ValueError("Cannot partition windows without 'row_in_window'.")
    output = frame.copy()
    output["window_id"] = (output["row_in_window"].astype("int64") // window_width).astype(
        "int64"
    )
    return output


def derive_instance_labels(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"src_ip", "window_id", "is_seeded_ddos"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Cannot derive instance labels; missing columns: " + ", ".join(sorted(missing))
        )
    labels = (
        frame.groupby(["src_ip", "window_id"], as_index=False)["is_seeded_ddos"]
        .max()
        .rename(columns={"is_seeded_ddos": "y_instance"})
    )
    labels["y_instance"] = labels["y_instance"].fillna(0).astype("int64")
    return labels


def derive_window_labels(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"window_id", "is_seeded_ddos"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Cannot derive window labels; missing columns: " + ", ".join(sorted(missing))
        )
    labels = (
        frame.groupby("window_id", as_index=False)["is_seeded_ddos"]
        .max()
        .rename(columns={"is_seeded_ddos": "y_window"})
    )
    labels["y_window"] = labels["y_window"].fillna(0).astype("int64")
    return labels


def derive_burst_windows(frame: pd.DataFrame) -> pd.DataFrame:
    if "burst_id" not in frame.columns or "window_id" not in frame.columns:
        return pd.DataFrame(
            columns=["burst_id", "window_id", "has_ramp_up", "has_seeded_ddos"]
        )
    burst_rows = frame[frame["burst_id"].notna()].copy()
    if burst_rows.empty:
        return pd.DataFrame(
            columns=["burst_id", "window_id", "has_ramp_up", "has_seeded_ddos"]
        )
    burst_rows["has_ramp_up"] = (
        burst_rows.get("burst_phase", pd.Series(index=burst_rows.index, dtype=object))
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("ramp-up")
    )
    burst_rows["has_seeded_ddos"] = burst_rows["is_seeded_ddos"].fillna(0).astype(int) == 1
    return (
        burst_rows.groupby(["burst_id", "window_id"], as_index=False)
        .agg(has_ramp_up=("has_ramp_up", "max"), has_seeded_ddos=("has_seeded_ddos", "max"))
        .sort_values(["burst_id", "window_id"])
    )
