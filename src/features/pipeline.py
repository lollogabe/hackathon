from __future__ import annotations

import pandas as pd

from src.features.base_features import compute_base_features
from src.features.correlation_features import add_correlation_features
from src.features.schema import FEATURE_COLUMNS


def compute_instance_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    base = compute_base_features(frame, config)
    with_correlations = add_correlation_features(frame, base, config)
    output = with_correlations[["src_ip", "window_id"] + FEATURE_COLUMNS].copy()
    output[FEATURE_COLUMNS] = output[FEATURE_COLUMNS].astype("float64").fillna(0.0)
    if output[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature matrix contains NaNs after feature engineering.")
    return output
