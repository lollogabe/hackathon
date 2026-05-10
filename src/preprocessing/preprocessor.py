from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.features.schema import FEATURE_COLUMNS, LOG_SCALE_COLUMNS

logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(
        self,
        n_components: int,
        variance_warning_threshold: float,
        feature_columns: list[str] | None = None,
        log_columns: list[str] | None = None,
    ) -> None:
        self.n_components = n_components
        self.variance_warning_threshold = variance_warning_threshold
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.log_columns = log_columns or LOG_SCALE_COLUMNS
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, features: pd.DataFrame | np.ndarray) -> "Preprocessor":
        matrix = self._to_matrix(features)
        if matrix.shape[0] < self.n_components:
            raise ValueError(
                f"PCA needs at least n_components rows; got {matrix.shape[0]} rows "
                f"for {self.n_components} components."
            )
        logged = self._log_scale(matrix)
        scaled = self.scaler.fit_transform(logged)
        self.pca.fit(scaled)
        explained = float(np.sum(self.pca.explained_variance_ratio_))
        logger.info("PCA cumulative explained variance ratio: %.4f", explained)
        if explained < self.variance_warning_threshold:
            logger.warning(
                "PCA explained variance %.4f is below %.4f; consider increasing "
                "pca_components.",
                explained,
                self.variance_warning_threshold,
            )
        self.is_fitted = True
        return self

    def transform(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        z = self.transform_to_z(features)
        return (2.0 * np.arctan(z)).astype("float64")

    def transform_to_z(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        matrix = self._to_matrix(features)
        logged = self._log_scale(matrix)
        scaled = self.scaler.transform(logged)
        return self.pca.transform(scaled).astype("float64")

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        with Path(path).open("rb") as handle:
            preprocessor = pickle.load(handle)
        if not isinstance(preprocessor, cls):
            raise TypeError(f"File does not contain a {cls.__name__}: {path}")
        return preprocessor

    def _to_matrix(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            missing = [column for column in self.feature_columns if column not in features]
            if missing:
                raise ValueError(
                    "Feature frame is missing column(s): " + ", ".join(missing)
                )
            matrix = features[self.feature_columns].to_numpy(dtype="float64")
        else:
            matrix = np.asarray(features, dtype="float64")
        if matrix.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape {matrix.shape}.")
        if matrix.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Expected {len(self.feature_columns)} features, got {matrix.shape[1]}."
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Feature matrix contains NaN or infinite values.")
        return matrix

    def _log_scale(self, matrix: np.ndarray) -> np.ndarray:
        output = matrix.copy()
        for column in self.log_columns:
            index = self.feature_columns.index(column)
            values = output[:, index]
            if np.any(values < 0.0):
                raise ValueError(f"Log-scaled feature '{column}' contains negative values.")
            output[:, index] = np.log1p(values)
        return output
