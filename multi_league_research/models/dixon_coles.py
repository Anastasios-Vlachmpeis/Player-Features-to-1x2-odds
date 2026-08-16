"""Standard time-weighted Dixon-Coles score model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.dixon_coles_core import DixonColesEstimator


class DixonColesModel:
    name = "dixon_coles"

    def __init__(self) -> None:
        self._estimator = DixonColesEstimator()

    def fit(self, train: pd.DataFrame) -> None:
        self._estimator.fit(train)

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        return self._estimator.predict_proba(test)

    def export_coefficients(self, test_season: str) -> pd.DataFrame | None:
        return self._estimator.export_parameters(self.name, test_season)
