# Predictor interface for walk-forward evaluation.

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class MatchPredictor(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> None: ...

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray: ...

    def export_coefficients(self, test_season: str) -> pd.DataFrame | None: ...
