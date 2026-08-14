# Multinomial logistic model on player-form differential features.

from __future__ import annotations

import numpy as np
import pandas as pd

from constants import PLAYER_FEATURES
from models.logistic import (
    aligned_model_probabilities,
    coefficient_rows,
    fit_logistic_model,
)


class PlayerFormModel:
    name = "player_form"

    def __init__(self, player_features=None, name="player_form") -> None:

        self._feature_columns = (PLAYER_FEATURES if player_features is None else list(player_features))
        self.name = name
        self._fitted_model: object | None = None

    def fit(
        self,
        train: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        self._fitted_model = fit_logistic_model(
            train,
            self._feature_columns,
            sample_weight=sample_weight,
        )

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        if self._fitted_model is None:
            raise RuntimeError("PlayerFormModel.fit must be called before predict_proba")
        return aligned_model_probabilities(self._fitted_model, test[self._feature_columns])

    def export_coefficients(self, test_season: str) -> pd.DataFrame | None:
        if self._fitted_model is None:
            return None

        return pd.DataFrame(coefficient_rows(self._fitted_model, self.name, test_season, self._feature_columns))
