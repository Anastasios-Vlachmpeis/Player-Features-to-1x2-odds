"""Dixon-Coles score model with direct player-form expected-goal adjustment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from constants import PLAYER_FEATURES
from models.dixon_coles_core import DixonColesEstimator, PLAYER_L2_STRENGTH


class DixonColesPlayerFormModel:
    name = "dixon_coles_player_form"

    def __init__(self, player_features=None, name="dixon_coles_player_form") -> None:

        selected = (PLAYER_FEATURES if player_features is None else list(player_features))

        self.name = name
        self._estimator = DixonColesEstimator(player_features=selected, player_l2_strength=PLAYER_L2_STRENGTH)

    def fit(self, train: pd.DataFrame) -> None:
        self._estimator.fit(train)

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        return self._estimator.predict_proba(test)

    def export_coefficients(self, test_season: str) -> pd.DataFrame | None:
        return self._estimator.export_parameters(self.name, test_season)
