# Multinomial logistic model on market log-odds and player-form features.

from __future__ import annotations

import pandas as pd

from constants import MARKET_FEATURES, PLAYER_FEATURES
from models.logistic import (
    aligned_model_probabilities,
    coefficient_rows,
    fit_logistic_model,
)

FEATURE_GROUPS = {
    "shooting": [
        "diff_npxg_per90_sum_5",
        "diff_shots_per90_sum_5",
    ],
    "chance_creation": [
        "diff_key_passes_per90_sum_5",
    ],
    "defending": [
        "diff_defensive_actions_per90_sum_5",
    ],
    "ratings": [
        "diff_rating_mean_5",
    ],
    "recent_experience": [
        "diff_recent_minutes_sum_5",
    ],
    "history_coverage": [
        "diff_starters_without_history",
        "diff_starters_without_full_window",
    ],
}

class MarketPlusPlayerFormModel:
    name = "market_plus_player_form"

    def __init__(self, player_features=None, name="market_plus_player_form") -> None:
        selected = (
            PLAYER_FEATURES
            if player_features is None
            else list(player_features)
        )
        self._fitted_model: object | None = None
        self._feature_columns = MARKET_FEATURES + selected
        self.name = name

    def fit(self, train: pd.DataFrame) -> None:
        self._fitted_model = fit_logistic_model(train, self._feature_columns)

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        if self._fitted_model is None:
            raise RuntimeError(
                "MarketPlusPlayerFormModel.fit must be called before predict_proba"
            )
        return aligned_model_probabilities(
            self._fitted_model,
            test[self._feature_columns],
        )

    def export_coefficients(self, test_season: str) -> pd.DataFrame | None:
        if self._fitted_model is None:
            return None
        return pd.DataFrame(
            coefficient_rows(
                self._fitted_model,
                self.name,
                test_season,
                self._feature_columns,
            )
        )
