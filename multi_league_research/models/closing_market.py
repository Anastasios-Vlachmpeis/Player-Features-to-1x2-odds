# De-vigged closing market probabilities as a non-learned comparator.

from __future__ import annotations

import numpy as np
import pandas as pd


class ClosingMarket:
    name = "closing_market"

    def fit(
        self,
        train: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        return None

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        return test[
            [
                "market_home_probability",
                "market_draw_probability",
                "market_away_probability",
            ]
        ].to_numpy(dtype=float)

    def export_coefficients(self, test_season: str) -> pd.DataFrame | None:
        return None
