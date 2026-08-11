# Historical outcome-frequency baseline.

from __future__ import annotations

import numpy as np
import pandas as pd

from constants import CLASS_ORDER


class FrequencyBaseline:
    name = "frequency_baseline"

    def __init__(self) -> None:
        self._class_probabilities: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> None:
        frequency = train["result_3way"].value_counts(normalize=True)
        self._class_probabilities = np.array(
            [frequency.get(label, 0.0) for label in CLASS_ORDER],
            dtype=float,
        )

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        if self._class_probabilities is None:
            raise RuntimeError("FrequencyBaseline.fit must be called before predict_proba")
        return np.tile(self._class_probabilities, (len(test), 1))

    def export_coefficients(self, test_season: str) -> pd.DataFrame | None:
        return None
