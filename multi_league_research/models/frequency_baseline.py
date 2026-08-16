# Historical outcome-frequency baseline.

from __future__ import annotations

import numpy as np
import pandas as pd

from constants import CLASS_ORDER


class FrequencyBaseline:
    name = "frequency_baseline"

    def __init__(self) -> None:
        self._class_probabilities: np.ndarray | None = None

    def fit(
        self,
        train: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        if sample_weight is None:
            frequency = train["result_3way"].value_counts(normalize=True)
        else:
            weights = pd.Series(np.asarray(sample_weight, dtype=float), index=train.index)
            weighted = weights.groupby(train["result_3way"]).sum()
            frequency = weighted / weighted.sum()
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


class LeagueFrequencyBaseline:
    """Historical H/D/A frequencies calculated independently by league."""

    name = "frequency_baseline"

    def __init__(self) -> None:
        self._probabilities: dict[str, np.ndarray] = {}

    def fit(
        self,
        train: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        if "league" not in train:
            raise ValueError("LeagueFrequencyBaseline requires a league column")
        weights = (
            np.ones(len(train), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        if weights.shape != (len(train),):
            raise ValueError("sample_weight must contain one value per training match")
        weighted = train[["league", "result_3way"]].copy()
        weighted["weight"] = weights
        for league, group in weighted.groupby("league", sort=True):
            totals = group.groupby("result_3way")["weight"].sum()
            probabilities = np.array(
                [totals.get(label, 0.0) for label in CLASS_ORDER],
                dtype=float,
            )
            if np.any(probabilities <= 0):
                raise ValueError(f"{league} training data does not contain all outcomes")
            self._probabilities[str(league)] = probabilities / probabilities.sum()

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(test["league"].astype(str)).difference(self._probabilities))
        if missing:
            raise ValueError(f"No historical frequency was fitted for: {missing}")
        return np.vstack(
            [self._probabilities[str(league)] for league in test["league"]]
        )

    def export_coefficients(self, test_season: str) -> pd.DataFrame | None:
        return None
