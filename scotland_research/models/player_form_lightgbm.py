# Multiclass LightGBM model on player-form differential features.

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from constants import CLASS_ORDER, PLAYER_FEATURES


class PlayerFormLightGBMModel:
    # Small, deterministic nonlinear (LightGBM based) player-form model for the Scotland sample

    name = "player_form_lightgbm"

    def __init__(self, player_features=None, name="player_form_lightgbm") -> None:
        self._feature_columns = (
            PLAYER_FEATURES if player_features is None else list(player_features)
        )
        self.name = name
        self._fitted_model: LGBMClassifier | None = None

    def fit(
        self,
        train: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        model = LGBMClassifier(
            objective="multiclass",
            num_class=len(CLASS_ORDER),
            n_estimators=100,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=30,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(
            train[self._feature_columns],
            train["result_3way"],
            sample_weight=sample_weight,
        )
        self._fitted_model = model

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        if self._fitted_model is None:
            raise RuntimeError(
                f"{type(self).__name__}.fit must be called before predict_proba"
            )

        probabilities = self._fitted_model.predict_proba(test[self._feature_columns])
        classes = list(self._fitted_model.classes_)
        return probabilities[:, [classes.index(label) for label in CLASS_ORDER]]

    def export_coefficients(self, test_season: str) -> pd.DataFrame | None:
        # Tree models have feature importances rather than class coefficients.
        return None
