"""Temporally tuned and calibrated LightGBM probability models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from evaluation.calibration import make_calibrator, multiclass_log_loss
from evaluation.lightgbm_tuning import DEFAULT_TUNING_TRIALS, LightGBMTuningResult, make_classifier, ordered_probabilities, tune_lightgbm


class TunedCalibratedLightGBMModel:
    """Fit once per outer fold and expose both raw and calibrated variants."""

    def __init__(self, representation: str, name: str, raw_name: str, n_trials: int = DEFAULT_TUNING_TRIALS, calibration_method: str = "temperature", random_seed: int = 42) -> None:
        self.representation = representation
        self.name = name
        self.raw_name = raw_name
        self.n_trials = n_trials
        self.calibration_method = calibration_method
        self.random_seed = random_seed
        self._tuning: LightGBMTuningResult | None = None
        self._model: LGBMClassifier | None = None
        self._calibrator = None
        self._raw_calibration_loss = float("nan")
        self._calibrated_calibration_loss = float("nan")

    def fit(self, train: pd.DataFrame) -> None:
        tuning = tune_lightgbm(train, self.representation, n_trials=self.n_trials, seed=self.random_seed)
        calibration = train[train["season"].astype(str).eq(tuning.calibration_season)].copy()
        pre_calibration = train[~train["season"].astype(str).eq(tuning.calibration_season)].copy()
        if calibration.empty or pre_calibration.empty:
            raise ValueError("Tuned LightGBM requires non-empty pre-calibration and calibration periods")

        # The tree count is fixed from earlier tuning folds. Calibration labels
        # therefore cannot influence early stopping or model capacity.
        calibration_model = make_classifier(tuning.selected_parameters, self.random_seed, tuning.fixed_estimators)
        calibration_model.fit(pre_calibration[tuning.feature_columns], pre_calibration["result_3way"])
        raw_calibration = ordered_probabilities(calibration_model, calibration, tuning.feature_columns)
        calibrator = make_calibrator(self.calibration_method)
        calibrator.fit(raw_calibration, calibration["result_3way"])
        calibrated = calibrator.transform(raw_calibration)
        self._raw_calibration_loss = multiclass_log_loss(calibration["result_3way"], raw_calibration)
        self._calibrated_calibration_loss = multiclass_log_loss(calibration["result_3way"], calibrated)

        # Refit the selected LightGBM on every outer-training season. The
        # calibrator remains frozen and the outer test season remains untouched.
        final_model = make_classifier(tuning.selected_parameters, self.random_seed, tuning.fixed_estimators)
        final_model.fit(train[tuning.feature_columns], train["result_3way"])
        self._tuning = tuning
        self._model = final_model
        self._calibrator = calibrator

    def predict_raw_proba(self, test: pd.DataFrame) -> np.ndarray:
        if self._model is None or self._tuning is None:
            raise RuntimeError("TunedCalibratedLightGBMModel.fit must be called before prediction")
        return ordered_probabilities(self._model, test, self._tuning.feature_columns)

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        if self._calibrator is None:
            raise RuntimeError("TunedCalibratedLightGBMModel.fit must be called before prediction")
        return self._calibrator.transform(self.predict_raw_proba(test))

    def predict_probability_variants(self, test: pd.DataFrame) -> dict[str, np.ndarray]:
        raw = self.predict_raw_proba(test)
        return {self.raw_name: raw, self.name: self._calibrator.transform(raw)}

    def export_coefficients(self, test_season: str) -> pd.DataFrame | None:
        if self._model is None or self._tuning is None:
            raise RuntimeError("TunedCalibratedLightGBMModel.fit must be called before export")
        return pd.DataFrame(
            {
                "model": self.name,
                "test_season": test_season,
                "result_class": "tree_importance_gain",
                "feature": self._tuning.feature_columns,
                "standardized_coefficient": self._model.booster_.feature_importance(importance_type="gain"),
            }
        )

    def export_tuning_artifacts(self, test_season: str) -> pd.DataFrame:
        if self._tuning is None or self._calibrator is None:
            raise RuntimeError("TunedCalibratedLightGBMModel.fit must be called before artifact export")
        trials = self._tuning.trials.copy()
        trials.insert(0, "artifact_type", "trial")
        trials.insert(0, "test_season", test_season)
        trials.insert(0, "model", self.name)
        trials["selected"] = trials.get("selected_after_stability", False)
        summary_rows: list[dict[str, object]] = []

        def add_summary(kind: str, key: str, value: object) -> None:
            summary_rows.append({"model": self.name, "test_season": test_season, "artifact_type": kind, "key": key, "value": value})

        add_summary("selection", "representation", self._tuning.representation)
        add_summary("selection", "feature_set", self._tuning.feature_set_name)
        add_summary("selection", "calibration_season", self._tuning.calibration_season)
        add_summary("selection", "fixed_estimators", self._tuning.fixed_estimators)
        for key, value in self._tuning.selected_parameters.items():
            add_summary("selected_parameter", key, value)
        add_summary("calibration", "method", self.calibration_method)
        for key, value in self._calibrator.parameters().items():
            add_summary("calibration_parameter", key, value)
        add_summary("calibration_diagnostic", "raw_log_loss", self._raw_calibration_loss)
        add_summary("calibration_diagnostic", "calibrated_log_loss", self._calibrated_calibration_loss)
        return pd.concat([trials, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)


class TunedPlayerFormLightGBMModel(TunedCalibratedLightGBMModel):
    def __init__(self, n_trials: int = DEFAULT_TUNING_TRIALS, calibration_method: str = "temperature") -> None:
        super().__init__(
            representation="diff",
            name=f"tuned_player_form_lightgbm_{calibration_method}",
            raw_name="tuned_player_form_lightgbm_raw",
            n_trials=n_trials,
            calibration_method=calibration_method,
        )


class TunedExpandedPlayerFormLightGBMModel(TunedCalibratedLightGBMModel):
    def __init__(self, n_trials: int = DEFAULT_TUNING_TRIALS, calibration_method: str = "temperature") -> None:
        super().__init__(
            representation="expanded",
            name=f"tuned_expanded_player_form_lightgbm_{calibration_method}",
            raw_name="tuned_expanded_player_form_lightgbm_raw",
            n_trials=n_trials,
            calibration_method=calibration_method,
        )
