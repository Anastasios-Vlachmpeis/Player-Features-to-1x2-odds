"""Baseline estimators and calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from superleague_baseline.constants import CLASS_ORDER, PROB_SUM_TOL
from superleague_baseline.modeling.metrics import evaluate_probs, reorder_probabilities
from superleague_baseline.splits import assert_classes_present


@dataclass
class PriorBaseline:
    class_probs_: np.ndarray

    @classmethod
    def fit(cls, y_train: list[str]) -> "PriorBaseline":
        counts = np.array([y_train.count(c) for c in CLASS_ORDER], dtype=float)
        if counts.sum() == 0:
            raise ValueError("Cannot fit prior baseline on empty labels")
        return cls(class_probs_=counts / counts.sum())

    def predict_proba(self, n: int) -> np.ndarray:
        return np.tile(self.class_probs_, (n, 1))


def build_logistic_pipeline(*, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_calibrated_logistic(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: list[str],
    x_cal: pd.DataFrame,
    y_cal: list[str],
):
    pipeline.fit(x_train, y_train)
    try:
        from sklearn.frozen import FrozenEstimator

        frozen = FrozenEstimator(pipeline)
        calibrated = CalibratedClassifierCV(frozen, method="sigmoid")
        calibrated.fit(x_cal, y_cal)
        return calibrated
    except ImportError:
        # Older sklearn: calibrate with prefit estimator via cv='prefit' fallback
        calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv="prefit")
        calibrated.fit(x_cal, y_cal)
        return calibrated


def predict_hda(model, x: pd.DataFrame) -> pd.DataFrame:
    raw = model.predict_proba(x)
    classes_ = list(getattr(model, "classes_", CLASS_ORDER))
    ordered = reorder_probabilities(raw, classes_)
    if not np.all(np.isfinite(ordered)):
        raise ValueError("Non-finite predicted probabilities")
    sums = ordered.sum(axis=1)
    if (sums <= 0).any():
        raise ValueError("Zero-sum probability vector")
    normalized = ordered / sums[:, None]
    if (np.abs(normalized.sum(axis=1) - 1.0) > PROB_SUM_TOL).any():
        raise ValueError("Probabilities failed normalization tolerance")
    return pd.DataFrame(normalized, columns=["p_home", "p_draw", "p_away"])


def train_and_evaluate(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    *,
    partition: pd.Series,
    seed: int,
) -> dict:
    label_col = "proxy_result_3way"
    complete = dataset["proxy_lineups_complete"].fillna(False)
    data = dataset.loc[complete].copy()
    part = partition.reindex(data.index)
    if part.isna().any():
        raise ValueError("Complete labeled rows are missing partition assignments")
    unknown = set(part.unique()) - {"train", "calibration", "test"}
    if unknown:
        raise ValueError(f"Unknown partition labels: {sorted(unknown)}")

    train = data.loc[part == "train"]
    cal = data.loc[part == "calibration"]
    test = data.loc[part == "test"]
    for name, frame in (("train", train), ("calibration", cal), ("test", test)):
        if frame.empty:
            raise ValueError(f"Partition {name} has no complete labeled rows")
        assert_classes_present(frame[label_col], name)

    y_train = train[label_col].astype(str).tolist()
    y_cal = cal[label_col].astype(str).tolist()
    y_test = test[label_col].astype(str).tolist()

    x_train = train[feature_cols]
    x_cal = cal[feature_cols]
    x_test = test[feature_cols]

    prior = PriorBaseline.fit(y_train)
    prior_probs = prior.predict_proba(len(y_test))

    pipeline = build_logistic_pipeline(seed=seed)
    calibrated = fit_calibrated_logistic(pipeline, x_train, y_train, x_cal, y_cal)
    model_probs = predict_hda(calibrated, x_test)

    metrics = {
        "prior": evaluate_probs(y_test, prior_probs),
        "logistic_calibrated": evaluate_probs(y_test, model_probs.to_numpy()),
        "counts": {
            "train": int(len(train)),
            "calibration": int(len(cal)),
            "test": int(len(test)),
        },
    }
    predictions = test[["match_id", "match_date", "home_team", "away_team", label_col]].copy()
    predictions = predictions.rename(columns={label_col: "label"})
    predictions = pd.concat([predictions.reset_index(drop=True), model_probs.reset_index(drop=True)], axis=1)
    return {"metrics": metrics, "predictions": predictions}
