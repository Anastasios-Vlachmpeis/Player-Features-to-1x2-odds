# Shared multinomial logistic pipeline for learned Scotland models.

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from constants import CLASS_ORDER


def build_logistic_pipeline() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=2_000),
    )


def aligned_model_probabilities(model: object, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features)
    classes = list(model.classes_)
    return raw[:, [classes.index(label) for label in CLASS_ORDER]]


def fit_logistic_model(
    train: pd.DataFrame,
    feature_columns: list[str],
    sample_weight: np.ndarray | None = None,
) -> object:
    model = build_logistic_pipeline()
    fit_parameters: dict[str, np.ndarray] = {}
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (len(train),):
            raise ValueError("sample_weight must contain one value per training match")
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("sample_weight must contain finite positive values")
        fit_parameters = {
            "standardscaler__sample_weight": weights,
            "logisticregression__sample_weight": weights,
        }
    model.fit(train[feature_columns], train["result_3way"], **fit_parameters)
    return model


def fit_logistic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, object]:
    model = fit_logistic_model(train, feature_columns, sample_weight=sample_weight)
    return aligned_model_probabilities(model, test[feature_columns]), model


def coefficient_rows(
    model: object,
    model_name: str,
    test_season: str,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    logistic = model.named_steps["logisticregression"]
    rows: list[dict[str, object]] = []
    for class_index, result_class in enumerate(logistic.classes_):
        for feature, coefficient in zip(feature_columns, logistic.coef_[class_index], strict=True):
            rows.append(
                {
                    "model": model_name,
                    "test_season": test_season,
                    "result_class": result_class,
                    "feature": feature,
                    "standardized_coefficient": coefficient,
                }
            )
    return rows
