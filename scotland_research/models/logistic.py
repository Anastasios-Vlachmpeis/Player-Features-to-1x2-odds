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


def fit_logistic_model(train: pd.DataFrame, feature_columns: list[str]) -> object:
    model = build_logistic_pipeline()
    model.fit(train[feature_columns], train["result_3way"])
    return model


def fit_logistic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, object]:
    model = fit_logistic_model(train, feature_columns)
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
