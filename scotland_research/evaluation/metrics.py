# Prediction scoring metrics.

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from constants import CLASS_ORDER


def probability_frame(probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in CLASS_ORDER])


def multiclass_brier(actual: pd.Series, probabilities: np.ndarray) -> float:
    observed = np.column_stack([(actual.to_numpy() == label) for label in CLASS_ORDER])
    return float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))


def ordered_log_loss(actual: pd.Series, probabilities: np.ndarray) -> float:
    # Calculate log loss using the explicit CLASS_ORDER probability columns.
    class_indexes = {label: index for index, label in enumerate(CLASS_ORDER)}
    actual_indexes = actual.map(class_indexes)
    if actual_indexes.isna().any():
        raise ValueError("Actual outcomes contain a class outside CLASS_ORDER")
    if probabilities.shape != (len(actual), len(CLASS_ORDER)):
        raise ValueError("Probability matrix shape does not match outcomes and classes")

    row_indexes = np.arange(len(actual))
    assigned_probabilities = probabilities[
        row_indexes, actual_indexes.to_numpy(dtype=int)
    ]
    assigned_probabilities = np.clip(
        assigned_probabilities,
        np.finfo(float).eps,
        1.0,
    )
    return float(-np.log(assigned_probabilities).mean())


def score_predictions(actual: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(CLASS_ORDER)[np.argmax(probabilities, axis=1)]
    return {
        "log_loss": ordered_log_loss(actual, probabilities),
        "brier_score": multiclass_brier(actual, probabilities),
        "accuracy": float(accuracy_score(actual, predicted)),
    }
