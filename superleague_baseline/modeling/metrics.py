"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss

from superleague_baseline.constants import CLASS_ORDER, CLASS_TO_INDEX


def reorder_probabilities(probs: np.ndarray, classes_: list[str]) -> np.ndarray:
    idx = [list(classes_).index(c) for c in CLASS_ORDER]
    return probs[:, idx]


def multiclass_brier(y_true: list[str], probs: np.ndarray) -> float:
    y_idx = np.array([CLASS_TO_INDEX[y] for y in y_true])
    one_hot = np.zeros((len(y_true), len(CLASS_ORDER)))
    one_hot[np.arange(len(y_true)), y_idx] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def evaluate_probs(y_true: list[str], probs: np.ndarray) -> dict[str, float]:
    lex_labels = sorted(CLASS_ORDER)
    lex_idx = [CLASS_ORDER.index(c) for c in lex_labels]
    lex_probs = probs[:, lex_idx]
    return {
        "log_loss": float(log_loss(y_true, lex_probs, labels=lex_labels)),
        "brier": multiclass_brier(y_true, probs),
    }
