"""Leakage-safe multiclass probability calibrators for temporal validation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from constants import CLASS_ORDER


PROBABILITY_EPSILON = 1e-12


def probability_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), PROBABILITY_EPSILON, 1.0)
    return np.log(clipped)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def encoded_targets(labels) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(CLASS_ORDER)}
    values = np.asarray(labels, dtype=str)
    if not set(values).issubset(mapping):
        raise ValueError("Calibration labels must contain only H, D, or A")
    return np.array([mapping[label] for label in values], dtype=int)


def multiclass_log_loss(labels, probabilities: np.ndarray) -> float:
    targets = encoded_targets(labels)
    clipped = np.clip(probabilities, PROBABILITY_EPSILON, 1.0)
    return -float(np.log(clipped[np.arange(len(targets)), targets]).mean())


class TemperatureCalibrator:
    """One-parameter scaling: calibrated = softmax(log(raw_probability) / T)."""

    name = "temperature"

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, probabilities: np.ndarray, labels) -> None:
        logits = probability_logits(probabilities)

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            return multiclass_log_loss(labels, softmax(logits / temperature))

        result = minimize_scalar(objective, bounds=(np.log(0.25), np.log(4.0)), method="bounded")
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("Temperature calibration optimization failed")
        self.temperature = float(np.exp(result.x))

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return softmax(probability_logits(probabilities) / self.temperature)

    def parameters(self) -> dict[str, float]:
        return {"temperature": self.temperature}


class VectorScalingCalibrator:
    """Regularized class-specific scaling of log probabilities."""

    name = "vector"

    def __init__(self, regularization: float = 10.0) -> None:
        if regularization < 0:
            raise ValueError("Vector calibration regularization must be non-negative")
        self.regularization = regularization
        self.slopes = np.ones(len(CLASS_ORDER), dtype=float)
        self.biases = np.zeros(len(CLASS_ORDER), dtype=float)

    def fit(self, probabilities: np.ndarray, labels) -> None:
        logits = probability_logits(probabilities)
        targets = encoded_targets(labels)

        # The final-class bias is fixed at zero for identifiability. L2 shrinkage
        # toward slopes=1 and biases=0 is essential with one calibration season.
        def unpack(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return vector[:3], np.array([vector[3], vector[4], 0.0])

        def objective(vector: np.ndarray) -> float:
            slopes, biases = unpack(vector)
            calibrated = softmax(logits * slopes + biases)
            loss = -np.log(np.clip(calibrated[np.arange(len(targets)), targets], PROBABILITY_EPSILON, 1.0)).mean()
            penalty = self.regularization * (np.square(slopes - 1.0).sum() + np.square(biases).sum()) / len(targets)
            return float(loss + penalty)

        result = minimize(objective, np.array([1.0, 1.0, 1.0, 0.0, 0.0]), method="L-BFGS-B", bounds=[(0.1, 4.0)] * 3 + [(-2.0, 2.0)] * 2)
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Vector calibration optimization failed: {result.message}")
        self.slopes, self.biases = unpack(result.x)

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return softmax(probability_logits(probabilities) * self.slopes + self.biases)

    def parameters(self) -> dict[str, float]:
        return {
            **{f"slope_{label}": float(value) for label, value in zip(CLASS_ORDER, self.slopes, strict=True)},
            **{f"bias_{label}": float(value) for label, value in zip(CLASS_ORDER, self.biases, strict=True)},
            "regularization": self.regularization,
        }


def make_calibrator(method: str):
    if method == "temperature":
        return TemperatureCalibrator()
    if method == "vector":
        return VectorScalingCalibrator()
    raise ValueError(f"Unknown calibration method: {method}")
