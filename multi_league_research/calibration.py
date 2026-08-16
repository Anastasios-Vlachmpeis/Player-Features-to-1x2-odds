"""Chronological post-hoc calibration for saved match predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from constants import CLASS_ORDER


PROBABILITY_COLUMNS = [f"prob_{label}" for label in CLASS_ORDER]
IDENTITY_COLUMNS = [
    "league",
    "match_id",
    "season",
    "match_date",
    "home_team",
    "away_team",
    "result_3way",
]
MAX_FOLD_LOG_LOSS_DETERIORATION = 0.001


@dataclass
class CalibrationResult:
    predictions: pd.DataFrame
    fitted_temperatures: pd.DataFrame


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Soften or sharpen probabilities while preserving their ordering."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(CLASS_ORDER):
        raise ValueError("Expected one H-D-A probability row per match")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Probabilities must be finite and positive")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Probabilities must sum to one")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    scaled_log = np.log(values) / temperature
    scaled_log -= scaled_log.max(axis=1, keepdims=True)
    scaled = np.exp(scaled_log)
    return scaled / scaled.sum(axis=1, keepdims=True)


def equal_league_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("league", sort=True).size()
    if counts.empty:
        raise ValueError("Cannot weight an empty calibration sample")
    target_total = len(frame) / len(counts)
    weights = frame["league"].map(target_total / counts).to_numpy(dtype=float)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Calibration weights must be finite and positive")
    return weights


def fit_temperature(
    probabilities: np.ndarray,
    actual: pd.Series,
    sample_weight: np.ndarray,
) -> float:
    class_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    actual_index = actual.map(class_index)
    if actual_index.isna().any():
        raise ValueError("Calibration outcomes contain an unknown result class")
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (len(actual),):
        raise ValueError("Calibration weights must contain one value per match")
    rows = np.arange(len(actual))
    indexes = actual_index.to_numpy(dtype=int)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        calibrated = apply_temperature(probabilities, temperature)
        losses = -np.log(
            np.clip(calibrated[rows, indexes], np.finfo(float).eps, 1.0)
        )
        return float(np.average(losses, weights=weights))

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.25), np.log(4.0)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Temperature fitting failed: {result.message}")
    return float(np.exp(result.x))


def chronological_temperature_scale(
    predictions: pd.DataFrame,
    *,
    training_scope: str,
    model: str,
) -> CalibrationResult:
    """Fit on earlier out-of-sample seasons and apply to the next season."""

    required = {
        "training_scope",
        "model",
        *IDENTITY_COLUMNS,
        *PROBABILITY_COLUMNS,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing calibration columns: {missing}")
    selected = predictions[
        predictions["training_scope"].eq(training_scope)
        & predictions["model"].eq(model)
    ].copy()
    if selected.empty:
        raise ValueError(f"No {training_scope} predictions found for {model}")
    seasons = sorted(selected["season"].astype(str).unique())
    if len(seasons) < 2:
        raise ValueError("Chronological calibration requires at least two prediction seasons")

    prediction_frames: list[pd.DataFrame] = []
    setting_rows: list[dict[str, object]] = []
    calibrated_name = f"{model}_temperature_scaled"
    for season_number, test_season in enumerate(seasons[1:], start=1):
        train_seasons = seasons[:season_number]
        train = selected[selected["season"].isin(train_seasons)].copy()
        test = selected[selected["season"].eq(test_season)].copy()
        weights = equal_league_weights(train)
        temperature = fit_temperature(
            train[PROBABILITY_COLUMNS].to_numpy(dtype=float),
            train["result_3way"],
            weights,
        )
        calibrated = apply_temperature(
            test[PROBABILITY_COLUMNS].to_numpy(dtype=float),
            temperature,
        )
        frame = test[IDENTITY_COLUMNS].reset_index(drop=True).copy()
        frame.insert(0, "model", calibrated_name)
        frame.insert(0, "training_scope", training_scope)
        for column_number, column in enumerate(PROBABILITY_COLUMNS):
            frame[column] = calibrated[:, column_number]
        frame["predicted_result"] = np.asarray(CLASS_ORDER)[
            np.argmax(calibrated, axis=1)
        ]
        frame["source_model"] = model
        frame["temperature"] = temperature
        prediction_frames.append(frame)
        setting_rows.append(
            {
                "training_scope": training_scope,
                "model": model,
                "calibrated_model": calibrated_name,
                "test_season": test_season,
                "calibration_seasons": ";".join(train_seasons),
                "calibration_matches": len(train),
                "test_matches": len(test),
                "temperature": temperature,
            }
        )
    return CalibrationResult(
        predictions=pd.concat(prediction_frames, ignore_index=True),
        fitted_temperatures=pd.DataFrame(setting_rows),
    )


def calibration_decision(
    paired_losses: pd.DataFrame,
    *,
    max_fold_deterioration: float = MAX_FOLD_LOG_LOSS_DETERIORATION,
) -> dict[str, object]:
    """Apply the fixed development-stage rule for retaining calibration."""

    required = {
        "league",
        "season",
        "improvement_log_loss",
        "improvement_brier_score",
    }
    missing = sorted(required.difference(paired_losses.columns))
    if missing:
        raise ValueError(f"Paired calibration losses are missing columns: {missing}")
    league_means = paired_losses.groupby("league", sort=True)[
        ["improvement_log_loss", "improvement_brier_score"]
    ].mean()
    fold_league = paired_losses.groupby(["season", "league"], sort=True)[
        ["improvement_log_loss", "improvement_brier_score"]
    ].mean()
    fold_equal = fold_league.groupby("season", sort=True).mean()
    overall_log = float(league_means["improvement_log_loss"].mean())
    overall_brier = float(league_means["improvement_brier_score"].mean())
    worst_fold_log = float(fold_equal["improvement_log_loss"].min())
    leagues_improved = int(league_means["improvement_log_loss"].gt(0).sum())
    majority = len(league_means) // 2 + 1
    rules = {
        "equal_league_log_loss_improved": overall_log > 0,
        "equal_league_brier_not_worse": overall_brier >= 0,
        "no_fold_log_loss_deterioration_over_limit": (
            worst_fold_log >= -max_fold_deterioration
        ),
        "log_loss_improved_in_league_majority": leagues_improved >= majority,
    }
    return {
        "retain_calibration": all(rules.values()),
        "equal_league_log_loss_improvement": overall_log,
        "equal_league_brier_improvement": overall_brier,
        "worst_fold_log_loss_improvement": worst_fold_log,
        "max_allowed_fold_deterioration": max_fold_deterioration,
        "leagues_improved": leagues_improved,
        "leagues_required": majority,
        **rules,
    }


def calibration_effect_table(
    paired_losses: pd.DataFrame,
    *,
    group_column: str,
    equal_league: bool = False,
) -> pd.DataFrame:
    """Summarize uncalibrated versus calibrated scores by fold or league."""

    score_columns = [
        *[f"baseline_{metric}" for metric in ("log_loss", "brier_score", "rps")],
        *[f"enhanced_{metric}" for metric in ("log_loss", "brier_score", "rps")],
        *[f"improvement_{metric}" for metric in ("log_loss", "brier_score", "rps")],
    ]
    required = {group_column, "league", *score_columns}
    missing = sorted(required.difference(paired_losses.columns))
    if missing:
        raise ValueError(f"Calibration effects are missing columns: {missing}")
    if equal_league:
        wide = (
            paired_losses.groupby([group_column, "league"], sort=True)[score_columns]
            .mean()
            .groupby(group_column, sort=True)
            .mean()
            .reset_index()
        )
    else:
        wide = paired_losses.groupby(group_column, sort=True)[score_columns].mean().reset_index()
    rows: list[dict[str, object]] = []
    for _, row in wide.iterrows():
        for metric in ("log_loss", "brier_score", "rps"):
            rows.append(
                {
                    group_column: row[group_column],
                    "metric": metric,
                    "uncalibrated_score": row[f"baseline_{metric}"],
                    "calibrated_score": row[f"enhanced_{metric}"],
                    "calibration_improvement": row[f"improvement_{metric}"],
                }
            )
    return pd.DataFrame(rows)
