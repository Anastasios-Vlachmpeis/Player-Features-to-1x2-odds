"""Nested chronological hyperparameter and feature-set tuning for LightGBM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from build_match_features import (
    ADJUSTED_MEAN_FEATURES,
    BASE_SUM_FEATURES,
    DISTRIBUTION_FEATURES,
    LINEUP_FEATURES,
    POSITION_FEATURES,
    RECENCY_MEAN_FEATURES,
    TEAM_FEATURES,
    TEAM_STRENGTH_FEATURES,
)
from constants import CLASS_ORDER
from evaluation.calibration import multiclass_log_loss


DEFAULT_TUNING_TRIALS = 50
DEFAULT_EARLY_STOPPING_ROUNDS = 100
MAX_ESTIMATORS = 2_000
STABILITY_CANDIDATES = 5
STABILITY_SEEDS = 3


CORE_FEATURES = [
    *BASE_SUM_FEATURES,
    "rating_mean_5",
    "starters_without_history",
    "starters_without_full_window",
]
ADJUSTED_FEATURES = list(ADJUSTED_MEAN_FEATURES)
RECENCY_FEATURES = list(RECENCY_MEAN_FEATURES)


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


FEATURE_SET_BASES = {
    "core": unique(CORE_FEATURES),
    "core_strength": unique(CORE_FEATURES + list(TEAM_STRENGTH_FEATURES)),
    "core_strength_adjusted": unique(CORE_FEATURES + list(TEAM_STRENGTH_FEATURES) + ADJUSTED_FEATURES),
    "core_strength_adjusted_lineup": unique(CORE_FEATURES + list(TEAM_STRENGTH_FEATURES) + ADJUSTED_FEATURES + list(LINEUP_FEATURES)),
    "core_strength_adjusted_lineup_position": unique(CORE_FEATURES + list(TEAM_STRENGTH_FEATURES) + ADJUSTED_FEATURES + list(LINEUP_FEATURES) + POSITION_FEATURES + DISTRIBUTION_FEATURES),
    "all": list(TEAM_FEATURES),
}


@dataclass
class LightGBMTuningResult:
    representation: str
    calibration_season: str
    feature_set_name: str
    feature_columns: list[str]
    selected_parameters: dict[str, object]
    fixed_estimators: int
    trials: pd.DataFrame


def representation_columns(base_features: list[str], representation: str) -> list[str]:
    if representation == "diff":
        return [f"diff_{feature}" for feature in base_features]
    if representation == "expanded":
        return [f"{side}_{feature}" for side in ("home", "away") for feature in base_features]
    raise ValueError("LightGBM representation must be diff or expanded")


def ordered_seasons(frame: pd.DataFrame) -> list[str]:
    date_column = "_match_datetime" if "_match_datetime" in frame else "match_date"
    dates = pd.to_datetime(frame[date_column], utc=True, errors="raise")
    first_dates = pd.DataFrame({"season": frame["season"].astype(str), "date": dates}).groupby("season")["date"].min()
    return first_dates.sort_values().index.tolist()


def chronological_tuning_splits(train: pd.DataFrame) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame, str]], str]:
    seasons = ordered_seasons(train)
    if len(seasons) < 2:
        raise ValueError("Tuned calibration requires at least two historical seasons")
    calibration_season = seasons[-1]
    tuning = train[~train["season"].astype(str).eq(calibration_season)].copy()
    tuning_seasons = ordered_seasons(tuning)
    splits: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []

    if len(tuning_seasons) >= 2:
        for validation_index in range(1, len(tuning_seasons)):
            fit_seasons = tuning_seasons[:validation_index]
            validation_season = tuning_seasons[validation_index]
            splits.append(
                (
                    tuning[tuning["season"].astype(str).isin(fit_seasons)].copy(),
                    tuning[tuning["season"].astype(str).eq(validation_season)].copy(),
                    validation_season,
                )
            )
    else:
        # The earliest outer fold has one season available for tuning after the
        # latest training season is reserved for calibration. Split by kickoff
        # date so simultaneous fixtures never cross the boundary.
        date_column = "_match_datetime" if "_match_datetime" in tuning else "match_date"
        dates = pd.to_datetime(tuning[date_column], utc=True, errors="raise")
        unique_dates = np.sort(dates.unique())
        cutoff = unique_dates[max(1, int(len(unique_dates) * 0.70)) - 1]
        fit = tuning[dates.le(cutoff)].copy()
        validation = tuning[dates.gt(cutoff)].copy()
        splits.append((fit, validation, f"{tuning_seasons[0]}_late_block"))

    for fit, validation, label in splits:
        if fit.empty or validation.empty or set(fit["result_3way"].astype(str)) != set(CLASS_ORDER):
            raise ValueError(f"Invalid chronological LightGBM tuning split: {label}")
    return splits, calibration_season


def sample_configuration(rng: np.random.Generator, trial_index: int) -> dict[str, object]:
    max_depth = int(rng.choice([2, 3, 4, 5, 6]))
    leaves = [value for value in [3, 5, 7, 11, 15, 23] if value <= 2**max_depth]
    feature_sets = list(FEATURE_SET_BASES)
    return {
        "feature_set": feature_sets[trial_index % len(feature_sets)],
        "learning_rate": float(np.exp(rng.uniform(np.log(0.005), np.log(0.08)))),
        "num_leaves": int(rng.choice(leaves)),
        "max_depth": max_depth,
        "min_child_samples": int(rng.choice([20, 40, 60, 90, 130])),
        "subsample": float(rng.choice([0.65, 0.8, 1.0])),
        "colsample_bytree": float(rng.choice([0.5, 0.7, 0.85, 1.0])),
        "reg_alpha": float(np.exp(rng.uniform(np.log(1e-3), np.log(10.0)))),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(100.0)))),
        "min_split_gain": float(rng.choice([0.0, 0.01, 0.05, 0.2])),
        "max_bin": int(rng.choice([31, 63, 127])),
    }


def parameters_from_row(row: pd.Series) -> dict[str, object]:
    return {
        "feature_set": str(row["feature_set"]),
        "learning_rate": float(row["learning_rate"]),
        "num_leaves": int(row["num_leaves"]),
        "max_depth": int(row["max_depth"]),
        "min_child_samples": int(row["min_child_samples"]),
        "subsample": float(row["subsample"]),
        "colsample_bytree": float(row["colsample_bytree"]),
        "reg_alpha": float(row["reg_alpha"]),
        "reg_lambda": float(row["reg_lambda"]),
        "min_split_gain": float(row["min_split_gain"]),
        "max_bin": int(row["max_bin"]),
    }


def make_classifier(parameters: dict[str, object], seed: int, n_estimators: int) -> LGBMClassifier:
    model_parameters = {key: value for key, value in parameters.items() if key != "feature_set"}
    subsample = float(model_parameters["subsample"])
    return LGBMClassifier(
        objective="multiclass",
        num_class=len(CLASS_ORDER),
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        subsample_freq=1 if subsample < 1.0 else 0,
        **model_parameters,
    )


def ordered_probabilities(model: LGBMClassifier, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    raw = model.predict_proba(frame[features])
    classes = list(model.classes_)
    return raw[:, [classes.index(label) for label in CLASS_ORDER]]


def tune_lightgbm(train: pd.DataFrame, representation: str, n_trials: int = DEFAULT_TUNING_TRIALS, seed: int = 42, early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS) -> LightGBMTuningResult:
    if n_trials < 1:
        raise ValueError("LightGBM tuning requires at least one trial")
    splits, calibration_season = chronological_tuning_splits(train)
    rng = np.random.default_rng(seed)
    trial_rows: list[dict[str, object]] = []

    for trial_index in range(n_trials):
        parameters = sample_configuration(rng, trial_index)
        feature_set = str(parameters["feature_set"])
        features = representation_columns(FEATURE_SET_BASES[feature_set], representation)
        missing = sorted(set(features).difference(train.columns))
        if missing:
            raise ValueError(f"Tuning feature set {feature_set} is missing columns: {missing[:10]}")
        fold_losses: list[float] = []
        best_iterations: list[int] = []
        fold_labels: list[str] = []
        for fold_index, (fit, validation, fold_label) in enumerate(splits):
            model = make_classifier(parameters, seed + fold_index, MAX_ESTIMATORS)
            model.fit(
                fit[features],
                fit["result_3way"],
                eval_set=[(validation[features], validation["result_3way"])],
                eval_metric="multi_logloss",
                callbacks=[early_stopping(early_stopping_rounds, verbose=False), log_evaluation(0)],
            )
            probabilities = ordered_probabilities(model, validation, features)
            fold_losses.append(multiclass_log_loss(validation["result_3way"], probabilities))
            best_iterations.append(int(model.best_iteration_ or MAX_ESTIMATORS))
            fold_labels.append(fold_label)
        trial_rows.append(
            {
                "trial": trial_index,
                **parameters,
                "mean_log_loss": float(np.mean(fold_losses)),
                "worst_fold_log_loss": float(np.max(fold_losses)),
                "best_iterations": ";".join(str(value) for value in best_iterations),
                "folds": ";".join(fold_labels),
            }
        )

    trials = pd.DataFrame(trial_rows).sort_values(["mean_log_loss", "worst_fold_log_loss", "trial"], kind="stable").reset_index(drop=True)
    trials["stability_mean_log_loss"] = np.nan
    trials["stability_worst_log_loss"] = np.nan
    trials["stability_best_iterations"] = ""

    # Refit the strongest temporal configurations across several seeds. This
    # guards against selecting a configuration that benefits from one random
    # row/column subsample on a small seasonal validation set.
    for candidate_index in range(min(STABILITY_CANDIDATES, len(trials))):
        parameters = parameters_from_row(trials.iloc[candidate_index])
        feature_set = str(parameters["feature_set"])
        features = representation_columns(FEATURE_SET_BASES[feature_set], representation)
        losses: list[float] = []
        iterations: list[int] = []
        for seed_offset in range(STABILITY_SEEDS):
            for fold_index, (fit, validation, _) in enumerate(splits):
                model = make_classifier(parameters, seed + 100 + seed_offset * 10 + fold_index, MAX_ESTIMATORS)
                model.fit(
                    fit[features],
                    fit["result_3way"],
                    eval_set=[(validation[features], validation["result_3way"])],
                    eval_metric="multi_logloss",
                    callbacks=[early_stopping(early_stopping_rounds, verbose=False), log_evaluation(0)],
                )
                losses.append(multiclass_log_loss(validation["result_3way"], ordered_probabilities(model, validation, features)))
                iterations.append(int(model.best_iteration_ or MAX_ESTIMATORS))
        trials.loc[candidate_index, "stability_mean_log_loss"] = float(np.mean(losses))
        trials.loc[candidate_index, "stability_worst_log_loss"] = float(np.max(losses))
        trials.loc[candidate_index, "stability_best_iterations"] = ";".join(str(value) for value in iterations)

    stable = trials[trials["stability_mean_log_loss"].notna()].sort_values(
        ["stability_mean_log_loss", "stability_worst_log_loss", "mean_log_loss", "trial"],
        kind="stable",
    )
    best = stable.iloc[0]
    best_trial = int(best["trial"])
    trials["selected_after_stability"] = trials["trial"].eq(best_trial)
    selected_configuration = parameters_from_row(best)
    selected_parameters = {key: value for key, value in selected_configuration.items() if key != "feature_set"}
    iteration_values = [int(value) for value in str(best["stability_best_iterations"]).split(";")]
    fixed_estimators = max(10, int(np.median(iteration_values)))
    feature_set_name = str(best["feature_set"])
    return LightGBMTuningResult(
        representation=representation,
        calibration_season=calibration_season,
        feature_set_name=feature_set_name,
        feature_columns=representation_columns(FEATURE_SET_BASES[feature_set_name], representation),
        selected_parameters=selected_parameters,
        fixed_estimators=fixed_estimators,
        trials=trials,
    )
