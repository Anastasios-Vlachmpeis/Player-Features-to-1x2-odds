"""Pooled and league-specific chronological evaluation for the five-league study."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature

import numpy as np
import pandas as pd

from constants import CLASS_ORDER, DEVELOPMENT_FOLDS, EXPECTED_LEAGUES
from evaluation.market_comparison import market_comparison_label
from evaluation.metrics import probability_frame, score_predictions
from evaluation.walk_forward import (
    select_development_leagues,
    validate_development_dataset,
    validate_fold,
)
from models.base import MatchPredictor
from models.publication_suite import PredictorFactory


POOLED_SCOPE = "pooled"
LEAGUE_SPECIFIC_SCOPE = "league_specific"
VALID_SCOPES = (POOLED_SCOPE, LEAGUE_SPECIFIC_SCOPE)
LEAGUE_EFFECT_PREFIX = "league_effect__"


@dataclass
class MultiLeagueResult:
    fold_metrics: pd.DataFrame
    fold_league_metrics: pd.DataFrame
    fold_equal_league_metrics: pd.DataFrame
    fold_league_counts: pd.DataFrame
    training_weight_audit: pd.DataFrame
    overall_metrics: pd.DataFrame
    overall_league_metrics: pd.DataFrame
    equal_league_metrics: pd.DataFrame
    predictions: pd.DataFrame
    coefficients: pd.DataFrame


def league_effect_column_names(
    leagues: set[str] | frozenset[str] = EXPECTED_LEAGUES,
) -> list[str]:
    ordered = sorted(leagues)
    if len(ordered) < 2:
        raise ValueError("Pooled evaluation requires at least two leagues")
    return [f"{LEAGUE_EFFECT_PREFIX}{league}" for league in ordered[1:]]


def add_league_effects(
    dataset: pd.DataFrame,
    leagues: set[str] | frozenset[str] = EXPECTED_LEAGUES,
) -> tuple[pd.DataFrame, list[str]]:
    """Add deterministic league indicators, leaving one reference league out."""

    ordered = sorted(leagues)
    output = dataset.copy()
    columns = league_effect_column_names(leagues)
    for league, column in zip(ordered[1:], columns, strict=True):
        output[column] = output["league"].eq(league).astype(float)
    return output, columns


def equal_league_training_weights(train: pd.DataFrame) -> np.ndarray:
    """Return positive mean-one weights with equal total weight per league."""

    counts = train.groupby("league", sort=True).size()
    if set(counts.index.astype(str)) != set(EXPECTED_LEAGUES):
        raise ValueError("Equal weighting requires every development league")
    league_count = len(counts)
    target_total = len(train) / league_count
    weight_by_league = target_total / counts
    weights = train["league"].map(weight_by_league).to_numpy(dtype=float)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Equal-league weights must be finite and positive")
    return weights


def training_weight_summary(
    train: pd.DataFrame,
    weights: np.ndarray,
    test_season: str,
) -> pd.DataFrame:
    audit = train[["league"]].copy()
    audit["training_weight"] = weights
    summary = audit.groupby("league", as_index=False).agg(
        train_matches=("league", "size"),
        total_training_weight=("training_weight", "sum"),
        mean_match_weight=("training_weight", "mean"),
    )
    summary.insert(0, "test_season", test_season)
    expected = summary["total_training_weight"].iloc[0]
    if not np.allclose(summary["total_training_weight"], expected, atol=1e-10):
        raise ValueError("Pooled training weights do not give every league equal importance")
    return summary


def _fit_predictor(
    predictor: MatchPredictor,
    train: pd.DataFrame,
    sample_weight: np.ndarray | None,
) -> None:
    parameters = signature(predictor.fit).parameters
    if sample_weight is not None:
        if "sample_weight" not in parameters:
            raise TypeError(
                f"{type(predictor).__name__} cannot be used in the equally weighted pooled suite"
            )
        predictor.fit(train, sample_weight=sample_weight)
    else:
        predictor.fit(train)


def _prediction_frame(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    model_name: str,
    training_scope: str,
) -> pd.DataFrame:
    if probabilities.shape != (len(test), len(CLASS_ORDER)):
        raise ValueError(f"{model_name} returned an invalid probability shape")
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
        raise ValueError(f"{model_name} returned non-positive or non-finite probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError(f"{model_name} probabilities do not sum to one")

    identity_columns = [
        "league",
        "match_id",
        "season",
        "match_date",
        "home_team",
        "away_team",
        "result_3way",
    ]
    output = test[identity_columns].reset_index(drop=True).copy()
    output.insert(0, "model", model_name)
    output.insert(0, "training_scope", training_scope)
    output = pd.concat(
        [output, probability_frame(probabilities).reset_index(drop=True)],
        axis=1,
    )
    output["predicted_result"] = np.asarray(CLASS_ORDER)[
        np.argmax(probabilities, axis=1)
    ]
    return output


def _coefficient_frame(
    predictor: MatchPredictor,
    test_season: str,
    scope: str,
    training_league: str,
) -> pd.DataFrame | None:
    coefficients = predictor.export_coefficients(test_season)
    if coefficients is None or coefficients.empty:
        return None
    output = coefficients.copy()
    output.insert(0, "training_league", training_league)
    output.insert(0, "training_scope", scope)
    return output


def _score_prediction_groups(
    predictions: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in predictions.groupby(grouper, sort=True):
        key_values = (keys,) if len(group_columns) == 1 else tuple(keys)
        probabilities = group[["prob_H", "prob_D", "prob_A"]].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_columns, key_values, strict=True)),
                "matches": len(group),
                **score_predictions(group["result_3way"], probabilities),
            }
        )
    return pd.DataFrame(rows)


def _add_market_reference(
    metrics: pd.DataFrame,
    comparison_groups: list[str],
) -> pd.DataFrame:
    market = metrics.loc[
        metrics["model"].eq("closing_market"),
        [*comparison_groups, "log_loss"],
    ].rename(columns={"log_loss": "closing_market_log_loss"})
    if market.duplicated(comparison_groups).any():
        raise ValueError("Multiple closing-market rows exist for a comparison group")
    output = metrics.merge(
        market,
        on=comparison_groups,
        how="left",
        validate="many_to_one",
    )
    if output["closing_market_log_loss"].isna().any():
        raise ValueError("A metric group has no closing-market benchmark")
    output["log_loss_vs_closing_market"] = (
        output["log_loss"] - output["closing_market_log_loss"]
    )
    output["log_loss_relative_to_closing_market"] = (
        output["log_loss_vs_closing_market"]
        / output["closing_market_log_loss"]
    )
    output["market_comparison"] = [
        market_comparison_label(model, difference)
        for model, difference in zip(
            output["model"],
            output["log_loss_relative_to_closing_market"],
            strict=True,
        )
    ]
    return output


def _equal_league_metrics(
    league_metrics: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    metric_columns = ["log_loss", "brier_score", "accuracy"]
    output = (
        league_metrics.groupby([*group_columns, "model"], as_index=False)
        .agg(
            leagues=("league", "nunique"),
            matches=("matches", "sum"),
            **{column: (column, "mean") for column in metric_columns},
        )
    )
    if not output["leagues"].eq(len(EXPECTED_LEAGUES)).all():
        raise ValueError("Equal-league metrics require every development league")
    return _add_market_reference(output, group_columns)


def run_multi_league_walk_forward(
    dataset: pd.DataFrame,
    pooled_factories: dict[str, PredictorFactory],
    league_specific_factories: dict[str, PredictorFactory],
    scopes: tuple[str, ...] = VALID_SCOPES,
) -> MultiLeagueResult:
    unknown_scopes = sorted(set(scopes).difference(VALID_SCOPES))
    if unknown_scopes:
        raise ValueError(f"Unknown training scope(s): {unknown_scopes}")

    selected = select_development_leagues(dataset)
    validate_development_dataset(selected)
    selected, _ = add_league_effects(selected)

    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    count_rows: list[dict[str, object]] = []
    weight_frames: list[pd.DataFrame] = []

    for test_season, train_seasons in DEVELOPMENT_FOLDS:
        train = selected[selected["season"].isin(train_seasons)].copy()
        test = selected[selected["season"].eq(test_season)].copy()
        validate_fold(train, test, test_season, train_seasons)

        for league in sorted(EXPECTED_LEAGUES):
            count_rows.append(
                {
                    "test_season": test_season,
                    "league": league,
                    "train_matches": int(train["league"].eq(league).sum()),
                    "test_matches": int(test["league"].eq(league).sum()),
                }
            )

        if POOLED_SCOPE in scopes:
            weights = equal_league_training_weights(train)
            weight_frames.append(training_weight_summary(train, weights, test_season))
            for expected_name, factory in pooled_factories.items():
                predictor = factory()
                if predictor.name != expected_name:
                    raise ValueError(
                        f"Pooled model registry name mismatch: {expected_name} != {predictor.name}"
                    )
                _fit_predictor(predictor, train, weights)
                probabilities = predictor.predict_proba(test)
                prediction_frames.append(
                    _prediction_frame(test, probabilities, predictor.name, POOLED_SCOPE)
                )
                coefficients = _coefficient_frame(
                    predictor,
                    test_season,
                    POOLED_SCOPE,
                    "all_leagues",
                )
                if coefficients is not None:
                    coefficient_frames.append(coefficients)

        if LEAGUE_SPECIFIC_SCOPE in scopes:
            for league in sorted(EXPECTED_LEAGUES):
                league_train = train[train["league"].eq(league)].copy()
                league_test = test[test["league"].eq(league)].copy()
                if set(league_train["result_3way"]) != set(CLASS_ORDER):
                    raise ValueError(
                        f"{league} training data before {test_season} lacks an outcome class"
                    )
                for expected_name, factory in league_specific_factories.items():
                    predictor = factory()
                    if predictor.name != expected_name:
                        raise ValueError(
                            "League-specific model registry name mismatch: "
                            f"{expected_name} != {predictor.name}"
                        )
                    _fit_predictor(predictor, league_train, None)
                    probabilities = predictor.predict_proba(league_test)
                    prediction_frames.append(
                        _prediction_frame(
                            league_test,
                            probabilities,
                            predictor.name,
                            LEAGUE_SPECIFIC_SCOPE,
                        )
                    )
                    coefficients = _coefficient_frame(
                        predictor,
                        test_season,
                        LEAGUE_SPECIFIC_SCOPE,
                        league,
                    )
                    if coefficients is not None:
                        coefficient_frames.append(coefficients)

    if not prediction_frames:
        raise ValueError("No predictions were generated")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    duplicate_keys = ["training_scope", "model", "season", "league", "match_id"]
    if predictions.duplicated(duplicate_keys).any():
        raise ValueError("A model produced duplicate predictions for a test match")

    fold_metrics = _add_market_reference(
        _score_prediction_groups(
            predictions,
            ["training_scope", "season", "model"],
        ).rename(columns={"season": "test_season"}),
        ["training_scope", "test_season"],
    )
    fold_league_metrics = _add_market_reference(
        _score_prediction_groups(
            predictions,
            ["training_scope", "season", "league", "model"],
        ).rename(columns={"season": "test_season"}),
        ["training_scope", "test_season", "league"],
    )
    overall_metrics = _add_market_reference(
        _score_prediction_groups(predictions, ["training_scope", "model"]),
        ["training_scope"],
    )
    overall_league_metrics = _add_market_reference(
        _score_prediction_groups(
            predictions,
            ["training_scope", "league", "model"],
        ),
        ["training_scope", "league"],
    )
    fold_equal_league_metrics = _equal_league_metrics(
        fold_league_metrics,
        ["training_scope", "test_season"],
    )
    equal_league_metrics = _equal_league_metrics(
        overall_league_metrics,
        ["training_scope"],
    )

    return MultiLeagueResult(
        fold_metrics=fold_metrics,
        fold_league_metrics=fold_league_metrics,
        fold_equal_league_metrics=fold_equal_league_metrics,
        fold_league_counts=pd.DataFrame(count_rows),
        training_weight_audit=(
            pd.concat(weight_frames, ignore_index=True)
            if weight_frames
            else pd.DataFrame()
        ),
        overall_metrics=overall_metrics,
        overall_league_metrics=overall_league_metrics,
        equal_league_metrics=equal_league_metrics,
        predictions=predictions,
        coefficients=(
            pd.concat(coefficient_frames, ignore_index=True)
            if coefficient_frames
            else pd.DataFrame()
        ),
    )
