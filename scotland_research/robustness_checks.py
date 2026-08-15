"""Focused robustness checks for the final pooled market-plus-player model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from models.closing_market import ClosingMarket
from models.market_plus_player_form import MarketPlusPlayerFormModel
from models.publication_suite import PredictorFactory


SHUFFLE_SEED = 42
HINDSIGHT_FEATURE = "hindsight_final_points_difference"

FEATURE_BLOCKS: dict[str, tuple[str, ...]] = {
    "attacking_output": (
        "key_passes_per90_sum_5",
        "shots_per90_sum_5",
    ),
    "defensive_output": ("defensive_actions_per90_sum_5",),
    "player_ratings": ("rating_mean_5",),
    "team_strength_context": ("expected_goals_strength",),
}

PLAYER_FEATURES_TO_SHUFFLE = tuple(
    feature
    for block_name, features in FEATURE_BLOCKS.items()
    if block_name != "team_strength_context"
    for feature in features
)


def validate_feature_blocks(selected_features: Sequence[str]) -> None:
    """Require every final feature to belong to exactly one declared block."""

    selected = list(selected_features)
    assigned = [feature for features in FEATURE_BLOCKS.values() for feature in features]
    duplicates = sorted({feature for feature in assigned if assigned.count(feature) > 1})
    missing = sorted(set(selected).difference(assigned))
    unexpected = sorted(set(assigned).difference(selected))
    if duplicates or missing or unexpected:
        raise ValueError(
            "Robustness feature blocks do not match the final selected features. "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )


def diff_columns(features: Sequence[str]) -> list[str]:
    return [f"diff_{feature}" for feature in features]


def shuffled_column(feature: str) -> str:
    return f"shuffled_diff_{feature}"


def add_within_league_season_shuffle(
    dataset: pd.DataFrame,
    *,
    seed: int = SHUFFLE_SEED,
) -> pd.DataFrame:
    """Shuffle the player block jointly within each league-season."""

    output = dataset.copy()
    source_columns = diff_columns(PLAYER_FEATURES_TO_SHUFFLE)
    missing = sorted(set(source_columns).difference(output.columns))
    if missing:
        raise ValueError(f"Cannot shuffle missing player columns: {missing}")

    target_columns = [shuffled_column(feature) for feature in PLAYER_FEATURES_TO_SHUFFLE]
    for target in target_columns:
        output[target] = np.nan

    rng = np.random.default_rng(seed)
    for _, group in output.groupby(["league", "season"], sort=True):
        row_indices = group.index.to_numpy()
        shuffled_indices = rng.permutation(row_indices)
        output.loc[row_indices, target_columns] = output.loc[
            shuffled_indices,
            source_columns,
        ].to_numpy()

    if output[target_columns].isna().any().any():
        raise ValueError("The within-league/season shuffle produced missing values")
    return output


def add_hindsight_final_points(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add a deliberately invalid feature built from complete-season results."""

    output = dataset.copy()
    home_points = output["result_3way"].map({"H": 3.0, "D": 1.0, "A": 0.0})
    away_points = output["result_3way"].map({"H": 0.0, "D": 1.0, "A": 3.0})
    if home_points.isna().any() or away_points.isna().any():
        raise ValueError("Cannot build hindsight points from an unknown result class")

    home = output[["league", "season", "home_team"]].rename(
        columns={"home_team": "team"}
    )
    home["points"] = home_points.to_numpy()
    away = output[["league", "season", "away_team"]].rename(
        columns={"away_team": "team"}
    )
    away["points"] = away_points.to_numpy()
    team_seasons = pd.concat([home, away], ignore_index=True)
    final_points = team_seasons.groupby(
        ["league", "season", "team"],
        sort=True,
    )["points"].agg(["sum", "size"])
    final_points["points_per_match"] = final_points["sum"] / final_points["size"]

    home_index = pd.MultiIndex.from_frame(
        output[["league", "season", "home_team"]].rename(
            columns={"home_team": "team"}
        )
    )
    away_index = pd.MultiIndex.from_frame(
        output[["league", "season", "away_team"]].rename(
            columns={"away_team": "team"}
        )
    )
    home_values = final_points["points_per_match"].reindex(home_index).to_numpy()
    away_values = final_points["points_per_match"].reindex(away_index).to_numpy()
    if not np.isfinite(home_values).all() or not np.isfinite(away_values).all():
        raise ValueError("The hindsight feature could not be mapped to every team")
    output[HINDSIGHT_FEATURE] = home_values - away_values
    return output


def build_robustness_factories(
    selected_features: Sequence[str],
    league_effect_columns: Sequence[str],
) -> dict[str, PredictorFactory]:
    """Build only the new pooled checks plus the required raw-market reference."""

    selected = list(selected_features)
    validate_feature_blocks(selected)
    league_effects = list(league_effect_columns)
    factories: dict[str, PredictorFactory] = {"closing_market": ClosingMarket}

    for block_name, removed_features in FEATURE_BLOCKS.items():
        remaining = [feature for feature in selected if feature not in removed_features]
        model_columns = [*diff_columns(remaining), *league_effects]
        model_name = f"without_{block_name}"
        factories[model_name] = (
            lambda columns=model_columns, name=model_name: MarketPlusPlayerFormModel(
                player_features=columns,
                name=name,
            )
        )

    shuffled_features = [
        shuffled_column(feature)
        if feature in PLAYER_FEATURES_TO_SHUFFLE
        else f"diff_{feature}"
        for feature in selected
    ]
    factories["shuffled_player_features"] = lambda: MarketPlusPlayerFormModel(
        player_features=[*shuffled_features, *league_effects],
        name="shuffled_player_features",
    )
    factories["deliberate_hindsight_model"] = lambda: MarketPlusPlayerFormModel(
        player_features=[*diff_columns(selected), HINDSIGHT_FEATURE, *league_effects],
        name="deliberate_hindsight_model",
    )
    return factories


def build_primary_summary(
    main_metrics: pd.DataFrame,
    check_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine existing primary results with new checks and compare with final."""

    primary_names = {
        "closing_market",
        "recalibrated_market",
        "market_plus_player_form",
    }
    primary = main_metrics[
        main_metrics["training_scope"].eq("pooled")
        & main_metrics["model"].isin(primary_names)
    ].copy()
    if set(primary["model"]) != primary_names:
        missing = sorted(primary_names.difference(primary["model"]))
        raise ValueError(f"Main evaluation is missing primary model results: {missing}")

    checks = check_metrics[
        check_metrics["training_scope"].eq("pooled")
        & ~check_metrics["model"].eq("closing_market")
    ].copy()
    combined = pd.concat([primary, checks], ignore_index=True)
    final_row = combined[combined["model"].eq("market_plus_player_form")]
    if len(final_row) != 1:
        raise ValueError("Expected one pooled final-model result")
    final_log_loss = float(final_row.iloc[0]["log_loss"])
    final_brier = float(final_row.iloc[0]["brier_score"])
    combined["log_loss_change_vs_final"] = combined["log_loss"] - final_log_loss
    combined["brier_change_vs_final"] = combined["brier_score"] - final_brier
    combined["interpretation"] = combined["model"].map(
        {
            "closing_market": "raw market reference",
            "recalibrated_market": "market-only reference",
            "market_plus_player_form": "final pooled model",
            "without_attacking_output": "positive change means attacking block helped",
            "without_defensive_output": "positive change means defensive block helped",
            "without_player_ratings": "positive change means ratings block helped",
            "without_team_strength_context": "positive change means team context helped",
            "shuffled_player_features": "positive change means real player ordering helped",
            "deliberate_hindsight_model": "invalid; improvement demonstrates hindsight leakage",
        }
    )
    return combined.sort_values("log_loss", kind="stable").reset_index(drop=True)


def validate_same_matches(
    main_predictions: pd.DataFrame,
    check_predictions: pd.DataFrame,
) -> None:
    """Confirm the old and new pooled comparisons use identical match keys."""

    keys = ["league", "season", "match_id"]
    main = main_predictions[
        main_predictions["training_scope"].eq("pooled")
        & main_predictions["model"].eq("market_plus_player_form")
    ][keys].drop_duplicates()
    checks = check_predictions[
        check_predictions["training_scope"].eq("pooled")
        & check_predictions["model"].eq("closing_market")
    ][keys].drop_duplicates()
    merged = main.merge(checks, on=keys, how="outer", indicator=True)
    if not merged["_merge"].eq("both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"Main and robustness evaluations use different matches: {counts}")


def settings_record(selected_features: Sequence[str]) -> Mapping[str, object]:
    return {
        "training_scope": "pooled only",
        "development_only": True,
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_groups": ["league", "season"],
        "shuffled_features": list(PLAYER_FEATURES_TO_SHUFFLE),
        "feature_blocks": {name: list(features) for name, features in FEATURE_BLOCKS.items()},
        "selected_features": list(selected_features),
        "hindsight_feature": HINDSIGHT_FEATURE,
        "hindsight_warning": (
            "Deliberately invalid complete-season result information; never use in rankings."
        ),
    }
