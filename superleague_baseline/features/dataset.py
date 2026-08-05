"""Assemble one-row-per-match historical dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from superleague_baseline.constants import DEFAULT_MIN_HISTORY
from superleague_baseline.features.lagged import compute_date_batched_features
from superleague_baseline.features.sources import (
    build_fixture_index,
    load_sofascore_sources,
)
from superleague_baseline.features.targets import build_proxy_targets
from superleague_baseline.features.team_match import (
    aggregate_lineup_team_match,
    aggregate_xg_team_match,
    attach_opponent_metrics,
    build_side_skeleton,
)


def _prefix_side(df: pd.DataFrame, side: str, prefix: str) -> pd.DataFrame:
    side_df = df[df["venue"] == side].copy()
    keep = {"match_id", "match_date", "home_team", "away_team"}
    rename = {c: f"{prefix}_{c}" for c in side_df.columns if c not in keep}
    return side_df.rename(columns=rename)


def pivot_one_row_per_match(team_features: pd.DataFrame) -> pd.DataFrame:
    """Merge home and away prefixed feature rows into one match row."""
    home = _prefix_side(team_features, "H", "home")
    away = _prefix_side(team_features, "A", "away")
    keys = ["match_id", "match_date", "home_team", "away_team"]
    out = home.merge(away, on=keys, how="inner", suffixes=("", ""))

    delta_pairs = [
        ("points_proxy_l5_mean", "delta_points_proxy_l5_mean"),
        ("xg_balance_l5_mean", "delta_xg_balance_l5_mean"),
        ("sot_balance_l5_mean", "delta_sot_balance_l5_mean"),
        ("pass_completion_l5_ratio", "delta_pass_completion_l5_ratio"),
        ("rating_l5_minutes_weighted", "delta_rating_l5_minutes_weighted"),
        ("rest_days", "delta_rest_days"),
    ]
    for base, delta in delta_pairs:
        hcol, acol = f"home_{base}", f"away_{base}"
        if hcol in out.columns and acol in out.columns:
            out[delta] = out[hcol] - out[acol]
    return out.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def build_historical_match_dataset(
    db_path: str | Path,
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
) -> pd.DataFrame:
    lineup, xg = load_sofascore_sources(db_path)
    fixtures = build_fixture_index(lineup)
    skeleton = build_side_skeleton(fixtures)
    lineup_agg = aggregate_lineup_team_match(lineup)
    xg_agg = aggregate_xg_team_match(xg)

    team_matches = skeleton.merge(
        lineup_agg,
        on=["match_id", "match_date", "home_team", "away_team", "player_team"],
        how="left",
    ).merge(
        xg_agg,
        on=["match_id", "player_team"],
        how="left",
    )
    team_matches = attach_opponent_metrics(team_matches)
    team_features = compute_date_batched_features(team_matches)
    match_features = pivot_one_row_per_match(team_features)
    targets = build_proxy_targets(team_matches)
    dataset = match_features.merge(targets, on=["match_id", "match_date", "home_team", "away_team"])

    eligible = (dataset["home_history_n"] >= min_history) & (
        dataset["away_history_n"] >= min_history
    )
    return dataset.loc[eligible].reset_index(drop=True)


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    exclude = {
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "proxy_home_goals",
        "proxy_away_goals",
        "proxy_result_3way",
        "proxy_lineups_complete",
        "target_is_official",
        "home_player_team",
        "away_player_team",
    }
    return [
        c
        for c in dataset.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(dataset[c])
    ]
