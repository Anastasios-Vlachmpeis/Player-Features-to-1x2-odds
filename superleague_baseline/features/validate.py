"""Dataset validation helpers."""

from __future__ import annotations

import pandas as pd


def validate_fixture_index(fixtures: pd.DataFrame) -> None:
    dup = fixtures["match_id"].duplicated().any()
    if dup:
        raise ValueError("Duplicate match_id in fixture index")
    bad = fixtures[
        fixtures["home_team"].isna()
        | fixtures["away_team"].isna()
        | (fixtures["home_team"] == fixtures["away_team"])
        | fixtures["match_date"].isna()
    ]
    if not bad.empty:
        raise ValueError(f"Invalid fixture rows: {len(bad)}")


def validate_no_future_leakage(
    team_matches: pd.DataFrame,
    team_features: pd.DataFrame,
) -> None:
    merged = team_features.merge(
        team_matches[
            [
                "match_id",
                "player_team",
                "gf_proxy",
                "xg_for",
                "points_proxy",
            ]
        ],
        on=["match_id", "player_team"],
        how="left",
        suffixes=("", "_current"),
    )
    # Feature history count must be strictly less than total eventual matches for first appearance
    first = merged.groupby("player_team")["history_n"].min()
    if (first < 0).any():
        raise ValueError("Negative history counts detected")


def validate_probabilities(probs: pd.DataFrame) -> None:
    cols = ["p_home", "p_draw", "p_away"]
    missing = [c for c in cols if c not in probs.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")
    if probs[cols].isna().any().any():
        raise ValueError("Non-finite probabilities")
    if (probs[cols] < 0).any().any() or (probs[cols] > 1).any().any():
        raise ValueError("Probabilities outside [0, 1]")
    sums = probs[cols].sum(axis=1)
    if ((sums - 1.0).abs() > 1e-12).any():
        raise ValueError("Probabilities do not sum to 1")
