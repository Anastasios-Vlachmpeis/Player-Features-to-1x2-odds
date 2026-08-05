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
    """Verify history counts include only fixtures from strictly earlier dates."""
    keys = ["match_id", "match_date", "player_team"]
    if team_features.duplicated(keys).any():
        raise ValueError("Duplicate team feature rows")

    date_counts = (
        team_matches.groupby(["player_team", "match_date"], as_index=False)
        .size()
        .sort_values(["player_team", "match_date"])
    )
    date_counts["expected_history_n"] = (
        date_counts.groupby("player_team")["size"].cumsum() - date_counts["size"]
    )
    checked = team_features.merge(
        date_counts[["player_team", "match_date", "expected_history_n"]],
        on=["player_team", "match_date"],
        how="left",
        validate="many_to_one",
    )
    if checked["expected_history_n"].isna().any():
        raise ValueError("Could not derive expected history count")
    if not checked["history_n"].eq(checked["expected_history_n"]).all():
        raise ValueError("Feature history includes current-date or future fixtures")


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
