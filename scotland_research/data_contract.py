"""Shared, auditable inclusion rules for the multi-league match sample."""

from __future__ import annotations

import pandas as pd


CONTRACT_CHECKS = (
    "completed_match",
    "football_data_match",
    "score_matches_football_data",
    "closing_odds_available",
    "player_data_available",
    "twenty_two_starters",
    "starter_identity_complete",
    "starter_minutes_complete",
    "player_team_ids_valid",
)

VALID_COMPETITION_PHASES = frozenset({"regular_season", "post_split", "playoffs"})

# Belgium reduced its regular season from 34 to 30 matchdays in 2023-24.
BELGIUM_REGULAR_SEASON_END = {
    "2020-21": 34,
    "2021-22": 34,
    "2022-23": 34,
    "2023-24": 30,
    "2024-25": 30,
    "2025-26": 30,
}

EXCLUSION_PRIORITY = (
    ("football_data_match", "not_top_division_match"),
    ("completed_match", "match_not_completed"),
    ("score_matches_football_data", "score_disagreement"),
    ("closing_odds_available", "missing_closing_odds"),
    ("player_data_available", "no_player_data"),
    ("twenty_two_starters", "invalid_starter_count"),
    ("starter_identity_complete", "missing_starter_identity"),
    ("starter_minutes_complete", "missing_starter_minutes"),
    ("player_team_ids_valid", "invalid_player_team_assignment"),
)


def competition_phase(
    league: str,
    season: str,
    matchday: int | float | str,
    stage_name: object = None,
) -> str:
    """Return the declared phase without using results or model performance."""
    day = pd.to_numeric(pd.Series([matchday]), errors="coerce").iloc[0]
    if pd.isna(day) or float(day) <= 0 or not float(day).is_integer():
        raise ValueError(
            f"Invalid matchday for phase labelling: league={league}, "
            f"season={season}, matchday={matchday!r}"
        )
    day = int(day)

    stage = "" if pd.isna(stage_name) else str(stage_name).strip().lower()
    if any(token in stage for token in ("playoff", "play-off", "championship", "relegation")):
        return "post_split" if league == "scotland" else "playoffs"

    if league == "scotland":
        return "regular_season" if day <= 33 else "post_split"
    if league == "belgium":
        if season not in BELGIUM_REGULAR_SEASON_END:
            raise ValueError(f"No Belgian phase rule declared for season {season}")
        return "regular_season" if day <= BELGIUM_REGULAR_SEASON_END[season] else "playoffs"
    if league in {"netherlands", "portugal"}:
        return "regular_season" if day <= 34 else "playoffs"
    if league in {"turkey", "greece"}:
        return "regular_season"
    raise ValueError(f"No competition-phase rule declared for league {league!r}")


def add_competition_phase(frame: pd.DataFrame, league: str) -> pd.DataFrame:
    required = {"season", "matchday"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot label competition phase; missing columns: {missing}")

    output = frame.copy()
    stages = output["stage_name"] if "stage_name" in output else pd.Series(None, index=output.index)
    output["matchday"] = pd.to_numeric(output["matchday"], errors="raise").astype(int)
    output["competition_phase"] = [
        competition_phase(league, season, matchday, stage)
        for season, matchday, stage in zip(output["season"], output["matchday"], stages)
    ]
    return output


def contract_ready(frame: pd.DataFrame) -> pd.Series:
    missing = sorted(set(CONTRACT_CHECKS).difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot apply data contract; missing checks: {missing}")
    return frame[list(CONTRACT_CHECKS)].fillna(False).astype(bool).all(axis=1)


def add_exclusion_reasons(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign one primary reason and retain every failed check for auditability."""
    output = frame.copy()
    expected_ready = contract_ready(output)
    if "model_ready" in output and not output["model_ready"].astype(bool).equals(expected_ready):
        raise ValueError("model_ready disagrees with the declared data contract")
    output["model_ready"] = expected_ready

    reason_by_check = dict(EXCLUSION_PRIORITY)
    output["failed_contract_checks"] = [
        "|".join(check for check in CONTRACT_CHECKS if not bool(row[check]))
        for _, row in output.iterrows()
    ]
    output["exclusion_reason"] = pd.NA
    for check, reason in reversed(EXCLUSION_PRIORITY):
        output.loc[~output[check].astype(bool), "exclusion_reason"] = reason

    excluded = ~output["model_ready"]
    unexplained = excluded & output["exclusion_reason"].isna()
    if unexplained.any():
        raise ValueError("At least one excluded match has no declared exclusion reason")
    if output.loc[~excluded, "failed_contract_checks"].ne("").any():
        raise ValueError("A model-ready match failed a declared contract check")
    if not set(output.loc[excluded, "exclusion_reason"]).issubset(set(reason_by_check.values())):
        raise ValueError("An undeclared exclusion reason was generated")
    return output
