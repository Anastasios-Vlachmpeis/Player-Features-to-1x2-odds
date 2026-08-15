from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


SCOTLAND_RESEARCH_DIR = Path(__file__).resolve().parents[1] / "scotland_research"
if str(SCOTLAND_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCOTLAND_RESEARCH_DIR))

from constants import DEVELOPMENT_EXCLUDED_LEAGUES, EXPECTED_LEAGUES  # noqa: E402
from match_rules import (  # noqa: E402
    MATCH_CHECKS,
    VALID_COMPETITION_PHASES,
    add_exclusion_reasons,
    competition_phase,
    meets_match_rules,
)


def match_rule_frame(**failures: bool) -> pd.DataFrame:
    values = {check: True for check in MATCH_CHECKS}
    values.update(failures)
    values["model_ready"] = all(values[check] for check in MATCH_CHECKS)
    return pd.DataFrame([values])


@pytest.mark.parametrize(
    ("league", "season", "matchday", "expected"),
    [
        ("scotland", "2024-25", 33, "regular_season"),
        ("scotland", "2024-25", 34, "post_split"),
        ("belgium", "2022-23", 34, "regular_season"),
        ("belgium", "2022-23", 35, "playoffs"),
        ("belgium", "2023-24", 30, "regular_season"),
        ("belgium", "2023-24", 31, "playoffs"),
        ("netherlands", "2024-25", 35, "playoffs"),
        ("portugal", "2024-25", 34, "regular_season"),
        ("turkey", "2020-21", 42, "regular_season"),
    ],
)
def test_competition_phase_rules(league, season, matchday, expected):
    phase = competition_phase(league, season, matchday)
    assert phase == expected
    assert phase in VALID_COMPETITION_PHASES


def test_invalid_matchday_is_rejected():
    with pytest.raises(ValueError, match="Invalid matchday"):
        competition_phase("scotland", "2024-25", None)


def test_every_ready_match_passes_every_inclusion_check():
    frame = match_rule_frame()
    assert meets_match_rules(frame).all()
    assert frame.loc[frame["model_ready"], list(MATCH_CHECKS)].all(axis=None)


def test_excluded_match_gets_one_primary_reason_and_all_failed_checks():
    frame = match_rule_frame(
        closing_odds_available=False,
        twenty_two_starters=False,
        starter_minutes_complete=False,
    )
    output = add_exclusion_reasons(frame)

    assert not output.loc[0, "model_ready"]
    assert output.loc[0, "exclusion_reason"] == "missing_closing_odds"
    assert output.loc[0, "failed_match_checks"] == (
        "closing_odds_available|twenty_two_starters|starter_minutes_complete"
    )


@pytest.mark.parametrize(
    ("failed_check", "expected_reason"),
    [
        ("completed_match", "match_not_completed"),
        ("score_matches_football_data", "score_disagreement"),
        ("player_data_available", "no_player_data"),
        ("twenty_two_starters", "invalid_starter_count"),
        ("starter_identity_complete", "missing_starter_identity"),
        ("starter_minutes_complete", "missing_starter_minutes"),
        ("player_team_ids_valid", "invalid_player_team_assignment"),
    ],
)
def test_granular_exclusion_reasons(failed_check, expected_reason):
    output = add_exclusion_reasons(match_rule_frame(**{failed_check: False}))
    assert output.loc[0, "exclusion_reason"] == expected_reason


def test_greece_is_explicitly_outside_the_five_league_rules():
    assert DEVELOPMENT_EXCLUDED_LEAGUES == frozenset({"greece"})
    assert EXPECTED_LEAGUES == {
        "scotland",
        "belgium",
        "netherlands",
        "portugal",
        "turkey",
    }
