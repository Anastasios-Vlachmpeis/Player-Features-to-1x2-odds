from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


SCOTLAND_RESEARCH_DIR = Path(__file__).resolve().parents[1] / "scotland_research"
if str(SCOTLAND_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCOTLAND_RESEARCH_DIR))

from build_all_leagues import add_league_identity, validate_combined  # noqa: E402
from league_config import DEVELOPMENT_SEASONS, FINAL_SEASON, LEAGUES  # noqa: E402
from validate_dataset import match_key  # noqa: E402


def sample_model_rows(season: str = "2024-25") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["mt_1"],
            "season": [season],
            "match_date": ["2025-01-01"],
            "home_team_id": ["tm_1"],
            "away_team_id": ["tm_2"],
        }
    )


def test_league_identity_preserves_source_ids_and_prefixes_model_ids():
    output = add_league_identity(sample_model_rows(), "scotland")

    assert output.loc[0, "league"] == "scotland"
    assert output.loc[0, "source_match_id"] == "mt_1"
    assert output.loc[0, "match_id"] == "scotland:mt_1"
    assert output.loc[0, "source_home_team_id"] == "tm_1"
    assert output.loc[0, "home_team_id"] == "scotland:tm_1"
    assert output.loc[0, "source_away_team_id"] == "tm_2"
    assert output.loc[0, "away_team_id"] == "scotland:tm_2"


def test_development_validation_rejects_final_season():
    final = add_league_identity(sample_model_rows(FINAL_SEASON), "scotland")

    with pytest.raises(ValueError, match="unexpected seasons|must not contain"):
        validate_combined(final, ["scotland"], include_final=False)


def test_development_seasons_do_not_include_final_season():
    assert FINAL_SEASON not in DEVELOPMENT_SEASONS


def test_team_aliases_are_league_specific():
    matches = pd.DataFrame(
        {
            "season": ["2024-25"],
            "match_date": ["2025-01-01"],
            "home_team": ["Olympiacos FC"],
            "away_team": ["AEK Athens"],
        }
    )

    greek_key = match_key(matches, LEAGUES["greece"].team_aliases).iloc[0]
    scottish_key = match_key(matches, LEAGUES["scotland"].team_aliases).iloc[0]

    assert greek_key.endswith("|olympiakos|aek")
    assert scottish_key.endswith("|olympiacos|aekathens")
