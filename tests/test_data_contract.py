import pytest

from superleague_baseline.features.sources import build_fixture_index, load_sofascore_sources
from superleague_baseline.features.validate import validate_fixture_index


def test_rejects_invalid_fixture_fields():
    import pandas as pd

    fixtures = pd.DataFrame(
        {
            "match_id": [1, 2],
            "match_date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "home_team": ["A", "B"],
            "away_team": ["B", "B"],
        }
    )
    try:
        validate_fixture_index(fixtures)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_fixture_index_unique(synthetic_db):
    lineup, _ = load_sofascore_sources(synthetic_db)
    fixtures = build_fixture_index(lineup)
    validate_fixture_index(fixtures)
    assert fixtures["match_id"].is_unique


def test_fixture_index_rejects_conflicting_source_metadata(synthetic_db):
    lineup, _ = load_sofascore_sources(synthetic_db)
    lineup.loc[lineup.index[0], "away_team"] = "Wrong Team"
    with pytest.raises(ValueError, match="Inconsistent or incomplete"):
        build_fixture_index(lineup)


def test_fixture_index_rejects_missing_match_side(synthetic_db):
    lineup, _ = load_sofascore_sources(synthetic_db)
    match_id = lineup["match_id"].iloc[0]
    side = lineup.loc[lineup["match_id"] == match_id, "player_team"].iloc[0]
    incomplete = lineup.loc[
        ~((lineup["match_id"] == match_id) & (lineup["player_team"] == side))
    ]
    with pytest.raises(ValueError, match="Inconsistent or incomplete"):
        build_fixture_index(incomplete)
