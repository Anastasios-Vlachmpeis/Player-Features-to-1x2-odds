import pytest

from football_data_scraper import (
    DEFAULT_LEAGUE_KEYS,
    LEAGUES,
    iter_seasons,
    parse_season_csv,
    resolve_league_keys,
    season_url,
)


def test_default_league_registry_covers_six_target_divisions():
    assert DEFAULT_LEAGUE_KEYS == (
        "greece",
        "turkey",
        "netherlands",
        "portugal",
        "belgium",
        "scotland",
    )
    assert LEAGUES["greece"]["division"] == "G1"
    assert LEAGUES["turkey"]["division"] == "T1"
    assert LEAGUES["netherlands"]["division"] == "N1"
    assert LEAGUES["portugal"]["division"] == "P1"
    assert LEAGUES["belgium"]["division"] == "B1"
    assert LEAGUES["scotland"]["division"] == "SC0"


def test_resolve_league_keys_preserves_registry_order():
    assert resolve_league_keys(["portugal", "greece"]) == ["greece", "portugal"]
    assert resolve_league_keys(None) == list(DEFAULT_LEAGUE_KEYS)


def test_unknown_league_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown league key"):
        resolve_league_keys(["greece", "spain"])


def test_calendar_range_maps_to_eleven_seasons():
    seasons = list(iter_seasons(2015, 2026))
    assert seasons[0] == (2015, 2016)
    assert seasons[-1] == (2025, 2026)
    assert len(seasons) == 11
    assert season_url(2015, 2016, "G1").endswith("/1516/G1.csv")
    assert season_url(2015, 2016, "T1").endswith("/1516/T1.csv")


def test_csv_parser_prefers_market_average_closing_and_removes_vig():
    text = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,"
        "AvgCH,AvgCD,AvgCA,B365CH,B365CD,B365CA\n"
        "G1,01/09/2025,Home FC,Away FC,2,1,H,2.0,3.5,4.0,1.9,3.4,3.8\n"
    )
    row = parse_season_csv(
        text,
        start_year=2025,
        end_year=2026,
        division="G1",
    )[0]
    assert row["result_3way"] == "H"
    assert row["division"] == "G1"
    assert row["odds_source"] == "market_average_closing"
    assert row["odds_is_closing"] is True
    assert row["match_date"] == "2025-09-01"
    assert row["market_p_home"] + row["market_p_draw"] + row["market_p_away"] == pytest.approx(1.0)


def test_preclosing_fallback_is_explicitly_flagged():
    text = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA\n"
        "N1,02/09/15,Home FC,Away FC,0,0,D,2.0,3.0,4.0\n"
    )
    row = parse_season_csv(
        text,
        start_year=2015,
        end_year=2016,
        division="N1",
    )[0]
    assert row["division"] == "N1"
    assert row["odds_source"] == "market_average_preclosing"
    assert row["odds_is_closing"] is False


def test_result_disagreement_is_rejected():
    text = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "G1,02/09/15,Home FC,Away FC,1,0,A\n"
    )
    with pytest.raises(ValueError, match="Result mismatch"):
        parse_season_csv(
            text,
            start_year=2015,
            end_year=2016,
            division="G1",
        )
