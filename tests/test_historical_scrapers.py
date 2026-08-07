import pytest

from football_data_scraper import (
    iter_seasons,
    parse_season_csv,
    season_url,
)


def test_calendar_range_maps_to_eleven_seasons():
    seasons = list(iter_seasons(2015, 2026))
    assert seasons[0] == (2015, 2016)
    assert seasons[-1] == (2025, 2026)
    assert len(seasons) == 11
    assert season_url(2015, 2016).endswith("/1516/G1.csv")


def test_csv_parser_prefers_market_average_closing_and_removes_vig():
    text = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,"
        "AvgCH,AvgCD,AvgCA,B365CH,B365CD,B365CA\n"
        "G1,01/09/2025,Home FC,Away FC,2,1,H,2.0,3.5,4.0,1.9,3.4,3.8\n"
    )
    row = parse_season_csv(text, start_year=2025, end_year=2026)[0]
    assert row["result_3way"] == "H"
    assert row["odds_source"] == "market_average_closing"
    assert row["odds_is_closing"] is True
    assert row["match_date"] == "2025-09-01"
    assert row["market_p_home"] + row["market_p_draw"] + row["market_p_away"] == pytest.approx(1.0)


def test_preclosing_fallback_is_explicitly_flagged():
    text = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA\n"
        "G1,02/09/15,Home FC,Away FC,0,0,D,2.0,3.0,4.0\n"
    )
    row = parse_season_csv(text, start_year=2015, end_year=2016)[0]
    assert row["odds_source"] == "market_average_preclosing"
    assert row["odds_is_closing"] is False


def test_result_disagreement_is_rejected():
    text = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "G1,02/09/15,Home FC,Away FC,1,0,A\n"
    )
    with pytest.raises(ValueError, match="Result mismatch"):
        parse_season_csv(text, start_year=2015, end_year=2016)
