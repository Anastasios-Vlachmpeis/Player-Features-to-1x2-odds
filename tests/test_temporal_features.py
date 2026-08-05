import sqlite3

import pandas as pd

from superleague_baseline.features.dataset import (
    build_historical_match_dataset,
    feature_columns,
)


def test_feature_output_is_deterministic(synthetic_db):
    a = build_historical_match_dataset(synthetic_db, min_history=3)
    b = build_historical_match_dataset(synthetic_db, min_history=3)
    pd.testing.assert_frame_equal(a, b)


def test_missing_history_is_flagged(synthetic_db):
    df = build_historical_match_dataset(synthetic_db, min_history=3)
    assert (df["home_history_n"] >= 3).all()
    assert (df["away_history_n"] >= 3).all()


def test_features_use_strictly_prior_dates(synthetic_db):
    before = build_historical_match_dataset(synthetic_db, min_history=1)
    target_date = pd.Timestamp("2025-10-22")

    with sqlite3.connect(synthetic_db) as conn:
        conn.execute(
            "UPDATE sofascore_match_stats SET goals = goals + 10 WHERE match_date = ?",
            (target_date.strftime("%Y-%m-%d"),),
        )
        conn.commit()

    after = build_historical_match_dataset(synthetic_db, min_history=1)
    cols = feature_columns(before)
    same_day = before["match_date"] == target_date
    future = before["match_date"] > target_date

    pd.testing.assert_frame_equal(
        before.loc[same_day, cols].reset_index(drop=True),
        after.loc[same_day, cols].reset_index(drop=True),
    )
    assert not before.loc[future, cols].reset_index(drop=True).equals(
        after.loc[future, cols].reset_index(drop=True)
    )


def test_incomplete_match_is_excluded_from_lineup_form(synthetic_db):
    with sqlite3.connect(synthetic_db) as conn:
        conn.execute(
            """
            UPDATE sofascore_match_stats
            SET is_starter = 0
            WHERE match_id = 1000 AND player_team = 'Beta' AND sofascore_id = 1100
            """
        )
        conn.commit()

    df = build_historical_match_dataset(synthetic_db, min_history=1)
    next_match = df.loc[df["match_id"] == 1002].iloc[0]
    assert next_match["home_lineup_for_l5_obs"] == 0
    assert next_match["away_lineup_for_l5_obs"] == 0
    assert pd.isna(next_match["home_points_proxy_l5_mean"])
    assert pd.isna(next_match["away_points_proxy_l5_mean"])
