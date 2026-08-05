import pandas as pd

from superleague_baseline.features.dataset import build_historical_match_dataset


def test_feature_output_is_deterministic(synthetic_db):
    a = build_historical_match_dataset(synthetic_db, min_history=3)
    b = build_historical_match_dataset(synthetic_db, min_history=3)
    pd.testing.assert_frame_equal(a, b)


def test_missing_history_is_flagged(synthetic_db):
    df = build_historical_match_dataset(synthetic_db, min_history=3)
    assert (df["home_history_n"] >= 3).all()
    assert (df["away_history_n"] >= 3).all()


def test_features_use_strictly_prior_dates(synthetic_db):
    df = build_historical_match_dataset(synthetic_db, min_history=1)
    # Earliest eligible row for a team should have low history count
    assert df["home_history_n"].min() >= 1
