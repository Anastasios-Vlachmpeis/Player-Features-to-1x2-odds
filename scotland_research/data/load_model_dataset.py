# Load and validate the combined six-league development dataset.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from constants import (
    CLASS_ORDER,
    EXPANDED_PLAYER_FEATURES,
    PLAYER_FEATURES,
    REQUIRED_COLUMNS,
)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Model dataset does not exist: {path}. Run step 4 first.")
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    if frame["match_id"].duplicated().any():
        raise ValueError("Model dataset contains duplicate match IDs")
    if frame["league"].isna().any() or frame["league"].astype(str).str.strip().eq("").any():
        raise ValueError("league must be populated for every match")
    frame["league"] = frame["league"].astype(str).str.strip().str.lower()
    if not frame["result_3way"].isin(CLASS_ORDER).all():
        raise ValueError("result_3way must contain only H, D, or A")

    for column in ("home_team_id", "away_team_id"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must be populated for every match")
        frame[column] = frame[column].astype(str)

    for column in ("home_score", "away_score"):
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or values.lt(0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain non-negative integer scores")
        frame[column] = values.astype(int)

    frame["_match_datetime"] = pd.to_datetime(
        frame["match_date"], utc=True, errors="coerce"
    )
    if frame["_match_datetime"].isna().any():
        raise ValueError("match_date contains an invalid or missing date")

    numeric_columns = PLAYER_FEATURES + EXPANDED_PLAYER_FEATURES + [
        "market_home_probability",
        "market_draw_probability",
        "market_away_probability",
    ]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    if frame[numeric_columns].isna().any().any():
        raise ValueError("Model inputs contain missing values")

    market_columns = [
        "market_home_probability",
        "market_draw_probability",
        "market_away_probability",
    ]
    if not np.allclose(frame[market_columns].sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Devigged market probabilities do not sum to one")
    if frame[market_columns].le(0).any().any():
        raise ValueError("Market probabilities must be positive")

    frame["market_log_home_vs_draw"] = np.log(
        frame["market_home_probability"] / frame["market_draw_probability"]
    )
    frame["market_log_away_vs_draw"] = np.log(
        frame["market_away_probability"] / frame["market_draw_probability"]
    )
    return frame.sort_values(["_match_datetime", "match_id"], kind="stable").reset_index(drop=True)
