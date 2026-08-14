# Shared constants for multi-league walk-forward model evaluation.

from __future__ import annotations

from pathlib import Path

from build_match_features import TEAM_FEATURES
from league_config import DEVELOPMENT_SEASONS, LEAGUES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "all_leagues"
    / "development_model_dataset.csv"
)
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "artifacts" / "five_league_development_evaluation"

# Greece is retained in the combined dataset but temporarily excluded from
# evaluation because 2020-21 has no valid starting lineups and 2021-22 has one.
DEVELOPMENT_EXCLUDED_LEAGUES = frozenset({"greece"})
EXPECTED_LEAGUES = frozenset(LEAGUES).difference(DEVELOPMENT_EXCLUDED_LEAGUES)

CLASS_ORDER = ["H", "D", "A"]
PLAYER_FEATURES = [f"diff_{feature}" for feature in TEAM_FEATURES]
EXPANDED_PLAYER_FEATURES = [
    f"{side}_{feature}"
    for side in ("home", "away")
    for feature in TEAM_FEATURES
]
MARKET_FEATURES = ["market_log_home_vs_draw", "market_log_away_vs_draw"]

# Fixed research thresholds: at most 1% worse is very close; at most 2% worse is close.
VERY_CLOSE_THRESHOLD = 0.01
CLOSE_THRESHOLD = 0.02

DEVELOPMENT_FOLDS = tuple(
    (test_season, DEVELOPMENT_SEASONS[:test_index])
    for test_index, test_season in enumerate(DEVELOPMENT_SEASONS[2:], start=2)
)

REQUIRED_COLUMNS = {
    "league",
    "match_id",
    "season",
    "match_date",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_score",
    "away_score",
    "result_3way",
    "market_home_probability",
    "market_draw_probability",
    "market_away_probability",
    *PLAYER_FEATURES,
    *EXPANDED_PLAYER_FEATURES,
}
