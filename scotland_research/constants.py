# Shared constants for Scotland walk-forward model evaluation.

from __future__ import annotations

from pathlib import Path

from build_match_features import TEAM_FEATURES
from build_match_dataset import DEFAULT_OUTPUT_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DATASET = DEFAULT_OUTPUT_DIR / "scotland_model_dataset.csv"
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "artifacts" / "scotland_model_evaluation"

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

FOLDS = [
    ("2022-23", ["2020-21", "2021-22"]),
    ("2023-24", ["2020-21", "2021-22", "2022-23"]),
    ("2024-25", ["2020-21", "2021-22", "2022-23", "2023-24"]),
]

REQUIRED_COLUMNS = {
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
