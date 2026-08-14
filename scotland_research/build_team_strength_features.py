"""Build leakage-safe, pre-match team-strength features for Scotland fixtures."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_match_dataset import DEFAULT_OUTPUT_DIR, MATCH_DATASET_NAME, write_csv_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCH_DATASET = DEFAULT_OUTPUT_DIR / MATCH_DATASET_NAME
TEAM_STRENGTH_NAME = "team_strength_features.csv"

# Elo uses the conventional 400-point logistic scale. A fixed 60-point home
# advantage is included before the match rather than estimated from its result.
ELO_INITIAL_RATING = 1500.0
ELO_HOME_ADVANTAGE = 60.0
ELO_K_FACTOR = 20.0

# Attack and defence are exponentially weighted goal rates. Alpha 0.20 gives
# the latest match 20% weight while retaining a stable multi-match baseline.
GOAL_RATE_INITIAL = 1.35
GOAL_RATE_ALPHA = 0.20
HOME_GOAL_MULTIPLIER = 1.10

FEATURE_COLUMNS = [
    "elo_rating",
    "opponent_elo_rating",
    "elo_difference",
    "elo_expected_result",
    "prior_team_matches",
    "attack_goal_rate_ewm",
    "defence_goal_rate_ewm",
    "expected_goals_strength",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dataset", type=Path, default=DEFAULT_MATCH_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def elo_expectation(rating: float, opponent_rating: float, home_adjustment: float) -> float:
    # E = 1 / (1 + 10^(-(rating + home advantage - opponent) / 400)).
    return 1.0 / (1.0 + 10.0 ** (-(rating + home_adjustment - opponent_rating) / 400.0))


def build_team_strength(matches: pd.DataFrame) -> pd.DataFrame:
    required = {"match_id", "utc_date", "home_team_id", "away_team_id", "home_score", "away_score"}
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(f"Match dataset is missing team-strength inputs: {', '.join(missing)}")

    ordered = matches.copy()
    ordered["_datetime"] = pd.to_datetime(ordered["utc_date"], utc=True, errors="raise")
    ordered[["home_score", "away_score"]] = ordered[["home_score", "away_score"]].apply(pd.to_numeric, errors="raise")
    ordered = ordered.sort_values(["_datetime", "match_id"], kind="stable")

    ratings: defaultdict[str, float] = defaultdict(lambda: ELO_INITIAL_RATING)
    attack: defaultdict[str, float] = defaultdict(lambda: GOAL_RATE_INITIAL)
    defence: defaultdict[str, float] = defaultdict(lambda: GOAL_RATE_INITIAL)
    appearances: defaultdict[str, int] = defaultdict(int)
    rows: list[dict[str, object]] = []

    # Fixtures sharing a kickoff timestamp are all emitted before any result in
    # that batch is applied, preventing same-time matches from leaking outcomes.
    for _, batch in ordered.groupby("_datetime", sort=True):
        pending_updates: list[tuple[str, str, float, float, float, float]] = []
        for match in batch.itertuples(index=False):
            home_id = str(match.home_team_id)
            away_id = str(match.away_team_id)
            home_rating = ratings[home_id]
            away_rating = ratings[away_id]
            home_expectation = elo_expectation(home_rating, away_rating, ELO_HOME_ADVANTAGE)
            away_expectation = 1.0 - home_expectation

            # Expected goals combine the team's scoring rate with the opponent's
            # conceding rate: attack * opponent defence / league baseline.
            home_xg = HOME_GOAL_MULTIPLIER * attack[home_id] * defence[away_id] / GOAL_RATE_INITIAL
            away_xg = attack[away_id] * defence[home_id] / GOAL_RATE_INITIAL
            side_values = (
                (home_id, away_id, "home", home_rating, away_rating, home_expectation, home_xg),
                (away_id, home_id, "away", away_rating, home_rating, away_expectation, away_xg),
            )
            for team_id, opponent_id, side, rating, opponent_rating, expectation, expected_goals in side_values:
                rows.append(
                    {
                        "match_id": str(match.match_id),
                        "team_id": team_id,
                        "opponent_team_id": opponent_id,
                        "team_side": side,
                        "elo_rating": rating,
                        "opponent_elo_rating": opponent_rating,
                        "elo_difference": rating - opponent_rating,
                        "elo_expected_result": expectation,
                        "prior_team_matches": appearances[team_id],
                        "attack_goal_rate_ewm": attack[team_id],
                        "defence_goal_rate_ewm": defence[team_id],
                        "expected_goals_strength": float(np.clip(expected_goals, 0.15, 4.0)),
                    }
                )

            home_goals = float(match.home_score)
            away_goals = float(match.away_score)
            pending_updates.append((home_id, away_id, home_goals, away_goals, home_expectation, away_expectation))

        for home_id, away_id, home_goals, away_goals, home_expectation, away_expectation in pending_updates:
            actual_home = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
            # A logarithmic goal-margin multiplier rewards decisive results without
            # letting one unusual scoreline dominate the sequential rating.
            margin = max(abs(home_goals - away_goals), 1.0)
            change = ELO_K_FACTOR * np.log1p(margin) * (actual_home - home_expectation)
            ratings[home_id] += change
            ratings[away_id] -= change
            attack[home_id] = (1.0 - GOAL_RATE_ALPHA) * attack[home_id] + GOAL_RATE_ALPHA * home_goals
            attack[away_id] = (1.0 - GOAL_RATE_ALPHA) * attack[away_id] + GOAL_RATE_ALPHA * away_goals
            defence[home_id] = (1.0 - GOAL_RATE_ALPHA) * defence[home_id] + GOAL_RATE_ALPHA * away_goals
            defence[away_id] = (1.0 - GOAL_RATE_ALPHA) * defence[away_id] + GOAL_RATE_ALPHA * home_goals
            appearances[home_id] += 1
            appearances[away_id] += 1

    output = pd.DataFrame(rows)
    if output.duplicated(["match_id", "team_id"]).any():
        raise ValueError("Team-strength output contains duplicate match/team rows")
    return output


def build_output(match_dataset_path: Path, output_dir: Path) -> Path:
    matches = pd.read_csv(match_dataset_path, dtype={"match_id": "string", "home_team_id": "string", "away_team_id": "string"})
    output = build_team_strength(matches)
    output_path = output_dir / TEAM_STRENGTH_NAME
    write_csv_atomic(output, output_path)
    print(f"Saved {len(output)} pre-match team-strength rows to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    build_output(args.match_dataset, args.output_dir)


if __name__ == "__main__":
    main()
