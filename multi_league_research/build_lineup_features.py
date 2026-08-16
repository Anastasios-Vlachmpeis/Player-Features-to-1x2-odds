"""Build pre-match lineup continuity, familiarity, and replacement features."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from build_match_dataset import DEFAULT_OUTPUT_DIR, write_csv_atomic
from build_player_form import PLAYER_FORM_NAME
from validate_dataset import PLAYER_STATS_CSV, as_bool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAYER_FORM = DEFAULT_OUTPUT_DIR / PLAYER_FORM_NAME
LINEUP_FEATURES_NAME = "lineup_features.csv"
REGULAR_WINDOW = 5

FEATURE_COLUMNS = [
    "retained_starters",
    "new_starters",
    "missing_regular_starters",
    "mean_pairwise_prior_starts",
    "mean_pairwise_prior_minutes",
    "new_player_pairs",
    "replacement_quality",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-form", type=Path, default=DEFAULT_PLAYER_FORM)
    parser.add_argument("--player-stats", type=Path, default=PLAYER_STATS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def pair_key(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


def build_lineup_features(player_form: pd.DataFrame, player_stats: pd.DataFrame) -> pd.DataFrame:
    required = {"match_id", "utc_date", "team_id", "team_side", "player_id", "form_rating_mean_5"}
    missing = sorted(required.difference(player_form.columns))
    if missing:
        raise ValueError(f"Player-form table is missing lineup inputs: {', '.join(missing)}")

    stats = player_stats.copy()
    stats["started"] = as_bool(stats["started"])
    stats = stats[stats["started"]][["match_id", "player_id", "minutes_played"]].copy()
    stats[["match_id", "player_id"]] = stats[["match_id", "player_id"]].astype("string")
    stats["minutes_played"] = pd.to_numeric(stats["minutes_played"], errors="coerce").fillna(0.0)

    players = player_form.copy()
    players[["match_id", "player_id", "team_id"]] = players[["match_id", "player_id", "team_id"]].astype("string")
    players["form_rating_mean_5"] = pd.to_numeric(players["form_rating_mean_5"], errors="raise")
    players = players.merge(stats, on=["match_id", "player_id"], how="left", validate="one_to_one")
    players["minutes_played"] = players["minutes_played"].fillna(0.0)
    players["_datetime"] = pd.to_datetime(players["utc_date"], utc=True, errors="raise")
    players = players.sort_values(["_datetime", "match_id", "team_side", "player_id"], kind="stable")

    previous_lineup: dict[str, set[str]] = defaultdict(set)
    recent_lineups: dict[str, deque[set[str]]] = defaultdict(lambda: deque(maxlen=REGULAR_WINDOW))
    pair_starts: dict[str, defaultdict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
    pair_minutes: dict[str, defaultdict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
    last_quality: dict[str, dict[str, float]] = defaultdict(dict)
    rows: list[dict[str, object]] = []

    for (match_id, team_id), group in players.groupby(["match_id", "team_id"], sort=False):
        current = set(group["player_id"])
        if len(current) != 11:
            raise ValueError(f"{match_id}/{team_id} does not contain exactly 11 starters")
        prior = previous_lineup[team_id]
        frequencies = Counter(player for lineup in recent_lineups[team_id] for player in lineup)

        # A regular started at least three of the preceding five team fixtures.
        regulars = {player for player, starts in frequencies.items() if starts >= 3}
        usual_xi = {player for player, _ in frequencies.most_common(11)}
        pairs = [pair_key(first, second) for first, second in combinations(sorted(current), 2)]
        current_quality = float(group["form_rating_mean_5"].mean())
        usual_quality_values = [last_quality[team_id][player] for player in usual_xi if player in last_quality[team_id]]
        usual_quality = float(np.mean(usual_quality_values)) if usual_quality_values else current_quality

        # Familiarity is the mean prior co-start count/minutes over the 55 pairs
        # in an XI. Replacement quality compares current and usual-XI mean form.
        rows.append(
            {
                "match_id": match_id,
                "team_id": team_id,
                "team_side": str(group["team_side"].iloc[0]),
                "retained_starters": len(current & prior),
                "new_starters": len(current - prior) if prior else 0,
                "missing_regular_starters": len(regulars - current),
                "mean_pairwise_prior_starts": float(np.mean([pair_starts[team_id][pair] for pair in pairs])),
                "mean_pairwise_prior_minutes": float(np.mean([pair_minutes[team_id][pair] for pair in pairs])),
                "new_player_pairs": sum(pair_starts[team_id][pair] == 0 for pair in pairs),
                "replacement_quality": current_quality - usual_quality,
            }
        )

        minute_lookup = dict(zip(group["player_id"], group["minutes_played"], strict=True))
        for first, second in pairs:
            pair_starts[team_id][(first, second)] += 1
            # Shared minutes are conservatively approximated by the smaller of
            # the two appearance-minute totals because substitution times are absent.
            pair_minutes[team_id][(first, second)] += min(minute_lookup[first], minute_lookup[second])
        for player, quality in zip(group["player_id"], group["form_rating_mean_5"], strict=True):
            last_quality[team_id][player] = float(quality)
        previous_lineup[team_id] = current
        recent_lineups[team_id].append(current)

    output = pd.DataFrame(rows)
    if output.duplicated(["match_id", "team_id"]).any():
        raise ValueError("Lineup feature output contains duplicate match/team rows")
    return output


def build_output(player_form_path: Path, player_stats_path: Path, output_dir: Path) -> Path:
    player_form = pd.read_csv(player_form_path, dtype="string", keep_default_na=True)
    player_stats = pd.read_csv(player_stats_path, dtype="string", keep_default_na=True)
    output = build_lineup_features(player_form, player_stats)
    output_path = output_dir / LINEUP_FEATURES_NAME
    write_csv_atomic(output, output_path)
    print(f"Saved {len(output)} pre-match lineup rows to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    build_output(args.player_form, args.player_stats, args.output_dir)


if __name__ == "__main__":
    main()
