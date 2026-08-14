"""Aggregate leakage-safe player, team-strength, and lineup features by match."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_lineup_features import FEATURE_COLUMNS as LINEUP_FEATURES
from build_lineup_features import LINEUP_FEATURES_NAME
from build_match_dataset import DEFAULT_OUTPUT_DIR, MATCH_DATASET_NAME, write_csv_atomic
from build_player_form import PLAYER_FORM_NAME
from build_team_strength_features import FEATURE_COLUMNS as TEAM_STRENGTH_FEATURES
from build_team_strength_features import TEAM_STRENGTH_NAME
from feature_req import USE_NPXG_FEATURE
from validate_dataset import as_bool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCH_DATASET = DEFAULT_OUTPUT_DIR / MATCH_DATASET_NAME
DEFAULT_PLAYER_FORM = DEFAULT_OUTPUT_DIR / PLAYER_FORM_NAME
DEFAULT_TEAM_STRENGTH = DEFAULT_OUTPUT_DIR / TEAM_STRENGTH_NAME
DEFAULT_LINEUP_FEATURES = DEFAULT_OUTPUT_DIR / LINEUP_FEATURES_NAME
MODEL_DATASET_NAME = "scotland_model_dataset.csv"

# Baseline aggregates preserve continuity with the original experiment.
BASE_SUM_FEATURES = {
    "key_passes_per90_sum_5": "form_key_passes_per90_5",
    "shots_per90_sum_5": "form_shots_per90_5",
    "defensive_actions_per90_sum_5": "form_defensive_actions_per90_5",
    "recent_minutes_sum_5": "form_minutes_5",
}
if USE_NPXG_FEATURE:
    BASE_SUM_FEATURES["npxg_per90_sum_5"] = "form_npxg_per90_5"

# A team mean at several horizons retains recency information that a single
# five-appearance total discards. Trend = one-appearance mean - five-appearance mean.
RECENCY_MEAN_FEATURES: dict[str, str] = {}
for stat in ("shots", "key_passes", "defensive_actions"):
    for horizon in ("1", "3", "ewm", "trend_1_5"):
        source = f"form_{stat}_trend_1_5" if horizon == "trend_1_5" else f"form_{stat}_per90_{horizon}"
        RECENCY_MEAN_FEATURES[f"{stat}_per90_lineup_mean_{horizon}"] = source
for horizon in ("1", "3", "ewm", "trend_1_5", "std_5"):
    source_suffix = f"mean_{horizon}" if horizon in {"1", "3"} else horizon
    RECENCY_MEAN_FEATURES[f"rating_lineup_mean_{horizon}"] = f"form_rating_{source_suffix}"

# Opponent-adjusted features are residuals: actual performance minus an
# expanding ridge expectation based on venue, position, and relative Elo.
ADJUSTED_MEAN_FEATURES: dict[str, str] = {}
for stat in ("shots", "key_passes", "defensive_actions", "rating"):
    for horizon in ("1", "3", "5", "ewm", "trend_1_5", "std_5"):
        source_suffix = f"mean_{horizon}" if horizon in {"1", "3", "5"} else horizon
        ADJUSTED_MEAN_FEATURES[f"adjusted_{stat}_lineup_mean_{horizon}"] = f"form_adjusted_{stat}_{source_suffix}"

DISTRIBUTION_FEATURES = [
    "rating_lineup_std_across_starters_5",
    "rating_lineup_min_5",
    "rating_lineup_max_5",
    "rating_lineup_top3_mean_5",
    "adjusted_shots_lineup_std_across_starters_5",
    "adjusted_key_passes_lineup_std_across_starters_5",
    "adjusted_defensive_actions_lineup_std_across_starters_5",
    "adjusted_rating_lineup_std_across_starters_5",
]

# Position-specific means preserve matchup structure instead of treating a
# forward's shots and a defender's actions as interchangeable lineup totals.
POSITION_SOURCES = {
    "gk": ("G", {"rating_mean_5": "form_rating_mean_5", "saves_per90_5": "form_saves_per90_5"}),
    "def": ("D", {"rating_mean_5": "form_rating_mean_5", "adjusted_defensive_actions_5": "form_adjusted_defensive_actions_mean_5"}),
    "mid": ("M", {"rating_mean_5": "form_rating_mean_5", "adjusted_key_passes_5": "form_adjusted_key_passes_mean_5"}),
    "fwd": ("F", {"rating_mean_5": "form_rating_mean_5", "adjusted_shots_5": "form_adjusted_shots_mean_5"}),
}
POSITION_FEATURES = [feature for label, (_, sources) in POSITION_SOURCES.items() for feature in [f"{label}_count", *[f"{label}_{name}" for name in sources]]]

PLAYER_TEAM_FEATURES = [
    *BASE_SUM_FEATURES,
    "rating_mean_5",
    "starters_without_history",
    "starters_without_full_window",
    *RECENCY_MEAN_FEATURES,
    *ADJUSTED_MEAN_FEATURES,
    *DISTRIBUTION_FEATURES,
    *POSITION_FEATURES,
]
TEAM_FEATURES = [*PLAYER_TEAM_FEATURES, *TEAM_STRENGTH_FEATURES, *LINEUP_FEATURES]

REQUIRED_PLAYER_COLUMNS = {
    "match_id",
    "team_side",
    "player_id",
    "position",
    "has_prior_history",
    "form_window_appearances_5",
    "form_rating_mean_5",
    "form_saves_per90_5",
    *BASE_SUM_FEATURES.values(),
    *RECENCY_MEAN_FEATURES.values(),
    *ADJUSTED_MEAN_FEATURES.values(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dataset", type=Path, default=DEFAULT_MATCH_DATASET)
    parser.add_argument("--player-form", type=Path, default=DEFAULT_PLAYER_FORM)
    parser.add_argument("--team-strength", type=Path, default=DEFAULT_TEAM_STRENGTH)
    parser.add_argument("--lineup-features", type=Path, default=DEFAULT_LINEUP_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def top_three_mean(values: pd.Series) -> float:
    return float(values.nlargest(min(3, len(values))).mean())


def aggregate_player_features(player_form: pd.DataFrame) -> pd.DataFrame:
    players = player_form.copy()
    players["has_prior_history"] = as_bool(players["has_prior_history"])
    numeric_columns = sorted((REQUIRED_PLAYER_COLUMNS - {"match_id", "team_side", "player_id", "position", "has_prior_history"}))
    players[numeric_columns] = players[numeric_columns].apply(pd.to_numeric, errors="raise")
    grouped = players.groupby(["match_id", "team_side"], sort=False)
    group_sizes = grouped.size()
    if not group_sizes.eq(11).all():
        raise ValueError(f"Match sides without exactly 11 starters:\n{group_sizes[~group_sizes.eq(11)].head(10)}")

    output = pd.DataFrame(index=group_sizes.index)
    for name, source in BASE_SUM_FEATURES.items():
        # min_count=11 ensures an enabled nullable feature cannot turn an
        # entirely missing lineup into a misleading zero.
        output[name] = grouped[source].sum(min_count=11 if source == "form_npxg_per90_5" else 1)
    output["rating_mean_5"] = grouped["form_rating_mean_5"].mean()
    output["starters_without_history"] = 11 - grouped["has_prior_history"].sum()
    output["starters_without_full_window"] = grouped["form_window_appearances_5"].apply(lambda values: values.lt(5).sum())
    for name, source in {**RECENCY_MEAN_FEATURES, **ADJUSTED_MEAN_FEATURES}.items():
        output[name] = grouped[source].mean()

    ratings = grouped["form_rating_mean_5"]
    output["rating_lineup_std_across_starters_5"] = ratings.std(ddof=0)
    output["rating_lineup_min_5"] = ratings.min()
    output["rating_lineup_max_5"] = ratings.max()
    output["rating_lineup_top3_mean_5"] = ratings.apply(top_three_mean)
    for stat in ("shots", "key_passes", "defensive_actions", "rating"):
        output[f"adjusted_{stat}_lineup_std_across_starters_5"] = grouped[f"form_adjusted_{stat}_mean_5"].std(ddof=0)

    for label, (position, sources) in POSITION_SOURCES.items():
        subset = players[players["position"].eq(position)]
        position_grouped = subset.groupby(["match_id", "team_side"], sort=False)
        output[f"{label}_count"] = position_grouped.size().reindex(output.index, fill_value=0)
        for name, source in sources.items():
            output[f"{label}_{name}"] = position_grouped[source].mean().reindex(output.index, fill_value=0.0)

    return output.reset_index()


def merge_long_feature_sets(player_features: pd.DataFrame, team_strength: pd.DataFrame, lineup_features: pd.DataFrame) -> pd.DataFrame:
    keys = ["match_id", "team_side"]
    require_columns(team_strength, set(keys + TEAM_STRENGTH_FEATURES), "team-strength table")
    require_columns(lineup_features, set(keys + LINEUP_FEATURES), "lineup table")
    combined = player_features.merge(team_strength[keys + TEAM_STRENGTH_FEATURES], on=keys, how="left", validate="one_to_one")
    combined = combined.merge(lineup_features[keys + LINEUP_FEATURES], on=keys, how="left", validate="one_to_one")
    if combined[TEAM_FEATURES].isna().any().any():
        missing = combined[TEAM_FEATURES].columns[combined[TEAM_FEATURES].isna().any()].tolist()
        raise ValueError(f"Combined feature table contains missing values in: {missing}")
    return combined


def side_features(team_features: pd.DataFrame, side: str) -> pd.DataFrame:
    side_frame = team_features[team_features["team_side"].eq(side)].drop(columns="team_side").copy()
    return side_frame.rename(columns={feature: f"{side}_{feature}" for feature in TEAM_FEATURES})


def build_model_dataset(matches: pd.DataFrame, player_form: pd.DataFrame, team_strength: pd.DataFrame, lineup_features: pd.DataFrame) -> pd.DataFrame:
    if matches["match_id"].duplicated().any():
        raise ValueError("Match dataset contains duplicate match IDs")
    require_columns(player_form, REQUIRED_PLAYER_COLUMNS, "player-form table")
    match_ids = set(matches["match_id"].astype(str))
    if set(player_form["match_id"].astype(str)) != match_ids:
        raise ValueError("Match IDs disagree between match and player-form inputs")

    player_teams = aggregate_player_features(player_form)
    team_features = merge_long_feature_sets(player_teams, team_strength, lineup_features)
    dataset = matches.merge(side_features(team_features, "home"), on="match_id", how="left", validate="one_to_one")
    dataset = dataset.merge(side_features(team_features, "away"), on="match_id", how="left", validate="one_to_one")
    for feature in TEAM_FEATURES:
        # Differences retain the signed home-away representation while absolute
        # columns preserve level effects and asymmetric nonlinear relationships.
        dataset[f"diff_{feature}"] = dataset[f"home_{feature}"] - dataset[f"away_{feature}"]
    ordered = [f"{side}_{feature}" for side in ("home", "away", "diff") for feature in TEAM_FEATURES]
    dataset = dataset[list(matches.columns) + ordered].sort_values(["utc_date", "match_id"], kind="stable").reset_index(drop=True)
    validate_model_dataset(dataset)
    return dataset


def validate_model_dataset(dataset: pd.DataFrame) -> None:
    feature_columns = [f"{side}_{feature}" for side in ("home", "away", "diff") for feature in TEAM_FEATURES]
    if dataset[feature_columns].isna().any().any():
        raise ValueError("Final model dataset contains missing engineered features")
    if not np.isfinite(dataset[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Final model dataset contains non-finite engineered features")
    for feature in TEAM_FEATURES:
        expected = dataset[f"home_{feature}"] - dataset[f"away_{feature}"]
        if not dataset[f"diff_{feature}"].sub(expected).abs().lt(1e-10).all():
            raise ValueError(f"Incorrect home-away difference for {feature}")


def build_output(
    match_dataset_path: Path,
    player_form_path: Path,
    team_strength_path: Path,
    lineup_features_path: Path,
    output_dir: Path,
    output_name: str = MODEL_DATASET_NAME,
) -> Path:
    matches = pd.read_csv(match_dataset_path, dtype={"match_id": "string"})
    player_form = pd.read_csv(player_form_path, dtype={"match_id": "string", "player_id": "string"})
    team_strength = pd.read_csv(team_strength_path, dtype={"match_id": "string"})
    lineup_features = pd.read_csv(lineup_features_path, dtype={"match_id": "string"})
    dataset = build_model_dataset(matches, player_form, team_strength, lineup_features)
    output_path = output_dir / output_name
    write_csv_atomic(dataset, output_path)
    print(f"Saved {len(dataset)} matches, {len(TEAM_FEATURES)} team features, and {len(dataset.columns)} total columns to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    build_output(args.match_dataset, args.player_form, args.team_strength, args.lineup_features, args.output_dir)


if __name__ == "__main__":
    main()
