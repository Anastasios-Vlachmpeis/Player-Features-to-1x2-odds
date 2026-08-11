#Aggregate starter rolling form into one modelling row per Scotland match.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_match_dataset import DEFAULT_OUTPUT_DIR, MATCH_DATASET_NAME
from build_player_form import PLAYER_FORM_NAME
from validate_dataset import as_bool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATCH_DATASET = DEFAULT_OUTPUT_DIR / MATCH_DATASET_NAME
DEFAULT_PLAYER_FORM = DEFAULT_OUTPUT_DIR / PLAYER_FORM_NAME
MODEL_DATASET_NAME = "scotland_model_dataset.csv"

SUM_FEATURES = {
    # Scotland's xG/xA fields are constant zero in the provider export.
    # Use populated npxG and key passes instead of meaningless zero predictors.
    "npxg_per90_sum_5": "form_npxg_per90_5",
    "key_passes_per90_sum_5": "form_key_passes_per90_5",
    "shots_per90_sum_5": "form_shots_per90_5",
    "defensive_actions_per90_sum_5": "form_defensive_actions_per90_5",
    "recent_minutes_sum_5": "form_minutes_5",
}

TEAM_FEATURES = [
    "npxg_per90_sum_5",
    "key_passes_per90_sum_5",
    "shots_per90_sum_5",
    "defensive_actions_per90_sum_5",
    "rating_mean_5",
    "recent_minutes_sum_5",
    "starters_without_history",
    "starters_without_full_window",
]

REQUIRED_PLAYER_COLUMNS = {
    "match_id",
    "team_side",
    "player_id",
    "has_prior_history",
    "form_window_appearances_5",
    "form_rating_mean_5",
    *SUM_FEATURES.values(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--match-dataset",
        type=Path,
        default=DEFAULT_MATCH_DATASET,
        help=f"Step-2 match/odds dataset (default: {DEFAULT_MATCH_DATASET})",
    )
    parser.add_argument(
        "--player-form",
        type=Path,
        default=DEFAULT_PLAYER_FORM,
        help=f"Step-3 starter form table (default: {DEFAULT_PLAYER_FORM})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Processed output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def aggregate_team_features(player_form: pd.DataFrame) -> pd.DataFrame:
    players = player_form.copy()
    players["has_prior_history"] = as_bool(players["has_prior_history"])
    numeric_columns = [
        "form_window_appearances_5",
        "form_rating_mean_5",
        *SUM_FEATURES.values(),
    ]
    for column in numeric_columns:
        players[column] = pd.to_numeric(players[column], errors="raise")

    group_sizes = players.groupby(["match_id", "team_side"]).size()
    if not group_sizes.eq(11).all():
        bad = group_sizes[~group_sizes.eq(11)]
        raise ValueError(f"Match sides without exactly 11 starters:\n{bad.head(10)}")
    if not set(players["team_side"]) == {"home", "away"}:
        raise ValueError("team_side must contain exactly home and away")

    grouped = players.groupby(["match_id", "team_side"], sort=False)
    team_features = grouped[list(SUM_FEATURES.values())].sum().rename(
        columns={source: output for output, source in SUM_FEATURES.items()}
    )
    team_features["rating_mean_5"] = grouped["form_rating_mean_5"].mean()
    team_features["starters_without_history"] = 11 - grouped["has_prior_history"].sum()
    team_features["starters_without_full_window"] = grouped[
        "form_window_appearances_5"
    ].apply(lambda values: values.lt(5).sum())
    return team_features.reset_index()


def side_features(team_features: pd.DataFrame, side: str) -> pd.DataFrame:
    side_frame = team_features[team_features["team_side"] == side].copy()
    side_frame = side_frame.drop(columns="team_side")
    return side_frame.rename(
        columns={feature: f"{side}_{feature}" for feature in TEAM_FEATURES}
    )


def build_model_dataset(
    matches: pd.DataFrame,
    player_form: pd.DataFrame,
) -> pd.DataFrame:
    if matches["match_id"].duplicated().any():
        raise ValueError("Match/odds dataset contains duplicate match IDs")
    if player_form.duplicated(["match_id", "player_id"]).any():
        raise ValueError("Player-form table contains duplicate match/player rows")

    match_ids = set(matches["match_id"])
    form_match_ids = set(player_form["match_id"])
    if form_match_ids != match_ids:
        missing = sorted(match_ids.difference(form_match_ids))
        extra = sorted(form_match_ids.difference(match_ids))
        raise ValueError(
            f"Match IDs disagree between inputs; missing={missing[:10]}, extra={extra[:10]}"
        )

    team_features = aggregate_team_features(player_form)
    home = side_features(team_features, "home")
    away = side_features(team_features, "away")
    dataset = matches.merge(home, on="match_id", how="left", validate="one_to_one")
    dataset = dataset.merge(away, on="match_id", how="left", validate="one_to_one")

    for feature in TEAM_FEATURES:
        dataset[f"diff_{feature}"] = dataset[f"home_{feature}"] - dataset[f"away_{feature}"]

    ordered_features = (
        [f"home_{feature}" for feature in TEAM_FEATURES]
        + [f"away_{feature}" for feature in TEAM_FEATURES]
        + [f"diff_{feature}" for feature in TEAM_FEATURES]
    )
    match_columns = list(matches.columns)
    dataset = dataset[match_columns + ordered_features].sort_values(
        ["utc_date", "match_id"],
        kind="stable",
    ).reset_index(drop=True)
    validate_model_dataset(dataset, expected_match_ids=match_ids)
    return dataset


def validate_model_dataset(dataset: pd.DataFrame, expected_match_ids: set[str]) -> None:
    if set(dataset["match_id"]) != expected_match_ids:
        raise ValueError("Final model dataset lost or added match IDs")
    if dataset["match_id"].duplicated().any():
        raise ValueError("Final model dataset contains duplicate matches")

    feature_columns = [
        f"{prefix}_{feature}"
        for prefix in ("home", "away", "diff")
        for feature in TEAM_FEATURES
    ]
    if dataset[feature_columns].isna().any().any():
        raise ValueError("Final model dataset contains missing player-form features")

    for feature in TEAM_FEATURES:
        expected = dataset[f"home_{feature}"] - dataset[f"away_{feature}"]
        if not dataset[f"diff_{feature}"].sub(expected).abs().lt(1e-12).all():
            raise ValueError(f"Incorrect home-minus-away difference for {feature}")

    count_features = ["starters_without_history", "starters_without_full_window"]
    for side in ("home", "away"):
        for feature in count_features:
            values = dataset[f"{side}_{feature}"]
            if not values.between(0, 11).all():
                raise ValueError(f"{side}_{feature} must be between 0 and 11")


def build_output(
    match_dataset_path: Path,
    player_form_path: Path,
    output_dir: Path,
) -> Path:
    for path in (match_dataset_path, player_form_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    matches = pd.read_csv(match_dataset_path)
    player_form = pd.read_csv(player_form_path, dtype={"match_id": "string", "player_id": "string"})
    require_columns(player_form, REQUIRED_PLAYER_COLUMNS, player_form_path)
    if "match_id" not in matches.columns:
        raise ValueError(f"{match_dataset_path} has no match_id column")

    dataset = build_model_dataset(matches, player_form)
    output_path = output_dir / MODEL_DATASET_NAME
    write_csv_atomic(dataset, output_path)

    print(f"Saved {len(dataset)} match rows and {len(dataset.columns)} columns to {output_path}")
    print("Player-derived feature columns:", len(TEAM_FEATURES) * 3)
    return output_path


def main() -> None:
    args = parse_args()
    build_output(
        resolve_project_path(args.match_dataset),
        resolve_project_path(args.player_form),
        resolve_project_path(args.output_dir),
    )


if __name__ == "__main__":
    main()
