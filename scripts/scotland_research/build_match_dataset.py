#Build the cleaned Scotland match/result/closing-odds modelling table.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from validate_dataset import build_match_validation, load_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "scotland"
MATCH_DATASET_NAME = "matches_with_closing_odds.csv"
EXCLUSIONS_NAME = "excluded_matches.csv"

MATCH_COLUMNS = [
    "match_id",
    "season",
    "match_date",
    "utc_date",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_score",
    "away_score",
    "result_3way",
    "odds_source",
    "home_closing_odds",
    "draw_closing_odds",
    "away_closing_odds",
    "market_overround",
    "market_home_probability",
    "market_draw_probability",
    "market_away_probability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Processed output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def add_market_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    odds_columns = ["home_odds", "draw_odds", "away_odds"]
    output[odds_columns] = output[odds_columns].apply(pd.to_numeric, errors="raise")
    if output[odds_columns].le(1.0).any().any():
        raise ValueError("Closing decimal odds must all be greater than 1.0")

    inverse = 1.0 / output[odds_columns]
    output["market_overround"] = inverse.sum(axis=1)
    output["market_home_probability"] = inverse["home_odds"] / output["market_overround"]
    output["market_draw_probability"] = inverse["draw_odds"] / output["market_overround"]
    output["market_away_probability"] = inverse["away_odds"] / output["market_overround"]
    output = output.rename(
        columns={
            "home_odds": "home_closing_odds",
            "draw_odds": "draw_closing_odds",
            "away_odds": "away_closing_odds",
        }
    )
    return output


def validate_output(dataset: pd.DataFrame) -> None:
    if dataset.empty:
        raise ValueError("The processed match dataset is empty")
    if dataset["match_id"].duplicated().any():
        raise ValueError("The processed match dataset contains duplicate match IDs")
    if not set(dataset["result_3way"].dropna()).issubset({"H", "D", "A"}):
        raise ValueError("Unexpected result_3way value")
    if dataset[MATCH_COLUMNS].isna().any().any():
        missing_columns = dataset[MATCH_COLUMNS].columns[
            dataset[MATCH_COLUMNS].isna().any()
        ].tolist()
        raise ValueError(f"Processed dataset contains missing values in: {missing_columns}")

    probabilities = dataset[
        [
            "market_home_probability",
            "market_draw_probability",
            "market_away_probability",
        ]
    ]
    if not probabilities.gt(0).all().all() or not probabilities.lt(1).all().all():
        raise ValueError("Devigged probabilities must be strictly between zero and one")
    if not probabilities.sum(axis=1).sub(1.0).abs().lt(1e-12).all():
        raise ValueError("Devigged probabilities do not sum to one")


def build_outputs(output_dir: Path) -> tuple[Path, Path]:
    matches, players, football_data = load_inputs()
    validation = build_match_validation(matches, players, football_data)

    dataset = validation[validation["model_ready"]].copy()
    dataset = add_market_probabilities(dataset)
    dataset = dataset[MATCH_COLUMNS].sort_values(["utc_date", "match_id"]).reset_index(drop=True)
    validate_output(dataset)

    exclusions = validation[~validation["model_ready"]].copy()
    exclusions["excluded_from_prediction_targets"] = True
    exclusions["retained_for_player_history"] = exclusions["player_data_available"]
    exclusion_columns = [
        "match_id",
        "season",
        "match_date",
        "home_team",
        "away_team",
        "football_data_match",
        "player_data_available",
        "home_starters",
        "away_starters",
        "exclusion_reason",
        "excluded_from_prediction_targets",
        "retained_for_player_history",
    ]
    exclusions = exclusions[exclusion_columns].sort_values(
        ["season", "match_date", "match_id"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / MATCH_DATASET_NAME
    exclusions_path = output_dir / EXCLUSIONS_NAME
    write_csv_atomic(dataset, dataset_path)
    write_csv_atomic(exclusions, exclusions_path)

    malformed_lineups = exclusions[exclusions["exclusion_reason"] == "invalid_starter_count"]
    print(f"Saved modelling matches: {len(dataset)}")
    print(f"Excluded malformed-lineup Premiership matches: {len(malformed_lineups)}")
    print(f"Other non-target fixtures recorded: {len(exclusions) - len(malformed_lineups)}")
    print(f"Match dataset: {dataset_path}")
    print(f"Exclusions log: {exclusions_path}")
    return dataset_path, exclusions_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    build_outputs(output_dir)


if __name__ == "__main__":
    main()
