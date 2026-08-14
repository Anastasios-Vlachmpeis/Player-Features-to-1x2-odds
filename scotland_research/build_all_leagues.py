"""Build six independent league datasets and combine their finished features.

By default this command is development-only and refuses to include 2025-26.
Pass ``--include-final`` only after the research specification is frozen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_lineup_features import build_output as build_lineup_output
from build_match_dataset import (
    build_outputs as build_match_outputs,
    write_csv_atomic,
)
from build_match_features import build_output as build_feature_output
from build_player_form import build_outputs as build_player_outputs
from build_team_strength_features import build_output as build_team_strength_output
from league_config import (
    ALL_RESEARCH_SEASONS,
    DEVELOPMENT_SEASONS,
    FINAL_SEASON,
    LEAGUES,
    PROJECT_ROOT,
)
from validate_dataset import write_reports


PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
COMBINED_OUTPUT_DIR = PROCESSED_ROOT / "all_leagues"
VALIDATION_ROOT = PROJECT_ROOT / "artifacts" / "all_leagues_data_validation"
PER_LEAGUE_MODEL_NAME = "model_dataset.csv"
COMBINED_MODEL_NAME = "all_leagues_model_dataset.csv"
DEVELOPMENT_MODEL_NAME = "development_model_dataset.csv"
FINAL_MODEL_NAME = "final_2025_26_model_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Include 2025-26 and create the held-out final dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=COMBINED_OUTPUT_DIR,
        help=f"Combined output directory (default: {COMBINED_OUTPUT_DIR})",
    )
    return parser.parse_args()


def add_league_identity(frame: pd.DataFrame, league: str) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "league", league)
    identifier_columns = ("match_id", "home_team_id", "away_team_id")
    for column in identifier_columns:
        if column not in output:
            raise ValueError(f"{league} model dataset is missing {column}")
        if output[column].isna().any() or output[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{league} model dataset contains an empty {column}")
        output.insert(
            output.columns.get_loc(column),
            f"source_{column}",
            output[column].astype(str),
        )
        output[column] = league + ":" + output[column].astype(str)
    return output


def validate_combined(
    combined: pd.DataFrame,
    league_keys: list[str],
    include_final: bool,
) -> None:
    if combined.empty:
        raise ValueError("The combined model dataset is empty")
    if combined["match_id"].duplicated().any():
        raise ValueError("The combined model dataset contains duplicate match IDs")
    observed_leagues = set(combined["league"])
    if observed_leagues != set(league_keys):
        raise ValueError(
            f"Combined league mismatch: expected={sorted(league_keys)}, "
            f"observed={sorted(observed_leagues)}"
        )
    allowed_seasons = set(
        ALL_RESEARCH_SEASONS if include_final else DEVELOPMENT_SEASONS
    )
    unexpected_seasons = sorted(set(combined["season"]).difference(allowed_seasons))
    if unexpected_seasons:
        raise ValueError(f"Combined dataset contains unexpected seasons: {unexpected_seasons}")
    if not include_final and combined["season"].eq(FINAL_SEASON).any():
        raise ValueError(f"Development build must not contain {FINAL_SEASON}")
    if include_final:
        final_leagues = set(combined.loc[combined["season"].eq(FINAL_SEASON), "league"])
        missing_final = sorted(set(league_keys).difference(final_leagues))
        if missing_final:
            raise ValueError(
                f"The final season has no model-ready matches for: {missing_final}"
            )


def write_combined_outputs(
    frames: list[pd.DataFrame],
    league_keys: list[str],
    output_dir: Path,
    include_final: bool,
) -> dict[str, Path]:
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["season", "match_date", "league", "match_id"],
        kind="stable",
    ).reset_index(drop=True)
    validate_combined(combined, league_keys, include_final)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "combined": output_dir / COMBINED_MODEL_NAME,
        "development": output_dir / DEVELOPMENT_MODEL_NAME,
    }
    write_csv_atomic(combined, paths["combined"])
    development = combined[combined["season"].isin(DEVELOPMENT_SEASONS)].copy()
    write_csv_atomic(development, paths["development"])

    if include_final:
        paths["final"] = output_dir / FINAL_MODEL_NAME
        final = combined[combined["season"].eq(FINAL_SEASON)].copy()
        write_csv_atomic(final, paths["final"])

    return paths


def build_all_leagues(
    output_dir: Path,
    include_final: bool,
) -> dict[str, Path]:
    configs = list(LEAGUES.values())
    seasons = ALL_RESEARCH_SEASONS if include_final else DEVELOPMENT_SEASONS
    model_frames: list[pd.DataFrame] = []

    for config in configs:
        print(f"\n=== Building {config.name} ===")
        processed_dir = PROCESSED_ROOT / config.key
        validation_dir = VALIDATION_ROOT / config.key

        write_reports(validation_dir, config, seasons)
        match_dataset_path, _ = build_match_outputs(
            processed_dir,
            config,
            seasons,
        )
        team_strength_path = build_team_strength_output(
            match_dataset_path,
            processed_dir,
        )
        player_form_path, _ = build_player_outputs(
            config.player_stats_csv,
            match_dataset_path,
            team_strength_path,
            processed_dir,
        )
        lineup_path = build_lineup_output(
            player_form_path,
            config.player_stats_csv,
            processed_dir,
        )
        model_path = build_feature_output(
            match_dataset_path,
            player_form_path,
            team_strength_path,
            lineup_path,
            processed_dir,
            output_name=PER_LEAGUE_MODEL_NAME,
        )
        frame = pd.read_csv(
            model_path,
            dtype={
                "match_id": "string",
                "home_team_id": "string",
                "away_team_id": "string",
            },
        )
        model_frames.append(add_league_identity(frame, config.key))

    resolved_output_dir = (
        output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    )
    paths = write_combined_outputs(
        model_frames,
        [config.key for config in configs],
        resolved_output_dir,
        include_final,
    )
    print("\n=== Combined outputs ===")
    for label, path in paths.items():
        print(f"{label}: {path}")
    return paths


def main() -> None:
    args = parse_args()
    build_all_leagues(args.output_dir, args.include_final)


if __name__ == "__main__":
    main()
