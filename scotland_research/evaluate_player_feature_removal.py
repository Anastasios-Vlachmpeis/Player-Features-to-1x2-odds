#Test what happens when each group of Scotland player features is removed

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_match_features import MODEL_DATASET_NAME
from data.load_model_dataset import load_dataset
from evaluation.report import write_evaluation_outputs
from evaluation.walk_forward import run_walk_forward
from models import FULL_PLAYER_MODEL_NAME, models_for_player_feature_removal_test

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DATASET = (REPO_ROOT / "data" / "processed" / "scotland" / MODEL_DATASET_NAME)
DEFAULT_OUTPUT_DIR = (REPO_ROOT / "artifacts" / "scotland_player_feature_removal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dataset",
        type=Path,
        default=DEFAULT_MODEL_DATASET,
        help=f"Scotland model dataset (default: {DEFAULT_MODEL_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def resolve_repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def explain_difference_from_full_model(
    metrics: pd.DataFrame,
    group_column: str | None = None,
) -> pd.DataFrame:
    #Add a comparison with the full player model

    compared = metrics.copy()
    full_rows = compared[compared["model"].eq(FULL_PLAYER_MODEL_NAME)]

    if group_column is None:
        if len(full_rows) != 1:
            raise ValueError("Expected exactly one overall full-player-model row")
        compared["full_player_model_log_loss"] = full_rows["log_loss"].iloc[0]
    
    else:
        full_by_group = full_rows.set_index(group_column)["log_loss"]
        if full_by_group.index.duplicated().any():
            raise ValueError(f"Multiple full-player-model rows for {group_column}")
        compared["full_player_model_log_loss"] = compared[group_column].map(
            full_by_group
        )
        if compared["full_player_model_log_loss"].isna().any():
            raise ValueError(f"A {group_column} group has no full-player-model row")

    compared["log_loss_vs_full_player_model"] = (compared["log_loss"] - compared["full_player_model_log_loss"])

    explanations: list[str] = []
    for model, difference in zip(
        compared["model"],
        compared["log_loss_vs_full_player_model"],
        strict=True,
    ):
        if model == "closing_market":
            explanations.append("closing market reference")
        elif model == FULL_PLAYER_MODEL_NAME:
            explanations.append("full player model reference")
        elif difference > 0:
            explanations.append("removed group helped the full model")
        elif difference < 0:
            explanations.append("removed group hurt the full model")
        else:
            explanations.append("removed group made no difference")

    compared["plain_language_result"] = explanations
    return compared


def main() -> None:
    args = parse_args()
    dataset = load_dataset(resolve_repository_path(args.model_dataset))
    result = run_walk_forward(dataset, models_for_player_feature_removal_test())
    result.fold_metrics = explain_difference_from_full_model(result.fold_metrics,group_column="test_season")
    result.overall_metrics = explain_difference_from_full_model(result.overall_metrics)

    output_dir = resolve_repository_path(args.output_dir)
    write_evaluation_outputs(result, output_dir)

    display_columns = ["model","log_loss","log_loss_vs_full_player_model","plain_language_result"]
    
    print(result.overall_metrics[display_columns].to_string(index=False))
    print(f"\nSaved player-feature removal results to {output_dir}")


if __name__ == "__main__":
    main()
