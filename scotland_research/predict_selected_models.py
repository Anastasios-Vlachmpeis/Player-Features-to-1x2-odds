# Generate chronological predictions for the frozen feature-selected models.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from constants import DEFAULT_MODEL_DATASET, PROJECT_ROOT
from data.load_model_dataset import load_dataset
from evaluation.report import write_evaluation_outputs
from evaluation.walk_forward import run_walk_forward
from selected_features import (
    SELECTED_FEATURES_PATH,
    load_selected_features,
)
from models.closing_market import ClosingMarket
from models.dixon_coles_player_form import DixonColesPlayerFormModel
from models.expanded_player_form_lightgbm import ExpandedPlayerFormLightGBMModel
from models.market_plus_player_form import MarketPlusPlayerFormModel
from models.player_form import PlayerFormModel
from models.player_form_lightgbm import PlayerFormLightGBMModel
from models.recalibrated_market import RecalibratedMarketModel


DEFAULT_SELECTED_FEATURES = SELECTED_FEATURES_PATH
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "scotland_selected_model_evaluation"

MODEL_ORDER = [
    "player_form_logistic",
    "market_plus_player_form",
    "dixon_coles_player_form",
    "player_form_lightgbm",
    "expanded_player_form_lightgbm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dataset", type=Path, default=DEFAULT_MODEL_DATASET)
    parser.add_argument(
        "--selected-features",
        type=Path,
        default=DEFAULT_SELECTED_FEATURES,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_selected_feature_map(path: Path) -> dict[str, list[str]]:
    selected = load_selected_features(path)
    return {model_name: selected.by_model[model_name] for model_name in MODEL_ORDER}


def diff_columns(base_features: list[str]) -> list[str]:
    # A differential feature is the home team value minus the away team value.
    return [f"diff_{feature}" for feature in base_features]


def expanded_columns(base_features: list[str]) -> list[str]:
    # Expanded models retain separate home and away values so trees can learn asymmetry.
    return [
        f"{side}_{feature}"
        for side in ("home", "away")
        for feature in base_features
    ]


def build_selected_predictors(feature_map: dict[str, list[str]]) -> list[object]:
    return [
        PlayerFormModel(
            player_features=diff_columns(feature_map["player_form_logistic"]),
            name="player_form_logistic",
        ),
        MarketPlusPlayerFormModel(
            player_features=diff_columns(feature_map["market_plus_player_form"]),
            name="market_plus_player_form",
        ),
        DixonColesPlayerFormModel(
            player_features=diff_columns(feature_map["dixon_coles_player_form"]),
            name="dixon_coles_player_form",
        ),
        PlayerFormLightGBMModel(
            player_features=diff_columns(feature_map["player_form_lightgbm"]),
            name="player_form_lightgbm",
        ),
        ExpandedPlayerFormLightGBMModel(
            player_features=expanded_columns(
                feature_map["expanded_player_form_lightgbm"]
            ),
            name="expanded_player_form_lightgbm",
        ),
    ]


def validate_selected_columns(
    dataset: pd.DataFrame,
    feature_map: dict[str, list[str]],
) -> None:
    required_columns: set[str] = set()
    for model_name, base_features in feature_map.items():
        if model_name == "expanded_player_form_lightgbm":
            required_columns.update(expanded_columns(base_features))
        else:
            required_columns.update(diff_columns(base_features))

    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise ValueError(
            "The model dataset is missing selected feature columns: "
            + ", ".join(missing_columns)
        )


def main() -> None:
    args = parse_args()
    dataset_path = resolve_project_path(args.model_dataset)
    selected_features_path = resolve_project_path(args.selected_features)
    output_dir = resolve_project_path(args.output_dir)

    dataset = load_dataset(dataset_path)
    feature_map = load_selected_feature_map(selected_features_path)
    validate_selected_columns(dataset, feature_map)

    # Market comparison reporting requires one closing-market prediction per match.
    predictors = [
        ClosingMarket(),
        RecalibratedMarketModel(),
        *build_selected_predictors(feature_map),
    ]

    # Existing walk-forward folds fit only on seasons preceding each test season.
    result = run_walk_forward(dataset, predictors)
    write_evaluation_outputs(result, output_dir)

    print(result.overall_metrics.to_string(index=False))
    print(f"\nSaved selected-model evaluation outputs to {output_dir}")
    print(f"Confidence input: {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
