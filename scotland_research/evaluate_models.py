# Pooled and league-specific walk-forward evaluation for five development leagues.

from __future__ import annotations

import argparse
from pathlib import Path

from constants import (
    DEFAULT_EVALUATION_DIR,
    DEFAULT_MODEL_DATASET,
    DEVELOPMENT_EXCLUDED_LEAGUES,
    PROJECT_ROOT,
)
from data.load_model_dataset import load_dataset
from evaluation.multi_league import (
    LEAGUE_SPECIFIC_SCOPE,
    POOLED_SCOPE,
    league_effect_column_names,
    run_multi_league_walk_forward,
)
from evaluation.report import write_multi_league_outputs
from selected_features import (
    SELECTED_FEATURES_PATH,
    load_selected_features,
    write_frozen_run_configuration,
)
from models.player_form_lightgbm import LIGHTGBM_SETTINGS
from models.publication_suite import (
    PUBLICATION_MODEL_NAMES,
    league_specific_model_factories,
    pooled_model_factories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dataset",
        type=Path,
        default=DEFAULT_MODEL_DATASET,
        help=f"Step-4 model dataset (default: {DEFAULT_MODEL_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EVALUATION_DIR,
        help=f"Evaluation output directory (default: {DEFAULT_EVALUATION_DIR})",
    )
    parser.add_argument(
        "--selected-features",
        type=Path,
        default=SELECTED_FEATURES_PATH,
        help=f"Selected model features CSV (default: {SELECTED_FEATURES_PATH})",
    )
    parser.add_argument(
        "--scope",
        choices=("both", "pooled", "league-specific"),
        default="both",
        help="Training scope to evaluate (default: both).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=PUBLICATION_MODEL_NAMES,
        help=(
            "Optional model subset. closing_market is added automatically because "
            "every learned model is compared with it."
        ),
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    selected = load_selected_features(
        resolve_project_path(args.selected_features)
    )
    dataset = load_dataset(resolve_project_path(args.model_dataset))
    excluded_present = sorted(
        set(dataset["league"]).intersection(DEVELOPMENT_EXCLUDED_LEAGUES)
    )
    if excluded_present:
        print(
            "Temporarily excluded from development evaluation: "
            + ", ".join(excluded_present)
        )
    selected_models = list(args.models) if args.models else None
    if selected_models is not None and "closing_market" not in selected_models:
        selected_models.insert(0, "closing_market")
    scopes = {
        "both": (POOLED_SCOPE, LEAGUE_SPECIFIC_SCOPE),
        "pooled": (POOLED_SCOPE,),
        "league-specific": (LEAGUE_SPECIFIC_SCOPE,),
    }[args.scope]
    pooled_factories = pooled_model_factories(
        league_effect_column_names(),
        selected_models,
        selected_features=selected.by_model,
    )
    separate_factories = league_specific_model_factories(
        selected_models,
        selected_features=selected.by_model,
    )
    result = run_multi_league_walk_forward(
        dataset,
        pooled_factories,
        separate_factories,
        scopes=scopes,
    )
    output_dir = resolve_project_path(args.output_dir)
    write_multi_league_outputs(result, output_dir)
    configuration_paths = write_frozen_run_configuration(
        selected,
        output_dir,
        lightgbm_settings=LIGHTGBM_SETTINGS,
    )
    print("\nEqual-league development results")
    print(result.equal_league_metrics.to_string(index=False))
    print("\nMatch-weighted development results")
    print(result.overall_metrics.to_string(index=False))
    print(f"\nSaved evaluation outputs to {output_dir}")
    print(
        "Saved configuration: "
        + ", ".join(str(path) for path in configuration_paths.values())
    )


if __name__ == "__main__":
    main()
