# Walk-forward evaluation of Scotland player-form models and baselines

from __future__ import annotations

import argparse
from pathlib import Path

from constants import DEFAULT_EVALUATION_DIR, DEFAULT_MODEL_DATASET, PROJECT_ROOT
from data.load_model_dataset import load_dataset
from evaluation.report import write_evaluation_outputs
from evaluation.walk_forward import run_walk_forward
from models import all_predictors


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
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    dataset = load_dataset(resolve_project_path(args.model_dataset))
    result = run_walk_forward(dataset, all_predictors())
    output_dir = resolve_project_path(args.output_dir)
    write_evaluation_outputs(result, output_dir)
    print(result.overall_metrics.to_string(index=False))
    print(f"\nSaved evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
