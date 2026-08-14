# Write walk-forward evaluation artifacts.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.walk_forward import WalkForwardResult
from evaluation.multi_league import MultiLeagueResult


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def write_evaluation_outputs(
    result: WalkForwardResult,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    paths = (
        output_dir / "fold_metrics.csv",
        output_dir / "fold_league_counts.csv",
        output_dir / "overall_metrics.csv",
        output_dir / "predictions.csv",
        output_dir / "feature_coefficients.csv",
        output_dir / "tuning_calibration_artifacts.csv",
    )
    for frame, path in zip(
        (
            result.fold_metrics,
            result.fold_league_counts,
            result.overall_metrics,
            result.predictions,
            result.coefficients,
            result.tuning_artifacts,
        ),
        paths,
        strict=True,
    ):
        write_csv_atomic(frame, path)
    return paths


def write_multi_league_outputs(
    result: MultiLeagueResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write every table required to audit pooled and separate evaluation."""

    frames = {
        "fold_metrics": result.fold_metrics,
        "fold_league_metrics": result.fold_league_metrics,
        "fold_equal_league_metrics": result.fold_equal_league_metrics,
        "fold_league_counts": result.fold_league_counts,
        "training_weight_audit": result.training_weight_audit,
        "overall_metrics": result.overall_metrics,
        "overall_league_metrics": result.overall_league_metrics,
        "equal_league_metrics": result.equal_league_metrics,
        "predictions": result.predictions,
        "feature_coefficients": result.coefficients,
        # Overwrite artifacts left by the retired tuned-model evaluator so an
        # old calibration table cannot be mistaken for part of this run.
        "tuning_calibration_artifacts": pd.DataFrame(
            columns=["training_scope", "model", "note"]
        ),
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output_dir / f"{name}.csv"
        write_csv_atomic(frame, path)
        paths[name] = path
    return paths
