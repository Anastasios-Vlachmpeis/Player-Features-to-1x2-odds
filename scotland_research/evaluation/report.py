# Write walk-forward evaluation artifacts.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.walk_forward import WalkForwardResult


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def write_evaluation_outputs(
    result: WalkForwardResult,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    paths = (
        output_dir / "fold_metrics.csv",
        output_dir / "overall_metrics.csv",
        output_dir / "predictions.csv",
        output_dir / "feature_coefficients.csv",
    )
    for frame, path in zip(
        (
            result.fold_metrics,
            result.overall_metrics,
            result.predictions,
            result.coefficients,
        ),
        paths,
        strict=True,
    ):
        write_csv_atomic(frame, path)
    return paths
