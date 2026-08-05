"""Chronological train / calibration / test partitions."""

from __future__ import annotations

import pandas as pd

from superleague_baseline.constants import (
    CLASS_ORDER,
    DEFAULT_CALIBRATION_END,
    DEFAULT_TEST_END,
    DEFAULT_TRAIN_END,
)


def assign_partition(
    dataset: pd.DataFrame,
    *,
    train_end: str = DEFAULT_TRAIN_END,
    calibration_end: str = DEFAULT_CALIBRATION_END,
    test_end: str = DEFAULT_TEST_END,
) -> pd.Series:
    dates = pd.to_datetime(dataset["match_date"])
    train_end_d = pd.Timestamp(train_end)
    cal_end_d = pd.Timestamp(calibration_end)
    test_end_d = pd.Timestamp(test_end)
    if not train_end_d < cal_end_d < test_end_d:
        raise ValueError(
            "Split boundaries must satisfy train_end < calibration_end < test_end"
        )

    part = pd.Series(index=dataset.index, dtype="object")
    part[dates <= train_end_d] = "train"
    part[(dates > train_end_d) & (dates <= cal_end_d)] = "calibration"
    part[(dates > cal_end_d) & (dates <= test_end_d)] = "test"
    if part.isna().any():
        raise ValueError("Dataset contains dates outside configured partitions")
    return part


def assert_disjoint(partitions: pd.Series) -> None:
    if partitions.isna().any():
        raise ValueError("Unassigned partition rows")


def assert_classes_present(y: pd.Series, partition_name: str) -> None:
    missing = [c for c in CLASS_ORDER if c not in set(y.dropna())]
    if missing:
        raise ValueError(f"Partition {partition_name} missing classes: {missing}")
