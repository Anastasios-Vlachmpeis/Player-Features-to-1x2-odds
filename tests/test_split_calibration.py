import pandas as pd
import pytest

from superleague_baseline.constants import CLASS_ORDER
from superleague_baseline.features.validate import validate_probabilities
from superleague_baseline.modeling.metrics import reorder_probabilities
from superleague_baseline.modeling.train import train_and_evaluate
from superleague_baseline.splits import assign_partition


def test_date_boundaries_are_disjoint():
    dataset = pd.DataFrame(
        {"match_date": pd.to_datetime(["2026-01-01", "2026-02-15", "2026-04-10"])}
    )
    part = assign_partition(dataset)
    assert set(part) == {"train", "calibration", "test"}


def test_rejects_non_monotonic_boundaries():
    dataset = pd.DataFrame({"match_date": pd.to_datetime(["2026-02-01"])})
    with pytest.raises(ValueError, match="train_end < calibration_end < test_end"):
        assign_partition(
            dataset,
            train_end="2026-03-31",
            calibration_end="2026-01-31",
            test_end="2026-05-21",
        )


def test_probabilities_are_normalized():
    probs = pd.DataFrame(
        {"p_home": [0.4], "p_draw": [0.3], "p_away": [0.3]}
    )
    validate_probabilities(probs)


def test_estimator_classes_are_reordered_to_hda():
    import numpy as np

    raw = np.array([[0.2, 0.5, 0.3]])
    ordered = reorder_probabilities(raw, ["A", "D", "H"])
    assert ordered.tolist() == [[0.3, 0.5, 0.2]]
    assert CLASS_ORDER == ("H", "D", "A")


def test_training_rejects_partitions_without_all_classes():
    dataset = pd.DataFrame(
        {
            "proxy_lineups_complete": [True, True, True],
            "proxy_result_3way": ["H", "H", "H"],
            "feature": [1.0, 2.0, 3.0],
        }
    )
    partition = pd.Series(["train", "calibration", "test"])
    with pytest.raises(ValueError, match="missing classes"):
        train_and_evaluate(dataset, ["feature"], partition=partition, seed=1)
