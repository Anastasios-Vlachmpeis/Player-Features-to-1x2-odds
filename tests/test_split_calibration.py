import pandas as pd

from superleague_baseline.constants import CLASS_ORDER
from superleague_baseline.features.validate import validate_probabilities
from superleague_baseline.modeling.metrics import reorder_probabilities
from superleague_baseline.splits import assign_partition


def test_date_boundaries_are_disjoint():
    dataset = pd.DataFrame(
        {"match_date": pd.to_datetime(["2026-01-01", "2026-02-15", "2026-04-10"])}
    )
    part = assign_partition(dataset)
    assert set(part) == {"train", "calibration", "test"}


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
