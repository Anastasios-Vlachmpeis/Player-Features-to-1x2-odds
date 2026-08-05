import hashlib
from pathlib import Path

import pytest

from superleague_baseline.audit import run_audit
from superleague_baseline.features.dataset import build_historical_match_dataset
from superleague_baseline.splits import assign_partition


DB_PATH = Path(__file__).resolve().parent.parent / "player_stats.db"


@pytest.mark.integration
@pytest.mark.skipif(not DB_PATH.exists(), reason="player_stats.db not present")
def test_live_db_contract():
    report = run_audit(DB_PATH)
    assert report["fixtures"] == 236
    assert report["eligible_feature_rows"] >= 190


@pytest.mark.integration
@pytest.mark.skipif(not DB_PATH.exists(), reason="player_stats.db not present")
def test_live_feature_rows_and_partitions():
    before = hashlib.sha256(DB_PATH.read_bytes()).hexdigest()
    dataset = build_historical_match_dataset(DB_PATH)
    part = assign_partition(dataset)
    after = hashlib.sha256(DB_PATH.read_bytes()).hexdigest()
    assert before == after
    assert len(dataset) >= 190
    assert set(part.unique()) <= {"train", "calibration", "test"}
