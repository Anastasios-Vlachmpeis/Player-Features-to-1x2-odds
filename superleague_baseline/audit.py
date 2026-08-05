"""Audit player_stats.db against baseline pipeline contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from superleague_baseline.constants import (
    DEFAULT_CALIBRATION_END,
    DEFAULT_TEST_END,
    DEFAULT_TRAIN_END,
)
from superleague_baseline.features.dataset import build_historical_match_dataset
from superleague_baseline.features.sources import build_fixture_index, load_sofascore_sources
from superleague_baseline.features.validate import validate_fixture_index
from superleague_baseline.splits import assign_partition


def run_audit(db_path: str | Path) -> dict:
    db_path = Path(db_path)
    lineup, xg = load_sofascore_sources(db_path)
    fixtures = build_fixture_index(lineup)
    validate_fixture_index(fixtures)

    with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as conn:
        tackles_nonzero = conn.execute(
            "SELECT COUNT(*) FROM sofascore_match_stats WHERE tackles != 0"
        ).fetchone()[0]
        fbref_rows = conn.execute("SELECT COUNT(*) FROM fbref_match_stats").fetchone()[0]

    dataset = build_historical_match_dataset(db_path)
    partition = assign_partition(dataset)

    return {
        "fixtures": int(len(fixtures)),
        "fixture_date_min": str(fixtures["match_date"].min().date()),
        "fixture_date_max": str(fixtures["match_date"].max().date()),
        "xg_rows": int(len(xg)),
        "eligible_feature_rows": int(len(dataset)),
        "complete_proxy_labels": int(dataset["proxy_lineups_complete"].sum()),
        "partitions": {
            "train": int((partition == "train").sum()),
            "calibration": int((partition == "calibration").sum()),
            "test": int((partition == "test").sum()),
        },
        "split_bounds": {
            "train_end": DEFAULT_TRAIN_END,
            "calibration_end": DEFAULT_CALIBRATION_END,
            "test_end": DEFAULT_TEST_END,
        },
        "warnings": [
            "No official match scores stored; proxy labels only.",
            "Transfermarkt snapshots are not safe for historical features.",
            f"tackles_nonzero_rows={tackles_nonzero} (field excluded from features).",
            f"fbref_rows={fbref_rows}.",
        ],
    }
