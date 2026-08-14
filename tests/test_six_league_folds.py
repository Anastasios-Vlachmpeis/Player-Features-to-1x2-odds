from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


SCOTLAND_RESEARCH_DIR = Path(__file__).resolve().parents[1] / "scotland_research"
if str(SCOTLAND_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCOTLAND_RESEARCH_DIR))

from constants import DEVELOPMENT_FOLDS, EXPECTED_LEAGUES  # noqa: E402
from evaluation.walk_forward import validate_development_dataset  # noqa: E402
from league_config import DEVELOPMENT_SEASONS, FINAL_SEASON  # noqa: E402


def complete_development_frame() -> pd.DataFrame:
    outcomes = ("H", "D", "A")
    rows = []
    row_number = 0
    for season in DEVELOPMENT_SEASONS:
        for league in sorted(EXPECTED_LEAGUES):
            rows.append(
                {
                    "league": league,
                    "season": season,
                    "match_id": f"{league}:{season}",
                    "result_3way": outcomes[row_number % len(outcomes)],
                }
            )
            row_number += 1
    return pd.DataFrame(rows)


def test_development_folds_are_expanding_and_stop_at_2024_25():
    assert DEVELOPMENT_FOLDS == (
        ("2022-23", ("2020-21", "2021-22")),
        ("2023-24", ("2020-21", "2021-22", "2022-23")),
        (
            "2024-25",
            ("2020-21", "2021-22", "2022-23", "2023-24"),
        ),
    )
    assert all(FINAL_SEASON not in seasons for _, seasons in DEVELOPMENT_FOLDS)


def test_complete_six_league_development_dataset_is_accepted():
    validate_development_dataset(complete_development_frame())


def test_missing_league_in_one_season_is_rejected():
    dataset = complete_development_frame()
    dataset = dataset.loc[
        ~(
            dataset["season"].eq("2021-22")
            & dataset["league"].eq("turkey")
        )
    ]

    with pytest.raises(ValueError, match="Season 2021-22 league mismatch"):
        validate_development_dataset(dataset)


def test_held_out_final_season_is_rejected():
    dataset = complete_development_frame()
    final_row = dataset.iloc[[0]].copy()
    final_row["season"] = FINAL_SEASON
    final_row["match_id"] = "scotland:held-out-final"

    with pytest.raises(ValueError, match="held-out final season"):
        validate_development_dataset(pd.concat([dataset, final_row], ignore_index=True))
