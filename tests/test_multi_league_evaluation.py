from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCOTLAND_RESEARCH_DIR = Path(__file__).resolve().parents[1] / "scotland_research"
if str(SCOTLAND_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCOTLAND_RESEARCH_DIR))

from constants import EXPECTED_LEAGUES  # noqa: E402
from evaluation.multi_league import (  # noqa: E402
    LEAGUE_SPECIFIC_SCOPE,
    POOLED_SCOPE,
    add_league_effects,
    equal_league_training_weights,
    run_multi_league_walk_forward,
)
from league_config import DEVELOPMENT_SEASONS  # noqa: E402
from models.closing_market import ClosingMarket  # noqa: E402
from models.publication_suite import (  # noqa: E402
    league_specific_model_factories,
    pooled_model_factories,
)


def minimal_dataset() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    outcomes = ("H", "D", "A")
    for season_number, season in enumerate(DEVELOPMENT_SEASONS):
        for league in sorted(EXPECTED_LEAGUES):
            for outcome_number, outcome in enumerate(outcomes):
                rows.append(
                    {
                        "league": league,
                        "season": season,
                        "match_id": f"{league}:{season}:{outcome}",
                        "match_date": f"{2021 + season_number}-01-{outcome_number + 1:02d}",
                        "home_team": f"{league} home",
                        "away_team": f"{league} away",
                        "result_3way": outcome,
                        "market_home_probability": 0.45,
                        "market_draw_probability": 0.28,
                        "market_away_probability": 0.27,
                    }
                )
    return pd.DataFrame(rows)


def test_equal_league_weights_have_equal_totals_and_mean_one():
    train = pd.DataFrame(
        {
            "league": ["belgium"] * 2
            + ["netherlands"] * 3
            + ["portugal"] * 4
            + ["scotland"] * 5
            + ["turkey"] * 6
        }
    )
    weights = equal_league_training_weights(train)
    totals = pd.DataFrame({"league": train["league"], "weight": weights}).groupby(
        "league"
    )["weight"].sum()

    assert np.allclose(totals, totals.iloc[0])
    assert np.isclose(weights.mean(), 1.0)


def test_pooled_league_effects_use_one_reference_league():
    dataset, columns = add_league_effects(minimal_dataset())

    assert len(columns) == len(EXPECTED_LEAGUES) - 1
    assert dataset[columns].isin([0.0, 1.0]).all().all()
    assert dataset[columns].sum(axis=1).isin([0.0, 1.0]).all()


FIT_LEAGUE_SETS: dict[str, list[set[str]]] = {
    POOLED_SCOPE: [],
    LEAGUE_SPECIFIC_SCOPE: [],
}


class RecordingPredictor:
    name = "recording_model"

    def __init__(self, scope: str) -> None:
        self.scope = scope

    def fit(
        self,
        train: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        FIT_LEAGUE_SETS[self.scope].append(set(train["league"]))

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        return np.tile(np.array([0.4, 0.3, 0.3]), (len(test), 1))

    def export_coefficients(self, test_season: str) -> pd.DataFrame | None:
        return None


def test_engine_pools_once_and_fits_separate_models_inside_each_league():
    FIT_LEAGUE_SETS[POOLED_SCOPE].clear()
    FIT_LEAGUE_SETS[LEAGUE_SPECIFIC_SCOPE].clear()
    result = run_multi_league_walk_forward(
        minimal_dataset(),
        pooled_factories={
            "closing_market": ClosingMarket,
            "recording_model": lambda: RecordingPredictor(POOLED_SCOPE),
        },
        league_specific_factories={
            "closing_market": ClosingMarket,
            "recording_model": lambda: RecordingPredictor(LEAGUE_SPECIFIC_SCOPE),
        },
    )

    assert len(FIT_LEAGUE_SETS[POOLED_SCOPE]) == 3
    assert all(leagues == set(EXPECTED_LEAGUES) for leagues in FIT_LEAGUE_SETS[POOLED_SCOPE])
    assert len(FIT_LEAGUE_SETS[LEAGUE_SPECIFIC_SCOPE]) == 3 * len(EXPECTED_LEAGUES)
    assert all(len(leagues) == 1 for leagues in FIT_LEAGUE_SETS[LEAGUE_SPECIFIC_SCOPE])
    assert set(result.predictions["training_scope"]) == {
        POOLED_SCOPE,
        LEAGUE_SPECIFIC_SCOPE,
    }
    assert not result.predictions.duplicated(
        ["training_scope", "model", "season", "league", "match_id"]
    ).any()


def test_dixon_coles_is_not_available_in_the_pooled_suite():
    factories = pooled_model_factories([])
    assert "dixon_coles" not in factories
    assert "dixon_coles_player_form" not in factories


def test_recalibrated_market_is_available_in_both_training_scopes():
    pooled = pooled_model_factories([])
    separate = league_specific_model_factories()

    assert "recalibrated_market" in pooled
    assert "recalibrated_market" in separate
