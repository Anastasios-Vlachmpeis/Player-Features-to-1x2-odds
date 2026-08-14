from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCOTLAND_RESEARCH_DIR = Path(__file__).resolve().parents[1] / "scotland_research"
if str(SCOTLAND_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCOTLAND_RESEARCH_DIR))

from models.dixon_coles_core import (  # noqa: E402
    MAX_EXPECTED_GOALS,
    MIN_EXPECTED_GOALS,
    TAU_EPSILON,
    DixonColesEstimator,
    DixonColesParameters,
    clip_expected_goal_rates,
    dixon_coles_corrections,
    stabilize_rho,
)


def test_expected_goal_rates_use_shared_bounds():
    home, away = clip_expected_goal_rates(
        np.array([0.0, 1.4, 20.0]),
        np.array([-1.0, 1.1, 30.0]),
    )

    np.testing.assert_allclose(home, [MIN_EXPECTED_GOALS, 1.4, MAX_EXPECTED_GOALS])
    np.testing.assert_allclose(away, [MIN_EXPECTED_GOALS, 1.1, MAX_EXPECTED_GOALS])


def test_rho_stabilization_makes_all_four_corrections_positive():
    home_rate = np.array([MAX_EXPECTED_GOALS, 1.2])
    away_rate = np.array([MAX_EXPECTED_GOALS, 1.0])
    effective_rho = stabilize_rho(home_rate, away_rate, rho=0.2)
    corrections = dixon_coles_corrections(home_rate, away_rate, effective_rho)

    assert effective_rho[0] < 0.2
    assert effective_rho[1] == 0.2
    assert np.all(corrections > TAU_EPSILON)


def test_prediction_records_rho_adjustment_for_extreme_rates():
    estimator = DixonColesEstimator()
    estimator.parameters = DixonColesParameters(
        team_ids=("A", "B"),
        team_names={"A": "Alpha", "B": "Beta"},
        goal_intercept=np.log(MAX_EXPECTED_GOALS),
        home_advantage=0.0,
        rho=0.2,
        attack=np.zeros(2),
        defence=np.zeros(2),
        player_coefficients=np.empty(0),
        player_feature_means=np.empty(0),
        player_feature_scales=np.empty(0),
    )
    matches = pd.DataFrame(
        {
            "home_team_id": ["A"],
            "away_team_id": ["B"],
            "home_team": ["Alpha"],
            "away_team": ["Beta"],
        }
    )

    probabilities = estimator.predict_proba(matches)
    diagnostics = estimator.last_prediction_diagnostics

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert diagnostics.loc[0, "rho_adjusted"]
    assert diagnostics.loc[0, "effective_rho"] < diagnostics.loc[0, "fitted_rho"]
    assert diagnostics.loc[0, "minimum_correction"] > TAU_EPSILON
