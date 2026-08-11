from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCOTLAND_RESEARCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "scotland_research"
if str(SCOTLAND_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCOTLAND_RESEARCH_DIR))

from models.dixon_coles_core import (  # noqa: E402
    DixonColesEstimator,
    DixonColesParameters,
    dixon_coles_tau,
    scoreline_to_outcome_probabilities,
)
from models import all_predictors  # noqa: E402


def test_dixon_coles_tau_corrects_only_four_low_scorelines():
    home_goals = np.array([0, 0, 1, 1, 2])
    away_goals = np.array([0, 1, 0, 1, 2])
    home_rate = np.full(5, 1.4)
    away_rate = np.full(5, 1.1)
    rho = -0.1

    actual = dixon_coles_tau(
        home_goals,
        away_goals,
        home_rate,
        away_rate,
        rho,
    )
    expected = np.array(
        [
            1 - 1.4 * 1.1 * rho,
            1 + 1.4 * rho,
            1 + 1.1 * rho,
            1 - rho,
            1.0,
        ]
    )
    assert np.allclose(actual, expected)


def test_scoreline_grid_returns_hda_probabilities_that_sum_to_one():
    probabilities = scoreline_to_outcome_probabilities(2.0, 0.6, rho=-0.08)
    assert probabilities.shape == (3,)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-12)
    assert probabilities[0] > probabilities[2]


def test_predictor_registry_contains_both_dixon_coles_models():
    names = [predictor.name for predictor in all_predictors()]
    assert names == [
        "frequency_baseline",
        "dixon_coles",
        "dixon_coles_player_form",
        "closing_market",
        "player_form",
        "market_plus_player_form",
    ]


def fitted_parameters(player_coefficients: np.ndarray) -> DixonColesParameters:
    feature_count = len(player_coefficients)
    return DixonColesParameters(
        team_ids=("A", "B"),
        team_names={"A": "Alpha", "B": "Beta"},
        goal_intercept=np.log(1.2),
        home_advantage=np.log(1.15),
        rho=-0.05,
        attack=np.array([0.2, -0.2]),
        defence=np.array([-0.1, 0.1]),
        player_coefficients=player_coefficients,
        player_feature_means=np.zeros(feature_count),
        player_feature_scales=np.ones(feature_count),
    )


def test_zero_player_coefficients_reproduce_standard_expected_goal_rates():
    matches = pd.DataFrame(
        {"home_team_id": ["A"], "away_team_id": ["B"], "feature": [3.0]}
    )
    standard = DixonColesEstimator()
    standard.parameters = fitted_parameters(np.empty(0))
    adjusted = DixonColesEstimator(player_features=["feature"])
    adjusted.parameters = fitted_parameters(np.zeros(1))

    standard_rates = standard.expected_goal_rates(matches)
    adjusted_rates = adjusted.expected_goal_rates(matches)
    assert np.allclose(standard_rates[0], adjusted_rates[0])
    assert np.allclose(standard_rates[1], adjusted_rates[1])


def test_unseen_teams_receive_neutral_attack_and_defence():
    estimator = DixonColesEstimator()
    estimator.parameters = fitted_parameters(np.empty(0))
    matches = pd.DataFrame(
        {"home_team_id": ["unseen-home"], "away_team_id": ["unseen-away"]}
    )
    home_rate, away_rate = estimator.expected_goal_rates(matches)
    assert np.allclose(home_rate, np.exp(np.log(1.2) + np.log(1.15)))
    assert np.allclose(away_rate, 1.2)


def test_player_scaling_is_fitted_from_training_rows():
    teams = ["A", "B", "C", "D"]
    rows = []
    for match_number in range(36):
        home = teams[match_number % len(teams)]
        away = teams[(match_number + 1 + match_number // len(teams)) % len(teams)]
        if away == home:
            away = teams[(teams.index(home) + 1) % len(teams)]
        rows.append(
            {
                "home_team_id": home,
                "away_team_id": away,
                "home_team": home,
                "away_team": away,
                "home_score": match_number % 3,
                "away_score": (match_number // 2) % 3,
                "_match_datetime": pd.Timestamp("2021-01-01", tz="UTC")
                + pd.Timedelta(days=7 * match_number),
                "feature": float(match_number),
            }
        )
    train = pd.DataFrame(rows)
    estimator = DixonColesEstimator(player_features=["feature"])
    estimator.fit(train)

    assert estimator.parameters is not None
    np.testing.assert_allclose(
        estimator.parameters.player_feature_means,
        [train["feature"].mean()],
    )
    np.testing.assert_allclose(
        estimator.parameters.player_feature_scales,
        [train["feature"].std(ddof=0)],
    )
