"""Shared Poisson generalized additive model for home and away goal rates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from models.dixon_coles_core import (
    RHO_BOUND,
    TAU_EPSILON,
    clip_expected_goal_rates,
    dixon_coles_corrections,
    dixon_coles_tau,
    scoreline_to_outcome_probabilities,
    stabilize_rho,
)


# Smooth only continuous quantities with a plausible curved relationship. Four
# knots give each effect limited flexibility while keeping the basis small for
# roughly one thousand Scottish Premiership matches.
SMOOTH_FEATURES = [
    "elo_advantage",
    "expected_goals_strength",
    "attack_goal_rate_ewm",
    "opponent_defence_goal_rate_ewm",
    "adjusted_shots_form",
    "adjusted_key_passes_form",
    "opponent_adjusted_defensive_form",
    "forward_adjusted_shots",
    "mean_pairwise_prior_minutes",
    "replacement_quality",
    "rating_form_trend",
]

# Counts and the venue indicator remain linear. With only a few integer support
# points, splining them would add parameters without a well-identified curve.
LINEAR_FEATURES = [
    "is_home",
    "retained_starters",
    "missing_regular_starters",
    "new_starters",
    "starters_without_history",
]

REQUIRED_MATCH_COLUMNS = {
    *[
        f"{side}_{feature}"
        for side in ("home", "away")
        for feature in (
            "elo_rating",
            "expected_goals_strength",
            "attack_goal_rate_ewm",
            "defence_goal_rate_ewm",
            "adjusted_shots_lineup_mean_5",
            "adjusted_key_passes_lineup_mean_5",
            "adjusted_defensive_actions_lineup_mean_5",
            "fwd_adjusted_shots_5",
            "mean_pairwise_prior_minutes",
            "replacement_quality",
            "rating_lineup_mean_trend_1_5",
            "retained_starters",
            "missing_regular_starters",
            "new_starters",
            "starters_without_history",
        )
    ],
}


def require_gam_columns(matches: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_MATCH_COLUMNS.difference(matches.columns))
    if missing:
        raise ValueError(f"Poisson GAM is missing required features: {', '.join(missing)}")


def attacking_rows(matches: pd.DataFrame, side: str) -> pd.DataFrame:
    """Orient one row per team so the same GAM is shared by home and away."""
    if side not in {"home", "away"}:
        raise ValueError("side must be home or away")
    opponent = "away" if side == "home" else "home"
    output = pd.DataFrame(index=matches.index)
    output["elo_advantage"] = matches[f"{side}_elo_rating"] - matches[f"{opponent}_elo_rating"]
    output["expected_goals_strength"] = matches[f"{side}_expected_goals_strength"]
    output["attack_goal_rate_ewm"] = matches[f"{side}_attack_goal_rate_ewm"]
    output["opponent_defence_goal_rate_ewm"] = matches[f"{opponent}_defence_goal_rate_ewm"]
    output["adjusted_shots_form"] = matches[f"{side}_adjusted_shots_lineup_mean_5"]
    output["adjusted_key_passes_form"] = matches[f"{side}_adjusted_key_passes_lineup_mean_5"]
    output["opponent_adjusted_defensive_form"] = matches[f"{opponent}_adjusted_defensive_actions_lineup_mean_5"]
    output["forward_adjusted_shots"] = matches[f"{side}_fwd_adjusted_shots_5"]
    output["mean_pairwise_prior_minutes"] = matches[f"{side}_mean_pairwise_prior_minutes"]
    output["replacement_quality"] = matches[f"{side}_replacement_quality"]
    output["rating_form_trend"] = matches[f"{side}_rating_lineup_mean_trend_1_5"]
    output["is_home"] = float(side == "home")
    output["retained_starters"] = matches[f"{side}_retained_starters"]
    output["missing_regular_starters"] = matches[f"{side}_missing_regular_starters"]
    output["new_starters"] = matches[f"{side}_new_starters"]
    output["starters_without_history"] = matches[f"{side}_starters_without_history"]
    return output[SMOOTH_FEATURES + LINEAR_FEATURES]


def stack_goal_rows(matches: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return home-team rows followed by away-team rows and their goal targets."""
    require_gam_columns(matches)
    missing_targets = sorted({"home_score", "away_score"}.difference(matches.columns))
    if missing_targets:
        raise ValueError(f"Poisson GAM fit is missing goal targets: {', '.join(missing_targets)}")
    features = pd.concat([attacking_rows(matches, "home"), attacking_rows(matches, "away")], ignore_index=True)
    targets = np.concatenate(
        [
            pd.to_numeric(matches["home_score"], errors="raise").to_numpy(dtype=float),
            pd.to_numeric(matches["away_score"], errors="raise").to_numpy(dtype=float),
        ]
    )
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Poisson GAM features contain missing or non-finite values")
    if np.any(targets < 0) or not np.equal(targets, np.floor(targets)).all():
        raise ValueError("Poisson GAM goal targets must be non-negative integers")
    return features, targets


class PoissonGAMModel:
    name = "poisson_gam"

    def __init__(self, n_knots: int = 4, degree: int = 3, alpha: float = 1.0) -> None:
        if n_knots < 3:
            raise ValueError("Poisson GAM requires at least three spline knots")
        if degree < 1:
            raise ValueError("Poisson GAM spline degree must be positive")
        if alpha < 0:
            raise ValueError("Poisson GAM regularization must be non-negative")
        self.n_knots = n_knots
        self.degree = degree
        self.alpha = alpha
        smooth_pipeline = Pipeline(
            [
                (
                    "splines",
                    SplineTransformer(
                        n_knots=n_knots,
                        degree=degree,
                        knots="quantile",
                        extrapolation="linear",
                        include_bias=False,
                    ),
                ),
                ("scale", StandardScaler()),
            ]
        )
        preprocessing = ColumnTransformer(
            [
                ("smooth", smooth_pipeline, SMOOTH_FEATURES),
                ("linear", StandardScaler(), LINEAR_FEATURES),
            ],
            remainder="drop",
        )
        self._pipeline = Pipeline(
            [
                ("features", preprocessing),
                ("poisson", PoissonRegressor(alpha=alpha, max_iter=2_000, tol=1e-8)),
            ]
        )
        self._rho = 0.0
        self._is_fitted = False
        self.last_prediction_diagnostics = pd.DataFrame()

    def fit(self, train: pd.DataFrame) -> None:
        features, goals = stack_goal_rows(train)
        self._pipeline.fit(features, goals)
        self._is_fitted = True
        home_rate, away_rate = self.expected_goal_rates(train)
        home_goals = train["home_score"].to_numpy(dtype=int)
        away_goals = train["away_score"].to_numpy(dtype=int)

        # Only rho is fitted after the GAM. The Poisson terms do not depend on
        # rho, so maximizing the Dixon-Coles likelihood reduces to log(tau).
        def rho_objective(rho: float) -> float:
            corrections = dixon_coles_corrections(home_rate, away_rate, rho)
            if not np.isfinite(corrections).all() or np.any(corrections <= TAU_EPSILON):
                return 1e20
            tau = dixon_coles_tau(home_goals, away_goals, home_rate, away_rate, rho)
            if not np.isfinite(tau).all() or np.any(tau <= TAU_EPSILON):
                return 1e20
            return -float(np.log(tau).sum())

        result = minimize_scalar(rho_objective, bounds=(-RHO_BOUND, RHO_BOUND), method="bounded")
        if result.success and np.isfinite(result.fun) and result.fun < rho_objective(0.0):
            self._rho = float(result.x)
        else:
            self._rho = 0.0

    def expected_goal_rates(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("PoissonGAMModel.fit must be called before prediction")
        require_gam_columns(matches)
        home_rate = self._pipeline.predict(attacking_rows(matches, "home"))
        away_rate = self._pipeline.predict(attacking_rows(matches, "away"))
        return clip_expected_goal_rates(home_rate, away_rate)

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        home_rate, away_rate = self.expected_goal_rates(test)
        effective_rhos = stabilize_rho(home_rate, away_rate, self._rho)
        corrections = dixon_coles_corrections(home_rate, away_rate, effective_rhos)
        diagnostics = pd.DataFrame(
            {
                "match_index": test.index.to_numpy(),
                "home_expected_goals": home_rate,
                "away_expected_goals": away_rate,
                "fitted_rho": self._rho,
                "effective_rho": effective_rhos,
                "rho_adjusted": ~np.isclose(effective_rhos, self._rho),
                "minimum_correction": corrections.min(axis=1),
            }
        )
        for column in ("season", "match_date", "home_team", "away_team"):
            if column in test.columns:
                diagnostics[column] = test[column].to_numpy()
        self.last_prediction_diagnostics = diagnostics
        return np.vstack(
            [
                scoreline_to_outcome_probabilities(home, away, effective_rho)
                for home, away, effective_rho in zip(
                    home_rate,
                    away_rate,
                    effective_rhos,
                    strict=True,
                )
            ]
        )

    def export_coefficients(self, test_season: str) -> pd.DataFrame | None:
        if not self._is_fitted:
            raise RuntimeError("PoissonGAMModel.fit must be called before export")
        feature_names = self._pipeline.named_steps["features"].get_feature_names_out()
        coefficients = self._pipeline.named_steps["poisson"].coef_
        rows = [
            {
                "model": self.name,
                "test_season": test_season,
                "result_class": "poisson_goal_rate",
                "feature": str(feature),
                "standardized_coefficient": float(coefficient),
            }
            for feature, coefficient in zip(feature_names, coefficients, strict=True)
        ]
        rows.append(
            {
                "model": self.name,
                "test_season": test_season,
                "result_class": "score_model",
                "feature": "dixon_coles_rho",
                "standardized_coefficient": self._rho,
            }
        )
        return pd.DataFrame(rows)
