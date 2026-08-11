"""Shared Dixon-Coles fitting and score-probability utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


HALF_LIFE_DAYS = 365.0
MAX_GOALS = 10
PLAYER_L2_STRENGTH = 1.0
RHO_BOUND = 0.20
INVALID_OBJECTIVE = 1e20
TAU_EPSILON = 1e-10


def dixon_coles_tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_rate: np.ndarray,
    away_rate: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Return the Dixon-Coles low-score correction for observed scorelines."""
    tau = np.ones_like(home_rate, dtype=float)
    zero_zero = (home_goals == 0) & (away_goals == 0)
    zero_one = (home_goals == 0) & (away_goals == 1)
    one_zero = (home_goals == 1) & (away_goals == 0)
    one_one = (home_goals == 1) & (away_goals == 1)
    tau[zero_zero] = 1.0 - home_rate[zero_zero] * away_rate[zero_zero] * rho
    tau[zero_one] = 1.0 + home_rate[zero_one] * rho
    tau[one_zero] = 1.0 + away_rate[one_zero] * rho
    tau[one_one] = 1.0 - rho
    return tau


def poisson_probabilities(rate: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    goals = np.arange(max_goals + 1, dtype=float)
    return np.exp(goals * np.log(rate) - rate - gammaln(goals + 1.0))


def scoreline_to_outcome_probabilities(
    home_rate: float,
    away_rate: float,
    rho: float,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """Build the corrected score grid and return probabilities in H, D, A order."""
    if not np.isfinite([home_rate, away_rate, rho]).all():
        raise ValueError("Dixon-Coles parameters must be finite")
    if home_rate <= 0 or away_rate <= 0:
        raise ValueError("Expected goal rates must be positive")

    home = poisson_probabilities(home_rate, max_goals)
    away = poisson_probabilities(away_rate, max_goals)
    grid = np.outer(home, away)
    corrections = np.array(
        [
            1.0 - home_rate * away_rate * rho,
            1.0 + home_rate * rho,
            1.0 + away_rate * rho,
            1.0 - rho,
        ]
    )
    if not np.isfinite(corrections).all() or np.any(corrections <= TAU_EPSILON):
        raise ValueError("Dixon-Coles low-score correction is not positive")
    grid[0, 0] *= corrections[0]
    grid[0, 1] *= corrections[1]
    grid[1, 0] *= corrections[2]
    grid[1, 1] *= corrections[3]

    total = grid.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Dixon-Coles score grid has invalid probability mass")
    grid /= total
    return np.array(
        [
            np.tril(grid, k=-1).sum(),
            np.trace(grid),
            np.triu(grid, k=1).sum(),
        ],
        dtype=float,
    )


@dataclass
class DixonColesParameters:
    team_ids: tuple[str, ...]
    team_names: dict[str, str]
    goal_intercept: float
    home_advantage: float
    rho: float
    attack: np.ndarray
    defence: np.ndarray
    player_coefficients: np.ndarray
    player_feature_means: np.ndarray
    player_feature_scales: np.ndarray


class DixonColesEstimator:
    """Time-weighted Dixon-Coles estimator with optional player-form adjustment."""

    def __init__(
        self,
        player_features: list[str] | None = None,
        player_l2_strength: float = PLAYER_L2_STRENGTH,
    ) -> None:
        self.player_features = list(player_features or [])
        self.player_l2_strength = player_l2_strength
        self.parameters: DixonColesParameters | None = None

    @staticmethod
    def _full_zero_sum(free_values: np.ndarray) -> np.ndarray:
        return np.append(free_values, -free_values.sum())

    def _unpack(
        self,
        vector: np.ndarray,
        team_count: int,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        free_count = team_count - 1
        goal_intercept = float(vector[0])
        home_advantage = float(vector[1])
        rho = float(vector[2])
        attack_start = 3
        defence_start = attack_start + free_count
        player_start = defence_start + free_count
        attack = self._full_zero_sum(vector[attack_start:defence_start])
        defence = self._full_zero_sum(vector[defence_start:player_start])
        player_coefficients = vector[player_start:]
        return (
            goal_intercept,
            home_advantage,
            rho,
            attack,
            defence,
            player_coefficients,
        )

    def fit(self, train: pd.DataFrame) -> None:
        team_ids = tuple(
            sorted(set(train["home_team_id"].astype(str)) | set(train["away_team_id"].astype(str)))
        )
        if len(team_ids) < 2:
            raise ValueError("Dixon-Coles requires at least two training teams")
        team_index = {team_id: index for index, team_id in enumerate(team_ids)}
        home_indexes = train["home_team_id"].astype(str).map(team_index).to_numpy(dtype=int)
        away_indexes = train["away_team_id"].astype(str).map(team_index).to_numpy(dtype=int)
        home_goals = train["home_score"].to_numpy(dtype=int)
        away_goals = train["away_score"].to_numpy(dtype=int)

        dates = pd.to_datetime(train["_match_datetime"], utc=True, errors="raise")
        age_days = (dates.max() - dates).dt.total_seconds().to_numpy() / 86_400.0
        weights = np.exp(-np.log(2.0) * age_days / HALF_LIFE_DAYS)
        weights /= weights.mean()

        if self.player_features:
            player_values = train[self.player_features].to_numpy(dtype=float)
            player_means = player_values.mean(axis=0)
            player_scales = player_values.std(axis=0, ddof=0)
            player_scales[player_scales == 0] = 1.0
            standardized_player = (player_values - player_means) / player_scales
        else:
            player_means = np.empty(0, dtype=float)
            player_scales = np.empty(0, dtype=float)
            standardized_player = np.empty((len(train), 0), dtype=float)

        mean_home = max(float(np.average(home_goals, weights=weights)), 0.05)
        mean_away = max(float(np.average(away_goals, weights=weights)), 0.05)
        initial = np.zeros(3 + 2 * (len(team_ids) - 1) + len(self.player_features))
        initial[0] = np.log(mean_away)
        initial[1] = np.clip(np.log(mean_home / mean_away), -1.0, 1.0)

        bounds = (
            [(-3.0, 2.0), (-1.0, 1.0), (-RHO_BOUND, RHO_BOUND)]
            + [(-2.5, 2.5)] * (2 * (len(team_ids) - 1))
            + [(-2.0, 2.0)] * len(self.player_features)
        )

        def objective(vector: np.ndarray) -> float:
            (
                goal_intercept,
                home_advantage,
                rho,
                attack,
                defence,
                player_coefficients,
            ) = self._unpack(vector, len(team_ids))
            player_shift = standardized_player @ player_coefficients
            home_linear = (
                goal_intercept
                + home_advantage
                + attack[home_indexes]
                + defence[away_indexes]
                + player_shift
            )
            away_linear = (
                goal_intercept
                + attack[away_indexes]
                + defence[home_indexes]
                - player_shift
            )
            if (
                not np.isfinite(home_linear).all()
                or not np.isfinite(away_linear).all()
                or np.any(np.abs(home_linear) > 10)
                or np.any(np.abs(away_linear) > 10)
            ):
                return INVALID_OBJECTIVE

            home_rate = np.exp(home_linear)
            away_rate = np.exp(away_linear)
            tau = dixon_coles_tau(home_goals, away_goals, home_rate, away_rate, rho)
            if not np.isfinite(tau).all() or np.any(tau <= TAU_EPSILON):
                return INVALID_OBJECTIVE

            log_probability = (
                home_goals * home_linear
                - home_rate
                - gammaln(home_goals + 1.0)
                + away_goals * away_linear
                - away_rate
                - gammaln(away_goals + 1.0)
                + np.log(tau)
            )
            penalty = self.player_l2_strength * np.dot(
                player_coefficients, player_coefficients
            )
            value = -float(np.dot(weights, log_probability)) + float(penalty)
            return value if np.isfinite(value) else INVALID_OBJECTIVE

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 2_000, "ftol": 1e-9},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Dixon-Coles optimization failed: {result.message}")

        (
            goal_intercept,
            home_advantage,
            rho,
            attack,
            defence,
            player_coefficients,
        ) = self._unpack(result.x, len(team_ids))
        team_names: dict[str, str] = {}
        for id_column, name_column in (
            ("home_team_id", "home_team"),
            ("away_team_id", "away_team"),
        ):
            for team_id, team_name in train[[id_column, name_column]].itertuples(index=False):
                team_names[str(team_id)] = str(team_name)

        self.parameters = DixonColesParameters(
            team_ids=team_ids,
            team_names=team_names,
            goal_intercept=goal_intercept,
            home_advantage=home_advantage,
            rho=rho,
            attack=attack,
            defence=defence,
            player_coefficients=player_coefficients,
            player_feature_means=player_means,
            player_feature_scales=player_scales,
        )

    def expected_goal_rates(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.parameters is None:
            raise RuntimeError("DixonColesEstimator.fit must be called before prediction")
        parameters = self.parameters
        team_index = {team_id: index for index, team_id in enumerate(parameters.team_ids)}

        def values_for(team_ids: pd.Series, parameter_values: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    parameter_values[team_index[team_id]] if team_id in team_index else 0.0
                    for team_id in team_ids.astype(str)
                ],
                dtype=float,
            )

        home_attack = values_for(matches["home_team_id"], parameters.attack)
        away_attack = values_for(matches["away_team_id"], parameters.attack)
        home_defence = values_for(matches["home_team_id"], parameters.defence)
        away_defence = values_for(matches["away_team_id"], parameters.defence)

        if self.player_features:
            player_values = matches[self.player_features].to_numpy(dtype=float)
            standardized = (
                player_values - parameters.player_feature_means
            ) / parameters.player_feature_scales
            player_shift = standardized @ parameters.player_coefficients
        else:
            player_shift = np.zeros(len(matches), dtype=float)

        home_rate = np.exp(
            parameters.goal_intercept
            + parameters.home_advantage
            + home_attack
            + away_defence
            + player_shift
        )
        away_rate = np.exp(
            parameters.goal_intercept
            + away_attack
            + home_defence
            - player_shift
        )
        return home_rate, away_rate

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        if self.parameters is None:
            raise RuntimeError("DixonColesEstimator.fit must be called before prediction")
        home_rates, away_rates = self.expected_goal_rates(matches)
        return np.vstack(
            [
                scoreline_to_outcome_probabilities(
                    home_rate,
                    away_rate,
                    self.parameters.rho,
                )
                for home_rate, away_rate in zip(home_rates, away_rates, strict=True)
            ]
        )

    def export_parameters(self, model_name: str, test_season: str) -> pd.DataFrame:
        if self.parameters is None:
            raise RuntimeError("DixonColesEstimator.fit must be called before export")
        parameters = self.parameters
        rows: list[dict[str, object]] = []

        def append(feature: str, value: float) -> None:
            rows.append(
                {
                    "model": model_name,
                    "test_season": test_season,
                    "result_class": "score_model",
                    "feature": feature,
                    "standardized_coefficient": value,
                }
            )

        append("goal_intercept", parameters.goal_intercept)
        append("home_advantage", parameters.home_advantage)
        append("rho", parameters.rho)
        for index, team_id in enumerate(parameters.team_ids):
            label = f"{team_id}:{parameters.team_names.get(team_id, '')}"
            append(f"attack[{label}]", parameters.attack[index])
            append(f"defence[{label}]", parameters.defence[index])
        for feature, coefficient in zip(
            self.player_features,
            parameters.player_coefficients,
            strict=True,
        ):
            append(f"player_form[{feature}]", coefficient)
        return pd.DataFrame(rows)
