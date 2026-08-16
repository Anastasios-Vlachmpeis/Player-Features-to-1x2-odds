"""Direct multinomial generalized additive model for H/D/A probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from constants import CLASS_ORDER


# The direct GAM preserves both strength difference and total match strength.
# Difference features explain home-away direction; totals allow the draw curve
# to change between low- and high-event fixtures with the same advantage.
SMOOTH_FEATURES = [
    "elo_difference",
    "expected_goal_difference",
    "expected_goal_total",
    "adjusted_shots_difference",
    "adjusted_key_passes_difference",
    "adjusted_defensive_difference",
    "forward_shots_difference",
    "pairwise_minutes_difference",
    "pairwise_minutes_mean",
    "replacement_quality_difference",
    "rating_trend_difference",
]

# Discrete lineup counts have only a handful of support points, so regularized
# linear effects are more stable than fitting separate spline curves to them.
LINEAR_FEATURES = [
    "retained_starters_difference",
    "retained_starters_mean",
    "missing_regulars_difference",
    "missing_regulars_total",
    "new_starters_difference",
    "new_starters_total",
    "missing_history_difference",
    "missing_history_total",
]

SOURCE_TEAM_FEATURES = [
    "elo_rating",
    "expected_goals_strength",
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
]

REQUIRED_MATCH_COLUMNS = {
    "result_3way",
    *[f"{side}_{feature}" for side in ("home", "away") for feature in SOURCE_TEAM_FEATURES],
}


def require_multinomial_gam_columns(matches: pd.DataFrame, require_target: bool = False) -> None:
    required = set(REQUIRED_MATCH_COLUMNS)
    if not require_target:
        required.discard("result_3way")
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(f"Multinomial GAM is missing required inputs: {', '.join(missing)}")


def direct_gam_features(matches: pd.DataFrame) -> pd.DataFrame:
    require_multinomial_gam_columns(matches)
    output = pd.DataFrame(index=matches.index)

    def difference(feature: str) -> pd.Series:
        return matches[f"home_{feature}"] - matches[f"away_{feature}"]

    def total(feature: str) -> pd.Series:
        return matches[f"home_{feature}"] + matches[f"away_{feature}"]

    def mean(feature: str) -> pd.Series:
        return total(feature) / 2.0

    # These formulae intentionally construct a compact representation instead
    # of splining all 249 home/away/difference columns from the feature dataset.
    output["elo_difference"] = difference("elo_rating")
    output["expected_goal_difference"] = difference("expected_goals_strength")
    output["expected_goal_total"] = total("expected_goals_strength")
    output["adjusted_shots_difference"] = difference("adjusted_shots_lineup_mean_5")
    output["adjusted_key_passes_difference"] = difference("adjusted_key_passes_lineup_mean_5")
    output["adjusted_defensive_difference"] = difference("adjusted_defensive_actions_lineup_mean_5")
    output["forward_shots_difference"] = difference("fwd_adjusted_shots_5")
    output["pairwise_minutes_difference"] = difference("mean_pairwise_prior_minutes")
    output["pairwise_minutes_mean"] = mean("mean_pairwise_prior_minutes")
    output["replacement_quality_difference"] = difference("replacement_quality")
    output["rating_trend_difference"] = difference("rating_lineup_mean_trend_1_5")
    output["retained_starters_difference"] = difference("retained_starters")
    output["retained_starters_mean"] = mean("retained_starters")
    output["missing_regulars_difference"] = difference("missing_regular_starters")
    output["missing_regulars_total"] = total("missing_regular_starters")
    output["new_starters_difference"] = difference("new_starters")
    output["new_starters_total"] = total("new_starters")
    output["missing_history_difference"] = difference("starters_without_history")
    output["missing_history_total"] = total("starters_without_history")
    output = output[SMOOTH_FEATURES + LINEAR_FEATURES]
    if output.isna().any().any() or not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ValueError("Multinomial GAM features contain missing or non-finite values")
    return output


class MultinomialGAMModel:
    name = "multinomial_gam"

    def __init__(self, n_knots: int = 4, degree: int = 3, regularization_c: float = 1.0) -> None:
        if n_knots < 3:
            raise ValueError("Multinomial GAM requires at least three spline knots")
        if degree < 1:
            raise ValueError("Multinomial GAM spline degree must be positive")
        if regularization_c <= 0:
            raise ValueError("Multinomial GAM C must be positive")
        self.n_knots = n_knots
        self.degree = degree
        self.regularization_c = regularization_c
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
                (
                    "multinomial",
                    LogisticRegression(
                        C=regularization_c,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=2_000,
                        tol=1e-8,
                    ),
                ),
            ]
        )
        self._is_fitted = False

    def fit(self, train: pd.DataFrame) -> None:
        require_multinomial_gam_columns(train, require_target=True)
        targets = train["result_3way"].astype(str)
        if set(targets) != set(CLASS_ORDER):
            raise ValueError("Multinomial GAM training data must contain H, D, and A")
        self._pipeline.fit(direct_gam_features(train), targets)
        self._is_fitted = True

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("MultinomialGAMModel.fit must be called before prediction")
        raw = self._pipeline.predict_proba(direct_gam_features(test))
        classes = self._pipeline.named_steps["multinomial"].classes_
        class_index = {label: index for index, label in enumerate(classes)}
        return raw[:, [class_index[label] for label in CLASS_ORDER]]

    def export_coefficients(self, test_season: str) -> pd.DataFrame | None:
        if not self._is_fitted:
            raise RuntimeError("MultinomialGAMModel.fit must be called before export")
        feature_names = self._pipeline.named_steps["features"].get_feature_names_out()
        estimator = self._pipeline.named_steps["multinomial"]
        rows: list[dict[str, object]] = []
        for class_label, coefficients, intercept in zip(estimator.classes_, estimator.coef_, estimator.intercept_, strict=True):
            rows.append(
                {
                    "model": self.name,
                    "test_season": test_season,
                    "result_class": str(class_label),
                    "feature": "intercept",
                    "standardized_coefficient": float(intercept),
                }
            )
            rows.extend(
                {
                    "model": self.name,
                    "test_season": test_season,
                    "result_class": str(class_label),
                    "feature": str(feature),
                    "standardized_coefficient": float(coefficient),
                }
                for feature, coefficient in zip(feature_names, coefficients, strict=True)
            )
        return pd.DataFrame(rows)
