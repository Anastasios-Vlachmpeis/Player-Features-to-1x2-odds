# Walk-forward fold orchestration.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from constants import CLASS_ORDER, FOLDS
from evaluation.market_comparison import add_market_comparison
from evaluation.metrics import probability_frame, score_predictions
from models.base import MatchPredictor


@dataclass
class WalkForwardResult:
    fold_metrics: pd.DataFrame
    overall_metrics: pd.DataFrame
    predictions: pd.DataFrame
    coefficients: pd.DataFrame


def run_walk_forward(
    dataset: pd.DataFrame,
    predictors: list[MatchPredictor],
) -> WalkForwardResult:
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []

    for test_season, train_seasons in FOLDS:
        train = dataset[dataset["season"].isin(train_seasons)].copy()
        test = dataset[dataset["season"].eq(test_season)].copy()
        if train.empty or test.empty:
            raise ValueError(f"Walk-forward fold {test_season} has no train or test rows")
        if set(train["result_3way"]) != set(CLASS_ORDER):
            raise ValueError(f"Training fold for {test_season} does not contain all outcomes")

        for predictor in predictors:
            predictor.fit(train)
            probabilities = predictor.predict_proba(test)
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
                raise ValueError(f"{predictor.name} probabilities do not sum to one")

            scores = score_predictions(test["result_3way"], probabilities)
            metric_rows.append(
                {
                    "test_season": test_season,
                    "train_seasons": ";".join(train_seasons),
                    "train_matches": len(train),
                    "test_matches": len(test),
                    "model": predictor.name,
                    **scores,
                }
            )

            output = test[
                ["match_id", "season", "match_date", "home_team", "away_team", "result_3way"]
            ].copy()
            output.insert(0, "model", predictor.name)
            output = pd.concat(
                [output.reset_index(drop=True), probability_frame(probabilities)],
                axis=1,
            )
            output["predicted_result"] = np.asarray(CLASS_ORDER)[
                np.argmax(probabilities, axis=1)
            ]
            prediction_frames.append(output)

            coefficients = predictor.export_coefficients(test_season)
            if coefficients is not None and not coefficients.empty:
                coefficient_frames.append(coefficients)

    fold_metrics = add_market_comparison(
        pd.DataFrame(metric_rows),
        group_column="test_season",
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    coefficients = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame()
    )

    overall_rows: list[dict[str, object]] = []
    for model_name, group in predictions.groupby("model", sort=False):
        probabilities = group[["prob_H", "prob_D", "prob_A"]].to_numpy()
        overall_rows.append(
            {
                "model": model_name,
                "out_of_sample_matches": len(group),
                **score_predictions(group["result_3way"], probabilities),
            }
        )
    overall_metrics = pd.DataFrame(overall_rows)
    overall_metrics = add_market_comparison(overall_metrics)
    overall_metrics = overall_metrics.sort_values("log_loss", kind="stable").reset_index(drop=True)

    return WalkForwardResult(
        fold_metrics=fold_metrics,
        overall_metrics=overall_metrics,
        predictions=predictions,
        coefficients=coefficients,
    )
