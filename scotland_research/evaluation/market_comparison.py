# Compare model log loss against the closing-market benchmark.

from __future__ import annotations

import pandas as pd

from constants import CLOSE_THRESHOLD, VERY_CLOSE_THRESHOLD


def market_comparison_label(model: str, relative_difference: float) -> str:
    if model == "closing_market":
        return "baseline"
    if relative_difference < 0:
        return "outperforms_market"
    if relative_difference <= VERY_CLOSE_THRESHOLD:
        return "within_1_percent"
    if relative_difference <= CLOSE_THRESHOLD:
        return "within_2_percent"
    return "worse_by_more_than_2_percent"


def add_market_comparison(
    metrics: pd.DataFrame,
    group_column: str | None = None,
) -> pd.DataFrame:
    compared = metrics.copy()
    if group_column is None:
        market_rows = compared[compared["model"].eq("closing_market")]
        if len(market_rows) != 1:
            raise ValueError("Overall metrics must contain one closing-market row")
        compared["closing_market_log_loss"] = market_rows["log_loss"].iloc[0]
    else:
        market_rows = compared[compared["model"].eq("closing_market")]
        market_by_group = market_rows.set_index(group_column)["log_loss"]
        if market_by_group.index.duplicated().any():
            raise ValueError(f"Multiple closing-market rows for {group_column}")
        compared["closing_market_log_loss"] = compared[group_column].map(market_by_group)
        if compared["closing_market_log_loss"].isna().any():
            raise ValueError(f"A {group_column} group has no closing-market row")

    compared["log_loss_vs_closing_market"] = (
        compared["log_loss"] - compared["closing_market_log_loss"]
    )
    compared["log_loss_relative_to_closing_market"] = (
        compared["log_loss_vs_closing_market"]
        / compared["closing_market_log_loss"]
    )
    compared["market_comparison"] = [
        market_comparison_label(model, relative_difference)
        for model, relative_difference in zip(
            compared["model"],
            compared["log_loss_relative_to_closing_market"],
            strict=True,
        )
    ]
    return compared
