"""Paired match-week uncertainty intervals for probabilistic match forecasts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from constants import CLASS_ORDER


PROBABILITY_COLUMNS = [f"prob_{label}" for label in CLASS_ORDER]
METRICS = ("log_loss", "brier_score", "rps")
PAIR_KEYS = [
    "league",
    "match_id",
    "season",
    "match_date",
    "result_3way",
]


def _per_match_scores(
    actual: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    class_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    actual_index = actual.map(class_index)
    if actual_index.isna().any():
        raise ValueError("Actual outcomes contain an unknown result class")
    if probabilities.shape != (len(actual), len(CLASS_ORDER)):
        raise ValueError("Probability matrix shape does not match the outcomes")
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
        raise ValueError("Probabilities must be finite and positive")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Probabilities must sum to one")

    rows = np.arange(len(actual))
    indexes = actual_index.to_numpy(dtype=int)
    observed = np.eye(len(CLASS_ORDER), dtype=float)[indexes]
    log_loss = -np.log(np.clip(probabilities[rows, indexes], np.finfo(float).eps, 1.0))
    brier = np.sum((probabilities - observed) ** 2, axis=1)
    # Normalized ranked probability score for the ordered H-D-A outcomes.
    rps = np.sum(
        (np.cumsum(probabilities, axis=1)[:, :-1] - np.cumsum(observed, axis=1)[:, :-1])
        ** 2,
        axis=1,
    ) / (len(CLASS_ORDER) - 1)
    return {"log_loss": log_loss, "brier_score": brier, "rps": rps}


def prepare_paired_comparison(
    predictions: pd.DataFrame,
    *,
    training_scope: str,
    baseline_model: str,
    enhanced_model: str,
) -> pd.DataFrame:
    """Return paired per-match losses; positive improvement favours enhanced."""

    required = {"training_scope", "model", *PAIR_KEYS, *PROBABILITY_COLUMNS}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing columns: {missing}")

    selected = predictions[predictions["training_scope"].eq(training_scope)]
    frames: dict[str, pd.DataFrame] = {}
    for role, model in (("baseline", baseline_model), ("enhanced", enhanced_model)):
        frame = selected[selected["model"].eq(model)][PAIR_KEYS + PROBABILITY_COLUMNS].copy()
        if frame.empty:
            raise ValueError(f"No {training_scope} predictions found for {model}")
        if frame.duplicated(PAIR_KEYS).any():
            raise ValueError(f"{model} contains duplicate match predictions")
        frame = frame.rename(
            columns={column: f"{role}_{column}" for column in PROBABILITY_COLUMNS}
        )
        frames[role] = frame

    paired = frames["baseline"].merge(
        frames["enhanced"],
        on=PAIR_KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        counts = paired["_merge"].value_counts().to_dict()
        raise ValueError(
            f"{baseline_model} and {enhanced_model} do not predict identical matches: {counts}"
        )
    paired = paired.drop(columns="_merge")
    paired["match_date"] = pd.to_datetime(paired["match_date"], errors="raise")
    paired["match_week"] = paired["match_date"].dt.to_period("W-MON").astype(str)

    for role in ("baseline", "enhanced"):
        probabilities = paired[
            [f"{role}_{column}" for column in PROBABILITY_COLUMNS]
        ].to_numpy(dtype=float)
        scores = _per_match_scores(paired["result_3way"], probabilities)
        for metric, values in scores.items():
            paired[f"{role}_{metric}"] = values
    for metric in METRICS:
        paired[f"improvement_{metric}"] = (
            paired[f"baseline_{metric}"] - paired[f"enhanced_{metric}"]
        )
    return paired


def _observed_values(data: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    league_means = data.groupby("league", sort=True)[
        [
            *[f"baseline_{metric}" for metric in METRICS],
            *[f"improvement_{metric}" for metric in METRICS],
        ]
    ].mean()
    equal_baseline = league_means[
        [f"baseline_{metric}" for metric in METRICS]
    ].mean().to_numpy(dtype=float)
    equal_improvement = league_means[
        [f"improvement_{metric}" for metric in METRICS]
    ].mean().to_numpy(dtype=float)
    weighted_baseline = np.array(
        [data[f"baseline_{metric}"].mean() for metric in METRICS],
        dtype=float,
    )
    weighted_improvement = np.array(
        [data[f"improvement_{metric}"].mean() for metric in METRICS],
        dtype=float,
    )
    return {
        "equal_league": {
            "baseline": equal_baseline,
            "improvement": equal_improvement,
        },
        "match_weighted": {
            "baseline": weighted_baseline,
            "improvement": weighted_improvement,
        },
    }


def resample_league_season_weeks(
    data: pd.DataFrame,
    *,
    repetitions: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired bootstrap of match weeks inside every league-season."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    sum_columns = [
        *[f"baseline_{metric}" for metric in METRICS],
        *[f"improvement_{metric}" for metric in METRICS],
    ]
    weekly = (
        data.groupby(["league", "season", "match_week"], as_index=False)
        .agg(matches=("match_id", "size"), **{column: (column, "sum") for column in sum_columns})
    )
    leagues = sorted(weekly["league"].unique())
    league_index = {league: index for index, league in enumerate(leagues)}
    blocks = [
        (
            league_index[str(league)],
            group[["matches", *sum_columns]].to_numpy(dtype=float),
        )
        for (league, _), group in weekly.groupby(["league", "season"], sort=True)
    ]
    if not blocks:
        raise ValueError("No league-season match-week blocks were created")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    metric_count = len(METRICS)
    for repetition in range(repetitions):
        league_matches = np.zeros(len(leagues), dtype=float)
        league_baseline = np.zeros((len(leagues), metric_count), dtype=float)
        league_improvement = np.zeros((len(leagues), metric_count), dtype=float)
        for current_league, values in blocks:
            sampled = values[rng.integers(0, len(values), size=len(values))]
            league_matches[current_league] += sampled[:, 0].sum()
            league_baseline[current_league] += sampled[:, 1 : 1 + metric_count].sum(axis=0)
            league_improvement[current_league] += sampled[:, 1 + metric_count :].sum(axis=0)

        league_baseline_means = league_baseline / league_matches[:, None]
        league_improvement_means = league_improvement / league_matches[:, None]
        values_by_weighting = {
            "equal_league": (
                league_baseline_means.mean(axis=0),
                league_improvement_means.mean(axis=0),
            ),
            "match_weighted": (
                league_baseline.sum(axis=0) / league_matches.sum(),
                league_improvement.sum(axis=0) / league_matches.sum(),
            ),
        }
        for weighting, (baseline, improvement) in values_by_weighting.items():
            for metric_index, metric in enumerate(METRICS):
                rows.append(
                    {
                        "repetition": repetition,
                        "weighting": weighting,
                        "metric": metric,
                        "absolute_improvement": improvement[metric_index],
                        "relative_improvement_pct": (
                            100 * improvement[metric_index] / baseline[metric_index]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_interval(
    data: pd.DataFrame,
    samples: pd.DataFrame,
    comparison: Mapping[str, str],
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    observed = _observed_values(data)
    rows: list[dict[str, object]] = []
    for weighting, weighting_values in observed.items():
        for metric_index, metric in enumerate(METRICS):
            metric_samples = samples[
                samples["weighting"].eq(weighting) & samples["metric"].eq(metric)
            ]
            absolute_limits = metric_samples["absolute_improvement"].quantile([0.025, 0.975])
            relative_limits = metric_samples["relative_improvement_pct"].quantile([0.025, 0.975])
            baseline = weighting_values["baseline"][metric_index]
            improvement = weighting_values["improvement"][metric_index]
            rows.append(
                {
                    **comparison,
                    "weighting": weighting,
                    "metric": metric,
                    "matches": len(data),
                    "leagues": data["league"].nunique(),
                    "observed_baseline": baseline,
                    "observed_enhanced": baseline - improvement,
                    "observed_absolute_improvement": improvement,
                    "observed_relative_improvement_pct": 100 * improvement / baseline,
                    "lower_95_absolute": absolute_limits.loc[0.025],
                    "upper_95_absolute": absolute_limits.loc[0.975],
                    "lower_95_relative_pct": relative_limits.loc[0.025],
                    "upper_95_relative_pct": relative_limits.loc[0.975],
                    "samples_favouring_enhanced_pct": (
                        100 * metric_samples["absolute_improvement"].gt(0).mean()
                    ),
                    "interval_excludes_zero": bool(
                        absolute_limits.loc[0.025] > 0 or absolute_limits.loc[0.975] < 0
                    ),
                    "repetitions": repetitions,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def summarize_individual_league_intervals(
    data: pd.DataFrame,
    comparison: Mapping[str, str],
    *,
    repetitions: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run paired season-week intervals separately inside each league."""

    frames: list[pd.DataFrame] = []
    for league_number, league in enumerate(sorted(data["league"].unique())):
        league_data = data[data["league"].eq(league)].copy()
        league_seed = seed + league_number
        samples = resample_league_season_weeks(
            league_data,
            repetitions=repetitions,
            seed=league_seed,
        )
        summary = summarize_interval(
            league_data,
            samples,
            comparison,
            repetitions=repetitions,
            seed=league_seed,
        )
        summary = summary[summary["weighting"].eq("match_weighted")].copy()
        summary.insert(6, "league", league)
        frames.append(summary)
    if not frames:
        raise ValueError("No leagues were available for individual intervals")
    return pd.concat(frames, ignore_index=True)
