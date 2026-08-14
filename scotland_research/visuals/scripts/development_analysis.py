"""Build five-league development-analysis visualizations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCOTLAND_RESEARCH_DIR = Path(__file__).resolve().parents[2]
if str(SCOTLAND_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCOTLAND_RESEARCH_DIR))

from constants import DEFAULT_EVALUATION_DIR, EXPECTED_LEAGUES, PROJECT_ROOT
from league_config import DEVELOPMENT_SEASONS


DEFAULT_VALIDATION_DIR = PROJECT_ROOT / "artifacts" / "all_leagues_data_validation"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = SCOTLAND_RESEARCH_DIR / "visuals" / "development_analysis"
PRIMARY_MODEL = "market_plus_player_form"
SCOPE_LABELS = {
    "pooled": "Shared model",
    "league_specific": "Separate models",
}
LEAGUE_LABELS = {
    "belgium": "Belgium",
    "netherlands": "Netherlands",
    "portugal": "Portugal",
    "scotland": "Scotland",
    "turkey": "Turkey",
}
COLORS = {
    "market": "#4c4c4c",
    "pooled": "#2878b5",
    "league_specific": "#d95f02",
    "accent": "#4daf4a",
    "grid": "#d9d9d9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-model", default=PRIMARY_MODEL)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def load_coverage(validation_dir: Path) -> pd.DataFrame:
    frames = []
    for league in sorted(EXPECTED_LEAGUES):
        path = validation_dir / league / "season_coverage.csv"
        if not path.exists():
            raise FileNotFoundError(f"Coverage report not found: {path}")
        frame = pd.read_csv(path)
        frame.insert(0, "league", league)
        frames.append(frame)
    coverage = pd.concat(frames, ignore_index=True)
    expected = pd.MultiIndex.from_product(
        [sorted(EXPECTED_LEAGUES), DEVELOPMENT_SEASONS],
        names=["league", "season"],
    )
    observed = pd.MultiIndex.from_frame(coverage[["league", "season"]])
    missing = expected.difference(observed)
    if len(missing):
        raise ValueError(f"Coverage reports are missing league-seasons: {missing.tolist()}")
    return coverage


def coverage_heatmap(coverage: pd.DataFrame, output_dir: Path) -> Path:
    panels = [
        ("player_match_coverage", "Player-match data"),
        ("twenty_two_starters_coverage", "Complete starting lineups"),
        ("model_ready_coverage", "Final model-ready matches"),
    ]
    leagues = sorted(EXPECTED_LEAGUES)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True, constrained_layout=True)
    image = None
    for axis, (column, title) in zip(axes, panels, strict=True):
        table = (
            coverage.pivot(index="league", columns="season", values=column)
            .reindex(index=leagues, columns=DEVELOPMENT_SEASONS)
        )
        values = table.to_numpy(dtype=float)
        image = axis.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for row in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row, column_index]
                color = "white" if value >= 0.72 else "black"
                axis.text(
                    column_index,
                    row,
                    f"{100 * value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                )
        axis.set_title(title)
        axis.set_xticks(range(len(DEVELOPMENT_SEASONS)), DEVELOPMENT_SEASONS, rotation=35, ha="right")
        axis.set_yticks(range(len(leagues)), [LEAGUE_LABELS[x] for x in leagues])
        axis.set_xlabel("Season")
    axes[0].set_ylabel("League")
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.72, label="Coverage")
    fig.suptitle("Five-league data coverage before model development", fontsize=15)
    fig.text(
        0.5,
        -0.02,
        "Greece is excluded under the predeclared lineup-coverage rule.",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, output_dir, "01_league_season_coverage")


def load_match_validation(validation_dir: Path) -> pd.DataFrame:
    frames = []
    for league in sorted(EXPECTED_LEAGUES):
        path = validation_dir / league / "match_validation.csv"
        if not path.exists():
            raise FileNotFoundError(f"Match-validation report not found: {path}")
        frame = pd.read_csv(path)
        frame.insert(0, "league", league)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def sample_construction_flow(validation: pd.DataFrame, output_dir: Path) -> Path:
    boolean_columns = [
        "football_data_match",
        "score_matches_football_data",
        "closing_odds_available",
        "player_data_available",
        "twenty_two_starters",
        "model_ready",
    ]
    for column in boolean_columns:
        validation[column] = validation[column].astype("string").str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        ).fillna(False)

    masks = []
    current = pd.Series(True, index=validation.index)
    for column in boolean_columns:
        current = current & validation[column]
        masks.append(current.copy())
    labels = [
        "Top-division fixtures linked",
        "Scores agree",
        "Valid closing odds",
        "Player data present",
        "Exactly 22 identified starters",
        "Final model-ready sample",
    ]
    totals = np.array([int(mask.sum()) for mask in masks])

    fig, (axis, table_axis) = plt.subplots(
        1,
        2,
        figsize=(13, 7),
        gridspec_kw={"width_ratios": [3.4, 1.6]},
        constrained_layout=True,
    )
    y = np.arange(len(labels))
    widths = totals / totals[0]
    bars = axis.barh(y, widths, color=plt.cm.Blues(np.linspace(0.45, 0.85, len(labels))))
    axis.invert_yaxis()
    axis.set_yticks(y, labels)
    axis.set_xlim(0, 1.04)
    axis.set_xlabel("Share of linked top-division fixtures retained")
    axis.set_title("Construction of the five-league analytical sample")
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    for bar, total, share in zip(bars, totals, widths, strict=True):
        axis.text(
            min(share + 0.012, 1.0),
            bar.get_y() + bar.get_height() / 2,
            f"{total:,}  ({100 * share:.1f}%)",
            va="center",
            fontsize=9,
        )

    per_league = []
    for league in sorted(EXPECTED_LEAGUES):
        group = validation[validation["league"].eq(league)]
        linked = int(group["football_data_match"].sum())
        ready = int(group["model_ready"].sum())
        per_league.append(
            [LEAGUE_LABELS[league], f"{linked:,}", f"{ready:,}", f"{100 * ready / linked:.1f}%"]
        )
    table_axis.axis("off")
    table_axis.set_title("Retention by league", fontsize=11, pad=10)
    table = table_axis.table(
        cellText=per_league,
        colLabels=["League", "Linked", "Final", "Retained"],
        cellLoc="right",
        colLoc="right",
        colWidths=[0.34, 0.22, 0.22, 0.22],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    return save_figure(fig, output_dir, "02_sample_construction_flow")


def experiment_timeline(output_dir: Path) -> Path:
    folds = [
        ("Development fold 1", ("2020-21", "2021-22"), "2022-23"),
        ("Development fold 2", ("2020-21", "2021-22", "2022-23"), "2023-24"),
        ("Development fold 3", ("2020-21", "2021-22", "2022-23", "2023-24"), "2024-25"),
        ("Final examination", tuple(DEVELOPMENT_SEASONS), "2025-26"),
    ]
    seasons = [*DEVELOPMENT_SEASONS, "2025-26"]
    fig, axis = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    axis.set_xlim(-0.5, len(seasons) - 0.5)
    axis.set_ylim(-0.7, len(folds) - 0.3)
    for row, (label, training, test) in enumerate(folds):
        for column, season in enumerate(seasons):
            if season in training:
                color, text = "#9ecae1", "Train"
            elif season == test:
                color, text = ("#ef8a62", "Final test") if test == "2025-26" else ("#74c476", "Predict")
            else:
                color, text = "#eeeeee", ""
            rectangle = plt.Rectangle(
                (column - 0.45, row - 0.34),
                0.90,
                0.68,
                facecolor=color,
                edgecolor="white",
                linewidth=1.5,
            )
            axis.add_patch(rectangle)
            axis.text(column, row, text, ha="center", va="center", fontsize=9)
    axis.axvline(4.5, color="#b2182b", linewidth=2.0, linestyle="--")
    axis.text(4.55, -0.48, "Specification frozen", color="#b2182b", ha="left", va="bottom", fontsize=9)
    axis.set_xticks(range(len(seasons)), seasons)
    axis.set_yticks(range(len(folds)), [row[0] for row in folds])
    axis.invert_yaxis()
    axis.set_xlabel("Season")
    axis.set_title("Chronological development and untouched final-season design")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    return save_figure(fig, output_dir, "03_chronological_experiment_timeline")


def load_predictions(evaluation_dir: Path) -> pd.DataFrame:
    path = evaluation_dir / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Combined predictions not found: {path}. Run evaluate_models.py first."
        )
    predictions = pd.read_csv(path)
    required = {
        "training_scope",
        "model",
        "league",
        "match_id",
        "season",
        "match_date",
        "result_3way",
        "prob_H",
        "prob_D",
        "prob_A",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing required columns: {missing}")
    predictions["match_date"] = pd.to_datetime(predictions["match_date"], errors="raise")
    return predictions


def paired_losses(
    predictions: pd.DataFrame,
    scope: str,
    model: str,
    league: str,
) -> pd.DataFrame:
    keys = ["league", "match_id", "season", "match_date", "result_3way"]
    probability_columns = ["prob_H", "prob_D", "prob_A"]
    subset = predictions[
        predictions["training_scope"].eq(scope) & predictions["league"].eq(league)
    ]
    market = subset[subset["model"].eq("closing_market")][keys + probability_columns]
    enhanced = subset[subset["model"].eq(model)][keys + probability_columns]
    data = market.merge(
        enhanced,
        on=keys,
        suffixes=("_market", "_model"),
        validate="one_to_one",
    )
    if len(data) != len(market) or len(data) != len(enhanced) or data.empty:
        raise ValueError(f"{scope}/{league}/{model} does not cover identical market matches")
    outcome_index = data["result_3way"].map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=int)
    row_index = np.arange(len(data))
    market_probabilities = data[[f"prob_{label}_market" for label in ("H", "D", "A")]].to_numpy()
    model_probabilities = data[[f"prob_{label}_model" for label in ("H", "D", "A")]].to_numpy()
    data["market_loss"] = -np.log(market_probabilities[row_index, outcome_index])
    data["model_loss"] = -np.log(model_probabilities[row_index, outcome_index])
    data["week"] = data["match_date"].dt.to_period("W-MON").astype(str)
    return data


def bootstrap_relative_difference(
    data: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    observed = 100 * (data["model_loss"].mean() - data["market_loss"].mean()) / data["market_loss"].mean()
    weekly = data.groupby(["season", "week"], as_index=False).agg(
        matches=("match_id", "size"),
        market_loss=("market_loss", "sum"),
        model_loss=("model_loss", "sum"),
    )
    season_blocks = [
        group[["matches", "market_loss", "model_loss"]].to_numpy(dtype=float)
        for _, group in weekly.groupby("season", sort=False)
    ]
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    for repetition in range(repetitions):
        selected = np.concatenate(
            [blocks[rng.integers(0, len(blocks), len(blocks))] for blocks in season_blocks]
        )
        market_total = selected[:, 1].sum()
        samples[repetition] = 100 * (selected[:, 2].sum() - market_total) / market_total
    return observed, samples


def market_comparison_plot(
    predictions: pd.DataFrame,
    output_dir: Path,
    model: str,
    repetitions: int,
) -> Path:
    leagues = sorted(EXPECTED_LEAGUES)
    labels = [LEAGUE_LABELS[league] for league in leagues] + ["Equal-league average"]
    fig, axis = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    offsets = {"pooled": -0.12, "league_specific": 0.12}
    for scope_index, scope in enumerate(("pooled", "league_specific")):
        observed_values = []
        sample_arrays = []
        for league_index, league in enumerate(leagues):
            data = paired_losses(predictions, scope, model, league)
            observed, samples = bootstrap_relative_difference(
                data,
                repetitions,
                seed=42 + 100 * scope_index + league_index,
            )
            observed_values.append(observed)
            sample_arrays.append(samples)
        observed_values.append(float(np.mean(observed_values)))
        sample_arrays.append(np.mean(np.vstack(sample_arrays), axis=0))
        lower = [np.quantile(samples, 0.025) for samples in sample_arrays]
        upper = [np.quantile(samples, 0.975) for samples in sample_arrays]
        y = np.arange(len(labels)) + offsets[scope]
        values = np.asarray(observed_values)
        axis.errorbar(
            values,
            y,
            xerr=np.vstack([values - lower, np.asarray(upper) - values]),
            fmt="o",
            markersize=6,
            capsize=3,
            color=COLORS[scope],
            label=SCOPE_LABELS[scope],
        )
    axis.axvline(0, color="black", linewidth=1.1)
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Log-loss change relative to closing market (%) — lower is better")
    axis.set_title("Does player information improve closing-market forecasts?")
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    axis.legend(frameon=False)
    return save_figure(fig, output_dir, "04_market_comparison_by_league")


def pooled_vs_separate_plot(
    predictions: pd.DataFrame,
    output_dir: Path,
    model: str,
) -> Path:
    leagues = sorted(EXPECTED_LEAGUES)
    labels = [LEAGUE_LABELS[league] for league in leagues] + ["Equal-league average"]
    scope_values: dict[str, list[float]] = {scope: [] for scope in SCOPE_LABELS}
    for scope in scope_values:
        for league in leagues:
            data = paired_losses(predictions, scope, model, league)
            scope_values[scope].append(float(data["model_loss"].mean()))
        scope_values[scope].append(float(np.mean(scope_values[scope])))

    fig, axis = plt.subplots(figsize=(9.5, 6), constrained_layout=True)
    y = np.arange(len(labels))
    for row in range(len(labels)):
        axis.plot(
            [scope_values["pooled"][row], scope_values["league_specific"][row]],
            [row, row],
            color="#aaaaaa",
            linewidth=1.5,
            zorder=1,
        )
    axis.scatter(scope_values["pooled"], y, color=COLORS["pooled"], s=48, label=SCOPE_LABELS["pooled"], zorder=2)
    axis.scatter(scope_values["league_specific"], y, color=COLORS["league_specific"], s=48, marker="s", label=SCOPE_LABELS["league_specific"], zorder=2)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Log loss — lower is better")
    axis.set_title("Shared versus separately trained market-plus-player models")
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    axis.legend(frameon=False)
    return save_figure(fig, output_dir, "05_pooled_vs_league_specific")


def reliability_table(frame: pd.DataFrame, outcome: str, bins: int = 8) -> pd.DataFrame:
    probability = frame[f"prob_{outcome}"].astype(float)
    observed = frame["result_3way"].eq(outcome).astype(float)
    bin_count = min(bins, max(2, len(frame) // 75))
    labels = pd.qcut(probability.rank(method="first"), q=bin_count, labels=False)
    table = pd.DataFrame({"probability": probability, "observed": observed, "bin": labels})
    return table.groupby("bin", as_index=False).agg(
        predicted=("probability", "mean"),
        actual=("observed", "mean"),
        matches=("observed", "size"),
    )


def reliability_plot(
    predictions: pd.DataFrame,
    output_dir: Path,
    model: str,
) -> Path:
    series = {
        "Closing market": predictions[
            predictions["training_scope"].eq("pooled")
            & predictions["model"].eq("closing_market")
        ],
        "Shared market + players": predictions[
            predictions["training_scope"].eq("pooled")
            & predictions["model"].eq(model)
        ],
        "Separate market + players": predictions[
            predictions["training_scope"].eq("league_specific")
            & predictions["model"].eq(model)
        ],
    }
    if any(frame.empty for frame in series.values()):
        missing = [label for label, frame in series.items() if frame.empty]
        raise ValueError(f"Reliability plot is missing predictions for: {missing}")
    series_colors = {
        "Closing market": COLORS["market"],
        "Shared market + players": COLORS["pooled"],
        "Separate market + players": COLORS["league_specific"],
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True, constrained_layout=True)
    for axis, outcome, title in zip(axes, ("H", "D", "A"), ("Home win", "Draw", "Away win"), strict=True):
        axis.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
        for label, frame in series.items():
            table = reliability_table(frame, outcome)
            axis.plot(
                table["predicted"],
                table["actual"],
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=series_colors[label],
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("Mean predicted probability")
        axis.set_xlim(0, 0.8)
        axis.set_ylim(0, 0.8)
        axis.grid(color=COLORS["grid"], linewidth=0.7)
    axes[0].set_ylabel("Observed frequency")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Reliability of development-period probabilities", fontsize=15)
    return save_figure(fig, output_dir, "06_probability_reliability")


def build_all_visuals(
    evaluation_dir: Path,
    validation_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    primary_model: str,
    bootstrap_repetitions: int,
) -> list[Path]:
    del processed_dir  # Reserved for future player-history panels; no hidden inputs.
    coverage = load_coverage(validation_dir)
    validation = load_match_validation(validation_dir)
    predictions = load_predictions(evaluation_dir)
    generated: list[Path] = []
    for path in (
        coverage_heatmap(coverage, output_dir),
        sample_construction_flow(validation, output_dir),
        experiment_timeline(output_dir),
        market_comparison_plot(predictions, output_dir, primary_model, bootstrap_repetitions),
        pooled_vs_separate_plot(predictions, output_dir, primary_model),
        reliability_plot(predictions, output_dir, primary_model),
    ):
        generated.append(path)
    return generated


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    generated = build_all_visuals(
        resolve(args.evaluation_dir),
        resolve(args.validation_dir),
        resolve(args.processed_dir),
        resolve(args.output_dir),
        args.primary_model,
        args.bootstrap_repetitions,
    )
    print("Generated development-analysis visualizations:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
