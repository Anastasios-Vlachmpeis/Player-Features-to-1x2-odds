"""Generate publication figures that are not produced by the analysis notebooks.

The script reads only saved evaluation and feature-selection tables. It does not
fit models, alter predictions, or rerun any experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "paper" / "figures"
FINAL_TABLE_DIR = PROJECT_ROOT / "scotland_research" / "visuals" / "final_report" / "tables"
POOLED_SELECTION_DIR = (
    PROJECT_ROOT / "scotland_research" / "visuals" / "pooled_feature_selection" / "tables"
)
DIXON_COLES_SELECTION_DIR = (
    PROJECT_ROOT
    / "scotland_research"
    / "visuals"
    / "dixon_coles_feature_selection"
    / "tables"
)
ROBUSTNESS_TABLE_DIR = (
    PROJECT_ROOT / "scotland_research" / "visuals" / "robustness_checks" / "tables"
)


MODEL_LABELS = {
    "frequency_baseline": "Historical frequency",
    "closing_market": "Raw closing market",
    "recalibrated_market": "Recalibrated market",
    "player_form": "Non-market logistic",
    "market_plus_player_form": "Enhanced market model",
    "player_form_lightgbm": "Player-feature LightGBM",
    "dixon_coles": "Dixon--Coles",
    "dixon_coles_player_form": "Dixon--Coles with players",
}

SELECTION_LABELS = {
    "player_form_logistic": "Non-market logistic",
    "market_plus_player_form": "Market + player logistic",
    "player_form_lightgbm": "Player-feature LightGBM",
    "expanded_player_form_lightgbm": "Expanded LightGBM",
    "dixon_coles_player_form": "Dixon--Coles + players",
}

GROUP_LABELS = {
    "legacy_player_form": "Recent player output",
    "team_strength": "Team strength",
    "opponent_adjusted_form": "Opponent-adjusted output",
    "lineup_continuity": "Line-up continuity",
    "position_and_distribution": "Position and distribution",
    "recency": "Alternative recent windows",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    fig.savefig(paths[0], bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def add_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    colour: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.0,
        edgecolor="black",
        facecolor=colour,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center")


def add_arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.1,
            color="black",
        )
    )


def prediction_pipeline(output_dir: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(11.5, 3.8), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    boxes = [
        (0.01, 0.13, "Earlier matches", "lightgray"),
        (0.16, 0.16, "Chronological player\nand team histories", "lightblue"),
        (0.35, 0.14, "Identified starting\nline-ups", "lightblue"),
        (0.52, 0.16, "Line-up features and\nclosing probabilities", "lightgreen"),
    ]
    height, y = 0.28, 0.56
    for x, width, label, colour in boxes:
        add_box(axis, x, y, width, height, label, colour)
    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        add_arrow(axis, (left[0] + left[1], y + height / 2), (right[0], y + height / 2))

    add_box(axis, 0.72, 0.68, 0.13, 0.20, "Recalibrated\nmarket model", "wheat")
    add_box(axis, 0.72, 0.38, 0.13, 0.20, "Enhanced market\nmodel", "lightgreen")
    add_arrow(axis, (0.68, y + height / 2), (0.72, 0.78))
    add_arrow(axis, (0.68, y + height / 2), (0.72, 0.48))
    add_box(axis, 0.89, 0.53, 0.10, 0.20, "1X2\nprobabilities", "lightskyblue")
    add_arrow(axis, (0.85, 0.78), (0.89, 0.66))
    add_arrow(axis, (0.85, 0.48), (0.89, 0.60))

    axis.text(
        0.02,
        0.18,
        "Every feature for a match was calculated from information available before that match.",
        ha="left",
        va="center",
        fontsize=11,
    )
    axis.set_title("Chronological prediction pipeline", pad=8)
    return save_figure(fig, output_dir, "prediction_pipeline")


def comparison_ladder(output_dir: Path) -> list[Path]:
    stages = [
        ("Historical frequency", "Does the method beat a basic prior?", "lightgray"),
        ("Dixon--Coles", "Does team-result history provide useful structure?", "lightblue"),
        ("Dixon--Coles + players", "Do player records help a team-strength model?", "lightgreen"),
        ("Raw closing market", "What did the market imply at closing?", "wheat"),
        ("Recalibrated market", "Can systematic market bias be corrected?", "khaki"),
        ("Recalibrated market + players", "Do the added records improve the same market model?", "lightgreen"),
        ("Flexible player models", "Can nonlinear models extract additional value?", "lightskyblue"),
    ]
    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, len(stages) + 0.3)
    axis.axis("off")
    for index, (model, question, colour) in enumerate(stages):
        y = len(stages) - index - 0.55
        add_box(axis, 0.06, y - 0.28, 0.33, 0.55, model, colour)
        axis.text(0.45, y, question, ha="left", va="center")
        if index < len(stages) - 1:
            add_arrow(axis, (0.225, y - 0.28), (0.225, y - 0.72))
    axis.set_title("Comparison ladder and the question answered at each step", pad=8)
    return save_figure(fig, output_dir, "comparison_ladder")


def development_fold_performance(output_dir: Path) -> list[Path]:
    seasons = ["2022/23", "2023/24", "2024/25", "All folds"]
    values = {
        "Raw closing market": [0.9164, 0.9163, 0.9407, 0.9267],
        "Recalibrated market": [0.9151, 0.9107, 0.9403, 0.9248],
        "Enhanced model": [0.9130, 0.9081, 0.9422, 0.9239],
    }
    colours = ["gray", "orange", "blue"]
    markers = ["o", "s", "^"]
    fig, axis = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    x = np.arange(len(seasons))
    for (label, scores), colour, marker in zip(values.items(), colours, markers, strict=True):
        axis.plot(x, scores, marker=marker, linewidth=1.7, color=colour, label=label)
    axis.set_xticks(x, seasons)
    axis.set_ylabel("Equal-league log loss")
    axis.set_title("Development performance changed across prediction seasons")
    axis.grid(axis="y", color="lightgray", linewidth=0.7)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    return save_figure(fig, output_dir, "development_fold_performance")


def feature_selection_overview(output_dir: Path) -> list[Path]:
    pooled = pd.read_csv(POOLED_SELECTION_DIR / "feature_group_combination_summary.csv")
    dixon = pd.read_csv(DIXON_COLES_SELECTION_DIR / "feature_group_combination_summary.csv")
    frames: list[tuple[str, pd.DataFrame, str]] = []
    for model in (
        "player_form_logistic",
        "market_plus_player_form",
        "player_form_lightgbm",
        "expanded_player_form_lightgbm",
    ):
        frames.append((model, pooled.loc[pooled["model"].eq(model)].copy(), "mean_log_loss"))
    frames.append(("dixon_coles_player_form", dixon.copy(), "equal_league_log_loss"))

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), constrained_layout=True)
    for axis, (model, frame, score_column) in zip(axes.flat, frames, strict=False):
        scores = frame[score_column].astype(float)
        gaps = 1000 * (scores - scores.min())
        x = frame["group_count"].astype(int)
        axis.scatter(x, gaps, s=28, color="blue", alpha=0.62, edgecolor="black", linewidth=0.35)
        best_by_count = pd.DataFrame({"groups": x, "gap": gaps}).groupby("groups", as_index=False)["gap"].min()
        axis.plot(best_by_count["groups"], best_by_count["gap"], color="red", marker="o", linewidth=1.2)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(SELECTION_LABELS[model])
        axis.set_xlabel("Feature groups included")
        axis.set_ylabel("Log-loss gap from family best (x1000)")
        axis.grid(color="lightgray", linewidth=0.6)
    axes.flat[-1].axis("off")
    fig.suptitle("Exhaustive feature-group searches during development", fontsize=14)
    return save_figure(fig, output_dir, "feature_selection_overview")


def feature_group_search_details(output_dir: Path) -> list[Path]:
    pooled = pd.read_csv(POOLED_SELECTION_DIR / "feature_group_combination_summary.csv")
    dixon = pd.read_csv(DIXON_COLES_SELECTION_DIR / "feature_group_combination_summary.csv")
    frames: list[tuple[str, pd.DataFrame, str]] = []
    for model in (
        "player_form_logistic",
        "market_plus_player_form",
        "player_form_lightgbm",
        "expanded_player_form_lightgbm",
    ):
        frames.append((model, pooled.loc[pooled["model"].eq(model)].copy(), "mean_log_loss"))
    frames.append(("dixon_coles_player_form", dixon.copy(), "equal_league_log_loss"))

    generated: list[Path] = []
    group_order = list(GROUP_LABELS)
    for model, frame, score_column in frames:
        frame = frame.sort_values(score_column).reset_index(drop=True)
        frame["gap"] = 1000 * (frame[score_column].astype(float) - frame[score_column].astype(float).min())
        y = np.arange(len(frame))
        fig, (matrix_axis, score_axis) = plt.subplots(
            1,
            2,
            figsize=(13.0, 7.5),
            gridspec_kw={"width_ratios": [1.65, 1]},
            constrained_layout=True,
        )
        selected_sets = [set(str(value).split("|")) for value in frame["selected_groups"]]
        for row_index, selected in enumerate(selected_sets):
            for group_index, group in enumerate(group_order):
                matrix_axis.scatter(
                    group_index,
                    row_index,
                    s=25 if group in selected else 12,
                    color="black" if group in selected else "lightgray",
                )
        matrix_axis.set_xticks(
            np.arange(len(group_order)),
            [GROUP_LABELS[group] for group in group_order],
            rotation=35,
            ha="right",
        )
        matrix_axis.set_yticks(y[::5], [f"#{index + 1}" for index in y[::5]])
        matrix_axis.set_ylabel("Combination rank (lower log loss is better)")
        matrix_axis.set_title("Included feature groups")
        matrix_axis.invert_yaxis()
        matrix_axis.grid(axis="y", color="lightgray", linewidth=0.5)

        colours = ["orange" if value <= 5 else "blue" for value in frame["gap"]]
        score_axis.scatter(frame["gap"], y, color=colours, s=28)
        score_axis.axvline(0, color="black", linewidth=0.8)
        score_axis.axvspan(0, 5, color="orange", alpha=0.10)
        score_axis.set_xlabel("Log-loss gap from best combination (x1000)")
        score_axis.set_yticks([])
        score_axis.set_title("Development performance")
        score_axis.invert_yaxis()
        score_axis.grid(axis="x", color="lightgray", linewidth=0.6)
        fig.suptitle(f"Exhaustive group search: {SELECTION_LABELS[model]}", fontsize=14)
        generated.extend(save_figure(fig, output_dir, f"feature_group_search_{model}"))
    return generated


def robustness_development(output_dir: Path) -> list[Path]:
    frame = pd.read_csv(ROBUSTNESS_TABLE_DIR / "robustness_summary.csv")
    keep = [
        "without_player_ratings",
        "market_plus_player_form",
        "without_attacking_output",
        "without_team_strength_context",
        "without_defensive_output",
        "recalibrated_market",
        "shuffled_player_features",
        "closing_market",
    ]
    labels = {
        "without_player_ratings": "Remove mean player rating",
        "market_plus_player_form": "Five-feature precursor",
        "without_attacking_output": "Remove attacking output",
        "without_team_strength_context": "Remove team-strength control",
        "without_defensive_output": "Remove defensive output",
        "recalibrated_market": "Recalibrated market only",
        "shuffled_player_features": "Shuffle players within league-season",
        "closing_market": "Raw closing market",
    }
    frame = frame.set_index("model").loc[keep].reset_index()
    changes = 1000 * frame["log_loss_change_vs_final"].astype(float).to_numpy()
    y = np.arange(len(frame))
    colours = ["green" if value < 0 else "gray" if value == 0 else "red" for value in changes]
    fig, axis = plt.subplots(figsize=(9.0, 5.1), constrained_layout=True)
    axis.barh(y, changes, color=colours)
    axis.axvline(0, color="black", linewidth=1.0)
    axis.set_yticks(y, [labels[model] for model in frame["model"]])
    axis.invert_yaxis()
    axis.set_xlabel("Log-loss change relative to the five-feature precursor (x1000)")
    axis.set_title("Development robustness and refinement checks")
    axis.grid(axis="x", color="lightgray", linewidth=0.7)
    return save_figure(fig, output_dir, "robustness_development")


def final_model_ranking(output_dir: Path) -> list[Path]:
    results = pd.read_csv(FINAL_TABLE_DIR / "all_model_results.csv")
    results = results.loc[results["aggregation"].eq("equal_league")].copy()
    requested = [
        ("league_specific", "market_plus_player_form"),
        ("league_specific", "recalibrated_market"),
        ("pooled", "recalibrated_market"),
        ("pooled", "market_plus_player_form"),
        ("league_specific", "closing_market"),
        ("pooled", "player_form_lightgbm"),
        ("pooled", "player_form"),
        ("league_specific", "dixon_coles"),
        ("league_specific", "dixon_coles_player_form"),
        ("league_specific", "frequency_baseline"),
    ]
    rows = []
    for scope, model in requested:
        selected = results.loc[results["training_scope"].eq(scope) & results["model"].eq(model)]
        if len(selected) != 1:
            raise ValueError(f"Expected one result for {scope}/{model}; found {len(selected)}")
        row = selected.iloc[0]
        scope_label = "Separate" if scope == "league_specific" else "Shared"
        label = MODEL_LABELS[model]
        if model not in {"closing_market", "frequency_baseline"}:
            label = f"{label} ({scope_label.lower()})"
        rows.append((label, float(row["log_loss"]), model))
    rows.sort(key=lambda item: item[1])

    labels = [row[0] for row in rows]
    scores = np.asarray([row[1] for row in rows])
    colours = ["green" if row[2] == "market_plus_player_form" else "orange" if row[2] == "recalibrated_market" else "blue" for row in rows]
    y = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    axis.scatter(scores, y, color=colours, s=55, zorder=2)
    axis.hlines(y, scores.min() - 0.004, scores, color="lightgray", linewidth=1.2, zorder=1)
    for score, y_value in zip(scores, y, strict=True):
        axis.text(score + 0.002, y_value, f"{score:.4f}", va="center", fontsize=9)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Equal-league log loss (lower is better)")
    axis.set_title("Final 2025/26 comparison across model families")
    axis.grid(axis="x", color="lightgray", linewidth=0.7)
    return save_figure(fig, output_dir, "final_model_ranking")


def final_uncertainty_intervals(output_dir: Path) -> list[Path]:
    frame = pd.read_csv(FINAL_TABLE_DIR / "uncertainty_intervals.csv")
    metric_labels = {"log_loss": "Log loss", "brier_score": "Brier score", "rps": "RPS"}
    weighting_labels = {"equal_league": "Equal league", "match_weighted": "Match weighted"}
    frame["label"] = frame.apply(
        lambda row: f"{weighting_labels[row['weighting']]} -- {metric_labels[row['metric']]}", axis=1
    )
    frame["weight_order"] = frame["weighting"].map({"equal_league": 0, "match_weighted": 1})
    frame["metric_order"] = frame["metric"].map({"log_loss": 0, "brier_score": 1, "rps": 2})
    frame = frame.sort_values(["weight_order", "metric_order"]).reset_index(drop=True)
    y = np.arange(len(frame))
    values = frame["observed_relative_improvement_pct"].to_numpy(float)
    lower = frame["lower_95_relative_pct"].to_numpy(float)
    upper = frame["upper_95_relative_pct"].to_numpy(float)

    fig, axis = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    axis.errorbar(
        values,
        y,
        xerr=np.vstack([values - lower, upper - values]),
        fmt="o",
        color="red",
        ecolor="red",
        capsize=3,
    )
    axis.axvline(0, color="black", linewidth=1.0)
    axis.set_yticks(y, frame["label"])
    axis.invert_yaxis()
    axis.set_xlabel("Relative improvement from the enhanced model (%)")
    axis.set_title("Final 2025/26 enhanced model versus recalibrated market")
    axis.grid(axis="x", color="lightgray", linewidth=0.7)
    return save_figure(fig, output_dir, "final_uncertainty_intervals")


def final_country_forest(output_dir: Path) -> list[Path]:
    frame = pd.read_csv(FINAL_TABLE_DIR / "uncertainty_by_league.csv")
    frame = frame.loc[frame["metric"].eq("log_loss")].copy()
    order = ["belgium", "netherlands", "portugal", "scotland", "turkey"]
    frame["league"] = pd.Categorical(frame["league"], categories=order, ordered=True)
    frame = frame.sort_values("league").reset_index(drop=True)
    overall = pd.read_csv(FINAL_TABLE_DIR / "uncertainty_intervals.csv")
    overall = overall.loc[
        overall["weighting"].eq("equal_league") & overall["metric"].eq("log_loss")
    ].iloc[0]

    labels = [str(value).title() for value in frame["league"].astype(str)] + ["Overall"]
    values = frame["observed_relative_improvement_pct"].to_list() + [float(overall["observed_relative_improvement_pct"])]
    lower = frame["lower_95_relative_pct"].to_list() + [float(overall["lower_95_relative_pct"])]
    upper = frame["upper_95_relative_pct"].to_list() + [float(overall["upper_95_relative_pct"])]
    y = np.arange(len(labels))
    colours = ["black"] * len(frame) + ["blue"]

    fig, axis = plt.subplots(figsize=(8.6, 5.7), constrained_layout=True)
    for index, (value, low, high, colour) in enumerate(zip(values, lower, upper, colours, strict=True)):
        axis.errorbar(value, index, xerr=[[value - low], [high - value]], fmt="o", color=colour, capsize=3)
    axis.axvline(0, color="black", linewidth=1.0)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Relative log-loss improvement from the enhanced model (%)")
    axis.set_title("Final 2025/26 result by country")
    axis.grid(axis="x", color="lightgray", linewidth=0.7)
    return save_figure(fig, output_dir, "final_country_forest")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    configure_style()
    generated: list[Path] = []
    for builder in (
        prediction_pipeline,
        comparison_ladder,
        development_fold_performance,
        feature_selection_overview,
        feature_group_search_details,
        robustness_development,
        final_model_ranking,
        final_uncertainty_intervals,
        final_country_forest,
    ):
        generated.extend(builder(output_dir))
    print("Generated publication figures:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
