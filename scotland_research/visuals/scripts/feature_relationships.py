"""Render feature-family relationships as PNGs without model evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "scotland" / "scotland_model_dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "scotland_research" / "visuals" / "out"
BACKGROUND = "#f7f9fb"
INK = "#183153"
POINT = "#2a6f97"
FIT = "#d1495b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def finite_limits(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    lower, upper = numeric.quantile([0.01, 0.99])
    if lower == upper:
        lower, upper = float(numeric.min()) - 1.0, float(numeric.max()) + 1.0
    return float(lower), float(upper)


def scale(value: float, lower: float, upper: float, start: int, end: int) -> int:
    clipped = min(max(value, lower), upper)
    return int(start + (clipped - lower) * (end - start) / (upper - lower))


def vertical_label(text: str, text_font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> Image.Image:
    label = Image.new("RGBA", (260, 30), (0, 0, 0, 0))
    ImageDraw.Draw(label).text((0, 5), text, fill=INK, font=text_font)
    return label.rotate(90, expand=True)


def draw_scatter(image: Image.Image, draw: ImageDraw.ImageDraw, frame: pd.DataFrame, bounds: tuple[int, int, int, int], x_column: str, y_column: str, title: str, x_label: str, y_label: str) -> None:
    left, top, right, bottom = bounds
    plot_left, plot_top, plot_right, plot_bottom = left + 75, top + 48, right - 18, bottom - 62
    draw.rectangle((left, top, right, bottom), fill="white", outline="#ccd6e0", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=INK, width=2)
    draw.text((left + 16, top + 12), title, fill=INK, font=font(19))
    draw.text((plot_left, bottom - 42), x_label, fill=INK, font=font(14))
    y_axis_label = vertical_label(y_label, font(13))
    image.paste(y_axis_label, (left + 12, plot_top + (plot_bottom - plot_top - y_axis_label.height) // 2), y_axis_label)

    x_low, x_high = finite_limits(frame[x_column])
    y_low, y_high = finite_limits(frame[y_column])
    valid = frame[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()
    for x_value, y_value in valid.itertuples(index=False):
        x = scale(float(x_value), x_low, x_high, plot_left, plot_right)
        y = scale(float(y_value), y_low, y_high, plot_bottom, plot_top)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=POINT)

    if len(valid) >= 2 and valid[x_column].nunique() > 1:
        slope, intercept = np.polyfit(valid[x_column], valid[y_column], 1)
        y_start = slope * x_low + intercept
        y_end = slope * x_high + intercept
        draw.line((plot_left, scale(float(y_start), y_low, y_high, plot_bottom, plot_top), plot_right, scale(float(y_end), y_low, y_high, plot_bottom, plot_top)), fill=FIT, width=3)
    draw.text((plot_left, plot_bottom + 5), f"{x_low:.2f}", fill="#526777", font=font(12))
    draw.text((plot_right - 45, plot_bottom + 5), f"{x_high:.2f}", fill="#526777", font=font(12))


def build_relationship_plot(frame: pd.DataFrame, output_dir: Path) -> Path:
    plot_frame = frame.copy()
    plot_frame["mean_retained"] = (plot_frame["home_retained_starters"] + plot_frame["away_retained_starters"]) / 2.0
    plot_frame["mean_pair_minutes"] = (plot_frame["home_mean_pairwise_prior_minutes"] + plot_frame["away_mean_pairwise_prior_minutes"]) / 2.0
    image = Image.new("RGB", (1500, 1080), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((45, 24), "Relationships among engineered Scotland feature sets", fill=INK, font=font(28))
    panels = [(40, 85, 735, 555), (765, 85, 1460, 555), (40, 585, 735, 1050), (765, 585, 1460, 1050)]
    specifications = [
        ("diff_elo_rating", "diff_expected_goals_strength", "Team strength components", "Home-away Elo", "Strength xG difference"),
        ("mean_retained", "mean_pair_minutes", "Continuity and familiarity", "Mean retained starters", "Prior pair minutes"),
        ("diff_shots_per90_sum_5", "diff_adjusted_shots_lineup_mean_5", "Raw and opponent-adjusted shooting", "Raw shot-rate difference", "Adjusted-shot residual"),
        ("diff_rating_lineup_std_across_starters_5", "diff_replacement_quality", "Distribution and replacement quality", "Rating spread difference", "Replacement quality"),
    ]
    for bounds, specification in zip(panels, specifications, strict=True):
        draw_scatter(image, draw, plot_frame, bounds, *specification)
    output_path = output_dir / "feature_set_relationships.png"
    image.save(output_path)
    return output_path


def correlation_color(value: float) -> tuple[int, int, int]:
    intensity = min(abs(value), 1.0)
    target = np.array((196, 54, 74) if value >= 0 else (42, 111, 151))
    return tuple((np.array((255, 255, 255)) * (1.0 - intensity) + target * intensity).astype(int))


def build_correlation_plot(frame: pd.DataFrame, output_dir: Path) -> Path:
    columns = {
        "diff_elo_rating": "Elo",
        "diff_expected_goals_strength": "Strength xG",
        "diff_retained_starters": "Retained XI",
        "diff_mean_pairwise_prior_minutes": "Pair minutes",
        "diff_replacement_quality": "Replacement",
        "diff_shots_per90_sum_5": "Raw shots",
        "diff_adjusted_shots_lineup_mean_5": "Adj shots",
        "diff_adjusted_key_passes_lineup_mean_5": "Adj key passes",
        "diff_fwd_adjusted_shots_5": "Forward shots",
        "diff_def_adjusted_defensive_actions_5": "Defence actions",
        "diff_rating_lineup_std_across_starters_5": "Rating spread",
        "diff_rating_lineup_mean_trend_1_5": "Rating trend",
    }
    correlation = frame[list(columns)].corr().rename(index=columns, columns=columns)
    cell = 64
    left, top = 230, 235
    size = len(correlation) * cell
    image = Image.new("RGB", (left + size + 70, top + size + 70), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((35, 25), "Correlation across engineered feature families", fill=INK, font=font(27))
    for row, row_name in enumerate(correlation.index):
        draw.text((20, top + row * cell + 22), row_name, fill=INK, font=font(14))
        for column, column_name in enumerate(correlation.columns):
            value = float(correlation.iat[row, column])
            x0, y0 = left + column * cell, top + row * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=correlation_color(value), outline="white")
            text_color = "white" if abs(value) > 0.58 else INK
            draw.text((x0 + 13, y0 + 22), f"{value:.2f}", fill=text_color, font=font(13))
    for column, name in enumerate(correlation.columns):
        label = Image.new("RGBA", (170, 28), (0, 0, 0, 0))
        ImageDraw.Draw(label).text((0, 4), name, fill=INK, font=font(13))
        rotated = label.rotate(90, expand=True)
        label_x = left + column * cell + (cell - rotated.width) // 2
        image.paste(rotated, (label_x, top - rotated.height - 10), rotated)
    output_path = output_dir / "feature_set_correlations.png"
    image.save(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.dataset)
    print(f"Saved feature relationship plot to {build_relationship_plot(frame, args.output_dir)}")
    print(f"Saved feature correlation plot to {build_correlation_plot(frame, args.output_dir)}")


if __name__ == "__main__":
    main()
