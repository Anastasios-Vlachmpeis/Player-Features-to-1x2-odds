"""Build the frozen five-league sample coverage and exclusion audit tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_match_dataset import MATCH_DATASET_NAME, write_csv_atomic
from constants import DEVELOPMENT_EXCLUDED_LEAGUES, EXPECTED_LEAGUES
from match_rules import VALID_COMPETITION_PHASES, add_exclusion_reasons
from league_config import DEVELOPMENT_SEASONS, LEAGUES, PROJECT_ROOT
from validate_dataset import build_match_validation, load_inputs


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "data_quality"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
OUTCOMES = ("H", "D", "A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def outcome_rates(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    valid = frame[frame["result_3way"].isin(OUTCOMES)]
    return {
        f"{prefix}_{label}_rate": safe_rate(valid["result_3way"].eq(outcome).sum(), len(valid))
        for label, outcome in zip(("home", "draw", "away"), OUTCOMES)
    }


def coverage_row(group: pd.DataFrame) -> dict[str, object]:
    target = group[group["football_data_match"]]
    included = target[target["model_ready"]]
    return {
        "target_matches": len(target),
        "model_ready_matches": len(included),
        "excluded_matches": len(target) - len(included),
        "retention_rate": safe_rate(len(included), len(target)),
        **outcome_rates(target, "target"),
        **outcome_rates(included, "included"),
    }


def grouped_coverage(validation: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for values, group in validation.groupby(keys, sort=True, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        rows.append({**dict(zip(keys, values)), **coverage_row(group)})
    return pd.DataFrame(rows)


def add_market_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    odds = output[["home_odds", "draw_odds", "away_odds"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = odds.notna().all(axis=1) & odds.gt(1.0).all(axis=1)
    output = output.loc[valid].copy()
    inverse = 1.0 / odds.loc[valid]
    probabilities = inverse.div(inverse.sum(axis=1), axis=0)
    output[["p_home", "p_draw", "p_away"]] = probabilities.to_numpy()
    return output


def market_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    probabilities = group[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    outcome_index = group["result_3way"].map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=int)
    observed = np.eye(3)[outcome_index]
    chosen = probabilities[np.arange(len(group)), outcome_index]
    return {
        "matches": len(group),
        "closing_market_log_loss": float(-np.log(np.clip(chosen, 1e-15, 1.0)).mean()),
        "closing_market_brier": float(np.square(probabilities - observed).sum(axis=1).mean()),
        "average_market_confidence": float(probabilities.max(axis=1).mean()),
        **outcome_rates(group, "outcome"),
    }


def included_vs_excluded_market(validation: pd.DataFrame) -> pd.DataFrame:
    comparable = validation[
        validation["football_data_match"]
        & validation["score_matches_football_data"]
        & validation["closing_odds_available"]
        & validation["result_3way"].isin(OUTCOMES)
    ].copy()
    comparable = add_market_probabilities(comparable)
    comparable["sample"] = np.where(comparable["model_ready"], "included", "excluded")

    rows: list[dict[str, object]] = []
    keys = ["league", "season", "sample"]
    for values, group in comparable.groupby(keys, sort=True):
        rows.append({**dict(zip(keys, values)), **market_metrics(group)})
    return pd.DataFrame(rows)


def validate_processed_membership(validation: pd.DataFrame) -> None:
    for league in sorted(EXPECTED_LEAGUES):
        path = PROCESSED_ROOT / league / MATCH_DATASET_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"Build the per-league match dataset before the data-quality report: {path}"
            )
        processed = pd.read_csv(path, dtype={"match_id": "string"})
        expected = set(
            validation.loc[
                validation["league"].eq(league) & validation["model_ready"], "match_id"
            ].astype(str)
        )
        observed = set(processed["match_id"].astype(str))
        if observed != expected:
            raise ValueError(
                f"{league} processed membership disagrees with the match inclusion rules: "
                f"missing={len(expected - observed)}, unexpected={len(observed - expected)}"
            )


def build_report(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    if DEVELOPMENT_EXCLUDED_LEAGUES != frozenset({"greece"}):
        raise ValueError("Development exclusions changed; review and refreeze the match inclusion rules")

    frames: list[pd.DataFrame] = []
    for league in sorted(EXPECTED_LEAGUES):
        config = LEAGUES[league]
        matches, players, football_data = load_inputs(config, DEVELOPMENT_SEASONS)
        validation = build_match_validation(matches, players, football_data, config=config)
        validation = add_exclusion_reasons(validation)
        validation.insert(0, "league", league)
        frames.append(validation)

    combined = pd.concat(frames, ignore_index=True)
    if not set(combined["competition_phase"]).issubset(VALID_COMPETITION_PHASES):
        raise ValueError("The report contains an undeclared competition phase")
    combined_ids = combined["league"] + ":" + combined["match_id"].astype(str)
    if combined_ids.duplicated().any():
        raise ValueError("League-qualified match identifiers are not unique")
    validate_processed_membership(combined)

    coverage = grouped_coverage(combined, ["league", "season"])
    phase_coverage = grouped_coverage(
        combined[combined["football_data_match"]],
        ["league", "season", "competition_phase"],
    )
    exclusions = (
        combined[combined["football_data_match"] & ~combined["model_ready"]]
        .groupby(
            ["league", "season", "competition_phase", "exclusion_reason"],
            sort=True,
            dropna=False,
        )
        .size()
        .rename("matches")
        .reset_index()
    )
    market = included_vs_excluded_market(combined)

    resolved = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    paths = {
        "coverage": resolved / "coverage_by_league_season.csv",
        "exclusions": resolved / "exclusion_reasons.csv",
        "market_comparison": resolved / "included_vs_excluded_market.csv",
        "phase_coverage": resolved / "competition_phase_coverage.csv",
    }
    for frame, path in zip((coverage, exclusions, market, phase_coverage), paths.values()):
        write_csv_atomic(frame, path)
    return paths


def main() -> None:
    paths = build_report(parse_args().output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
