"""Run the one-time, fixed 2025-26 final evaluation.

This command is intentionally separate from development evaluation. It trains
on 2020-21 through 2024-25, predicts 2025-26 once, and refuses to overwrite an
existing output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from inspect import signature
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from constants import CLASS_ORDER, DEFAULT_MODEL_DATASET, EXPECTED_LEAGUES, PROJECT_ROOT
from data.load_model_dataset import load_dataset
from evaluation.metrics import probability_frame
from evaluation.multi_league import (
    LEAGUE_SPECIFIC_SCOPE,
    POOLED_SCOPE,
    add_league_effects,
    equal_league_training_weights,
    training_weight_summary,
)
from evaluation.report import write_csv_atomic
from league_config import DEVELOPMENT_SEASONS, FINAL_SEASON
from models.base import MatchPredictor
from models.player_form_lightgbm import LIGHTGBM_SETTINGS
from models.publication_suite import (
    COMMON_MODEL_NAMES,
    PUBLICATION_MODEL_NAMES,
    PredictorFactory,
    league_specific_model_factories,
    pooled_model_factories,
)
from selected_features import (
    SELECTED_FEATURES_PATH,
    SelectedFeatures,
    load_selected_features,
    write_frozen_run_configuration,
)


DEFAULT_SETTINGS_PATH = (
    PROJECT_ROOT / "multi_league_research" / "config" / "final_evaluation.json"
)
DEFAULT_FINAL_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "all_leagues"
    / "final_2025_26_model_dataset.csv"
)
REQUIRED_POST_HOC_CALIBRATION = "none"
REQUIRED_PRIMARY_METRIC = "log_loss"
REQUIRED_SECONDARY_METRICS = ("brier_score", "rps")
REQUIRED_PRIMARY_AGGREGATION = "equal_league"
REQUIRED_SECONDARY_AGGREGATIONS = ("match_weighted", "individual_league")
REQUIRED_DEFAULT_OUTPUT_DIRECTORY = "artifacts/final_2025_26_evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-dataset",
        type=Path,
        default=DEFAULT_MODEL_DATASET,
        help=f"Five-season training dataset (default: {DEFAULT_MODEL_DATASET})",
    )
    parser.add_argument(
        "--final-dataset",
        type=Path,
        default=DEFAULT_FINAL_DATASET,
        help=f"Held-out 2025-26 dataset (default: {DEFAULT_FINAL_DATASET})",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
        help=f"Fixed final-evaluation settings (default: {DEFAULT_SETTINGS_PATH})",
    )
    parser.add_argument(
        "--selected-features",
        type=Path,
        default=SELECTED_FEATURES_PATH,
        help=f"Fixed selected features (default: {SELECTED_FEATURES_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to the directory fixed in the settings file.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _require_equal(label: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise ValueError(f"Final settings changed for {label}: expected={expected}, observed={observed}")


def load_and_validate_settings(
    path: Path,
    selected: SelectedFeatures,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Final evaluation settings do not exist: {path}")
    settings = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Final evaluation settings must be a JSON object")

    _require_equal(
        "specification status",
        settings.get("specification_status"),
        "fixed_for_final_evaluation",
    )
    _require_equal(
        "training seasons",
        tuple(settings.get("training_seasons", ())),
        tuple(DEVELOPMENT_SEASONS),
    )
    _require_equal("test season", settings.get("test_season"), FINAL_SEASON)
    _require_equal(
        "included leagues",
        tuple(settings.get("included_leagues", ())),
        tuple(sorted(EXPECTED_LEAGUES)),
    )
    _require_equal("excluded leagues", tuple(settings.get("excluded_leagues", ())), ("greece",))
    _require_equal(
        "selected features checksum",
        settings.get("selected_features_sha256"),
        selected.semantic_sha256,
    )

    models = settings.get("models")
    if not isinstance(models, dict):
        raise ValueError("Final settings must contain a models object")
    _require_equal("pooled model list", tuple(models.get("pooled", ())), COMMON_MODEL_NAMES)
    _require_equal(
        "league-specific model list",
        tuple(models.get("league_specific", ())),
        PUBLICATION_MODEL_NAMES,
    )
    _require_equal(
        "primary comparison",
        settings.get("primary_comparison"),
        {
            "training_scope": POOLED_SCOPE,
            "baseline_model": "recalibrated_market",
            "player_model": "market_plus_player_form",
        },
    )
    _require_equal(
        "post-hoc calibration",
        settings.get("post_hoc_calibration"),
        REQUIRED_POST_HOC_CALIBRATION,
    )

    metrics = settings.get("metrics", {})
    _require_equal("primary metric", metrics.get("primary"), REQUIRED_PRIMARY_METRIC)
    _require_equal(
        "secondary metrics",
        tuple(metrics.get("secondary", ())),
        REQUIRED_SECONDARY_METRICS,
    )
    aggregation = settings.get("aggregation", {})
    _require_equal(
        "primary aggregation",
        aggregation.get("primary"),
        REQUIRED_PRIMARY_AGGREGATION,
    )
    _require_equal(
        "secondary aggregations",
        tuple(aggregation.get("secondary", ())),
        REQUIRED_SECONDARY_AGGREGATIONS,
    )

    uncertainty = settings.get("uncertainty", {})
    _require_equal(
        "uncertainty method",
        uncertainty.get("method"),
        "paired_match_week_bootstrap",
    )
    _require_equal(
        "uncertainty resampling groups",
        tuple(uncertainty.get("resample_within", ())),
        ("league", "season"),
    )
    _require_equal("bootstrap repetitions", uncertainty.get("repetitions"), 10_000)
    _require_equal("bootstrap seed", uncertainty.get("seed"), 42)
    random_seeds = settings.get("random_seeds", {})
    _require_equal(
        "model seed",
        random_seeds.get("model"),
        LIGHTGBM_SETTINGS["random_state"],
    )
    _require_equal("recorded bootstrap seed", random_seeds.get("bootstrap"), 42)

    _require_equal(
        "default output directory",
        settings.get("default_output_directory"),
        REQUIRED_DEFAULT_OUTPUT_DIRECTORY,
    )
    return settings


def _validate_season_coverage(
    frame: pd.DataFrame,
    seasons: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    observed_seasons = set(frame["season"].astype(str))
    if observed_seasons != set(seasons):
        raise ValueError(
            f"{label} season mismatch: expected={list(seasons)}, "
            f"observed={sorted(observed_seasons)}"
        )

    selected = frame[frame["league"].isin(EXPECTED_LEAGUES)].copy()
    observed_leagues = set(selected["league"].astype(str))
    if observed_leagues != set(EXPECTED_LEAGUES):
        raise ValueError(
            f"{label} league mismatch: expected={sorted(EXPECTED_LEAGUES)}, "
            f"observed={sorted(observed_leagues)}"
        )
    counts = selected.groupby(["league", "season"], sort=True).size()
    missing = [
        (league, season)
        for league in sorted(EXPECTED_LEAGUES)
        for season in seasons
        if (league, season) not in counts.index
    ]
    if missing:
        raise ValueError(f"{label} has no eligible matches for league-season pairs: {missing}")
    return selected


def load_final_samples(
    training_path: Path,
    final_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training_all = load_dataset(training_path)
    final_all = load_dataset(final_path)
    training = _validate_season_coverage(
        training_all,
        tuple(DEVELOPMENT_SEASONS),
        "Training dataset",
    )
    final = _validate_season_coverage(final_all, (FINAL_SEASON,), "Final dataset")

    if set(training["season"]).intersection(final["season"]):
        raise ValueError("Training and final seasons overlap")
    overlapping_matches = set(training["match_id"]).intersection(final["match_id"])
    if overlapping_matches:
        raise ValueError(
            "Training and final datasets share match IDs: "
            f"{sorted(overlapping_matches)[:5]}"
        )
    if training["_match_datetime"].max() >= final["_match_datetime"].min():
        raise ValueError("Final matches must occur strictly after all training matches")

    coverage = pd.concat(
        [
            training.groupby(["league", "season"], as_index=False)
            .size()
            .rename(columns={"size": "matches"})
            .assign(sample="training"),
            final.groupby(["league", "season"], as_index=False)
            .size()
            .rename(columns={"size": "matches"})
            .assign(sample="final"),
        ],
        ignore_index=True,
    )[["sample", "league", "season", "matches"]]
    return training, final, coverage


def _fit_predictor(
    predictor: MatchPredictor,
    train: pd.DataFrame,
    sample_weight: np.ndarray | None,
) -> None:
    if sample_weight is None:
        predictor.fit(train)
        return
    if "sample_weight" not in signature(predictor.fit).parameters:
        raise TypeError(
            f"{type(predictor).__name__} cannot use equal-league training weights"
        )
    predictor.fit(train, sample_weight=sample_weight)


def _prediction_frame(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    model_name: str,
    scope: str,
) -> pd.DataFrame:
    if probabilities.shape != (len(test), len(CLASS_ORDER)):
        raise ValueError(f"{model_name} returned an invalid probability shape")
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
        raise ValueError(f"{model_name} returned non-positive or non-finite probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError(f"{model_name} probabilities do not sum to one")

    identity_columns = [
        "league",
        "match_id",
        "season",
        "match_date",
        "home_team",
        "away_team",
        "result_3way",
    ]
    output = test[identity_columns].reset_index(drop=True).copy()
    output.insert(0, "model", model_name)
    output.insert(0, "training_scope", scope)
    output = pd.concat(
        [output, probability_frame(probabilities).reset_index(drop=True)],
        axis=1,
    )
    output["predicted_result"] = np.asarray(CLASS_ORDER)[
        np.argmax(probabilities, axis=1)
    ]
    return output


def _coefficient_frame(
    predictor: MatchPredictor,
    scope: str,
    training_league: str,
) -> pd.DataFrame | None:
    coefficients = predictor.export_coefficients(FINAL_SEASON)
    if coefficients is None or coefficients.empty:
        return None
    output = coefficients.copy()
    output.insert(0, "training_league", training_league)
    output.insert(0, "training_scope", scope)
    return output


def run_final_models(
    training: pd.DataFrame,
    final: pd.DataFrame,
    selected: SelectedFeatures,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training, league_columns = add_league_effects(training)
    final, final_league_columns = add_league_effects(final)
    if league_columns != final_league_columns:
        raise ValueError("Training and final league indicators differ")

    pooled_factories = pooled_model_factories(
        league_columns,
        settings["models"]["pooled"],
        selected_features=selected.by_model,
    )
    separate_factories = league_specific_model_factories(
        settings["models"]["league_specific"],
        selected_features=selected.by_model,
    )
    predictions: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []

    weights = equal_league_training_weights(training)
    weight_audit = training_weight_summary(training, weights, FINAL_SEASON)
    for registered_name, factory in pooled_factories.items():
        predictor = factory()
        if predictor.name != registered_name:
            raise ValueError(
                f"Pooled model registry name mismatch: {registered_name} != {predictor.name}"
            )
        _fit_predictor(predictor, training, weights)
        predictions.append(
            _prediction_frame(
                final,
                predictor.predict_proba(final),
                predictor.name,
                POOLED_SCOPE,
            )
        )
        exported = _coefficient_frame(predictor, POOLED_SCOPE, "all_leagues")
        if exported is not None:
            coefficients.append(exported)

    for league in sorted(EXPECTED_LEAGUES):
        league_training = training[training["league"].eq(league)].copy()
        league_final = final[final["league"].eq(league)].copy()
        if set(league_training["result_3way"]) != set(CLASS_ORDER):
            raise ValueError(f"{league} training data lacks an outcome class")
        for registered_name, factory in separate_factories.items():
            predictor = factory()
            if predictor.name != registered_name:
                raise ValueError(
                    "League-specific model registry name mismatch: "
                    f"{registered_name} != {predictor.name}"
                )
            _fit_predictor(predictor, league_training, None)
            predictions.append(
                _prediction_frame(
                    league_final,
                    predictor.predict_proba(league_final),
                    predictor.name,
                    LEAGUE_SPECIFIC_SCOPE,
                )
            )
            exported = _coefficient_frame(
                predictor,
                LEAGUE_SPECIFIC_SCOPE,
                league,
            )
            if exported is not None:
                coefficients.append(exported)

    combined_predictions = pd.concat(predictions, ignore_index=True)
    duplicate_keys = ["training_scope", "model", "league", "match_id"]
    if combined_predictions.duplicated(duplicate_keys).any():
        raise ValueError("A model produced duplicate final predictions")
    expected_rows = len(final) * (len(COMMON_MODEL_NAMES) + len(PUBLICATION_MODEL_NAMES))
    if len(combined_predictions) != expected_rows:
        raise ValueError(
            f"Final prediction count mismatch: expected={expected_rows}, "
            f"observed={len(combined_predictions)}"
        )
    coefficient_table = (
        pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame()
    )
    return combined_predictions, coefficient_table, weight_audit


def _per_match_scores(group: pd.DataFrame) -> dict[str, np.ndarray]:
    probabilities = group[["prob_H", "prob_D", "prob_A"]].to_numpy(dtype=float)
    class_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    indexes = group["result_3way"].map(class_index)
    if indexes.isna().any():
        raise ValueError("Final outcomes contain an unknown class")
    indexes_array = indexes.to_numpy(dtype=int)
    rows = np.arange(len(group))
    observed = np.eye(len(CLASS_ORDER), dtype=float)[indexes_array]
    return {
        "log_loss": -np.log(
            np.clip(probabilities[rows, indexes_array], np.finfo(float).eps, 1.0)
        ),
        "brier_score": np.sum((probabilities - observed) ** 2, axis=1),
        "rps": np.sum(
            (
                np.cumsum(probabilities, axis=1)[:, :-1]
                - np.cumsum(observed, axis=1)[:, :-1]
            )
            ** 2,
            axis=1,
        )
        / (len(CLASS_ORDER) - 1),
    }


def _score_groups(predictions: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(groups, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        scores = _per_match_scores(group)
        rows.append(
            {
                **dict(zip(groups, key_values, strict=True)),
                "matches": len(group),
                **{metric: float(values.mean()) for metric, values in scores.items()},
            }
        )
    return pd.DataFrame(rows)


def build_metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    match_weighted = _score_groups(predictions, ["training_scope", "model"])
    individual_league = _score_groups(
        predictions,
        ["training_scope", "league", "model"],
    )
    equal_league = (
        individual_league.groupby(["training_scope", "model"], as_index=False)
        .agg(
            leagues=("league", "nunique"),
            matches=("matches", "sum"),
            log_loss=("log_loss", "mean"),
            brier_score=("brier_score", "mean"),
            rps=("rps", "mean"),
        )
    )
    if not equal_league["leagues"].eq(len(EXPECTED_LEAGUES)).all():
        raise ValueError("Equal-league final metrics require every included league")
    return match_weighted, individual_league, equal_league


def build_primary_comparison(
    match_weighted: pd.DataFrame,
    individual_league: pd.DataFrame,
    equal_league: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    comparison = settings["primary_comparison"]
    scope = comparison["training_scope"]
    baseline = comparison["baseline_model"]
    player = comparison["player_model"]
    rows: list[dict[str, object]] = []

    sources = (
        ("equal_league", equal_league, []),
        ("match_weighted", match_weighted, []),
        ("individual_league", individual_league, ["league"]),
    )
    for aggregation, source, group_columns in sources:
        filtered = source[source["training_scope"].eq(scope)]
        baseline_rows = filtered[filtered["model"].eq(baseline)]
        player_rows = filtered[filtered["model"].eq(player)]
        if group_columns:
            paired = baseline_rows.merge(
                player_rows,
                on=group_columns,
                how="outer",
                validate="one_to_one",
                suffixes=("_baseline", "_player"),
                indicator=True,
            )
            if not paired["_merge"].eq("both").all():
                raise ValueError("Primary models do not cover identical leagues")
        else:
            if len(baseline_rows) != 1 or len(player_rows) != 1:
                raise ValueError(f"Primary comparison is incomplete for {aggregation}")
            paired = pd.DataFrame(
                {
                    **{
                        f"{metric}_baseline": [baseline_rows.iloc[0][metric]]
                        for metric in ("log_loss", "brier_score", "rps")
                    },
                    **{
                        f"{metric}_player": [player_rows.iloc[0][metric]]
                        for metric in ("log_loss", "brier_score", "rps")
                    },
                }
            )
        for record in paired.to_dict("records"):
            for metric in ("log_loss", "brier_score", "rps"):
                baseline_score = float(record[f"{metric}_baseline"])
                player_score = float(record[f"{metric}_player"])
                row: dict[str, object] = {
                    "aggregation": aggregation,
                    "training_scope": scope,
                    "baseline_model": baseline,
                    "player_model": player,
                    "metric": metric,
                    "baseline_score": baseline_score,
                    "player_score": player_score,
                    "improvement": baseline_score - player_score,
                    "relative_improvement_pct": 100
                    * (baseline_score - player_score)
                    / baseline_score,
                }
                for column in group_columns:
                    row[column] = record[column]
                rows.append(row)
    return pd.DataFrame(rows)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_final_outputs(
    output_dir: Path,
    settings_path: Path,
    selected: SelectedFeatures,
    training_path: Path,
    final_path: Path,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    weight_audit: pd.DataFrame,
    coverage: pd.DataFrame,
    match_weighted: pd.DataFrame,
    individual_league: pd.DataFrame,
    equal_league: pd.DataFrame,
    primary_comparison: pd.DataFrame,
) -> dict[str, Path]:
    if output_dir.exists():
        raise FileExistsError(
            f"Final output directory already exists and will not be overwritten: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    tables = {
        "predictions": predictions,
        "match_weighted_metrics": match_weighted,
        "individual_league_metrics": individual_league,
        "equal_league_metrics": equal_league,
        "primary_comparison": primary_comparison,
        "feature_coefficients": coefficients,
        "training_weight_audit": weight_audit,
        "sample_counts": coverage,
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        write_csv_atomic(table, path)
        paths[name] = path

    settings_copy = output_dir / "final_evaluation.json"
    settings_temporary = settings_copy.with_suffix(".json.tmp")
    settings_temporary.write_bytes(settings_path.read_bytes())
    settings_temporary.replace(settings_copy)
    paths["final_evaluation"] = settings_copy
    paths.update(
        write_frozen_run_configuration(
            selected,
            output_dir,
            lightgbm_settings=LIGHTGBM_SETTINGS,
        )
    )

    settings_digest = hashlib.sha256(settings_path.read_bytes()).hexdigest()
    run_record = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings_source": str(settings_path),
        "settings_file_sha256": settings_digest,
        "selected_features_sha256": selected.semantic_sha256,
        "training_dataset": str(training_path),
        "training_dataset_sha256": _file_sha256(training_path),
        "final_dataset": str(final_path),
        "final_dataset_sha256": _file_sha256(final_path),
        "training_matches": int(
            coverage.loc[coverage["sample"].eq("training"), "matches"].sum()
        ),
        "final_matches": int(
            coverage.loc[coverage["sample"].eq("final"), "matches"].sum()
        ),
        "post_hoc_calibration": REQUIRED_POST_HOC_CALIBRATION,
    }
    run_record_path = output_dir / "run_record.json"
    _write_json_atomic(run_record, run_record_path)
    paths["run_record"] = run_record_path
    return paths


def main() -> None:
    args = parse_args()
    settings_path = resolve_project_path(args.settings)
    selected = load_selected_features(resolve_project_path(args.selected_features))
    settings = load_and_validate_settings(settings_path, selected)
    output_dir = resolve_project_path(
        args.output_dir or Path(settings["default_output_directory"])
    )
    if output_dir.exists():
        raise FileExistsError(
            f"Final output directory already exists and will not be overwritten: {output_dir}"
        )

    training_path = resolve_project_path(args.training_dataset)
    final_path = resolve_project_path(args.final_dataset)
    training, final, coverage = load_final_samples(training_path, final_path)
    predictions, coefficients, weight_audit = run_final_models(
        training,
        final,
        selected,
        settings,
    )
    match_weighted, individual_league, equal_league = build_metric_tables(predictions)
    primary_comparison = build_primary_comparison(
        match_weighted,
        individual_league,
        equal_league,
        settings,
    )
    paths = write_final_outputs(
        output_dir,
        settings_path,
        selected,
        training_path,
        final_path,
        predictions,
        coefficients,
        weight_audit,
        coverage,
        match_weighted,
        individual_league,
        equal_league,
        primary_comparison,
    )

    print("\nFinal equal-league results")
    print(equal_league.to_string(index=False))
    print("\nPrimary comparison (positive improvement favours player information)")
    print(primary_comparison.to_string(index=False))
    print(f"\nSaved the one-time final evaluation to {output_dir}")
    print("Saved files: " + ", ".join(str(path) for path in paths.values()))


if __name__ == "__main__":
    main()
