# Walk-forward evaluation of Scotland player-form models and baselines

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_match_features import MODEL_DATASET_NAME, TEAM_FEATURES
from build_match_dataset import DEFAULT_OUTPUT_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DATASET = DEFAULT_OUTPUT_DIR / MODEL_DATASET_NAME
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "artifacts" / "scotland_model_evaluation"

CLASS_ORDER = ["H", "D", "A"]
PLAYER_FEATURES = [f"diff_{feature}" for feature in TEAM_FEATURES]
MARKET_FEATURES = ["market_log_home_vs_draw", "market_log_away_vs_draw"]
FOLDS = [
    ("2022-23", ["2020-21", "2021-22"]),
    ("2023-24", ["2020-21", "2021-22", "2022-23"]),
    ("2024-25", ["2020-21", "2021-22", "2022-23", "2023-24"]),
]

REQUIRED_COLUMNS = {
    "match_id",
    "season",
    "match_date",
    "home_team",
    "away_team",
    "result_3way",
    "market_home_probability",
    "market_draw_probability",
    "market_away_probability",
    *PLAYER_FEATURES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dataset",
        type=Path,
        default=DEFAULT_MODEL_DATASET,
        help=f"Step-4 model dataset (default: {DEFAULT_MODEL_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EVALUATION_DIR,
        help=f"Evaluation output directory (default: {DEFAULT_EVALUATION_DIR})",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Model dataset does not exist: {path}. Run step 4 first.")
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    if frame["match_id"].duplicated().any():
        raise ValueError("Model dataset contains duplicate match IDs")
    if not frame["result_3way"].isin(CLASS_ORDER).all():
        raise ValueError("result_3way must contain only H, D, or A")

    numeric_columns = PLAYER_FEATURES + [
        "market_home_probability",
        "market_draw_probability",
        "market_away_probability",
    ]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    if frame[numeric_columns].isna().any().any():
        raise ValueError("Model inputs contain missing values")

    market_columns = [
        "market_home_probability",
        "market_draw_probability",
        "market_away_probability",
    ]
    if not np.allclose(frame[market_columns].sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Devigged market probabilities do not sum to one")
    if frame[market_columns].le(0).any().any():
        raise ValueError("Market probabilities must be positive")

    frame["market_log_home_vs_draw"] = np.log(
        frame["market_home_probability"] / frame["market_draw_probability"]
    )
    frame["market_log_away_vs_draw"] = np.log(
        frame["market_away_probability"] / frame["market_draw_probability"]
    )
    return frame.sort_values(["match_date", "match_id"], kind="stable").reset_index(drop=True)


def probability_frame(probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in CLASS_ORDER])


def aligned_model_probabilities(model: object, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features)
    classes = list(model.classes_)
    return raw[:, [classes.index(label) for label in CLASS_ORDER]]


def fit_logistic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, object]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=2_000),
    )
    model.fit(train[feature_columns], train["result_3way"])
    return aligned_model_probabilities(model, test[feature_columns]), model


def multiclass_brier(actual: pd.Series, probabilities: np.ndarray) -> float:
    observed = np.column_stack([(actual.to_numpy() == label) for label in CLASS_ORDER])
    return float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))


def ordered_log_loss(actual: pd.Series, probabilities: np.ndarray) -> float:
    """Calculate log loss using the explicit CLASS_ORDER probability columns."""
    class_indexes = {label: index for index, label in enumerate(CLASS_ORDER)}
    actual_indexes = actual.map(class_indexes)
    if actual_indexes.isna().any():
        raise ValueError("Actual outcomes contain a class outside CLASS_ORDER")
    if probabilities.shape != (len(actual), len(CLASS_ORDER)):
        raise ValueError("Probability matrix shape does not match outcomes and classes")

    row_indexes = np.arange(len(actual))
    assigned_probabilities = probabilities[
        row_indexes, actual_indexes.to_numpy(dtype=int)
    ]
    assigned_probabilities = np.clip(
        assigned_probabilities,
        np.finfo(float).eps,
        1.0,
    )
    return float(-np.log(assigned_probabilities).mean())


def score_predictions(actual: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(CLASS_ORDER)[np.argmax(probabilities, axis=1)]
    return {
        "log_loss": ordered_log_loss(actual, probabilities),
        "brier_score": multiclass_brier(actual, probabilities),
        "accuracy": float(accuracy_score(actual, predicted)),
    }


def coefficient_rows(
    model: object,
    model_name: str,
    test_season: str,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    logistic = model.named_steps["logisticregression"]
    rows: list[dict[str, object]] = []
    for class_index, result_class in enumerate(logistic.classes_):
        for feature, coefficient in zip(feature_columns, logistic.coef_[class_index], strict=True):
            rows.append(
                {
                    "model": model_name,
                    "test_season": test_season,
                    "result_class": result_class,
                    "feature": feature,
                    "standardized_coefficient": coefficient,
                }
            )
    return rows


def evaluate(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    coefficient_output: list[dict[str, object]] = []

    for test_season, train_seasons in FOLDS:
        train = dataset[dataset["season"].isin(train_seasons)].copy()
        test = dataset[dataset["season"].eq(test_season)].copy()
        if train.empty or test.empty:
            raise ValueError(f"Walk-forward fold {test_season} has no train or test rows")
        if set(train["result_3way"]) != set(CLASS_ORDER):
            raise ValueError(f"Training fold for {test_season} does not contain all outcomes")

        frequency = train["result_3way"].value_counts(normalize=True)
        frequency_probabilities = np.tile(
            [frequency.get(label, 0.0) for label in CLASS_ORDER],
            (len(test), 1),
        )
        market_probabilities = test[
            [
                "market_home_probability",
                "market_draw_probability",
                "market_away_probability",
            ]
        ].to_numpy()
        player_probabilities, player_model = fit_logistic(
            train, test, PLAYER_FEATURES
        )
        combined_features = MARKET_FEATURES + PLAYER_FEATURES
        combined_probabilities, combined_model = fit_logistic(
            train, test, combined_features
        )

        models = {
            "frequency_baseline": frequency_probabilities,
            "closing_market": market_probabilities,
            "player_form": player_probabilities,
            "market_plus_player_form": combined_probabilities,
        }
        for model_name, probabilities in models.items():
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
                raise ValueError(f"{model_name} probabilities do not sum to one")
            scores = score_predictions(test["result_3way"], probabilities)
            metric_rows.append(
                {
                    "test_season": test_season,
                    "train_seasons": ";".join(train_seasons),
                    "train_matches": len(train),
                    "test_matches": len(test),
                    "model": model_name,
                    **scores,
                }
            )

            output = test[
                ["match_id", "season", "match_date", "home_team", "away_team", "result_3way"]
            ].copy()
            output.insert(0, "model", model_name)
            output = pd.concat(
                [output.reset_index(drop=True), probability_frame(probabilities)], axis=1
            )
            output["predicted_result"] = np.asarray(CLASS_ORDER)[
                np.argmax(probabilities, axis=1)
            ]
            prediction_frames.append(output)

        coefficient_output.extend(
            coefficient_rows(player_model, "player_form", test_season, PLAYER_FEATURES)
        )
        coefficient_output.extend(
            coefficient_rows(
                combined_model,
                "market_plus_player_form",
                test_season,
                combined_features,
            )
        )

    fold_metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_output)

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
    market_log_loss = overall_metrics.loc[
        overall_metrics["model"].eq("closing_market"), "log_loss"
    ].iloc[0]
    overall_metrics["log_loss_vs_closing_market"] = (
        overall_metrics["log_loss"] - market_log_loss
    )
    overall_metrics = overall_metrics.sort_values("log_loss", kind="stable").reset_index(drop=True)
    return fold_metrics, overall_metrics, predictions, coefficients


def build_outputs(dataset_path: Path, output_dir: Path) -> tuple[Path, Path, Path, Path]:
    dataset = load_dataset(dataset_path)
    fold_metrics, overall_metrics, predictions, coefficients = evaluate(dataset)

    paths = (
        output_dir / "fold_metrics.csv",
        output_dir / "overall_metrics.csv",
        output_dir / "predictions.csv",
        output_dir / "feature_coefficients.csv",
    )
    for frame, path in zip(
        (fold_metrics, overall_metrics, predictions, coefficients), paths, strict=True
    ):
        write_csv_atomic(frame, path)

    print(overall_metrics.to_string(index=False))
    print(f"\nSaved evaluation outputs to {output_dir}")
    return paths


def main() -> None:
    args = parse_args()
    build_outputs(
        resolve_project_path(args.model_dataset),
        resolve_project_path(args.output_dir),
    )


if __name__ == "__main__":
    main()
