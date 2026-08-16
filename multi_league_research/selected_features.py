"""Load and record the frozen publication feature selections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from constants import PLAYER_FEATURES, PROJECT_ROOT


SELECTED_FEATURES_PATH = (
    PROJECT_ROOT / "multi_league_research" / "config" / "selected_features.csv"
)
SELECTED_FEATURES_SHA256 = (
    "2f3f3a757a26bcd3df93eb798ae8dd8135d82bb40ab4c695a39f0846c4655df4"
)
FEATURE_MODELS = (
    "player_form_logistic",
    "market_plus_player_form",
    "player_form_lightgbm",
    "expanded_player_form_lightgbm",
    "dixon_coles_player_form",
)
EXPECTED_FEATURE_COUNTS = {
    "player_form_logistic": 8,
    "market_plus_player_form": 4,
    "player_form_lightgbm": 19,
    "expanded_player_form_lightgbm": 10,
    "dixon_coles_player_form": 7,
}
VALID_BASE_FEATURES = frozenset(
    feature.removeprefix("diff_") for feature in PLAYER_FEATURES
)


@dataclass(frozen=True)
class SelectedFeatures:
    path: Path
    frame: pd.DataFrame
    by_model: dict[str, list[str]]
    semantic_sha256: str


def selected_features_hash(frame: pd.DataFrame) -> str:
    payload = "\n".join(
        frame["model"].astype(str) + "," + frame["feature"].astype(str)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_selected_features(
    frame: pd.DataFrame,
    *,
    expected_sha256: str | None = None,
) -> tuple[pd.DataFrame, str]:
    required = {"model", "feature"}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Selected features file is missing columns: {missing_columns}")

    normalized = frame[["model", "feature"]].copy()
    for column in ("model", "feature"):
        normalized[column] = normalized[column].astype("string").str.strip()
    if normalized.empty or normalized.isna().any().any():
        raise ValueError("Selected features file must contain non-empty model-feature rows")
    if normalized.eq("").any().any():
        raise ValueError("Selected features file contains an empty model or feature name")
    if normalized.duplicated(["model", "feature"]).any():
        duplicates = normalized.loc[
            normalized.duplicated(["model", "feature"], keep=False)
        ]
        raise ValueError(
            "Selected features file contains duplicate model-feature rows: "
            f"{duplicates.head(5).to_dict('records')}"
        )

    observed_models = set(normalized["model"])
    expected_models = set(FEATURE_MODELS)
    if observed_models != expected_models:
        raise ValueError(
            "Selected features model mismatch: "
            f"missing={sorted(expected_models - observed_models)}, "
            f"unexpected={sorted(observed_models - expected_models)}"
        )

    counts = normalized.groupby("model", sort=False).size().to_dict()
    if counts != EXPECTED_FEATURE_COUNTS:
        raise ValueError(
            "Selected features count mismatch: "
            f"expected={EXPECTED_FEATURE_COUNTS}, observed={counts}"
        )

    unknown_features = sorted(set(normalized["feature"]).difference(VALID_BASE_FEATURES))
    if unknown_features:
        raise ValueError(
            "Selected features file contains names outside the model dataset registry: "
            f"{unknown_features}"
        )

    digest = selected_features_hash(normalized)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            "Selected features checksum changed: "
            f"expected={expected_sha256}, observed={digest}. "
            "Review the selection deliberately and update the registered checksum."
        )
    return normalized, digest


def load_selected_features(
    path: Path = SELECTED_FEATURES_PATH,
) -> SelectedFeatures:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"Selected features file does not exist: {resolved}")

    frame = pd.read_csv(resolved, dtype="string")
    verify_frozen = resolved.resolve() == SELECTED_FEATURES_PATH.resolve()
    normalized, digest = validate_selected_features(
        frame,
        expected_sha256=SELECTED_FEATURES_SHA256 if verify_frozen else None,
    )
    by_model = {
        model: normalized.loc[normalized["model"].eq(model), "feature"].tolist()
        for model in FEATURE_MODELS
    }
    return SelectedFeatures(
        path=resolved,
        frame=normalized,
        by_model=by_model,
        semantic_sha256=digest,
    )


def write_frozen_run_configuration(
    selected: SelectedFeatures,
    output_dir: Path,
    *,
    lightgbm_settings: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features_output = output_dir / "selected_features.csv"
    features_temporary = features_output.with_suffix(".csv.tmp")
    selected.frame.to_csv(features_temporary, index=False)
    features_temporary.replace(features_output)

    try:
        source_path = str(selected.path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        source_path = str(selected.path.resolve())
    settings = {
        "selected_features": {
            "source": source_path,
            "semantic_sha256": selected.semantic_sha256,
            "feature_counts": {
                model: len(features) for model, features in selected.by_model.items()
            },
        },
        "lightgbm": lightgbm_settings,
    }
    settings_output = output_dir / "model_settings.json"
    settings_temporary = settings_output.with_suffix(".json.tmp")
    settings_temporary.write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    settings_temporary.replace(settings_output)
    return {"selected_features": features_output, "model_settings": settings_output}
