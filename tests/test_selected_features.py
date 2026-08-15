from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCOTLAND_RESEARCH_DIR = Path(__file__).resolve().parents[1] / "scotland_research"
if str(SCOTLAND_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCOTLAND_RESEARCH_DIR))

from constants import MARKET_FEATURES  # noqa: E402
from selected_features import (  # noqa: E402
    EXPECTED_FEATURE_COUNTS,
    SELECTED_FEATURES_SHA256,
    load_selected_features,
    validate_selected_features,
    write_frozen_run_configuration,
)
from models.player_form_lightgbm import LIGHTGBM_SETTINGS  # noqa: E402
from models.publication_suite import (  # noqa: E402
    league_specific_model_factories,
    pooled_model_factories,
)


def test_selected_features_have_expected_models_counts_and_checksum():
    selected = load_selected_features()

    assert {model: len(features) for model, features in selected.by_model.items()} == (
        EXPECTED_FEATURE_COUNTS
    )
    assert selected.semantic_sha256 == SELECTED_FEATURES_SHA256


def test_selected_features_reject_duplicates_missing_models_and_unknown_names():
    selected = load_selected_features()

    duplicate = pd.concat([selected.frame, selected.frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_selected_features(duplicate)

    missing_model = selected.frame[
        ~selected.frame["model"].eq("dixon_coles_player_form")
    ]
    with pytest.raises(ValueError, match="model mismatch"):
        validate_selected_features(missing_model)

    unknown = selected.frame.copy()
    unknown.loc[0, "feature"] = "feature_that_does_not_exist"
    with pytest.raises(ValueError, match="outside the model dataset registry"):
        validate_selected_features(unknown)


def test_frozen_checksum_rejects_a_silent_change():
    selected = load_selected_features()

    with pytest.raises(ValueError, match="checksum changed"):
        validate_selected_features(selected.frame, expected_sha256="0" * 64)


def test_pooled_and_separate_models_receive_the_same_selected_base_features():
    selected = load_selected_features()
    indicators = ["league_effect_belgium", "league_effect_scotland"]
    pooled = pooled_model_factories(
        indicators,
        selected_features=selected.by_model,
    )
    separate = league_specific_model_factories(
        selected_features=selected.by_model,
    )

    pooled_player = pooled["player_form"]()
    separate_player = separate["player_form"]()
    expected_player = [
        f"diff_{feature}" for feature in selected.by_model["player_form_logistic"]
    ]
    assert separate_player._feature_columns == expected_player
    assert pooled_player._feature_columns == [*expected_player, *indicators]

    pooled_market = pooled["market_plus_player_form"]()
    separate_market = separate["market_plus_player_form"]()
    expected_market_players = [
        f"diff_{feature}"
        for feature in selected.by_model["market_plus_player_form"]
    ]
    assert separate_market._feature_columns == [*MARKET_FEATURES, *expected_market_players]
    assert pooled_market._feature_columns == [
        *MARKET_FEATURES,
        *expected_market_players,
        *indicators,
    ]

    pooled_recalibrated = pooled["recalibrated_market"]()
    separate_recalibrated = separate["recalibrated_market"]()
    assert separate_recalibrated._feature_columns == MARKET_FEATURES
    assert pooled_recalibrated._feature_columns == [*MARKET_FEATURES, *indicators]

    pooled_expanded = pooled["expanded_player_form_lightgbm"]()
    separate_expanded = separate["expanded_player_form_lightgbm"]()
    expected_expanded = [
        f"{side}_{feature}"
        for side in ("home", "away")
        for feature in selected.by_model["expanded_player_form_lightgbm"]
    ]
    assert separate_expanded._feature_columns == expected_expanded
    assert pooled_expanded._feature_columns == [*expected_expanded, *indicators]


def test_dixon_coles_remains_separate_and_uses_its_own_selected_features():
    selected = load_selected_features()
    pooled = pooled_model_factories([], selected_features=selected.by_model)
    separate = league_specific_model_factories(selected_features=selected.by_model)

    assert "dixon_coles" not in pooled
    assert "dixon_coles_player_form" not in pooled
    assert separate["dixon_coles"]()._estimator.player_features == []
    assert separate["dixon_coles_player_form"]()._estimator.player_features == [
        f"diff_{feature}"
        for feature in selected.by_model["dixon_coles_player_form"]
    ]


def test_evaluation_configuration_is_saved_beside_results(tmp_path):
    selected = load_selected_features()
    paths = write_frozen_run_configuration(
        selected,
        tmp_path,
        lightgbm_settings=LIGHTGBM_SETTINGS,
    )

    saved_features = pd.read_csv(paths["selected_features"])
    saved_settings = json.loads(paths["model_settings"].read_text(encoding="utf-8"))
    pd.testing.assert_frame_equal(saved_features, selected.frame, check_dtype=False)
    assert saved_settings["selected_features"]["semantic_sha256"] == (
        SELECTED_FEATURES_SHA256
    )
    assert saved_settings["lightgbm"] == LIGHTGBM_SETTINGS
    assert saved_settings["lightgbm"]["random_state"] == 42
    assert saved_settings["lightgbm"]["n_estimators"] == 100
