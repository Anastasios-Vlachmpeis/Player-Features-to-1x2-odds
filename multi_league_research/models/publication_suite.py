"""Frozen model factories for pooled and league-specific comparisons."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from constants import EXPANDED_PLAYER_FEATURES, PLAYER_FEATURES
from models.base import MatchPredictor
from models.closing_market import ClosingMarket
from models.dixon_coles import DixonColesModel
from models.dixon_coles_player_form import DixonColesPlayerFormModel
from models.expanded_player_form_lightgbm import ExpandedPlayerFormLightGBMModel
from models.frequency_baseline import FrequencyBaseline, LeagueFrequencyBaseline
from models.market_plus_player_form import MarketPlusPlayerFormModel
from models.player_form import PlayerFormModel
from models.player_form_lightgbm import PlayerFormLightGBMModel
from models.recalibrated_market import RecalibratedMarketModel


PredictorFactory = Callable[[], MatchPredictor]
SelectedFeatureMap = Mapping[str, list[str]]

COMMON_MODEL_NAMES = (
    "frequency_baseline",
    "closing_market",
    "recalibrated_market",
    "player_form",
    "market_plus_player_form",
    "player_form_lightgbm",
    "expanded_player_form_lightgbm",
)
SEPARATE_ONLY_MODEL_NAMES = (
    "dixon_coles",
    "dixon_coles_player_form",
)
PUBLICATION_MODEL_NAMES = (*COMMON_MODEL_NAMES, *SEPARATE_ONLY_MODEL_NAMES)


def _select_factories(
    factories: dict[str, PredictorFactory],
    selected_models: Iterable[str] | None,
) -> dict[str, PredictorFactory]:
    if selected_models is None:
        return factories
    requested = list(dict.fromkeys(selected_models))
    unknown = sorted(set(requested).difference(PUBLICATION_MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown publication model(s): {unknown}")
    return {name: factories[name] for name in requested if name in factories}


def _selected_columns(
    selected_features: SelectedFeatureMap | None,
    feature_model: str,
    *,
    expanded: bool = False,
) -> list[str]:
    if selected_features is None:
        return list(EXPANDED_PLAYER_FEATURES if expanded else PLAYER_FEATURES)
    if feature_model not in selected_features:
        raise ValueError(f"Selected features file has no entry for {feature_model}")
    base_features = list(selected_features[feature_model])
    if not base_features:
        raise ValueError(f"Selected features file has no features for {feature_model}")
    if expanded:
        return [
            f"{side}_{feature}"
            for side in ("home", "away")
            for feature in base_features
        ]
    return [f"diff_{feature}" for feature in base_features]


def pooled_model_factories(
    league_effect_columns: list[str],
    selected_models: Iterable[str] | None = None,
    *,
    selected_features: SelectedFeatureMap | None = None,
) -> dict[str, PredictorFactory]:
    """Models that share fitted parameters across leagues.

    League indicator columns allow different starting outcome environments while
    all player-feature effects remain shared. Every learned model here supports
    equal-league sample weights.
    """

    player_form_columns = [
        *_selected_columns(selected_features, "player_form_logistic"),
        *league_effect_columns,
    ]
    market_player_columns = [
        *_selected_columns(selected_features, "market_plus_player_form"),
        *league_effect_columns,
    ]
    lightgbm_columns = [
        *_selected_columns(selected_features, "player_form_lightgbm"),
        *league_effect_columns,
    ]
    expanded_columns = [
        *_selected_columns(
            selected_features,
            "expanded_player_form_lightgbm",
            expanded=True,
        ),
        *league_effect_columns,
    ]
    factories: dict[str, PredictorFactory] = {
        "frequency_baseline": LeagueFrequencyBaseline,
        "closing_market": ClosingMarket,
        "recalibrated_market": lambda: RecalibratedMarketModel(
            context_features=league_effect_columns
        ),
        "player_form": lambda: PlayerFormModel(
            player_features=player_form_columns
        ),
        "market_plus_player_form": lambda: MarketPlusPlayerFormModel(
            player_features=market_player_columns
        ),
        "player_form_lightgbm": lambda: PlayerFormLightGBMModel(
            player_features=lightgbm_columns
        ),
        "expanded_player_form_lightgbm": lambda: ExpandedPlayerFormLightGBMModel(
            player_features=expanded_columns
        ),
    }
    return _select_factories(factories, selected_models)


def league_specific_model_factories(
    selected_models: Iterable[str] | None = None,
    *,
    selected_features: SelectedFeatureMap | None = None,
) -> dict[str, PredictorFactory]:
    """Identically configured models fitted independently inside each league."""

    player_form_columns = _selected_columns(
        selected_features,
        "player_form_logistic",
    )
    market_player_columns = _selected_columns(
        selected_features,
        "market_plus_player_form",
    )
    lightgbm_columns = _selected_columns(
        selected_features,
        "player_form_lightgbm",
    )
    expanded_columns = _selected_columns(
        selected_features,
        "expanded_player_form_lightgbm",
        expanded=True,
    )
    dixon_coles_player_columns = _selected_columns(
        selected_features,
        "dixon_coles_player_form",
    )
    factories: dict[str, PredictorFactory] = {
        "frequency_baseline": FrequencyBaseline,
        "closing_market": ClosingMarket,
        "recalibrated_market": RecalibratedMarketModel,
        "player_form": lambda: PlayerFormModel(
            player_features=player_form_columns
        ),
        "market_plus_player_form": lambda: MarketPlusPlayerFormModel(
            player_features=market_player_columns
        ),
        "player_form_lightgbm": lambda: PlayerFormLightGBMModel(
            player_features=lightgbm_columns
        ),
        "expanded_player_form_lightgbm": lambda: ExpandedPlayerFormLightGBMModel(
            player_features=expanded_columns
        ),
        "dixon_coles": DixonColesModel,
        "dixon_coles_player_form": lambda: DixonColesPlayerFormModel(
            player_features=dixon_coles_player_columns
        ),
    }
    return _select_factories(factories, selected_models)
