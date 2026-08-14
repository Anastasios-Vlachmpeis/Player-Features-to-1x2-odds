"""Frozen model factories for pooled and league-specific comparisons."""

from __future__ import annotations

from collections.abc import Callable, Iterable

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


PredictorFactory = Callable[[], MatchPredictor]

COMMON_MODEL_NAMES = (
    "frequency_baseline",
    "closing_market",
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


def pooled_model_factories(
    league_effect_columns: list[str],
    selected_models: Iterable[str] | None = None,
) -> dict[str, PredictorFactory]:
    """Models that share fitted parameters across leagues.

    League indicator columns allow different starting outcome environments while
    all player-feature effects remain shared. Every learned model here supports
    equal-league sample weights.
    """

    player_columns = [*PLAYER_FEATURES, *league_effect_columns]
    expanded_columns = [*EXPANDED_PLAYER_FEATURES, *league_effect_columns]
    factories: dict[str, PredictorFactory] = {
        "frequency_baseline": LeagueFrequencyBaseline,
        "closing_market": ClosingMarket,
        "player_form": lambda: PlayerFormModel(player_features=player_columns),
        "market_plus_player_form": lambda: MarketPlusPlayerFormModel(
            player_features=player_columns
        ),
        "player_form_lightgbm": lambda: PlayerFormLightGBMModel(
            player_features=player_columns
        ),
        "expanded_player_form_lightgbm": lambda: ExpandedPlayerFormLightGBMModel(
            player_features=expanded_columns
        ),
    }
    return _select_factories(factories, selected_models)


def league_specific_model_factories(
    selected_models: Iterable[str] | None = None,
) -> dict[str, PredictorFactory]:
    """Identically configured models fitted independently inside each league."""

    factories: dict[str, PredictorFactory] = {
        "frequency_baseline": FrequencyBaseline,
        "closing_market": ClosingMarket,
        "player_form": PlayerFormModel,
        "market_plus_player_form": MarketPlusPlayerFormModel,
        "player_form_lightgbm": PlayerFormLightGBMModel,
        "expanded_player_form_lightgbm": ExpandedPlayerFormLightGBMModel,
        "dixon_coles": DixonColesModel,
        "dixon_coles_player_form": DixonColesPlayerFormModel,
    }
    return _select_factories(factories, selected_models)
