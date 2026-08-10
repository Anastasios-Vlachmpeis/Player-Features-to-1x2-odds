# Registry of Scotland walk-forward predictors.

from __future__ import annotations

from models.base import MatchPredictor
from models.closing_market import ClosingMarket
from models.frequency_baseline import FrequencyBaseline
from models.market_plus_player_form import MarketPlusPlayerFormModel
from models.player_form import PlayerFormModel


def all_predictors() -> list[MatchPredictor]:
    return [
        FrequencyBaseline(),
        ClosingMarket(),
        PlayerFormModel(),
        MarketPlusPlayerFormModel(),
    ]
