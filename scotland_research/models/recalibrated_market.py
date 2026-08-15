"""Logistic recalibration of closing-market probabilities without player data."""

from __future__ import annotations

from models.market_plus_player_form import MarketPlusPlayerFormModel


class RecalibratedMarketModel(MarketPlusPlayerFormModel):
    name = "recalibrated_market"

    def __init__(self, context_features=None) -> None:
        # The pooled evaluation supplies league indicators here. Player
        # performance features are never supplied to this model.
        super().__init__(
            player_features=[] if context_features is None else context_features,
            name=self.name,
        )
