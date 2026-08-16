# Multiclass LightGBM model on separate home and away player-form features.

from __future__ import annotations

from constants import EXPANDED_PLAYER_FEATURES
from models.player_form_lightgbm import PlayerFormLightGBMModel


class ExpandedPlayerFormLightGBMModel(PlayerFormLightGBMModel):
    #Nonlinear player-form model that retains absolute home/away values

    name = "expanded_player_form_lightgbm"

    def __init__(self, player_features=None, name="expanded_player_form_lightgbm",) -> None:
        super().__init__(player_features=(EXPANDED_PLAYER_FEATURES if player_features is None else list(player_features)),name=name)