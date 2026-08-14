# Registry of Scotland walk-forward predictors.

from __future__ import annotations

from constants import PLAYER_FEATURES
from models.base import MatchPredictor
from models.closing_market import ClosingMarket
from models.dixon_coles import DixonColesModel
from models.dixon_coles_player_form import DixonColesPlayerFormModel
from models.expanded_player_form_lightgbm import ExpandedPlayerFormLightGBMModel
from models.frequency_baseline import FrequencyBaseline
from models.market_plus_player_form import MarketPlusPlayerFormModel
from models.multinomial_gam import MultinomialGAMModel
from models.player_form import PlayerFormModel
from models.player_form_lightgbm import PlayerFormLightGBMModel
from models.poisson_gam import PoissonGAMModel
from models.tuned_lightgbm import TunedExpandedPlayerFormLightGBMModel, TunedPlayerFormLightGBMModel


PLAYER_FEATURE_GROUPS = {
    "shooting": [
        "diff_npxg_per90_sum_5",
        "diff_shots_per90_sum_5",
    ],
    "chance_creation": [
        "diff_key_passes_per90_sum_5",
    ],
    "defending": [
        "diff_defensive_actions_per90_sum_5",
    ],
    "ratings": [
        "diff_rating_mean_5",
    ],
    "recent_experience": [
        "diff_recent_minutes_sum_5",
    ],
    "history_coverage": [
        "diff_starters_without_history",
        "diff_starters_without_full_window",
    ],
}

FULL_PLAYER_MODEL_NAME = "market_plus_all_player_features"


def all_predictors() -> list[MatchPredictor]:
    return [
        FrequencyBaseline(),
        DixonColesModel(),
        DixonColesPlayerFormModel(),
        ClosingMarket(),
        PlayerFormModel(),
        PlayerFormLightGBMModel(),
        ExpandedPlayerFormLightGBMModel(),
        TunedPlayerFormLightGBMModel(),
        TunedExpandedPlayerFormLightGBMModel(),
        PoissonGAMModel(),
        MultinomialGAMModel(),
        MarketPlusPlayerFormModel(),
    ]


def models_for_player_feature_removal_test() -> list[MatchPredictor]:
    #Compare the full market/player model with one player group removed at a time

    models: list[MatchPredictor] = [
        ClosingMarket(),
        MarketPlusPlayerFormModel(player_features=PLAYER_FEATURES,name=FULL_PLAYER_MODEL_NAME,),
    ]

    for group_name, columns_to_remove in PLAYER_FEATURE_GROUPS.items():
        
        remaining_columns = [column for column in PLAYER_FEATURES if column not in columns_to_remove]
        models.append(MarketPlusPlayerFormModel(player_features=remaining_columns,name=f"market_without_{group_name}"))

    return models

def models_for_individual_feature_removal_test() -> list[MatchPredictor]:
    models: list[MatchPredictor] = [ClosingMarket()]

    families = [
        ("player_form", PlayerFormModel),
        ("market_plus_player_form", MarketPlusPlayerFormModel),
        ("dixon_coles_player_form", DixonColesPlayerFormModel),
    ]

    for family_name, model_class in families:
        models.append(
            model_class(
                player_features=PLAYER_FEATURES,
                name=f"{family_name}_all_features",
            )
        )

        for removed_feature in PLAYER_FEATURES:
            remaining_features = [
                feature
                for feature in PLAYER_FEATURES
                if feature != removed_feature
            ]

            short_name = removed_feature.removeprefix("diff_")

            models.append(
                model_class(
                    player_features=remaining_features,
                    name=f"{family_name}_without_{short_name}",
                )
            )

    return models
