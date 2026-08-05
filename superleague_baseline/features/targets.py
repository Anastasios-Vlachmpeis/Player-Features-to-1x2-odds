"""Match outcome targets."""

from __future__ import annotations

import pandas as pd

from superleague_baseline.constants import CLASS_ORDER


def build_proxy_targets(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Build exploratory proxy labels from summed player goals."""
    home = team_matches[team_matches["venue"] == "H"][
        ["match_id", "match_date", "home_team", "away_team", "gf_proxy", "lineup_complete"]
    ].rename(columns={"gf_proxy": "proxy_home_goals", "lineup_complete": "proxy_home_lineup_complete"})
    away = team_matches[team_matches["venue"] == "A"][
        ["match_id", "gf_proxy", "lineup_complete"]
    ].rename(columns={"gf_proxy": "proxy_away_goals", "lineup_complete": "proxy_away_lineup_complete"})
    out = home.merge(away, on="match_id", how="inner")
    out["proxy_lineups_complete"] = out["proxy_home_lineup_complete"] & out[
        "proxy_away_lineup_complete"
    ]
    out["proxy_result_3way"] = out.apply(_result_from_goals, axis=1)
    out["target_is_official"] = False
    return out[
        [
            "match_id",
            "match_date",
            "home_team",
            "away_team",
            "proxy_home_goals",
            "proxy_away_goals",
            "proxy_result_3way",
            "proxy_lineups_complete",
            "target_is_official",
        ]
    ]


def _result_from_goals(row) -> str | None:
    if not row["proxy_lineups_complete"]:
        return None
    hg, ag = row["proxy_home_goals"], row["proxy_away_goals"]
    if hg > ag:
        return "H"
    if hg == ag:
        return "D"
    if hg < ag:
        return "A"
    return None


def require_label_source(label_source: str | None) -> str:
    if label_source == "player-goals-proxy":
        return label_source
    raise ValueError(
        "No official match outcomes are stored in player_stats.db. "
        "Pass --label-source player-goals-proxy for exploratory training only."
    )


def labels_to_array(labels: pd.Series) -> list[str]:
    missing = labels[labels.isna() | ~labels.isin(CLASS_ORDER)]
    if not missing.empty:
        raise ValueError(f"Invalid or missing labels: {missing.head().tolist()}")
    return labels.astype(str).tolist()
