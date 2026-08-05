"""Strictly lagged rolling features computed in date batches."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from superleague_baseline.constants import DEFAULT_L5_WINDOW, DEFAULT_VENUE_L3_WINDOW


@dataclass
class _HistoryRow:
    match_date: pd.Timestamp
    venue: str
    match_lineups_complete: bool
    points_proxy: float | None
    gf_proxy: float
    ga_proxy: float
    xg_for: float | None
    xg_against: float | None
    xgot_for: float | None
    xgot_against: float | None
    shots_for: float | None
    shots_against: float | None
    sot_for: float | None
    sot_against: float | None
    key_passes: float
    passes_attempted: float
    passes_completed: float
    interceptions: float
    clearances: float
    aerial_won: float
    aerial_total: float
    rating_num: float
    rating_den: float
    players_used: float


def _mean_last_n(values: list[float | None], n: int) -> float | None:
    picked = [v for v in values[-n:] if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not picked:
        return None
    return float(np.mean(picked))


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return float(num / den)


def _side_features(history: list[_HistoryRow], venue: str, l5: int, l3: int) -> dict[str, Any]:
    if not history:
        return {
            "history_n": 0,
            "rest_days": None,
            "lineup_for_l5_obs": 0,
            "points_proxy_l5_mean": None,
            "gf_proxy_l5_mean": None,
            "ga_proxy_l5_mean": None,
            "xg_for_l5_mean": None,
            "xg_against_l5_mean": None,
            "xg_balance_l5_mean": None,
            "sot_for_l5_mean": None,
            "sot_against_l5_mean": None,
            "sot_balance_l5_mean": None,
            "key_passes_l5_mean": None,
            "pass_completion_l5_ratio": None,
            "interceptions_l5_mean": None,
            "clearances_l5_mean": None,
            "aerial_win_l5_ratio": None,
            "rating_l5_minutes_weighted": None,
            "players_used_l5_mean": None,
            "same_venue_history_n": 0,
            "points_proxy_same_venue_l3_mean": None,
            "gf_proxy_same_venue_l3_mean": None,
            "ga_proxy_same_venue_l3_mean": None,
            "xg_for_same_venue_l3_mean": None,
            "xg_against_same_venue_l3_mean": None,
        }

    last = history[-1]
    rest_days = None  # filled by caller with target date

    l5_rows = history[-l5:]
    lineup_rows = [r for r in l5_rows if r.match_lineups_complete]
    xg_bal = [
        (r.xg_for - r.xg_against)
        if r.xg_for is not None and r.xg_against is not None
        else None
        for r in l5_rows
    ]
    sot_bal = [
        (r.sot_for - r.sot_against)
        if r.sot_for is not None and r.sot_against is not None
        else None
        for r in l5_rows
    ]

    pass_comp = _ratio(
        sum(r.passes_completed for r in lineup_rows),
        sum(r.passes_attempted for r in lineup_rows),
    )
    aerial_ratio = _ratio(
        sum(r.aerial_won for r in lineup_rows),
        sum(r.aerial_total for r in lineup_rows),
    )
    rating_den = sum(r.rating_den for r in lineup_rows)
    rating_weighted = _ratio(sum(r.rating_num for r in lineup_rows), rating_den)

    venue_rows = [r for r in history if r.venue == venue][-l3:]
    venue_lineup_rows = [r for r in venue_rows if r.match_lineups_complete]

    return {
        "history_n": len(history),
        "rest_days": rest_days,
        "lineup_for_l5_obs": len(lineup_rows),
        "points_proxy_l5_mean": _mean_last_n([r.points_proxy for r in lineup_rows], l5),
        "gf_proxy_l5_mean": _mean_last_n([r.gf_proxy for r in lineup_rows], l5),
        "ga_proxy_l5_mean": _mean_last_n([r.ga_proxy for r in lineup_rows], l5),
        "xg_for_l5_mean": _mean_last_n([r.xg_for for r in l5_rows], l5),
        "xg_against_l5_mean": _mean_last_n([r.xg_against for r in l5_rows], l5),
        "xg_balance_l5_mean": _mean_last_n(xg_bal, l5),
        "sot_for_l5_mean": _mean_last_n([r.sot_for for r in l5_rows], l5),
        "sot_against_l5_mean": _mean_last_n([r.sot_against for r in l5_rows], l5),
        "sot_balance_l5_mean": _mean_last_n(sot_bal, l5),
        "key_passes_l5_mean": _mean_last_n([r.key_passes for r in lineup_rows], l5),
        "pass_completion_l5_ratio": pass_comp,
        "interceptions_l5_mean": _mean_last_n([r.interceptions for r in lineup_rows], l5),
        "clearances_l5_mean": _mean_last_n([r.clearances for r in lineup_rows], l5),
        "aerial_win_l5_ratio": aerial_ratio,
        "rating_l5_minutes_weighted": rating_weighted,
        "players_used_l5_mean": _mean_last_n([r.players_used for r in lineup_rows], l5),
        "same_venue_history_n": len([r for r in history if r.venue == venue]),
        "points_proxy_same_venue_l3_mean": _mean_last_n(
            [r.points_proxy for r in venue_lineup_rows], l3
        ),
        "gf_proxy_same_venue_l3_mean": _mean_last_n(
            [r.gf_proxy for r in venue_lineup_rows], l3
        ),
        "ga_proxy_same_venue_l3_mean": _mean_last_n(
            [r.ga_proxy for r in venue_lineup_rows], l3
        ),
        "xg_for_same_venue_l3_mean": _mean_last_n([r.xg_for for r in venue_rows], l3),
        "xg_against_same_venue_l3_mean": _mean_last_n(
            [r.xg_against for r in venue_rows], l3
        ),
    }


def _row_to_history(row: pd.Series) -> _HistoryRow:
    return _HistoryRow(
        match_date=row["match_date"],
        venue=row["venue"],
        match_lineups_complete=bool(row["match_lineups_complete"]),
        points_proxy=row["points_proxy"] if pd.notna(row.get("points_proxy")) else None,
        gf_proxy=float(row["gf_proxy"]),
        ga_proxy=float(row["ga_proxy"]),
        xg_for=row["xg_for"] if pd.notna(row.get("xg_for")) else None,
        xg_against=row["xg_against"] if pd.notna(row.get("xg_against")) else None,
        xgot_for=row["xgot_for"] if pd.notna(row.get("xgot_for")) else None,
        xgot_against=row["xgot_against"] if pd.notna(row.get("xgot_against")) else None,
        shots_for=row["shots_for"] if pd.notna(row.get("shots_for")) else None,
        shots_against=row["shots_against"] if pd.notna(row.get("shots_against")) else None,
        sot_for=row["sot_for"] if pd.notna(row.get("sot_for")) else None,
        sot_against=row["sot_against"] if pd.notna(row.get("sot_against")) else None,
        key_passes=float(row["key_passes"]),
        passes_attempted=float(row["passes_attempted"]),
        passes_completed=float(row["passes_completed"]),
        interceptions=float(row["interceptions"]),
        clearances=float(row["clearances"]),
        aerial_won=float(row["aerial_won"]),
        aerial_total=float(row["aerial_total"]),
        rating_num=float(row["rating_num"]),
        rating_den=float(row["rating_den"]),
        players_used=float(row["players_used"]),
    )


def compute_date_batched_features(
    team_matches: pd.DataFrame,
    *,
    l5_window: int = DEFAULT_L5_WINDOW,
    venue_l3_window: int = DEFAULT_VENUE_L3_WINDOW,
) -> pd.DataFrame:
    """Compute lagged features using only matches strictly before each fixture date."""
    df = team_matches.sort_values(["match_date", "match_id", "player_team"]).copy()
    history: dict[str, list[_HistoryRow]] = defaultdict(list)
    feature_rows: list[dict[str, Any]] = []

    for match_date, day_df in df.groupby("match_date", sort=True):
        day_df = day_df.sort_values(["match_id", "player_team"])
        for _, row in day_df.iterrows():
            team = row["player_team"]
            feats = _side_features(history[team], row["venue"], l5_window, venue_l3_window)
            if history[team]:
                feats["rest_days"] = int((match_date - history[team][-1].match_date).days)
            else:
                feats["rest_days"] = None
            prefixed = {f"{k}": v for k, v in feats.items()}
            feature_rows.append(
                {
                    "match_id": row["match_id"],
                    "match_date": row["match_date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "player_team": row["player_team"],
                    "venue": row["venue"],
                    **prefixed,
                }
            )

        for _, row in day_df.iterrows():
            history[row["player_team"]].append(_row_to_history(row))

    return pd.DataFrame(feature_rows)
