"""
Export player_stats.db → static JSON for the squad graph frontend.

Run from repo root:
    python graph_viz/build_graph.py

Writes:
    graph_viz/frontend/public/data/graph.json
    graph_viz/frontend/public/data/players.json
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import PLAYER_DB_PATH  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "frontend" / "public" / "data"

LEAGUE_ID = "league:super-league"
LEAGUE_LABEL = "Super League"


def _team_id(club: str) -> str:
    return f"team:{club}"


def _player_id(tm_id: int) -> str:
    return f"player:{tm_id}"


def _node_val(market_value_eur: int | None) -> float:
    """Scale market value to a graph node size (log-scaled, clamped)."""
    mv = market_value_eur or 0
    if mv <= 0:
        return 2.0
    import math

    return max(2.0, min(12.0, 2.0 + math.log10(mv + 1) * 1.5))


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(PLAYER_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        players = conn.execute(
            """
            SELECT tm_id, full_name, dob, age, nationality,
                   primary_position, secondary_position,
                   market_value_eur, club, shirt_number
            FROM tm_players
            ORDER BY club, full_name
            """
        ).fetchall()

        injuries_by_player: dict[int, list[dict]] = {}
        for row in conn.execute(
            """
            SELECT tm_id, injury_type, date_from, date_to, matches_missed
            FROM tm_injuries
            ORDER BY tm_id, date_from DESC
            """
        ):
            injuries_by_player.setdefault(row["tm_id"], []).append(
                {
                    "injury_type": row["injury_type"],
                    "date_from": row["date_from"],
                    "date_to": row["date_to"],
                    "matches_missed": row["matches_missed"],
                }
            )

    clubs = sorted({p["club"] for p in players if p["club"]})
    nodes: list[dict] = []
    links: list[dict] = []
    players_detail: dict[str, dict] = {}

    nodes.append(
        {
            "id": LEAGUE_ID,
            "type": "league",
            "label": LEAGUE_LABEL,
            "club": "",
            "val": 36,
            "fx": 0,
            "fy": 0,
        }
    )

    for club in clubs:
        nodes.append(
            {
                "id": _team_id(club),
                "type": "team",
                "label": club,
                "club": club,
                "val": 24,
            }
        )
        links.append(
            {
                "source": LEAGUE_ID,
                "target": _team_id(club),
                "type": "league",
            }
        )

    for p in players:
        tm_id = p["tm_id"]
        club = p["club"] or "Unknown"
        pid = _player_id(tm_id)
        nodes.append(
            {
                "id": pid,
                "type": "player",
                "label": p["full_name"] or f"Player {tm_id}",
                "club": club,
                "position": p["primary_position"],
                "val": _node_val(p["market_value_eur"]),
                "tm_id": tm_id,
            }
        )
        links.append(
            {
                "source": pid,
                "target": _team_id(club),
                "type": "membership",
            }
        )
        players_detail[str(tm_id)] = {
            "tm_id": tm_id,
            "full_name": p["full_name"],
            "dob": p["dob"],
            "age": p["age"],
            "nationality": p["nationality"],
            "primary_position": p["primary_position"],
            "secondary_position": p["secondary_position"],
            "market_value_eur": p["market_value_eur"],
            "club": club,
            "shirt_number": p["shirt_number"],
            "injuries": injuries_by_player.get(tm_id, []),
        }

    graph = {
        "meta": {
            "teams": len(clubs),
            "players": len(players),
            "links": len(links),
        },
        "nodes": nodes,
        "links": links,
    }

    graph_path = OUT_DIR / "graph.json"
    players_path = OUT_DIR / "players.json"
    graph_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    players_path.write_text(
        json.dumps(players_detail, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[build_graph] {len(clubs)} teams, {len(players)} players, {len(links)} links")
    print(f"[build_graph] wrote {graph_path}")
    print(f"[build_graph] wrote {players_path}")


if __name__ == "__main__":
    build()
