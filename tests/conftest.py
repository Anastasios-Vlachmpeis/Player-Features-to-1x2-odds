from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def synthetic_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE sofascore_match_stats (
            sofascore_id INTEGER,
            match_id INTEGER,
            match_date TEXT,
            home_team TEXT,
            away_team TEXT,
            player_team TEXT,
            rating REAL,
            minutes_played INTEGER,
            goals INTEGER,
            assists INTEGER,
            key_passes INTEGER,
            total_passes INTEGER,
            accurate_passes INTEGER,
            tackles INTEGER,
            interceptions INTEGER,
            clearances INTEGER,
            aerial_won INTEGER,
            aerial_total INTEGER,
            is_starter BOOLEAN,
            scraped_at TEXT
        );
        CREATE TABLE sofascore_xg (
            sofascore_id INTEGER,
            match_id INTEGER,
            match_date TEXT,
            player_team TEXT,
            xg REAL,
            xgot REAL,
            shots INTEGER,
            sot INTEGER,
            scraped_at TEXT
        );
        """
    )

    rows = []
    xg_rows = []
    dates = ["2025-10-01", "2025-10-08", "2025-10-15", "2025-10-22", "2025-10-29", "2025-11-05"]
    teams = [("Alpha", "Beta"), ("Gamma", "Delta")]
    match_id = 1000
    for d in dates:
        for home, away in teams:
            for team, opp, is_home in ((home, away, True), (away, home, False)):
                for i in range(11):
                    rows.append(
                        (
                            100 + i,
                            match_id,
                            d,
                            home,
                            away,
                            team,
                            6.5,
                            90,
                            1 if i == 0 and is_home else 0,
                            0,
                            1,
                            20,
                            15,
                            0,
                            1,
                            1,
                            1,
                            2,
                            1,
                            "2026-01-01",
                        )
                    )
                xg_rows.append((200, match_id, d, team, 1.0, 0.5, 5, 2, "2026-01-01"))
            match_id += 1

    conn.executemany(
        """
        INSERT INTO sofascore_match_stats VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.executemany(
        "INSERT INTO sofascore_xg VALUES (?,?,?,?,?,?,?,?,?)",
        xg_rows,
    )
    conn.commit()
    conn.close()
    return db_path
