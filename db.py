import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "odds.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS match_odds (
                match_id   TEXT,
                home_team  TEXT,
                away_team  TEXT,
                match_date TEXT,
                bookmaker  TEXT,
                home_win   REAL,
                draw       REAL,
                away_win   REAL,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def insert_odds(rows):
    """
    rows: list of (match_id, home_team, away_team, match_date,
                   bookmaker, home_win, draw, away_win)
    """
    if not rows:
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """INSERT INTO match_odds
               (match_id, home_team, away_team, match_date,
                bookmaker, home_win, draw, away_win)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
    print(f"[db] Inserted {len(rows)} rows")
