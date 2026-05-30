import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "odds.db"
PLAYER_DB_PATH = Path(__file__).parent / "player_stats.db"


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


# ---------------------------------------------------------------------------
# Player stats DB (player_stats.db)
# ---------------------------------------------------------------------------

def init_player_db() -> None:
    with sqlite3.connect(PLAYER_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tm_players (
                tm_id              INTEGER PRIMARY KEY,
                full_name          TEXT,
                dob                TEXT,
                age                INTEGER,
                nationality        TEXT,
                primary_position   TEXT,
                secondary_position TEXT,
                market_value_eur   INTEGER,
                club               TEXT,
                shirt_number       INTEGER,
                scraped_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tm_injuries (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                tm_id         INTEGER,
                injury_type   TEXT,
                date_from     TEXT,
                date_to       TEXT,
                matches_missed INTEGER,
                scraped_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tm_id) REFERENCES tm_players(tm_id)
            )
        """)
        conn.commit()


def upsert_player(player: dict) -> None:
    """Insert or update a player row keyed on tm_id."""
    with sqlite3.connect(PLAYER_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO tm_players
                (tm_id, full_name, dob, age, nationality, primary_position,
                 secondary_position, market_value_eur, club, shirt_number, scraped_at)
            VALUES
                (:tm_id, :full_name, :dob, :age, :nationality, :primary_position,
                 :secondary_position, :market_value_eur, :club, :shirt_number,
                 CURRENT_TIMESTAMP)
            ON CONFLICT(tm_id) DO UPDATE SET
                full_name          = excluded.full_name,
                dob                = excluded.dob,
                age                = excluded.age,
                nationality        = excluded.nationality,
                primary_position   = excluded.primary_position,
                secondary_position = excluded.secondary_position,
                market_value_eur   = excluded.market_value_eur,
                club               = excluded.club,
                shirt_number       = excluded.shirt_number,
                scraped_at         = CURRENT_TIMESTAMP
            """,
            player,
        )
        conn.commit()


def upsert_injuries(tm_id: int, injuries: list) -> None:
    """Replace all injury rows for a player — delete-then-insert on each run."""
    with sqlite3.connect(PLAYER_DB_PATH) as conn:
        conn.execute("DELETE FROM tm_injuries WHERE tm_id = ?", (tm_id,))
        if injuries:
            conn.executemany(
                """
                INSERT INTO tm_injuries
                    (tm_id, injury_type, date_from, date_to, matches_missed, scraped_at)
                VALUES
                    (:tm_id, :injury_type, :date_from, :date_to, :matches_missed,
                     CURRENT_TIMESTAMP)
                """,
                injuries,
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Sofascore per-match stats (same player_stats.db — new tables only)
# ---------------------------------------------------------------------------

def init_sofascore_db() -> None:
    with sqlite3.connect(PLAYER_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sofascore_players (
                sofascore_id INTEGER,
                player_name  TEXT,
                PRIMARY KEY (sofascore_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sofascore_match_stats (
                sofascore_id    INTEGER,
                match_id        INTEGER,
                match_date      TEXT,
                home_team       TEXT,
                away_team       TEXT,
                player_team     TEXT,
                rating          REAL,
                minutes_played  INTEGER,
                goals           INTEGER,
                assists         INTEGER,
                key_passes      INTEGER,
                total_passes    INTEGER,
                accurate_passes INTEGER,
                tackles         INTEGER,
                interceptions   INTEGER,
                clearances      INTEGER,
                aerial_won      INTEGER,
                aerial_total    INTEGER,
                is_starter      BOOLEAN,
                scraped_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (sofascore_id, match_id)
            )
        """)
        conn.commit()


def upsert_sofascore_player(sofascore_id: int, player_name: str) -> None:
    """Insert or update the player-name lookup row keyed on sofascore_id."""
    with sqlite3.connect(PLAYER_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO sofascore_players (sofascore_id, player_name)
            VALUES (:sofascore_id, :player_name)
            ON CONFLICT(sofascore_id) DO UPDATE SET
                player_name = excluded.player_name
            """,
            {"sofascore_id": sofascore_id, "player_name": player_name},
        )
        conn.commit()


def upsert_sofascore_match_stats(rows: list) -> None:
    """
    Upsert per-player per-match stat rows on composite key (sofascore_id, match_id).

    rows: list of dicts matching the sofascore_match_stats columns.
    """
    if not rows:
        return
    with sqlite3.connect(PLAYER_DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO sofascore_match_stats
                (sofascore_id, match_id, match_date, home_team, away_team,
                 player_team, rating, minutes_played, goals, assists,
                 key_passes, total_passes, accurate_passes, tackles,
                 interceptions, clearances, aerial_won, aerial_total,
                 is_starter, scraped_at)
            VALUES
                (:sofascore_id, :match_id, :match_date, :home_team, :away_team,
                 :player_team, :rating, :minutes_played, :goals, :assists,
                 :key_passes, :total_passes, :accurate_passes, :tackles,
                 :interceptions, :clearances, :aerial_won, :aerial_total,
                 :is_starter, CURRENT_TIMESTAMP)
            ON CONFLICT(sofascore_id, match_id) DO UPDATE SET
                match_date      = excluded.match_date,
                home_team       = excluded.home_team,
                away_team       = excluded.away_team,
                player_team     = excluded.player_team,
                rating          = excluded.rating,
                minutes_played  = excluded.minutes_played,
                goals           = excluded.goals,
                assists         = excluded.assists,
                key_passes      = excluded.key_passes,
                total_passes    = excluded.total_passes,
                accurate_passes = excluded.accurate_passes,
                tackles         = excluded.tackles,
                interceptions   = excluded.interceptions,
                clearances      = excluded.clearances,
                aerial_won      = excluded.aerial_won,
                aerial_total    = excluded.aerial_total,
                is_starter      = excluded.is_starter,
                scraped_at      = CURRENT_TIMESTAMP
            """,
            rows,
        )
        conn.commit()
