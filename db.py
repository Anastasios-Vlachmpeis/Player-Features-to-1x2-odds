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


def init_historical_results_odds_db() -> None:
    """Create the Football-Data historical results and odds table."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_results_odds (
                source          TEXT NOT NULL,
                season          TEXT NOT NULL,
                division        TEXT,
                match_date      TEXT NOT NULL,
                home_team       TEXT NOT NULL,
                away_team       TEXT NOT NULL,
                full_time_home  INTEGER NOT NULL,
                full_time_away  INTEGER NOT NULL,
                result_3way     TEXT NOT NULL,
                odds_source     TEXT,
                odds_is_closing BOOLEAN NOT NULL DEFAULT 0,
                home_odds       REAL,
                draw_odds       REAL,
                away_odds       REAL,
                market_p_home   REAL,
                market_p_draw   REAL,
                market_p_away   REAL,
                source_url      TEXT NOT NULL,
                scraped_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (
                    source, season, match_date, home_team, away_team
                )
            )
        """)
        conn.commit()


def upsert_historical_results_odds(rows: list[dict]) -> None:
    """Upsert normalized Football-Data result/odds records."""
    if not rows:
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO historical_results_odds
                (source, season, division, match_date, home_team, away_team,
                 full_time_home, full_time_away, result_3way, odds_source,
                 odds_is_closing, home_odds, draw_odds, away_odds,
                 market_p_home, market_p_draw, market_p_away, source_url,
                 scraped_at)
            VALUES
                (:source, :season, :division, :match_date, :home_team, :away_team,
                 :full_time_home, :full_time_away, :result_3way, :odds_source,
                 :odds_is_closing, :home_odds, :draw_odds, :away_odds,
                 :market_p_home, :market_p_draw, :market_p_away, :source_url,
                 CURRENT_TIMESTAMP)
            ON CONFLICT(source, season, match_date, home_team, away_team)
            DO UPDATE SET
                division        = excluded.division,
                full_time_home  = excluded.full_time_home,
                full_time_away  = excluded.full_time_away,
                result_3way     = excluded.result_3way,
                odds_source     = excluded.odds_source,
                odds_is_closing = excluded.odds_is_closing,
                home_odds       = excluded.home_odds,
                draw_odds       = excluded.draw_odds,
                away_odds       = excluded.away_odds,
                market_p_home   = excluded.market_p_home,
                market_p_draw   = excluded.market_p_draw,
                market_p_away   = excluded.market_p_away,
                source_url      = excluded.source_url,
                scraped_at      = CURRENT_TIMESTAMP
            """,
            rows,
        )
        conn.commit()


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
            CREATE TABLE IF NOT EXISTS sofascore_matches (
                match_id      INTEGER PRIMARY KEY,
                season_id     INTEGER,
                season_name   TEXT,
                match_date    TEXT,
                home_team     TEXT,
                away_team     TEXT,
                home_score    INTEGER,
                away_score    INTEGER,
                result_3way   TEXT,
                scraped_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
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


def upsert_sofascore_matches(events: list[dict]) -> None:
    """Store official fixture scores discovered by the Sofascore collector."""
    if not events:
        return

    rows = []
    for event in events:
        home_score = event.get("home_score")
        away_score = event.get("away_score")
        if home_score is None or away_score is None:
            result = None
        elif home_score > away_score:
            result = "H"
        elif home_score < away_score:
            result = "A"
        else:
            result = "D"
        rows.append({**event, "result_3way": result})

    with sqlite3.connect(PLAYER_DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO sofascore_matches
                (match_id, season_id, season_name, match_date, home_team,
                 away_team, home_score, away_score, result_3way, scraped_at)
            VALUES
                (:match_id, :season_id, :season_name, :match_date, :home_team,
                 :away_team, :home_score, :away_score, :result_3way,
                 CURRENT_TIMESTAMP)
            ON CONFLICT(match_id) DO UPDATE SET
                season_id   = excluded.season_id,
                season_name = excluded.season_name,
                match_date  = excluded.match_date,
                home_team   = excluded.home_team,
                away_team   = excluded.away_team,
                home_score  = excluded.home_score,
                away_score  = excluded.away_score,
                result_3way = excluded.result_3way,
                scraped_at  = CURRENT_TIMESTAMP
            """,
            rows,
        )
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


# ---------------------------------------------------------------------------
# FBref advanced metrics (same player_stats.db — new tables only)
# Supplementary source; sparse coverage is expected, missing metrics = NULL.
# ---------------------------------------------------------------------------

def init_fbref_db() -> None:
    with sqlite3.connect(PLAYER_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fbref_players (
                fbref_id    TEXT PRIMARY KEY,
                player_name TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fbref_match_stats (
                fbref_id            TEXT,
                match_id            TEXT,
                match_date          TEXT,
                home_team           TEXT,
                away_team           TEXT,
                player_team         TEXT,
                xg                  REAL,
                xa                  REAL,
                npxg                REAL,
                sca                 REAL,
                gca                 REAL,
                progressive_carries INTEGER,
                progressive_passes  INTEGER,
                touches_att_pen     INTEGER,
                carries_final_third INTEGER,
                scraped_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (fbref_id, match_id)
            )
        """)
        conn.commit()


def upsert_fbref_player(fbref_id: str, player_name: str) -> None:
    """Insert or update the player-name lookup row keyed on fbref_id."""
    with sqlite3.connect(PLAYER_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO fbref_players (fbref_id, player_name)
            VALUES (:fbref_id, :player_name)
            ON CONFLICT(fbref_id) DO UPDATE SET
                player_name = excluded.player_name
            """,
            {"fbref_id": fbref_id, "player_name": player_name},
        )
        conn.commit()


def upsert_fbref_match_stats(rows: list) -> None:
    """
    Upsert per-player per-match advanced metrics on key (fbref_id, match_id).

    rows: list of dicts matching the fbref_match_stats columns. Any metric the
    match page didn't provide should already be None in the dict — it is stored
    as NULL, never treated as an error.
    """
    if not rows:
        return
    with sqlite3.connect(PLAYER_DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO fbref_match_stats
                (fbref_id, match_id, match_date, home_team, away_team,
                 player_team, xg, xa, npxg, sca, gca, progressive_carries,
                 progressive_passes, touches_att_pen, carries_final_third,
                 scraped_at)
            VALUES
                (:fbref_id, :match_id, :match_date, :home_team, :away_team,
                 :player_team, :xg, :xa, :npxg, :sca, :gca, :progressive_carries,
                 :progressive_passes, :touches_att_pen, :carries_final_third,
                 CURRENT_TIMESTAMP)
            ON CONFLICT(fbref_id, match_id) DO UPDATE SET
                match_date          = excluded.match_date,
                home_team           = excluded.home_team,
                away_team           = excluded.away_team,
                player_team         = excluded.player_team,
                xg                  = excluded.xg,
                xa                  = excluded.xa,
                npxg                = excluded.npxg,
                sca                 = excluded.sca,
                gca                 = excluded.gca,
                progressive_carries = excluded.progressive_carries,
                progressive_passes  = excluded.progressive_passes,
                touches_att_pen     = excluded.touches_att_pen,
                carries_final_third = excluded.carries_final_third,
                scraped_at          = CURRENT_TIMESTAMP
            """,
            rows,
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Sofascore xG / xGOT (aggregated from the per-match shotmap endpoint)
# Same player_stats.db — new table only. Players with no shots get no row.
# ---------------------------------------------------------------------------

def init_sofascore_xg_db() -> None:
    with sqlite3.connect(PLAYER_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sofascore_xg (
                sofascore_id INTEGER,
                match_id     INTEGER,
                match_date   TEXT,
                player_team  TEXT,
                xg           REAL,
                xgot         REAL,
                shots        INTEGER,
                sot          INTEGER,
                scraped_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (sofascore_id, match_id)
            )
        """)
        # Add sot to pre-existing tables created before this column existed.
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sofascore_xg)")]
        if "sot" not in cols:
            conn.execute("ALTER TABLE sofascore_xg ADD COLUMN sot INTEGER")
        conn.commit()


def upsert_sofascore_xg(rows: list) -> None:
    """
    Upsert per-player per-match xG aggregates on key (sofascore_id, match_id).

    rows: list of dicts with keys sofascore_id, match_id, match_date,
    player_team, xg, xgot, shots, sot.
    """
    if not rows:
        return
    with sqlite3.connect(PLAYER_DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO sofascore_xg
                (sofascore_id, match_id, match_date, player_team,
                 xg, xgot, shots, sot, scraped_at)
            VALUES
                (:sofascore_id, :match_id, :match_date, :player_team,
                 :xg, :xgot, :shots, :sot, CURRENT_TIMESTAMP)
            ON CONFLICT(sofascore_id, match_id) DO UPDATE SET
                match_date  = excluded.match_date,
                player_team = excluded.player_team,
                xg          = excluded.xg,
                xgot        = excluded.xgot,
                shots       = excluded.shots,
                sot         = excluded.sot,
                scraped_at  = CURRENT_TIMESTAMP
            """,
            rows,
        )
        conn.commit()
