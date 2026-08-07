"""
Entry point: python scrape_transfermarkt.py

Runs the full Transfermarkt pipeline for Greek Super League 1:
  1. Fetch all clubs from the league page
  2. For each club, fetch the squad roster
  3. For each player, fetch the profile (secondary position, enriched values)
     and the full injury history
  4. Upsert everything into player_stats.db
"""

import logging
import sys

from db import init_player_db, upsert_player, upsert_injuries
from tm_scraper import get_clubs, get_squad, get_player_profile, get_player_injuries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def run() -> None:
    init_player_db()

    clubs = get_clubs()
    if not clubs:
        log.error("No clubs found — check the league page selector or connectivity")
        sys.exit(1)

    total_players = 0
    total_failures = 0

    for club in clubs:
        club_name = club["club_name"]
        log.info(">> %s", club_name)

        squad = get_squad(club["club_id"], club["club_slug"], club_name)
        log.info("   %d players in squad", len(squad))

        club_ok = 0
        for stub in squad:
            tm_id = stub["tm_id"]
            slug = stub["slug"]
            try:
                # Enrich stub with profile-page data (secondary position, etc.)
                profile_extras = get_player_profile(tm_id, slug)
                player = {**stub, **profile_extras}

                # Drop the internal routing key before writing to DB
                player.pop("slug", None)

                upsert_player(player)

                injuries = get_player_injuries(tm_id, slug)
                upsert_injuries(tm_id, injuries)

                club_ok += 1
            except Exception as exc:
                total_failures += 1
                log.error(
                    "   FAILED tm_id=%s (%s): %s",
                    tm_id,
                    stub.get("full_name", "?"),
                    exc,
                )

        total_players += club_ok
        log.info("   saved %d / %d players", club_ok, len(squad))

    log.info(
        "Done — %d players saved, %d failures",
        total_players,
        total_failures,
    )


if __name__ == "__main__":
    run()
