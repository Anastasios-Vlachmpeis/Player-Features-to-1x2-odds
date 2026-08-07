# Transfermarkt (archived)

Squad, injury, and market-value scrapers plus an interactive squad graph explorer. **Not used in the baseline** (snapshot leakage on historical fixtures).

- `scrape_transfermarkt.py`, `tm_scraper.py` — ingestion into `player_stats.db` / `tm_*` tables
- `transfermarkt_squad_viz/` — Vite/React graph UI over Transfermarkt data
