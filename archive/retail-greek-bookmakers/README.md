# Retail Greek bookmaker odds (archived)

Forward-only 1X2 snapshots from Stoiximan and Novibet via Selenium. **Not used for historical backtests** — closing odds come from [Football-Data.co.uk](https://www.football-data.co.uk/) (`AvgCH/D/A`).

- `collect_odds.py` — daily collector → `odds.db` / `match_odds`
- `stoiximan_scraper.py`, `novibet_scraper.py` — browser scrapers (Greece VPN required)
