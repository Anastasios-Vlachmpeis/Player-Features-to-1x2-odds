# Greek Super League ~ player-level odds research

Research project to predict **1X2 match outcomes** for the Greek Super League from **player-level match data**, then benchmark against **bookmaker closing odds**. The long-term goal is a graph-based model (GNN); the current focus is a **leakage-safe baseline** and proper market comparison before adding complexity.

## Research question

Can player-level features (lineups, ratings, xG, form) produce calibrated 1X2 probabilities that **beat the market** on out-of-sample fixtures?

Evaluation targets:

- Log loss and Brier score vs a class prior and vs **closing implied probabilities**
- Calibration on a held-out time slice
- Eventually: comparison to Stoiximan / Novibet retail lines (forward collection)

## Current status

| Layer | Status | Notes |
|-------|--------|-------|
| Sofascore ingestion | Done | ~236 fixtures, lineups + shot-level xG in `player_stats.db` |
| Leakage-safe features | Done | Date-batched lagged team form (`superleague_baseline`) |
| Calibrated logistic baseline | Done | Train / cal / test split, 17 tests passing |
| Official match results | Not in DB | Proxy labels from summed player goals only (exploratory) |
| Closing odds benchmark | Planned | `odds.db` empty; use Football-Data / OddsPortal for history |
| Greek bookmaker scrapers | Built, misconfigured | Stoiximan/Novibet pointed at Champions League for testing |
| FBref advanced stats | Scraper exists | 0 rows ingested |
| Transfermarkt | In DB | Exploration only — not used in baseline (snapshot leakage) |
| GNN / graph model | Not started | Deferred until labels + odds benchmark exist |

## Repository layout

```
player_stats.db          # Sofascore lineups, xG, Transfermarkt (read-only for baseline)
odds.db                # Stoiximan / Novibet snapshots (forward collection)

scrape_sofascore.py    # Greek Super League player match stats
scrape_sofascore_xg.py # Per-match xG from Sofascore shotmaps
scrape_fbref.py        # FBref advanced metrics (patchy Greek coverage)
scrape_transfermarkt.py
collect_odds.py        # Daily 1X2 odds from Greek bookmakers

superleague_baseline/  # Feature pipeline, splits, calibrated models, CLI
tests/                 # Contract, leakage, split, and integration tests
graph_viz/             # Interactive squad graph explorer (Transfermarkt)
Agent Summaries/       # Scraper endpoint notes and design docs
```

## Setup

**Modeling pipeline** (Python 3.12+):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
```

**Odds scrapers** (separate Selenium deps — see `requirements.txt`):

```powershell
pip install -r requirements.txt
```

Scrapers are geo-restricted; a **Greece VPN** is required for Stoiximan and Novibet.

## Baseline pipeline

Audit the database, build match-level features, and train a calibrated multinomial logistic model:

```powershell
python -m superleague_baseline audit --db player_stats.db
python -m superleague_baseline build-features --db player_stats.db
python -m superleague_baseline train-evaluate --db player_stats.db --label-source player-goals-proxy
```

Outputs land in `artifacts/` (`match_features.csv`, `baseline_run/predictions.csv`, `metrics.json`).

**Important:** `--label-source player-goals-proxy` is required today because official FT scores are not stored in `player_stats.db`. Metrics from proxy labels are sanity checks only, not publishable results.

Default chronological split:

- Train through **2026-01-31**
- Calibration through **2026-03-31**
- Test through **2026-05-21**

Features use only matches **strictly before** each fixture date (same-day leakage guarded in tests).

## Data collection

```powershell
python scrape_sofascore.py          # Refresh Sofascore lineups
python scrape_sofascore_xg.py         # Refresh xG aggregates
python collect_odds.py              # Snapshot upcoming Greek SL odds (after repointing scrapers)
```

Before running `collect_odds.py`, set competition URLs back to **Greek Super League** in `stoiximan_scraper.py` and `novibet_scraper.py` (they currently target UEFA Champions League from scraper testing).

## Tests

```powershell
python -m pytest -q -m "not integration"
python -m pytest -q -m integration    # requires player_stats.db
```

## Design constraints

- **No Transfermarkt in historical features** — current snapshots leak future information on past fixtures.
- **Proxy labels only until official scores are ingested** — e.g. from [Football-Data.co.uk](https://www.football-data.co.uk/greecem.php) Greece CSVs (same 236-match season available).
- **Closing odds for backtests** — bookmakers do not publish historical closes; use Football-Data (`B365CH/D/A`, `AvgCH/D/A`, `PSCH/D/A`) or OddsPortal archives, not live Stoiximan/Novibet scrapes alone.

## Roadmap

1. Ingest **official results** and **closing odds** (Football-Data Greece CSV → join to Sofascore fixtures)
2. Add **model vs market** evaluation to `superleague_baseline`
3. Repoint and schedule **Greek bookmaker** odds collection for forward fixtures
4. Ingest **FBref** where coverage exists
5. **GNN** over player/team graphs once steps 1–2 are solid
