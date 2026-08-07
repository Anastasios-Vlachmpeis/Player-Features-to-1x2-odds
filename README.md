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
| Sofascore ingestion | Current season ingested | Multi-season collector supports 2015/16--2025/26; historical backfill not yet run |
| Leakage-safe features | Done | Date-batched lagged team form (`superleague_baseline`) |
| Calibrated logistic baseline | Done | Train / cal / test split, 17 tests passing |
| Official match results | Collector ready | Sofascore scores and Football-Data results are not backfilled until you run the collectors |
| Closing odds benchmark | Collector ready | Football-Data closing/pre-closing distinction is stored explicitly in `odds.db` |
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
scrape_historical_results_odds.py # Official results + historical 1X2 odds
football_data_scraper.py          # Football-Data download and normalization logic
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
The Sofascore and Football-Data historical collectors do not use Selenium.

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

Historical official results and odds for seasons 2015/16 through 2025/26:

```powershell
python scrape_historical_results_odds.py --start-year 2015 --end-year 2026
```

Historical Sofascore player statistics and shotmap xG for the same seasons:

```powershell
python scrape_sofascore.py --start-year 2015 --end-year 2026
python scrape_sofascore_xg.py --start-year 2015 --end-year 2026
```

These commands are restart-safe because database writes are upserts. Run the
base Sofascore command before the xG command; it also fills `sofascore_matches`
with official scores. Older matches may lack lineup or shotmap data, which is
logged and skipped rather than fabricated.

For a smaller retry, combine a single-season range with dates:

```powershell
python scrape_sofascore.py --start-year 2020 --end-year 2021 --from 2020-09-01 --to 2021-06-30
```

Forward bookmaker snapshots remain separate:

```powershell
python collect_odds.py
```

Before running `collect_odds.py`, set competition URLs back to **Greek Super League** in `stoiximan_scraper.py` and `novibet_scraper.py` (they currently target UEFA Champions League from scraper testing).

## Tests

```powershell
python -m pytest -q -m "not integration"
python -m pytest -q -m integration    # requires player_stats.db
```

## Design constraints

- **Historical odds type is explicit** -- use only `odds_is_closing = 1` for the closing benchmark. Pre-closing fallbacks are stored but flagged and must not be mixed into that evaluation.

- **No Transfermarkt in historical features** — current snapshots leak future information on past fixtures.
- **Proxy labels only until official scores are ingested** — e.g. from [Football-Data.co.uk](https://www.football-data.co.uk/greecem.php) Greece CSVs (same 236-match season available).
- **Closing odds for backtests** — bookmakers do not publish historical closes; use Football-Data (`B365CH/D/A`, `AvgCH/D/A`, `PSCH/D/A`) or OddsPortal archives, not live Stoiximan/Novibet scrapes alone.

## Roadmap

1. Ingest **official results** and **closing odds** (Football-Data Greece CSV → join to Sofascore fixtures)
2. Add **model vs market** evaluation to `superleague_baseline`
3. Repoint and schedule **Greek bookmaker** odds collection for forward fixtures
4. Ingest **FBref** where coverage exists
5. **GNN** over player/team graphs once steps 1–2 are solid
