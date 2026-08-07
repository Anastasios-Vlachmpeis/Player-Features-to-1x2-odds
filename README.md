# Medium-sized leagues based player-level odds research

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
| Player data ingest | Pivoting to TheStatsAPI | Legacy Sofascore collectors moved to `archive/sofascore/` |
| Greek retail odds | Archived | Selenium Stoiximan/Novibet in `archive/retail-greek-bookmakers/` (forward-only) |
| GNN / graph model | Not started | Deferred until labels + odds benchmark exist |

## Repository layout

```
player_stats.db          # Player lineups, xG (Sofascore today; TheStatsAPI planned)
odds.db                  # Football-Data historical results + closing odds

scrape_historical_results_odds.py # Official results + historical 1X2 odds (Football-Data)
football_data_scraper.py          # Football-Data download and normalization logic

superleague_baseline/    # Feature pipeline, splits, calibrated models, CLI
tests/                   # Contract, leakage, split, and integration tests
archive/                 # Retired scrapers (Sofascore, FBref, TM, retail bookmakers)
```

## Setup

**Modeling pipeline** (Python 3.12+):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
```

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

Historical official results and closing odds (Football-Data G1) for seasons 2015/16 through 2025/26:

```powershell
python scrape_historical_results_odds.py --start-year 2015 --end-year 2026
```

Player-level ingest is moving to **TheStatsAPI**. Legacy Sofascore collectors live under `archive/sofascore/` (see `archive/README.md`).

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

1. Ingest **official results** and **closing odds** (Football-Data, multi-league)
2. Ingest **player data** via TheStatsAPI and join to Football-Data fixtures
3. Add **model vs market** evaluation to `superleague_baseline`
4. **GNN** over player/team graphs once steps 1–3 are solid
