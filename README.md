# Super League baseline modeling

Leakage-safe historical match features and calibrated 1X2 baselines built from `player_stats.db`.

## Setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
```

## Commands

```powershell
python -m superleague_baseline audit --db player_stats.db
python -m superleague_baseline build-features --db player_stats.db
python -m superleague_baseline train-evaluate --db player_stats.db --label-source player-goals-proxy
```

Proxy labels are exploratory only until official match scores are ingested.

## Tests

```powershell
python -m pytest -q -m "not integration"
python -m pytest -q -m integration
```
