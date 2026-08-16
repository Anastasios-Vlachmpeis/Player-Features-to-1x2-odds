# The Incremental Value of Player Information in Football Match Prediction

## Research question

Does the recent recorded performance of the announced starting players improve
1X2 match predictions beyond bookmaker closing odds? Recent performance is
measured from each starter's appearances before the predicted match.

The final study covers Belgium, the Netherlands, Portugal, Scotland, and Turkey.
Greece is built and audited but excluded from modelling because its first two
seasons do not satisfy the declared starting-lineup coverage rule.

## Research design

All match and player histories are constructed chronologically and separately
inside each league. Development decisions use three expanding-window folds:

| Prediction season | Training seasons |
|---|---|
| 2022/23 | 2020/21-2021/22 |
| 2023/24 | 2020/21-2022/23 |
| 2024/25 | 2020/21-2023/24 |

After the features and model settings were fixed, the models were trained on
2020/21-2024/25 and evaluated once on the held-out 2025/26 season.

The principal comparison is between two otherwise identical models:

- closing bookmaker probabilities recalibrated from earlier matches;
- the same recalibrated probabilities plus selected recent-player features.

Historical frequency, Dixon-Coles, Dixon-Coles with player information,
non-market logistic regression, and LightGBM models provide additional
comparisons. Log loss is primary; Brier score and ranked probability score are
secondary. Results are reported with equal league weight, match weight, and by
country.

## Repository layout

```text
multi_league_research/
  config/                              Fixed features and final settings
  data/                                Model-dataset loading helpers
  evaluation/                          Walk-forward scoring and reporting
  models/                              Market, logistic, LightGBM and Dixon-Coles models
  exploration_validation_notebooks/   Development and validation notebooks
  visuals/                             Saved tables, figures and figure scripts
  build_all_leagues.py                 Build and combine league datasets
  report_data_quality.py               Audit inclusion and exclusion rules
  evaluate_models.py                   Run development evaluation
  evaluate_final.py                    Run the fixed 2025/26 evaluation

statsapi_scripts/statsapi_leagues/     Six-league 2025/26 data collection
scrapers/                              Football-Data results and odds ingestion
data/processed/all_leagues/            Combined modelling tables
artifacts/                             Saved evaluation reports and manuscript
odds.db                                Historical results and closing odds
```

## Setup

Python 3.12 or newer is required.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
```

TheStatsAPI collectors read `THESTATSAPI_KEY` from the environment or `.env`.
The active collector covers all six stored leagues:

```powershell
.venv\Scripts\python statsapi_scripts\statsapi_leagues\fetch_matches.py
.venv\Scripts\python statsapi_scripts\statsapi_leagues\fetch_player_stats.py
```

## Build the datasets

The standard build includes only the five development seasons:

```powershell
.venv\Scripts\python multi_league_research\build_all_leagues.py
```

It writes
`data/processed/all_leagues/development_model_dataset.csv`. The held-out season
is added only when explicitly requested:

```powershell
.venv\Scripts\python multi_league_research\build_all_leagues.py --include-final
```

This additionally writes
`data/processed/all_leagues/final_2025_26_model_dataset.csv`.

## Data-quality audit

```powershell
.venv\Scripts\python multi_league_research\report_data_quality.py
```

A match is eligible only when it is a completed top-division fixture with
matching scores, valid closing odds, player data, exactly eleven identified
starters per team, starter minutes, and valid team assignments. Exclusions are
recorded rather than silently discarded.

## Development evaluation

```powershell
.venv\Scripts\python multi_league_research\evaluate_models.py
```

The evaluator loads the fixed model-specific inputs from
`multi_league_research/config/selected_features.csv`. It runs pooled models with
equal league training weight and identically configured league-specific models.
Dixon-Coles remains league-specific.

## Final evaluation

```powershell
.venv\Scripts\python multi_league_research\evaluate_final.py
```

The final evaluator verifies the fixed settings, selected-feature checksum,
training and test seasons, and output location before fitting. It refuses to
silently overwrite an existing final evaluation.

## Writeup (Pending arXiv publication)

The LaTex manuscript explaining the research is stored as `writeup.pdf` in the repository's root.

## Leakage safeguards

- Player form excludes the current match and uses only earlier appearances.
- Team histories are updated only from earlier matches in the same league.
- Every model is compared on identical eligible matches.
- Only valid closing odds are used for the market benchmark.
- Development evaluation rejects 2025/26 rows.
- Final settings and selected features are checked before the holdout is used.
