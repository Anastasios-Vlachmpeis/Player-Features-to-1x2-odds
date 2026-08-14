# Medium-Large European Leagues player-form odds research

Research project testing whether recent player-level performance improves 1X2 predictions in Belgium, the Netherlands, Portugal, Scotland, and Turkey, either on its own or when added to bookmaker closing probabilities. Greece is built and audited but excluded from model development under the predeclared lineup-coverage rule.

The active work is a leakage-safe, walk-forward comparison across the 2020/21 to 2024/25 development seasons. The untouched 2025/26 season is reserved for the final examination. The main question is whether information from the announced starters' previous appearances adds predictive value beyond the closing market.

## Current experiment

The current modelling dataset joins, independently inside each league:

- Match results and player-match statistics from TheStatsAPI
- Historical 1X2 closing odds from Football-Data.co.uk, stored in `odds.db`
- Rolling player form calculated only from appearances before each target match

For every starter, form is calculated over their previous (five) appearances. The starter-level values are then aggregated into home-team, away-team, and home-minus-away features for:

- non-penalty expected goals
- shots
- key passes
- tackles plus interceptions
- average player rating
- recent minutes
- availability of prior player history

The Scotland export has no useful variation in its xG and xA fields, so we use populated non-penalty xG and key passes instead.

## Evaluation design

Models are evaluated out of sample with expanding training windows:

| Test season | Training seasons |
|---|---|
| 2022/23 | 2020/21–2021/22 |
| 2023/24 | 2020/21–2022/23 |
| 2024/25 | 2020/21–2023/24 |

The comparison includes:

- historical result-frequency baseline
- Dixon–Coles score model
- Dixon–Coles with player-form adjustments
- player-form multinomial model
- bookmaker closing market
- closing market plus player-form features

Performance is reported with log loss, Brier score, and accuracy. Log loss is the main comparison, with every model measured directly against the closing market.

The current combined evaluation covers 665 out-of-sample matches. In the latest saved results, the closing market has the best log loss (0.918), with the strongest non-market model being Dixon–Coles (0.937). We are currently identifying whether particular groups of player features help or hurt the market-plus-player model.

## Repository layout

```text
scotland_research/
  validate_dataset.py                 Validate match, odds, lineup, and player coverage
  build_match_dataset.py              Join clean matches, results, and closing odds
  build_player_form.py                Build prior-five-appearance starter form
  build_match_features.py             Aggregate player form to match-level features
  evaluate_models.py                  Run the main walk-forward model comparison
  evaluate_player_feature_removal.py  Remove one player feature group at a time
  models/                             Market, logistic, and Dixon–Coles predictors
  evaluation/                         Walk-forward scoring and reports
  exploration_validation_notebooks    Notebooks used to explore and validate the data  

statsapi_scripts/
  statsapi_scotland/                   Scotland match and player-stat collection
  statsapi_remaining_leagues/          Collection work for the other target leagues

scrapers/                              Football-Data results and odds ingestion
data/processed/scotland/               Generated modelling tables
artifacts/                             Generated validation and evaluation reports
odds.db                                Historical results and odds database
```

## Setup

Python 3.12 or newer is recommended.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
```

To collect Scotland data, put the API key in `.env`:

```text
THESTATSAPI_KEY=your_key_here
```

Then run the match collector followed by the resumable player-stat collector:

```powershell
.venv\Scripts\python statsapi_scripts\statsapi_scotland\fetch_matches.py
.venv\Scripts\python statsapi_scripts\statsapi_scotland\fetch_player_stats.py
```

## Build the six-league modelling dataset

The shared builder runs the existing leakage-safe feature pipeline independently
for Scotland, Greece, Belgium, Portugal, the Netherlands, and Turkey, then adds a
`league` column and combines the finished model tables.

During development, build only seasons through 2024/25:

```powershell
.venv\Scripts\python scotland_research\build_all_leagues.py
```

This writes the combined development dataset to:

```text
data/processed/all_leagues/development_model_dataset.csv
```

After the model and feature specification is frozen and all 2025/26 source data
has been collected, build the held-out final table explicitly:

```powershell
.venv\Scripts\python scotland_research\build_all_leagues.py --include-final
```

This additionally writes:

```text
data/processed/all_leagues/final_2025_26_model_dataset.csv
```

Omitting `--include-final` is a deliberate safeguard: a normal development build
cannot contain 2025/26 rows.

## Run the five-league development evaluation

With `data/processed/all_leagues/development_model_dataset.csv` already built:

```powershell
.venv\Scripts\python scotland_research\evaluate_models.py
```

The default command runs both research branches:

- one equally weighted shared model with league indicators;
- five identically configured models fitted separately by league.

Dixon-Coles and Dixon-Coles with player form are fitted only in the
league-specific branch. To run a quick primary-comparison pass while developing
the reporting pipeline:

```powershell
.venv\Scripts\python scotland_research\evaluate_models.py `
  --models closing_market market_plus_player_form
```

Outputs include match-level predictions, league and equal-league metrics,
fold-level metrics, coefficient tables, and an audit proving that every league
receives equal total weight in pooled training.

The evaluator currently pools Scotland, Belgium, Portugal, the Netherlands,
and Turkey inside each chronological development fold. Greece remains in the
combined source dataset but is temporarily excluded because its 2020/21 and
2021/22 data do not provide usable starting lineups.
It refuses to run if any league-season is absent or if the held-out 2025/26
season is present. Outputs are written to
`artifacts/five_league_development_evaluation`. The
`fold_league_counts.csv` output makes each country's train/test match count
visible in every fold.

## Build the development-analysis visualizations

After pooled and league-specific predictions have been generated:

```powershell
.venv\Scripts\python scotland_research\visuals\scripts\development_analysis.py
```

PNG files are written to
`scotland_research/visuals/development_analysis`:

1. league-season coverage heatmaps;
2. analytical-sample construction flow;
3. chronological experiment timeline;
4. player-model comparison with the closing market by league;
5. shared versus separately trained model comparison;
6. home/draw/away probability reliability plots.

## Data safeguards

- Rolling form excludes the current match and uses only earlier appearances.
- Each modelling fixture must have exactly 11 identified starters per team.
- Match scores must agree between TheStatsAPI and Football-Data.
- Only rows explicitly identified as closing odds are used for the market benchmark.
- Fixtures that fail validation are recorded separately instead of silently entering the model.
