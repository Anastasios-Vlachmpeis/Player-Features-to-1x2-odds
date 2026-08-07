# Archive

Retired collectors and side experiments, grouped by purpose. Active research uses **Football-Data CSV** (labels + closing odds) and **TheStatsAPI** (player data) instead of these paths.

| Directory | Contents |
|-----------|----------|
| [`retail-greek-bookmakers/`](retail-greek-bookmakers/) | Selenium scrapers for forward Stoiximan/Novibet 1X2 snapshots |
| [`fbref/`](fbref/) | FBref HTML scraper for patchy Greek advanced stats |
| [`transfermarkt/`](transfermarkt/) | Transfermarkt squad/injury scraper + squad graph viz |
| [`sofascore/`](sofascore/) | Sofascore JSON API collectors (lineups, xG) — superseded by TheStatsAPI |
| [`exploratory/`](exploratory/) | Early notebooks and manual labeling files |

Archived entry scripts import `db.py` from the repo root. To run one for reference:

```powershell
$env:PYTHONPATH = "archive\<folder>;."
python archive\<folder>\<script>.py
```
