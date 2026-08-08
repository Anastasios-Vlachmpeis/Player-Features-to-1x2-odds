# Archive

Retired collectors and side experiments, grouped by purpose. Active research uses **Football-Data CSV** (labels + closing odds) and **TheStatsAPI** (player data) instead of these paths.

| Directory | Contents |
|-----------|----------|
| [`sofascore/`](sofascore/) | Sofascore JSON API collectors (lineups, xG) — superseded by TheStatsAPI |

Archived entry scripts import `db.py` from the repo root. To run one for reference:

```powershell
$env:PYTHONPATH = "archive\<folder>;."
python archive\<folder>\<script>.py
```
