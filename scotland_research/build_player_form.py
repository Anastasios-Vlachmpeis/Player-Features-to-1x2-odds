# Leakage-safe rolling form for starters in clean Scotland matches.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_match_dataset import DEFAULT_OUTPUT_DIR, MATCH_DATASET_NAME
from build_team_strength_features import TEAM_STRENGTH_NAME
from validate_dataset import PLAYER_STATS_CSV, as_bool
from feature_rules import NPXG_FIELD, NPXG_MIN_VALID_APPEARANCES, NPXG_MIN_VALID_MINUTES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCH_DATASET = DEFAULT_OUTPUT_DIR / MATCH_DATASET_NAME
DEFAULT_TEAM_STRENGTH = DEFAULT_OUTPUT_DIR / TEAM_STRENGTH_NAME
PLAYER_FORM_NAME = "player_rolling_form.csv"
PLAYER_FORM_COVERAGE_NAME = "player_form_coverage.csv"
ROLLING_APPEARANCES = 5
FORM_WINDOWS = (1, 3, 5)
OPPONENT_RIDGE_STRENGTH = 25.0
EWM_HALFLIFE_APPEARANCES = 2.0

EVENT_STATS = {
    "shooting.goals": "goals",
    "passing.assists": "assists",
    # Scotland's expected_goals and expected_assists fields are constant zero.
    # np_expected_goals is populated, and key passes remain a usable creation proxy.
    "shooting.np_expected_goals": "npxg",
    "shooting.total_shots": "shots",
    "passing.key_passes": "key_passes",
    "goalkeeping.saves": "saves",
}

# These targets receive expanding pre-match expectations. Event targets are
# converted to per-90 rates; rating is already an appearance-level measure.
ADJUSTED_TARGETS = {
    "shots": ("shooting.total_shots", True, 0.0),
    "key_passes": ("passing.key_passes", True, 0.0),
    "defensive_actions": ("defensive_actions", True, 0.0),
    "rating": ("rating", False, 6.5),
}

REQUIRED_PLAYER_COLUMNS = {
    "match_id",
    "season",
    "utc_date",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "team_id",
    "player_id",
    "player_name",
    "position",
    "played",
    "started",
    "minutes_played",
    "rating",
    "defending.tackles",
    "defending.interceptions",
    *EVENT_STATS,
}

OUTPUT_COLUMNS = [
    "match_id",
    "season",
    "utc_date",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "team_id",
    "team_side",
    "player_id",
    "player_name",
    "position",
    "previous_appearance_utc_date",
    "days_since_previous_appearance",
    "prior_appearances",
    "form_window_appearances_5",
    "has_prior_history",
    "form_minutes_5",
    "form_starts_5",
    "form_rating_mean_5",
    "form_goals_per90_5",
    "form_assists_per90_5",
    "form_npxg_per90_5",
    "form_npxg_observations_5",
    "form_npxg_minutes_5",
    "form_shots_per90_5",
    "form_key_passes_per90_5",
    "form_defensive_actions_per90_5",
    "form_saves_per90_5",
]

# Retain several horizons instead of forcing all form into an equal-weight
# five-appearance average. The 1-vs-5 difference measures short-term direction.
for stat_name in ("shots", "key_passes", "defensive_actions"):
    OUTPUT_COLUMNS.extend(
        [
            f"form_{stat_name}_per90_1",
            f"form_{stat_name}_per90_3",
            f"form_{stat_name}_per90_ewm",
            f"form_{stat_name}_trend_1_5",
        ]
    )
OUTPUT_COLUMNS.extend(["form_rating_mean_1", "form_rating_mean_3", "form_rating_ewm", "form_rating_trend_1_5", "form_rating_std_5"])
for stat_name in ADJUSTED_TARGETS:
    OUTPUT_COLUMNS.extend(
        [
            f"form_adjusted_{stat_name}_mean_1",
            f"form_adjusted_{stat_name}_mean_3",
            f"form_adjusted_{stat_name}_mean_5",
            f"form_adjusted_{stat_name}_ewm",
            f"form_adjusted_{stat_name}_trend_1_5",
            f"form_adjusted_{stat_name}_std_5",
        ]
    )


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--match-dataset",
        type=Path,
        default=DEFAULT_MATCH_DATASET,
        help=f"Clean match table from step 2 (default: {DEFAULT_MATCH_DATASET})",
    )
    parser.add_argument(
        "--team-strength",
        type=Path,
        default=DEFAULT_TEAM_STRENGTH,
        help=f"Pre-match team-strength table (default: {DEFAULT_TEAM_STRENGTH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Processed output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def rolling_previous_sum(history: pd.DataFrame, column: str, window: int = ROLLING_APPEARANCES) -> pd.Series:
    
    return history.groupby("player_id", sort=False)[column].transform(
        lambda values: values.shift(1).rolling(
            window,
            min_periods=1,
        ).sum()
    ).fillna(0.0)


def rolling_previous_observed_sum(history: pd.DataFrame, column: str, window: int = ROLLING_APPEARANCES) -> pd.Series:
    # Sum only observed values while retaining a separate coverage count
    return history.groupby("player_id", sort=False)[column].transform(lambda values: values.shift(1).rolling(window, min_periods=1).sum()).fillna(0.0)


def rolling_previous_mean(history: pd.DataFrame, column: str, window: int = ROLLING_APPEARANCES) -> pd.Series:
    
    return history.groupby("player_id", sort=False)[column].transform(
        lambda values: values.shift(1).rolling(
            window,
            min_periods=1,
        ).mean()
    ).fillna(0.0)


def rolling_previous_observed_count(history: pd.DataFrame, column: str) -> pd.Series:
    # Count genuine observations, including observed zero values
    return history.groupby("player_id", sort=False)[column].transform(lambda values: values.shift(1).rolling(ROLLING_APPEARANCES, min_periods=1).count()).fillna(0).astype(int)


def rolling_previous_std(history: pd.DataFrame, column: str, window: int = ROLLING_APPEARANCES) -> pd.Series:
    # Population standard deviation describes consistency; a one-match history
    # has zero observed dispersion rather than an undefined sample statistic.
    return history.groupby("player_id", sort=False)[column].transform(lambda values: values.shift(1).rolling(window, min_periods=1).std(ddof=0)).fillna(0.0)


def previous_ewm(history: pd.DataFrame, column: str) -> pd.Series:
    # EWM is shifted first, so the current appearance receives zero weight in
    # its own feature; half-life two makes recent appearances matter more.
    return history.groupby("player_id", sort=False)[column].transform(lambda values: values.shift(1).ewm(halflife=EWM_HALFLIFE_APPEARANCES, adjust=False).mean()).fillna(0.0)


def load_history(player_stats_path: Path) -> pd.DataFrame:
    
    if not player_stats_path.exists():
        raise FileNotFoundError(f"Player-stat source does not exist: {player_stats_path}")

    players = pd.read_csv(player_stats_path, dtype="string", keep_default_na=True)
    require_columns(players, REQUIRED_PLAYER_COLUMNS, player_stats_path)
    players["played"] = as_bool(players["played"])
    players["started"] = as_bool(players["started"])
    players = players[players["played"] & players["player_id"].notna()].copy()
    if players.empty:
        raise ValueError("No played player appearances were found")

    numeric_columns = [
        "minutes_played",
        "rating",
        "defending.tackles",
        "defending.interceptions",
        *EVENT_STATS,
    ]
    for column in numeric_columns:
        players[column] = pd.to_numeric(players[column], errors="coerce")

    event_columns = [
        "minutes_played",
        "defending.tackles",
        "defending.interceptions",
        *[column for column in EVENT_STATS if column != NPXG_FIELD]
    ]
    players[event_columns] = players[event_columns].fillna(0.0)
    players["started_numeric"] = players["started"].astype(int)
    players["defensive_actions"] = (
        players["defending.tackles"] + players["defending.interceptions"]
    )
    players["utc_datetime"] = pd.to_datetime(players["utc_date"], utc=True, errors="raise")
    players = players.sort_values(
        ["player_id", "utc_datetime", "match_id"],
        kind="stable",
    ).reset_index(drop=True)
    return players


def add_opponent_adjusted_performance(history: pd.DataFrame, team_strength: pd.DataFrame) -> pd.DataFrame:
    """Residualize each appearance against context estimated from earlier rows."""
    required_strength = {"match_id", "team_id", "elo_rating", "opponent_elo_rating"}
    missing = sorted(required_strength.difference(team_strength.columns))
    if missing:
        raise ValueError(f"Team-strength table is missing: {', '.join(missing)}")

    form = history.copy()
    strengths = team_strength[list(required_strength)].copy()
    strengths[["match_id", "team_id"]] = strengths[["match_id", "team_id"]].astype("string")
    form[["match_id", "team_id"]] = form[["match_id", "team_id"]].astype("string")
    form = form.merge(strengths, on=["match_id", "team_id"], how="left", validate="many_to_one")
    form["elo_rating"] = pd.to_numeric(form["elo_rating"], errors="coerce").fillna(1500.0)
    form["opponent_elo_rating"] = pd.to_numeric(form["opponent_elo_rating"], errors="coerce").fillna(1500.0)

    # X contains an intercept, venue, relative team strength, and position. A
    # ridge system is updated after each kickoff batch: beta=(X'WX+lambda I)^-1 X'Wy.
    position_order = ("D", "F", "G", "M")
    predictor_count = 3 + len(position_order)
    cross_products = {name: OPPONENT_RIDGE_STRENGTH * np.eye(predictor_count) for name in ADJUSTED_TARGETS}
    target_products = {name: np.zeros(predictor_count) for name in ADJUSTED_TARGETS}
    form = form.sort_values(["utc_datetime", "match_id", "player_id"], kind="stable")

    for _, batch in form.groupby("utc_datetime", sort=True):
        indexes = batch.index
        is_home = batch["team_id"].eq(batch["home_team_id"]).astype(float).to_numpy()
        relative_elo = ((batch["elo_rating"] - batch["opponent_elo_rating"]) / 400.0).to_numpy()
        position_values = batch["position"].astype("string")
        design = np.column_stack(
            [
                np.ones(len(batch)),
                is_home,
                relative_elo,
                *[position_values.eq(position).astype(float).to_numpy() for position in position_order],
            ]
        )
        update_payload: list[tuple[str, np.ndarray, np.ndarray]] = []
        minutes = batch["minutes_played"].to_numpy(dtype=float)
        exposure = np.clip(minutes / 90.0, 0.0, 1.0)

        for name, (source, per90, baseline) in ADJUSTED_TARGETS.items():
            beta = np.linalg.solve(cross_products[name], target_products[name])
            expected = baseline + design @ beta
            if per90:
                expected = np.clip(expected, 0.0, None)
                actual = np.divide(90.0 * batch[source].to_numpy(dtype=float), minutes, out=np.zeros(len(batch)), where=minutes > 0)
            else:
                actual = batch[source].fillna(baseline).to_numpy(dtype=float)
            residual = actual - expected
            form.loc[indexes, f"adjusted_{name}"] = residual

            valid = batch[source].notna().to_numpy()
            weights = exposure * valid
            centered_target = actual - baseline
            update_payload.append((name, weights, centered_target))

        # All players at this timestamp were predicted from the same historical
        # state. Only now are their observed performances added to the ridge fit.
        for name, weights, centered_target in update_payload:
            cross_products[name] += design.T @ (weights[:, None] * design)
            target_products[name] += design.T @ (weights * centered_target)

    return form.sort_values(["player_id", "utc_datetime", "match_id"], kind="stable").reset_index(drop=True)


def add_rolling_form(history: pd.DataFrame) -> pd.DataFrame:
    
    form = history.copy()
    player_groups = form.groupby("player_id", sort=False)
    form["prior_appearances"] = player_groups.cumcount()
    form["form_window_appearances_5"] = form["prior_appearances"].clip(
        upper=ROLLING_APPEARANCES
    )
    form["has_prior_history"] = form["prior_appearances"] > 0

    form["previous_appearance_datetime"] = player_groups["utc_datetime"].shift(1)
    form["previous_appearance_utc_date"] = form[
        "previous_appearance_datetime"
    ].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    form["days_since_previous_appearance"] = (
        form["utc_datetime"] - form["previous_appearance_datetime"]
    ).dt.total_seconds().div(86_400).fillna(0.0)

    form["form_minutes_5"] = rolling_previous_sum(form, "minutes_played")
    form["form_starts_5"] = rolling_previous_sum(form, "started_numeric").astype(int)
    form["form_rating_mean_5"] = rolling_previous_mean(form, "rating")

    # Process npxG separately because it has nullable provider coverage
    rolling_stats: dict[str, pd.Series] = {
        output_name: rolling_previous_sum(form, source_name)
        for source_name, output_name in EVENT_STATS.items() if source_name != NPXG_FIELD
    }
    rolling_stats["defensive_actions"] = rolling_previous_sum(form, "defensive_actions")

    has_minutes = form["form_minutes_5"] > 0
    for feature_name, rolling_total in rolling_stats.items():
        output_column = f"form_{feature_name}_per90_5"
        form[output_column] = 0.0
        form.loc[has_minutes, output_column] = 90.0 * rolling_total.loc[has_minutes] / form.loc[has_minutes, "form_minutes_5"]
    
    # Calculate npxG using only minutes belonging to observed npxG appearances
    form["npxg_observed_minutes"] = form["minutes_played"].where(form[NPXG_FIELD].notna())
    form["form_npxg_sum_5"] = rolling_previous_observed_sum(form, NPXG_FIELD)
    form["form_npxg_minutes_5"] = rolling_previous_observed_sum(form, "npxg_observed_minutes")
    form["form_npxg_observations_5"] = rolling_previous_observed_count(form, NPXG_FIELD)
    valid_npxg = form["form_npxg_observations_5"].ge(NPXG_MIN_VALID_APPEARANCES) & form["form_npxg_minutes_5"].ge(NPXG_MIN_VALID_MINUTES)
    form["form_npxg_per90_5"] = float("nan")
    form.loc[valid_npxg, "form_npxg_per90_5"] = 90.0 * form.loc[valid_npxg, "form_npxg_sum_5"] / form.loc[valid_npxg, "form_npxg_minutes_5"]

    # Multiple horizons preserve recency shape. Per-90 rates use total events /
    # total minutes in each shifted window, avoiding an average of noisy rates.
    recency_sources = {
        "shots": "shooting.total_shots",
        "key_passes": "passing.key_passes",
        "defensive_actions": "defensive_actions",
    }
    for window in FORM_WINDOWS:
        minutes_column = f"_form_minutes_{window}"
        form[minutes_column] = rolling_previous_sum(form, "minutes_played", window)
        form[f"form_rating_mean_{window}"] = rolling_previous_mean(form, "rating", window)
        for stat_name, source in recency_sources.items():
            total = rolling_previous_sum(form, source, window)
            output = f"form_{stat_name}_per90_{window}"
            form[output] = 0.0
            valid_minutes = form[minutes_column].gt(0)
            form.loc[valid_minutes, output] = 90.0 * total.loc[valid_minutes] / form.loc[valid_minutes, minutes_column]

    form["form_rating_ewm"] = previous_ewm(form, "rating")
    form["form_rating_trend_1_5"] = form["form_rating_mean_1"] - form["form_rating_mean_5"]
    form["form_rating_std_5"] = rolling_previous_std(form, "rating")
    for stat_name, source in recency_sources.items():
        appearance_rate = f"_appearance_{stat_name}_per90"
        form[appearance_rate] = np.divide(90.0 * form[source], form["minutes_played"], out=np.zeros(len(form)), where=form["minutes_played"].gt(0))
        form[f"form_{stat_name}_per90_ewm"] = previous_ewm(form, appearance_rate)
        form[f"form_{stat_name}_trend_1_5"] = form[f"form_{stat_name}_per90_1"] - form[f"form_{stat_name}_per90_5"]

    # Opponent-adjusted values are residuals. Event residuals are averaged with
    # minute exposure weights; rating residuals use an ordinary appearance mean.
    for stat_name, (_, per90, _) in ADJUSTED_TARGETS.items():
        source = f"adjusted_{stat_name}"
        if source not in form:
            form[source] = 0.0
        weighted_source = f"_weighted_{source}"
        form[weighted_source] = form[source] * form["minutes_played"]
        for window in FORM_WINDOWS:
            if per90:
                weighted_total = rolling_previous_sum(form, weighted_source, window)
                denominator = form[f"_form_minutes_{window}"]
                output = np.divide(weighted_total, denominator, out=np.zeros(len(form)), where=denominator.gt(0))
                form[f"form_adjusted_{stat_name}_mean_{window}"] = output
            else:
                form[f"form_adjusted_{stat_name}_mean_{window}"] = rolling_previous_mean(form, source, window)
        form[f"form_adjusted_{stat_name}_ewm"] = previous_ewm(form, source)
        form[f"form_adjusted_{stat_name}_trend_1_5"] = form[f"form_adjusted_{stat_name}_mean_1"] - form[f"form_adjusted_{stat_name}_mean_5"]
        form[f"form_adjusted_{stat_name}_std_5"] = rolling_previous_std(form, source)
    
    return form


def select_target_starters(form: pd.DataFrame, match_dataset_path: Path) -> pd.DataFrame:
    
    if not match_dataset_path.exists():
        raise FileNotFoundError(
            f"Clean match dataset does not exist: {match_dataset_path}. Run step 2 first."
        )
    matches = pd.read_csv(match_dataset_path, dtype="string", keep_default_na=True)
    if "match_id" not in matches.columns:
        raise ValueError(f"{match_dataset_path} has no match_id column")
    if matches["match_id"].duplicated().any():
        raise ValueError(f"{match_dataset_path} contains duplicate match IDs")

    target_ids = set(matches["match_id"])
    starters = form[form["match_id"].isin(target_ids) & form["started"]].copy()
    starters["team_side"] = ""
    starters.loc[starters["team_id"].eq(starters["home_team_id"]), "team_side"] = "home"
    starters.loc[starters["team_id"].eq(starters["away_team_id"]), "team_side"] = "away"
    if starters["team_side"].eq("").any():
        bad = starters.loc[starters["team_side"].eq(""), ["match_id", "player_id", "team_id"]]
        raise ValueError(f"Starter team did not match home or away team:\n{bad.head(10)}")

    starters = starters[OUTPUT_COLUMNS].sort_values(
        ["utc_date", "match_id", "team_side", "player_id"],
        kind="stable",
    ).reset_index(drop=True)
    validate_starter_form(starters, expected_match_ids=target_ids)
    return starters


def validate_starter_form(starters: pd.DataFrame, expected_match_ids: set[str]) -> None:
    
    if set(starters["match_id"]) != expected_match_ids:
        missing = sorted(expected_match_ids.difference(starters["match_id"]))
        raise ValueError(f"Player form is missing target matches: {missing[:10]}")
    
    if starters.duplicated(["match_id", "player_id"]).any():
        raise ValueError("Player form contains duplicate match/player rows")

    per_match = starters.groupby("match_id").size()
    
    if not per_match.eq(22).all():
        bad = per_match[~per_match.eq(22)]
        raise ValueError(f"Target matches without exactly 22 starter rows:\n{bad.head(10)}")
    per_side = starters.groupby(["match_id", "team_side"]).size()
    
    if not per_side.eq(11).all():
        bad = per_side[~per_side.eq(11)]
        raise ValueError(f"Target match sides without exactly 11 starters:\n{bad.head(10)}")

    if starters["form_window_appearances_5"].gt(ROLLING_APPEARANCES).any():
        raise ValueError("Rolling appearance count exceeds the five-appearance window")
    
    if starters["form_window_appearances_5"].gt(starters["prior_appearances"]).any():
        raise ValueError("Rolling appearance count exceeds total prior appearances")
    
    if starters["days_since_previous_appearance"].lt(0).any():
        raise ValueError("A previous appearance occurs after the target match")

    no_history = ~starters["has_prior_history"]
    zero_history_columns = [
        "form_window_appearances_5",
        "form_minutes_5",
        "form_starts_5",
        "form_rating_mean_5",
        "form_goals_per90_5",
        "form_assists_per90_5",
        "form_npxg_observations_5",
        "form_npxg_minutes_5",
        "form_shots_per90_5",
        "form_key_passes_per90_5",
        "form_defensive_actions_per90_5",
        "form_saves_per90_5",
    ]
    
    if not starters.loc[no_history, zero_history_columns].eq(0).all().all():
        raise ValueError("A first appearance contains non-zero historical form")

    # A player without prior history must have unknown rather than zero npxG form
    if starters.loc[no_history, "form_npxg_per90_5"].notna().any(): raise ValueError("A first appearance contains a populated historical npxG rate")
    

def build_coverage(starters: pd.DataFrame) -> pd.DataFrame:
    
    rows: list[dict[str, object]] = []
    for season, group in starters.groupby("season", sort=True):
        starter_rows = len(group)
        any_history = group["has_prior_history"].sum()
        full_window = group["form_window_appearances_5"].eq(ROLLING_APPEARANCES).sum()
        prior_minutes = group["form_minutes_5"].gt(0).sum()
        rows.append(
            {
                "season": season,
                "target_matches": group["match_id"].nunique(),
                "starter_rows": starter_rows,
                "starters_with_any_prior_history": any_history,
                "any_prior_history_coverage": any_history / starter_rows,
                "starters_with_full_five_appearance_window": full_window,
                "full_five_appearance_window_coverage": full_window / starter_rows,
                "starters_with_prior_minutes": prior_minutes,
                "prior_minutes_coverage": prior_minutes / starter_rows,
                "mean_prior_appearances": group["prior_appearances"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_outputs(player_stats_path: Path, match_dataset_path: Path, team_strength_path: Path, output_dir: Path) -> tuple[Path, Path]:
    
    history = load_history(player_stats_path)
    if not team_strength_path.exists():
        raise FileNotFoundError(f"Team-strength source does not exist: {team_strength_path}")
    team_strength = pd.read_csv(team_strength_path, dtype={"match_id": "string", "team_id": "string"})
    history = add_opponent_adjusted_performance(history, team_strength)
    form = add_rolling_form(history)
    starters = select_target_starters(form, match_dataset_path)
    coverage = build_coverage(starters)

    output_dir.mkdir(parents=True, exist_ok=True)
    player_form_path = output_dir / PLAYER_FORM_NAME
    coverage_path = output_dir / PLAYER_FORM_COVERAGE_NAME
    write_csv_atomic(starters, player_form_path)
    write_csv_atomic(coverage, coverage_path)

    print(coverage.to_string(index=False))
    print(f"\nSaved {len(starters)} starter-form rows to {player_form_path}")
    print(f"Coverage report: {coverage_path}")
    return player_form_path, coverage_path


def main() -> None:
    
    args = parse_args()
    match_dataset_path = resolve_project_path(args.match_dataset)
    team_strength_path = resolve_project_path(args.team_strength)
    output_dir = resolve_project_path(args.output_dir)
    build_outputs(PLAYER_STATS_CSV, match_dataset_path, team_strength_path, output_dir)


if __name__ == "__main__":
    main()
