"""CLI entry points."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

from superleague_baseline.audit import run_audit
from superleague_baseline.constants import (
    DEFAULT_CALIBRATION_END,
    DEFAULT_MIN_HISTORY,
    DEFAULT_TEST_END,
    DEFAULT_TRAIN_END,
)
from superleague_baseline.features.dataset import (
    build_historical_match_dataset,
    feature_columns,
)
from superleague_baseline.features.targets import require_label_source
from superleague_baseline.modeling.train import train_and_evaluate
from superleague_baseline.splits import assign_partition


def _default_db() -> Path:
    return Path(__file__).resolve().parent.parent / "player_stats.db"


def cmd_audit(args: argparse.Namespace) -> int:
    report = run_audit(args.db)
    print(json.dumps(report, indent=2))
    return 0


def cmd_build_features(args: argparse.Namespace) -> int:
    dataset = build_historical_match_dataset(args.db, min_history=args.min_history)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out, index=False)
    print(f"Wrote {len(dataset)} rows to {out}")
    return 0


def cmd_train_evaluate(args: argparse.Namespace) -> int:
    require_label_source(args.label_source)
    warnings.warn(
        "Training on player-goals proxy labels is exploratory only; "
        "metrics are not production-reliable until official scores exist.",
        stacklevel=1,
    )
    dataset = build_historical_match_dataset(args.db, min_history=args.min_history)
    partition = assign_partition(
        dataset,
        train_end=args.train_end,
        calibration_end=args.calibration_end,
        test_end=args.test_end,
    )
    cols = feature_columns(dataset)
    result = train_and_evaluate(dataset, cols, partition=partition, seed=args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.csv"
    metrics_path = out_dir / "metrics.json"
    result["predictions"].to_csv(pred_path, index=False)
    metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Wrote {pred_path}")
    print(f"Wrote {metrics_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="superleague-baseline")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Validate database contract")
    audit.add_argument("--db", type=Path, default=_default_db())
    audit.set_defaults(func=cmd_audit)

    build = sub.add_parser("build-features", help="Build leakage-safe match feature table")
    build.add_argument("--db", type=Path, default=_default_db())
    build.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/match_features.csv"),
    )
    build.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    build.set_defaults(func=cmd_build_features)

    train = sub.add_parser("train-evaluate", help="Train calibrated baseline and evaluate")
    train.add_argument("--db", type=Path, default=_default_db())
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline_run"))
    train.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    train.add_argument("--calibration-end", default=DEFAULT_CALIBRATION_END)
    train.add_argument("--test-end", default=DEFAULT_TEST_END)
    train.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    train.add_argument("--seed", type=int, default=20260805)
    train.add_argument("--label-source", required=True)
    train.set_defaults(func=cmd_train_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
