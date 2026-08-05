"""Shared constants for feature building and baseline modeling."""

from __future__ import annotations

CLASS_ORDER = ("H", "D", "A")
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_ORDER)}

DEFAULT_TRAIN_END = "2026-01-31"
DEFAULT_CALIBRATION_END = "2026-03-31"
DEFAULT_TEST_END = "2026-05-21"

DEFAULT_MIN_HISTORY = 5
DEFAULT_L5_WINDOW = 5
DEFAULT_VENUE_L3_WINDOW = 3

PROB_SUM_TOL = 1e-12
