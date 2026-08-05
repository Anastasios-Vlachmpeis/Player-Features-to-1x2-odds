import json
import subprocess
import sys


def test_help_works():
    proc = subprocess.run(
        [sys.executable, "-m", "superleague_baseline", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "audit" in proc.stdout


def test_train_requires_label_source():
    proc = subprocess.run(
        [sys.executable, "-m", "superleague_baseline", "train-evaluate"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
