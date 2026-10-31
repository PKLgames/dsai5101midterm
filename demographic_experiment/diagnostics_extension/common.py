"""Shared fixed-input validation and I/O for retrospective diagnostic extensions."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parent
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(ROOT))

from experiment import digest, verify_freeze, verify_original, write_csv, write_json  # noqa: E402


def verify_inputs() -> dict[str, Any]:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    freeze = verify_freeze()
    verify_original()
    evaluation = json.loads((EXPERIMENT / "evaluation_provenance.json").read_text(encoding="utf-8"))
    if evaluation["freeze_sha256"] != digest(EXPERIMENT / "freeze.json"):
        raise ValueError("Demographic experiment evaluation freeze mismatch.")
    for name, expected in evaluation["outputs_sha256"].items():
        if digest(EXPERIMENT / name) != expected:
            raise ValueError(f"Demographic experiment output mismatch: {name}")
    control = json.loads((EXPERIMENT / "control_provenance.json").read_text(encoding="utf-8"))
    if (control["status"] != "passed_operational_control"
            or control["freeze_sha256"] != digest(EXPERIMENT / "freeze.json")
            or control["results_sha256"] != digest(EXPERIMENT / "control_results.csv")
            or control["source_sha256"] != digest(EXPERIMENT / "control.py")):
        raise ValueError("Demographic operational control provenance mismatch.")
    return freeze


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))
