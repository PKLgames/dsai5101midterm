"""Repeat a check in a temporary working copy, preserving recorded logs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


COPY_DIRECTORIES = ("src", "tests", "legacy", "features", "models", "tables", "artifacts", "figures")
COPY_FILES = (
    "decisions.yaml", "protocol.md", "project-plan-source.txt", "claims.csv", "claims_final.csv",
    "contributions.csv", "README.md", "references.csv", "requirements-stage2.txt",
    "requirements-stage3.txt", "requirements-stage4.txt", "requirements-stage5.txt",
    "requirements-stage7.txt", "requirements-stage11.txt", "requirements-y-gpu.txt",
)


def run_check(root: Path, output: Path, mode: str = "statistics") -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Choose a new reproduction log path; existing logs are preserved.")
    if mode not in {"statistics", "H", "C", "Y"}:
        raise ValueError("Unknown reproduction mode.")
    with tempfile.TemporaryDirectory(prefix="group19-reproduction-") as directory:
        working = Path(directory) / "project"
        working.mkdir()
        for name in COPY_DIRECTORIES:
            path = root / name
            if path.is_dir():
                shutil.copytree(path, working / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in COPY_FILES:
            if (root / name).is_file():
                shutil.copyfile(root / name, working / name)
        (working / "dataset").mkdir()
        for name in ("train.csv", "test.csv"):
            shutil.copyfile(root / "dataset" / name, working / "dataset" / name)
        if mode == "statistics":
            target = working / "artifacts/reproduction_statistical.json"
            target.unlink(missing_ok=True)
            command = [sys.executable, "-m", "src.reproduce"]
        else:
            if not (root / "dataset/images").is_dir():
                raise FileNotFoundError("Raw photographs are required for feature regeneration; see REPRODUCTION.md.")
            shutil.copytree(root / "dataset/images", working / "dataset/images")
            if (root / ".cache").is_dir():
                shutil.copytree(root / ".cache", working / ".cache")
            target = working / "artifacts" / f"feature_regeneration_{mode}.json"
            target.unlink(missing_ok=True)
            command = [sys.executable, "-m", "src.feature_regeneration", "--block", mode]
        result = subprocess.run(command, cwd=working, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Fresh working-copy reproduction failed with exit code {result.returncode}.")
        report = json.loads(target.read_text(encoding="utf-8"))
        if report["status"] != "passed":
            raise ValueError("Fresh working-copy check did not pass.")
        report["fresh_temporary_working_copy"] = True
        report["requested_mode"] = mode
        report["original_logs_preserved"] = True
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["statistics", "H", "C", "Y"], default="statistics")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve()
    try:
        report = run_check(root, output, args.mode)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Fresh working-copy reproduction failed: {exc}", file=sys.stderr)
        return 2
    print(f"Fresh working-copy {report['requested_mode']} reproduction passed; recorded artifacts unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
