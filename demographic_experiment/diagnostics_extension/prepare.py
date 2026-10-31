"""Snapshot all pre-existing project files before the diagnostic extension."""

from __future__ import annotations

import os
from pathlib import Path

from common import HERE, ROOT, digest, verify_inputs, write_json


def protected_snapshot() -> dict[str, dict[str, object]]:
    entries = {}
    for directory, subdirs, names in os.walk(ROOT):
        base = Path(directory)
        if base == ROOT:
            subdirs[:] = [name for name in subdirs if name not in {".git", ".venv", ".cache"}]
        subdirs[:] = [name for name in subdirs if name not in {"__pycache__", "matplotlib_config"}
                      and base / name != HERE]
        for name in names:
            path = base / name
            if path.is_file():
                entries[str(path.relative_to(ROOT))] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return entries


def run() -> None:
    target = HERE / "original_hashes_before.json"
    if target.exists():
        raise FileExistsError("Diagnostic snapshot already exists.")
    verify_inputs()
    before = protected_snapshot()
    write_json(target, before)
    if protected_snapshot() != before:
        raise RuntimeError("Project originals changed while recording diagnostic snapshot.")
    print(f"Protected {len(before)} existing project files; original evaluation and demographic freezes verified.")


if __name__ == "__main__":
    run()
