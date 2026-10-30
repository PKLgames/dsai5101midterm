"""Copy selected G3 outputs into presentation evidence and verify isolation."""

from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from prepare import EXPERIMENT, ROOT, digest, snapshot, write_csv, write_json


def tree_lines(directory: Path, prefix: str = "") -> list[str]:
    children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold()))
    lines = []
    for index, path in enumerate(children):
        last = index == len(children) - 1
        lines.append(prefix + ("`-- " if last else "|-- ") + path.name + ("\\" if path.is_dir() else ""))
        if path.is_dir():
            lines.extend(tree_lines(path, prefix + ("    " if last else "|   ")))
    return lines


def publish() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    evidence = ROOT / "Presentation_Evidence"
    stage = Path("By_Stage") / "12_Retrospective_G3_Experiment"
    flow = Path("By_Presentation_Flow") / "03_Diagnostics" / "06_Retrospective_G3"
    metadata = [evidence / "g3_selection_manifest.csv", evidence / "g3_verification.json",
                EXPERIMENT / "quarantine_verification.json"]
    if any(path.exists() for path in metadata) or (evidence / stage).exists() or (evidence / flow).exists():
        raise FileExistsError("G3 evidence publication already started; refusing to overwrite.")
    before = json.loads((EXPERIMENT / "root_hashes_before.json").read_text(encoding="utf-8"))
    evidence_before = json.loads((EXPERIMENT / "evidence_hashes_before.json").read_text(encoding="utf-8"))
    exclusions = {".git", ".venv", ".cache", "Presentation_Evidence", "g3_experiment"}
    current = snapshot(ROOT, exclusions)
    if before != current:
        changed = [p for p in sorted(set(before) | set(current)) if before.get(p) != current.get(p)]
        raise RuntimeError("Project originals changed since G3 preparation: " + repr(changed[:20]))
    if snapshot(evidence, set()) != evidence_before:
        raise RuntimeError("Previous presentation evidence changed since G3 preparation.")
    provenance = json.loads((EXPERIMENT / "evaluation_provenance.json").read_text(encoding="utf-8"))
    if provenance["status"] != "completed_once_post_freeze_retrospective_exploratory":
        raise ValueError("G3 evaluation has not completed.")
    for name, expected in provenance["output_sha256"].items():
        if digest(EXPERIMENT / name) != expected:
            raise ValueError(f"G3 output hash mismatch: {name}")
    roles = {
        "presentation_g3_results.pdf": "Exploratory known-cat G3 predictive results and uncertainty.",
        "test_results.csv": "Complete matched test metrics by model.",
        "test_contrasts.csv": "Descriptive paired model contrasts and photo-cluster intervals.",
        "validation_results.csv": "Validation-only model selection results.",
        "coverage.json": "Outcome-blind reservation coverage and eligibility gate.",
        "protocol.json": "Retrospective scope, fixed selection rules, and nonclaims.",
        "evaluation_provenance.json": "Experiment freeze, test access, output hashes, and interpretation limits.",
    }
    source_hashes = {name: digest(EXPERIMENT / name) for name in roles}
    for directory in (stage, flow):
        (evidence / directory).mkdir(parents=True, exist_ok=False)
    rows = []
    for name, role in roles.items():
        source = EXPERIMENT / name
        for directory in (stage, flow):
            target = evidence / directory / name
            shutil.copy2(source, target)
            if digest(target) != source_hashes[name]:
                raise RuntimeError(f"Evidence copy hash mismatch: {target}")
        rows.append({"source": str(Path("g3_experiment") / name), "by_stage_copy": str(stage / name),
                     "by_flow_copy": str(flow / name), "bytes": source.stat().st_size,
                     "sha256": source_hashes[name], "presentation_role": role})
    if snapshot(ROOT, exclusions) != before:
        raise RuntimeError("Project originals changed during evidence copying.")
    for name, record in evidence_before.items():
        if digest(evidence / name) != record["sha256"]:
            raise RuntimeError(f"Previous evidence changed during copying: {name}")
    for name, expected in source_hashes.items():
        if digest(EXPERIMENT / name) != expected:
            raise RuntimeError(f"G3 source changed during copying: {name}")
    manifest = evidence / "g3_selection_manifest.csv"
    write_csv(manifest, rows, ["source", "by_stage_copy", "by_flow_copy", "bytes", "sha256", "presentation_role"])
    verification = {
        "status": "all_checks_passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "Copied G3 presentation evidence; originals were not moved or changed.",
        "original_project_files_verified_unchanged": len(before),
        "previous_evidence_files_verified_unchanged_before_tree_refresh": len(evidence_before),
        "g3_sources_verified_unchanged": len(rows), "g3_copies_verified": 2 * len(rows),
        "g3_selection_manifest_sha256": digest(manifest),
        "g3_freeze_sha256": digest(EXPERIMENT / "freeze.json"),
        "g2_freeze_sha256": digest(ROOT / "artifacts" / "model_freeze.json"),
        "original_scope_exclusions": sorted(exclusions),
        "note": "The original and abstention evidence manifests are historical and remain unchanged. The folder tree is refreshed after this check. This retrospective G3 experiment does not revise the G2 report, ZIP, or claims.",
        "copies": [{"source": row["source"], "copy": row[key], "sha256": row["sha256"]}
                   for row in rows for key in ("by_stage_copy", "by_flow_copy")],
    }
    write_json(evidence / "g3_verification.json", verification)
    write_json(EXPERIMENT / "quarantine_verification.json", verification)
    tree = "Presentation_Evidence\\\n" + "\n".join(tree_lines(evidence)) + "\n"
    (evidence / "folder_tree.txt").write_text(tree, encoding="utf-8")
    print(tree)
    print(f"Verified {len(before)} original project files, {len(evidence_before)} prior evidence files, "
          f"and {2 * len(rows)} new copies.")


if __name__ == "__main__":
    publish()
