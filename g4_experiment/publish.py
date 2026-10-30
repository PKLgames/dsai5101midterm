"""Publish G4 evidence by copying, while preserving all previous records."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate import verify_freeze
from shared import HERE, ROOT, digest, snapshot, write_csv, write_json

sys.path.insert(0, str(ROOT / "g3_experiment"))
from publish_evidence import tree_lines  # noqa: E402


def publish() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    evidence = ROOT / "Presentation_Evidence"
    stage = Path("By_Stage") / "13_Retrospective_G4_Experiment"
    flow = Path("By_Presentation_Flow") / "03_Diagnostics" / "07_Retrospective_G4"
    manifest = evidence / "g4_selection_manifest.csv"
    verification_file = evidence / "g4_verification.json"
    if any(p.exists() for p in (evidence / stage, evidence / flow, manifest, verification_file,
                                 HERE / "publication_verification.json")):
        raise FileExistsError("G4 evidence publication already started; refusing to overwrite.")
    exclusions = {".git", ".venv", ".cache", "Presentation_Evidence", "g4_experiment"}
    originals = json.loads((HERE / "root_hashes_before.json").read_text(encoding="utf-8"))
    prior_evidence = json.loads((HERE / "evidence_hashes_before.json").read_text(encoding="utf-8"))
    if snapshot(ROOT, exclusions) != originals:
        raise RuntimeError("Original project or G3 files changed since the G4 reservation.")
    if snapshot(evidence, set()) != prior_evidence:
        raise RuntimeError("Previous presentation evidence changed since the G4 reservation.")
    verify_freeze()
    provenance = json.loads((HERE / "evaluation_provenance.json").read_text(encoding="utf-8"))
    if provenance["status"] != "completed_once_post_freeze_retrospective_exploratory":
        raise ValueError("G4 evaluation has not completed.")
    if provenance["freeze_sha256"] != digest(HERE / "freeze.json"):
        raise ValueError("G4 evaluation freeze mismatch.")
    if provenance["test_access_record_sha256"] != digest(HERE / "evaluation_started.json"):
        raise ValueError("G4 test access record hash mismatch.")
    for name, expected in provenance["output_sha256"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"G4 evaluation output hash mismatch: {name}")
    roles = {
        "presentation_g4_results.pdf": "Unseen-cat G4 predictive results and descriptive uncertainty.",
        "test_results.csv": "Complete matched G4 test metrics by model.",
        "test_contrasts.csv": "Paired G4 model contrasts and held-out-cat intervals.",
        "validation_results.csv": "G4 validation-only model selection results.",
        "coverage.json": "Outcome-blind cat reservation coverage and eligibility gate.",
        "protocol.json": "Retrospective scope, cold-start convention, fixed selection rules and nonclaims.",
        "evaluation_provenance.json": "G4 freeze, test access, output hashes and interpretation limits.",
    }
    source_hashes = {name: digest(HERE / name) for name in roles}
    for directory in (stage, flow):
        (evidence / directory).mkdir(parents=True, exist_ok=False)
    rows = []
    for name, role in roles.items():
        source = HERE / name
        for directory in (stage, flow):
            target = evidence / directory / name
            shutil.copy2(source, target)
            if digest(target) != source_hashes[name]:
                raise RuntimeError(f"G4 evidence copy hash mismatch: {target}")
        rows.append({"source": str(source.relative_to(ROOT)), "by_stage_copy": str(stage / name),
                     "by_flow_copy": str(flow / name), "bytes": source.stat().st_size,
                     "sha256": source_hashes[name], "presentation_role": role})
    if snapshot(ROOT, exclusions) != originals:
        raise RuntimeError("Original project or G3 files changed during G4 evidence copying.")
    for name, record in prior_evidence.items():
        if digest(evidence / name) != record["sha256"]:
            raise RuntimeError(f"Previous evidence changed during G4 copying: {name}")
    if any(digest(HERE / name) != expected for name, expected in source_hashes.items()):
        raise RuntimeError("A G4 evidence source changed during copying.")
    write_csv(manifest, rows, ["source", "by_stage_copy", "by_flow_copy", "bytes", "sha256", "presentation_role"])
    report = {
        "status": "all_checks_passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "Copied G4 evidence into both views; original sources were not moved or edited.",
        "original_project_files_including_g3_unchanged": len(originals),
        "previous_evidence_files_unchanged_before_tree_refresh": len(prior_evidence),
        "g4_sources_verified_unchanged": len(rows), "g4_copies_verified": len(rows) * 2,
        "selection_manifest_sha256": digest(manifest),
        "g2_freeze_sha256": digest(ROOT / "artifacts" / "model_freeze.json"),
        "g3_freeze_sha256": digest(ROOT / "g3_experiment" / "freeze.json"),
        "g4_freeze_sha256": digest(HERE / "freeze.json"),
        "scope_exclusions": sorted(exclusions),
        "note": "The original, abstention, G3 and within-cat evidence records remain historical and unchanged. The tree alone is refreshed. G2/G3 claims, reports and frozen evaluation outputs are not updated.",
        "copies": [{"source": row["source"], "copy": row[key], "sha256": row["sha256"]}
                   for row in rows for key in ("by_stage_copy", "by_flow_copy")],
    }
    write_json(verification_file, report)
    write_json(HERE / "publication_verification.json", report)
    tree = "Presentation_Evidence\\\n" + "\n".join(tree_lines(evidence)) + "\n"
    (evidence / "folder_tree.txt").write_text(tree, encoding="utf-8")
    print(f"Verified {len(originals)} original project files including G3, {len(prior_evidence)} prior evidence files, "
          f"and {2 * len(rows)} new copies.")
    print(f"Updated tree: {evidence / 'folder_tree.txt'}")


if __name__ == "__main__":
    publish()
