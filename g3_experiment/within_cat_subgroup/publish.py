"""Publish subgroup evidence by copying, while preserving all prior records."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from analyze import EXCLUDED_ROOT, G3, HERE, ROOT, digest, snapshot, verify_inputs, write_csv, write_json

sys.path.insert(0, str(G3))
from publish_evidence import tree_lines  # noqa: E402


def publish() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    evidence = ROOT / "Presentation_Evidence"
    stage = Path("By_Stage") / "12_Retrospective_G3_Experiment" / "Within_Cat_Subgroup"
    flow = Path("By_Presentation_Flow") / "03_Diagnostics" / "06_Retrospective_G3" / "Within_Cat_Subgroup"
    manifest = evidence / "g3_within_cat_selection_manifest.csv"
    verification_file = evidence / "g3_within_cat_verification.json"
    if any(p.exists() for p in (evidence / stage, evidence / flow, manifest, verification_file,
                                 HERE / "publication_verification.json")):
        raise FileExistsError("Within-cat evidence publication already started; refusing to overwrite.")
    originals = json.loads((HERE / "original_hashes_before.json").read_text(encoding="utf-8"))
    prior_evidence = json.loads((HERE / "evidence_hashes_before.json").read_text(encoding="utf-8"))
    if snapshot(ROOT, EXCLUDED_ROOT, HERE) != originals:
        raise RuntimeError("Original project files changed since the subgroup analysis started.")
    if snapshot(evidence, set()) != prior_evidence:
        raise RuntimeError("Previous presentation evidence changed since the subgroup analysis started.")
    verify_inputs()
    provenance = json.loads((HERE / "analysis_provenance.json").read_text(encoding="utf-8"))
    for name, expected in provenance["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Within-cat source hash mismatch: {name}")
    for name, expected in provenance["output_sha256"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"Within-cat output hash mismatch: {name}")
    previous = json.loads((evidence / "g3_verification.json").read_text(encoding="utf-8"))
    if digest(evidence / "g3_selection_manifest.csv") != previous["g3_selection_manifest_sha256"]:
        raise ValueError("Earlier G3 evidence manifest no longer matches its verification.")
    roles = {
        "presentation_within_cat_g3.pdf": "Within-cat frozen-model comparison and paired exploratory intervals.",
        "within_cat_test_results.csv": "Within-cat log loss, Brier score, accuracy and support by model.",
        "within_cat_contrasts.csv": "Paired within-cat log-loss differences and held-out-photo intervals.",
        "within_cat_summary.json": "Subgroup support, prior outcome exposure and M1 interpretation.",
        "protocol.json": "Saved-prediction-only analysis rules, support thresholds and limitations.",
        "analysis_provenance.json": "Source hashes, output hashes and frozen-artifact safeguards.",
    }
    source_hashes = {name: digest(HERE / name) for name in roles}
    for directory in (stage, flow):
        (evidence / directory).mkdir(parents=False, exist_ok=False)
    rows = []
    for name, role in roles.items():
        source = HERE / name
        for directory in (stage, flow):
            target = evidence / directory / name
            shutil.copy2(source, target)
            if digest(target) != source_hashes[name]:
                raise RuntimeError(f"Evidence copy hash mismatch: {target}")
        rows.append({"source": str(source.relative_to(ROOT)), "by_stage_copy": str(stage / name),
                     "by_flow_copy": str(flow / name), "bytes": source.stat().st_size,
                     "sha256": source_hashes[name], "presentation_role": role})
    if snapshot(ROOT, EXCLUDED_ROOT, HERE) != originals:
        raise RuntimeError("Original project files changed during evidence copying.")
    for name, record in prior_evidence.items():
        if digest(evidence / name) != record["sha256"]:
            raise RuntimeError(f"Previous evidence changed during copying: {name}")
    if any(digest(HERE / name) != expected for name, expected in source_hashes.items()):
        raise RuntimeError("A subgroup source changed during evidence copying.")
    write_csv(manifest, rows, ["source", "by_stage_copy", "by_flow_copy", "bytes", "sha256", "presentation_role"])
    report = {
        "status": "all_checks_passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "Copied within-cat G3 evidence to both views; original sources were not moved or edited.",
        "original_project_files_unchanged": len(originals),
        "previous_evidence_files_unchanged_before_tree_refresh": len(prior_evidence),
        "subgroup_sources_verified_unchanged": len(rows), "subgroup_copies_verified": len(rows) * 2,
        "selection_manifest_sha256": digest(manifest),
        "g2_freeze_sha256": digest(ROOT / "artifacts" / "model_freeze.json"),
        "g3_freeze_sha256": digest(G3 / "freeze.json"),
        "scope_exclusions": sorted(EXCLUDED_ROOT) + [str(HERE.relative_to(ROOT))],
        "note": "The original, abstention and G3 evidence verification records remain historical and unchanged. The tree alone is refreshed. G2/G3 claims, reports and frozen evaluation outputs are not updated.",
        "copies": [{"source": row["source"], "copy": row[key], "sha256": row["sha256"]}
                   for row in rows for key in ("by_stage_copy", "by_flow_copy")],
    }
    write_json(verification_file, report)
    write_json(HERE / "publication_verification.json", report)
    tree = "Presentation_Evidence\\\n" + "\n".join(tree_lines(evidence)) + "\n"
    (evidence / "folder_tree.txt").write_text(tree, encoding="utf-8")
    print(f"Verified {len(originals)} original project files, {len(prior_evidence)} prior evidence files, "
          f"and {2 * len(rows)} new copies.")
    print(f"Updated tree: {evidence / 'folder_tree.txt'}")


if __name__ == "__main__":
    publish()
