"""Publish completed diagnostic evidence without rewriting historical manifests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from common import HERE, ROOT, digest, verify_inputs, write_csv, write_json
from prepare import protected_snapshot


def tree_lines(directory: Path, prefix: str = "") -> list[str]:
    children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold()))
    lines = []
    for i, path in enumerate(children):
        last = i == len(children)-1
        lines.append(prefix + ("`-- " if last else "|-- ") + path.name + ("\\" if path.is_dir() else ""))
        if path.is_dir():
            lines.extend(tree_lines(path, prefix + ("    " if last else "|   ")))
    return lines


def run() -> None:
    verify_inputs()
    evidence = ROOT / "Presentation_Evidence"
    stage = Path("By_Stage") / "14_Retrospective_All_Demographics" / "Full_Diagnostic_Parity"
    flow = Path("By_Presentation_Flow") / "03_Diagnostics" / "08_Retrospective_All_Demographics" / "Full_Diagnostic_Parity"
    manifest = evidence / "demographic_diagnostics_selection_manifest.csv"
    audit = evidence / "demographic_diagnostics_verification.json"
    if any(p.exists() for p in (evidence/stage, evidence/flow, manifest, audit, HERE/"publication_verification.json")):
        raise FileExistsError("Diagnostic publication already started.")
    before = json.loads((HERE / "original_hashes_before.json").read_text(encoding="utf-8"))
    if protected_snapshot() != before:
        raise RuntimeError("Pre-existing project files changed during diagnostic analysis.")
    summary = json.loads((HERE / "diagnostic_summary.json").read_text(encoding="utf-8"))
    verified = json.loads((HERE / "verification.json").read_text(encoding="utf-8"))
    if summary["status"] != "all_requested_additional_diagnostics_available" or verified["status"] != "independent_replay_passed":
        raise ValueError("Diagnostics and independent verification must both complete.")
    if verified["summary_sha256"] != digest(HERE / "diagnostic_summary.json"):
        raise ValueError("Diagnostic verification references a different summary.")
    names = (
        "DIAGNOSTIC_EVIDENCE.md", "protocol.json", "diagnostic_summary.json", "verification.json",
        "presentation_fixed_diagnostics.pdf", "presentation_abstention_full.pdf", "presentation_learning_penalty_transfer.pdf",
        "temperature_candidates.csv", "temperature_results.csv", "abstention_utility.csv",
        "subgroup_calibration.csv", "test_support_slices.csv", "test_support_calibration.csv",
        "validation_support_slices.csv", "validation_support_calibration.csv",
        "validation_demographic_slices.csv", "validation_demographic_calibration.csv",
        "learning_curves.csv", "learning_curve_support.csv", "penalty_path_refits.csv",
        "g3_results.csv", "g3_contrasts.csv", "g4_results.csv", "g4_contrasts.csv",
        "g3_provenance.json", "g4_provenance.json", "abstention_figure_provenance.json",
        "full_size_reuse_provenance.json", "resource_adjustment.json", "fixed_diagnostics_provenance.json",
        "subgroup_calibration_provenance.json", "test_support_slices_provenance.json",
        "validation_support_provenance.json", "validation_demographic_provenance.json",
        "penalty_path_provenance.json", "temperature_freeze.json", "temperature_freeze.sha256",
    )
    hashes = {name: digest(HERE/name) for name in names}
    for directory in (stage, flow):
        (evidence/directory).mkdir(parents=False, exist_ok=False)
    entries = []
    for name in names:
        for directory in (stage, flow):
            target = evidence/directory/name
            shutil.copy2(HERE/name, target)
            if digest(target) != hashes[name]:
                raise RuntimeError(f"Copied evidence mismatch: {target}")
        source = HERE/name
        entries.append({"source": str(source.relative_to(ROOT)), "by_stage_copy": str(stage/name),
                        "by_flow_copy": str(flow/name), "bytes": source.stat().st_size,
                        "sha256": hashes[name],
                        "presentation_role": "Post-test demographic diagnostic parity; predictive, not causal or independently confirmed."})
    for name, record in before.items():
        if digest(ROOT/name) != record["sha256"]:
            raise RuntimeError(f"Prior project file changed during copy: {name}")
    if any(digest(HERE/name) != hashes[name] for name in names):
        raise RuntimeError("Diagnostic source changed during copy.")
    write_csv(manifest, entries, list(entries[0]))
    report = {"status": "all_checks_passed", "previous_project_files_verified_unchanged": len(before),
              "selected_sources": len(names), "copies_verified": 2*len(names),
              "selection_manifest_sha256": digest(manifest), "diagnostic_summary_sha256": digest(HERE/"diagnostic_summary.json"),
              "verification_sha256": digest(HERE/"verification.json"),
              "note": "Historical selection/verification records are retained; folder_tree.txt alone is refreshed.",
              "original_evaluations_or_deployment_changed": False}
    write_json(audit, report)
    write_json(HERE/"publication_verification.json", report)
    (evidence/"folder_tree.txt").write_text("Presentation_Evidence\\\n"+"\n".join(tree_lines(evidence))+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
