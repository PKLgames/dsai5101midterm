"""Copy selected kernel evidence to both presentation views."""

import json
import shutil
from pathlib import Path

from study import EXCLUDED, HERE, ROOT, digest, snapshot, verify_base, write_csv, write_json


def tree_lines(directory, prefix=""):
    children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold()))
    lines = []
    for i, path in enumerate(children):
        last = i == len(children)-1
        lines.append(prefix+("`-- " if last else "|-- ")+path.name+("\\" if path.is_dir() else ""))
        if path.is_dir():
            lines.extend(tree_lines(path, prefix+("    " if last else "|   ")))
    return lines


def run():
    verify_base()
    evidence = ROOT/"Presentation_Evidence"
    stage = Path("By_Stage")/"15_Retrospective_Kernel_CNN"
    flow = Path("By_Presentation_Flow")/"03_Diagnostics"/"09_Retrospective_Kernel_CNN"
    manifest, audit = evidence/"kernel_selection_manifest.csv", evidence/"kernel_verification.json"
    if any(p.exists() for p in (evidence/stage, evidence/flow, manifest, audit, HERE/"publication_verification.json")):
        raise FileExistsError("Kernel evidence publication already started.")
    before = json.loads((HERE/"original_hashes_before.json").read_text(encoding="utf-8"))
    evidence_before = json.loads((HERE/"evidence_hashes_before.json").read_text(encoding="utf-8"))
    if snapshot(ROOT, EXCLUDED) != before or snapshot(evidence, set()) != evidence_before:
        raise RuntimeError("Pre-existing project or presentation evidence changed during kernel study.")
    summary = json.loads((HERE/"summary.json").read_text(encoding="utf-8"))
    verification = json.loads((HERE/"verification.json").read_text(encoding="utf-8"))
    additional = json.loads((HERE/"additional_verification.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete_retrospective_exploratory" or verification["status"] != "passed" or additional["status"] != "passed":
        raise ValueError("Kernel study or verification is incomplete.")
    if summary["verification_sha256"] != digest(HERE/"verification.json"):
        raise ValueError("Kernel verification record mismatch.")
    for name, value in summary["outputs_sha256"].items():
        if digest(HERE/name) != value:
            raise ValueError(f"Kernel summary output mismatch: {name}")
    roles = {
        "RESULTS.md": "Evidence-linked synthesis and interpretation limitations.",
        "protocol.json": "Retrospective photo-level RBF Nyström fitting and validation rules.",
        "presentation_kernel_study.pdf": "G2/G3/G4, learning, penalty, calibration and utility figures.",
        "presentation_kernel_abstention.pdf": "Coverage, selective accuracy and fixed-cutoff counts.",
        "g2_results.csv": "Matched original G2 and kernel metrics.",
        "g2_contrasts.csv": "Paired original G2 comparator differences and intervals.",
        "g2_pair_weighted.csv": "Pair-weighted G2 kernel and linear baseline losses.",
        "g2_subgroups.csv": "Within/between-cat G2 metrics.",
        "g2_slices.csv": "Original-style G2 support and feature-quality slices.",
        "g2_calibration.csv": "Overall G2 calibration bins.",
        "g2_slice_calibration.csv": "Support and quality subgroup calibration bins.",
        "validation_slices.csv": "Development support and feature-quality slices.",
        "validation_calibration.csv": "Development supported-subgroup reliability bins.",
        "learning_curves.csv": "Two-seed nested fitting-size learning curves.",
        "g2_attempts.csv": "Complete validation width/ridge candidate log.",
        "penalty_refits.csv": "Independent 21-candidate penalty-path refits.",
        "temperature_candidates.csv": "Validation-only fixed temperature grid.",
        "g2_abstention_utility.csv": "Fixed-confidence coverage, selective accuracy and utility grid.",
        "fair_coin_control.csv": "Twenty fixed-setting independent fair-coin controls.",
        "fair_coin_provenance.json": "Random-label control status, scope and hashes.",
        "g3_results.csv": "Retrospective G3 matched test metrics.",
        "g3_contrasts.csv": "G3 held-out-photo paired intervals.",
        "g3_within_results.csv": "G3 within-cat model metrics.",
        "g3_within_contrasts.csv": "G3 within-cat held-out-photo paired intervals.",
        "g4_results.csv": "Retrospective one-unseen-cat G4 test metrics.",
        "g4_contrasts.csv": "G4 held-out-cat paired intervals.",
        "additional_paired_contrasts.csv": "Matched kernel versus all-demographic CNN comparisons; inputs differ.",
        "verification.json": "Independent selected-model and cluster-bootstrap replay.",
        "additional_verification.json": "Penalty, support and demographic-comparison verification.",
        "summary.json": "Kernel study scope and summary-output hashes.",
        "resource_adjustment.json": "Preserved interrupted curve attempts and sequential completion.",
    }
    hashes = {name: digest(HERE/name) for name in roles}
    for directory in (stage, flow):
        (evidence/directory).mkdir(parents=True, exist_ok=False)
    entries = []
    for name, role in roles.items():
        for directory in (stage, flow):
            target = evidence/directory/name
            shutil.copy2(HERE/name, target)
            if digest(target) != hashes[name]:
                raise RuntimeError(f"Copy mismatch: {target}")
        entries.append({"source": str(Path("kernel_experiment")/name), "by_stage_copy": str(stage/name),
                        "by_flow_copy": str(flow/name), "bytes": (HERE/name).stat().st_size,
                        "sha256": hashes[name], "presentation_role": role})
    if snapshot(ROOT, EXCLUDED) != before:
        raise RuntimeError("Pre-existing project files changed during copying.")
    for name, details in evidence_before.items():
        if digest(evidence/name) != details["sha256"]:
            raise RuntimeError(f"Pre-existing presentation evidence changed: {name}")
    if any(digest(HERE/name) != value for name, value in hashes.items()):
        raise RuntimeError("Kernel evidence source changed during copying.")
    write_csv(manifest, entries, list(entries[0]))
    report = {"status": "all_checks_passed", "original_project_files_unchanged": len(before),
              "previous_evidence_files_unchanged_before_tree_refresh": len(evidence_before),
              "selected_sources": len(roles), "copies_verified": 2*len(roles),
              "manifest_sha256": digest(manifest), "kernel_summary_sha256": digest(HERE/"summary.json"),
              "primary_claims_models_and_deployment_changed": False,
              "note": "Historical evidence manifests remain unchanged; only folder_tree.txt refreshed."}
    write_json(audit, report)
    write_json(HERE/"publication_verification.json", report)
    (evidence/"folder_tree.txt").write_text("Presentation_Evidence\\\n"+"\n".join(tree_lines(evidence))+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
