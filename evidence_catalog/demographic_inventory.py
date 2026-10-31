"""Extend the historical catalog without modifying its original inventory."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from pypdf import PdfReader

from catalog import ROOT, HERE, files, sha


PURPOSES = {
    "all_fit_attempts.csv": "Complete 94-candidate development selection and numerical acceptance log.",
    "calibration.csv": "Four requested models' 40 overall test calibration bins and support flags.",
    "coefficients.csv": "Full signed fitted side, pooled score and interaction coefficients with feature labels.",
    "control.py": "Source for fixed-setting fair-coin operational controls.",
    "control_provenance.json": "Fair-coin control status, 80 fits, trigger and source/results hashes.",
    "control_results.csv": "Twenty fair-coin fit/validation repetitions per requested procedure.",
    "design_conditioning.json": "Fitting demographic side-design rank and redundant missing indicators.",
    "development_access.json": "Original fitting/validation-only access and test-outcome exclusion record.",
    "development_support.csv": "Fitting and validation demographic categories, counts, pair support and fallback states.",
    "evaluation_provenance.json": "Frozen saved-test analysis source/output hashes and post-test limitations.",
    "evaluation_started.json": "Write-once saved-test-outcome analysis access record.",
    "evidence_hashes_before.json": "Historical pre-experiment presentation evidence hash snapshot.",
    "experiment.py": "Source for separate fitting, selection, evaluation, chart and publication.",
    "freeze.json": "Pre-saved-test-analysis exploratory model, selection and source hash freeze.",
    "freeze.sha256": "SHA-256 seal of the exploratory freeze record.",
    "INTERPRETATION.md": "Detailed results, matched contrasts, limitations and next discriminating evidence.",
    "original_hashes_before.json": "Historical pre-experiment original project-file hash snapshot.",
    "presentation_demographic_models.pdf": "Two-page model/contrast and supported-bin calibration figure.",
    "protocol.json": "Fixed retrospective four-model, support, selection and bootstrap protocol.",
    "publication_checks_verification.json": "Supplementary interpretation/control evidence copy verification.",
    "publication_verification.json": "Initial demographic presentation evidence copy verification.",
    "README.md": "Experiment model definitions, interpretation limits and reproduction instructions.",
    "selected_models.json": "Seven validation-selected procedures, demographic encoding and fit-only preprocessing.",
    "subgroups.csv": "112 model/demographic or comparison-type subgroup support and metric records.",
    "tests.py": "Focused design, encoding, numerical gradient and fallback unit tests.",
    "test_contrasts.csv": "Nine matched paired comparisons and 95% pointwise bootstrap intervals.",
    "test_predictions.parquet": "Saved model predictions and outcomes for matched G2 test rows.",
    "test_results.csv": "Four requested models, three side-adjusted comparators and seven original comparators.",
    "test_support.csv": "Test demographic-category response and pair support with unsupported fallback states.",
    "validation_results.csv": "Seven validation-selected model results and hyperparameters.",
    "verification.json": "Recorded independent validation/test-prediction and bootstrap replay.",
    "verify.py": "Source for independent saved-prediction replay and one-evaluation guard checks.",
}
LIMIT = ("Post-test retrospective predictive association, not a causal or independent confirmatory result. "
         "Demographic side effects and score interactions are distinct; overlapping missing indicators are not separately identifiable.")


def describe(path: str) -> tuple[str, str, str, str]:
    parts = Path(path).parts
    if parts[0] == "demographic_experiment":
        name = parts[-1]
        if name not in PURPOSES:
            raise ValueError(f"Unclassified demographic experiment file: {path}")
        return "post-test demographic experiment", "exploratory all-demographic prediction", PURPOSES[name], LIMIT
    if path == "Presentation_Evidence/folder_tree.txt":
        return ("presentation evidence", "navigation and provenance",
                "Updated tree including both demographic evidence views.",
                "Tree is a navigation record and does not add observations.")
    if parts[0] == "Presentation_Evidence":
        return ("presentation evidence metadata", "demographic evidence provenance",
                "Copy-selection manifest or historical verification for the demographic evidence batch.",
                "Integrity and navigation record; copies are not independent evidence.")
    raise ValueError(f"Unexpected new file: {path}")


def run() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    inventory = HERE / "demographic_inventory.csv"
    audit_path = HERE / "demographic_integrity_audit.json"
    if inventory.exists() or audit_path.exists():
        raise FileExistsError("Demographic catalog update already exists.")
    original_audit = json.loads((HERE / "integrity_audit.json").read_text(encoding="utf-8"))
    old_inventory = HERE / "file_inventory.csv"
    if sha(old_inventory) != original_audit["inventory_sha256"]:
        raise ValueError("Historical catalog inventory no longer matches its audit.")
    with old_inventory.open(encoding="utf-8", newline="") as stream:
        old_rows = list(csv.DictReader(stream))
    old = {r["path"]: r for r in old_rows}
    if len(old) != len(old_rows):
        raise ValueError("Historical catalog contains duplicate paths.")
    original_paths = files()
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in original_paths}
    changed = [name for name, row in old.items() if before.get(name) != row["sha256"]]
    if changed != ["Presentation_Evidence/folder_tree.txt"]:
        raise ValueError(f"Unexpected changes to historical evidence: {changed}")
    new = sorted(set(before) - set(old))
    if any(not name.startswith(("demographic_experiment/", "Presentation_Evidence/")) for name in new):
        raise ValueError("Unclassified new project files require review.")
    copies = {}
    evidence = ROOT / "Presentation_Evidence"
    for stem in ("demographic", "demographic_checks"):
        manifest = evidence / f"{stem}_selection_manifest.csv"
        proof = json.loads((evidence / f"{stem}_verification.json").read_text(encoding="utf-8"))
        if sha(manifest) != proof["manifest_sha256"]:
            raise ValueError(f"Selection manifest seal mismatch: {stem}")
        with manifest.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                source = row["source"].replace("\\", "/")
                if before.get(source) != row["sha256"]:
                    raise ValueError(f"Presentation source mismatch: {source}")
                for key in ("by_stage_copy", "by_flow_copy"):
                    destination = "Presentation_Evidence/" + row[key].replace("\\", "/")
                    if destination in copies or before.get(destination) != row["sha256"]:
                        raise ValueError(f"Presentation copy mismatch: {destination}")
                    copies[destination] = source
    if len(copies) != 28 or len(set(copies.values())) != 14:
        raise ValueError("Demographic evidence copy/source count mismatch.")
    experiment = ROOT / "demographic_experiment"
    freeze_path = experiment / "freeze.json"
    if sha(freeze_path) != (experiment / "freeze.sha256").read_text(encoding="ascii").strip():
        raise ValueError("Demographic experiment freeze seal mismatch.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for name, expected in freeze["source_output_sha256"].items():
        if sha(experiment / name) != expected:
            raise ValueError(f"Demographic frozen artifact mismatch: {name}")
    for name, expected in freeze["root_input_sha256"].items():
        if sha(ROOT.joinpath(*name.replace("\\", "/").split("/"))) != expected:
            raise ValueError(f"Demographic frozen root input mismatch: {name}")
    evaluation = json.loads((experiment / "evaluation_provenance.json").read_text(encoding="utf-8"))
    if evaluation["freeze_sha256"] != sha(freeze_path):
        raise ValueError("Demographic evaluation freeze mismatch.")
    for name, expected in evaluation["outputs_sha256"].items():
        if sha(experiment / name) != expected:
            raise ValueError(f"Demographic output mismatch: {name}")
    control = json.loads((experiment / "control_provenance.json").read_text(encoding="utf-8"))
    if (control["status"] != "passed_operational_control" or control["freeze_sha256"] != sha(freeze_path)
            or control["results_sha256"] != sha(experiment / "control_results.csv")
            or control["source_sha256"] != sha(experiment / "control.py")):
        raise ValueError("Demographic fair-coin control provenance mismatch.")
    verification = json.loads((experiment / "verification.json").read_text(encoding="utf-8"))
    if verification["status"] != "independent_replay_passed" or verification["prediction_sha256"] != sha(experiment / "test_predictions.parquet"):
        raise ValueError("Demographic independent replay record mismatch.")
    rows = []
    for path in sorted(new + changed):
        source = copies.get(path, path)
        stage, claim, purpose, limitation = describe(source)
        target = ROOT.joinpath(*path.split("/"))
        row_count = columns = pages = ""
        if target.suffix.lower() == ".csv":
            with target.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader, [])
                columns, row_count = len(header), sum(1 for _ in reader)
        elif target.suffix.lower() == ".parquet":
            metadata = pq.read_metadata(target)
            row_count, columns = metadata.num_rows, metadata.num_columns
        elif target.suffix.lower() == ".pdf":
            pages = len(PdfReader(target).pages)
        rows.append({"path": path, "canonical_source": source, "presentation_copy": path in copies,
                     "stage": stage, "supports_claim": claim, "what_it_is": purpose,
                     "interpretation_limit": limitation, "bytes": target.stat().st_size,
                     "sha256": before[path], "data_rows": row_count, "columns": columns, "pdf_pages": pages,
                     "update_type": "updated_historical_file" if path in old else "new_file"})
    after = {p.relative_to(ROOT).as_posix(): sha(p) for p in files()}
    if before != after:
        raise RuntimeError("Original project files changed during the catalog update.")
    with inventory.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "historical_inventory_sha256": sha(old_inventory), "historical_integrity_audit_sha256": sha(HERE / "integrity_audit.json"),
        "historical_files_verified_unchanged": len(old) - len(changed),
        "historical_files_updated": changed, "new_files_inventoried": len(new),
        "current_total_project_files_in_scope": len(before), "demographic_presentation_sources": 14,
        "demographic_presentation_copies_verified": len(copies),
        "demographic_freeze_sha256": sha(freeze_path), "demographic_frozen_and_evaluation_hashes_verified": True,
        "fair_coin_control_provenance_verified": True, "historical_independent_replay_record_verified": True,
        "project_files_unchanged_during_catalog_update": True, "tests_or_evaluations_rerun": False,
        "raw_outcomes_reopened": False, "supplemental_inventory_sha256": sha(inventory),
        "lineage": "Original inventory/audit remain historical. Read the supplemental inventory as additions plus the updated folder_tree.txt replacement."
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    run()
