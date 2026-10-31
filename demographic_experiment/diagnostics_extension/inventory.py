"""Extend the evidence-catalog lineage after diagnostic publication."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from pypdf import PdfReader

from common import HERE, ROOT, digest, verify_inputs, write_json

CATALOG = ROOT / "evidence_catalog"
sys.path.insert(0, str(CATALOG))
from catalog import files  # noqa: E402


def describe(path: str) -> tuple[str, str, str, str]:
    name = Path(path).name
    limitations = ("Retrospective predictive diagnostic after prior outcome exposure; "
                   "not a causal or independent confirmatory result.")
    if path == "Presentation_Evidence/folder_tree.txt":
        return "presentation evidence", "navigation", "Updated folder tree including demographic diagnostic parity.", "Navigation is not new empirical evidence."
    if path.startswith("Presentation_Evidence/"):
        return "presentation metadata", "copy provenance", "Diagnostic copy manifest or verification record.", "Copies are not independent observations."
    if not path.startswith("demographic_experiment/diagnostics_extension/"):
        raise ValueError(f"Unclassified addition: {path}")
    if name.startswith("curve_") and name.endswith("_models.json"):
        purpose = "One nested-pair fitting subset's retuned models, category support and unique-fitting-photo preprocessing."
    elif name.startswith("curve_") and name.endswith("_interrupted_attempts.csv"):
        purpose = "Preserved incomplete candidate log from an interrupted run; not a final candidate set."
    elif name.startswith("curve_") and name.endswith("_attempts.csv"):
        purpose = "All development penalty/interaction candidates and numerical acceptance for one nested fitting subset."
    elif name in {"g3_models.json", "g4_models.json"}:
        purpose = "Separate validation-selected transfer models, fitting-only demographic encoding and feature preprocessing."
    elif name in {"g3_predictions.parquet", "g4_predictions.parquet"}:
        purpose = "Matched saved transfer predictions for exploratory models and original comparators."
    elif name.endswith("_results.csv") and name.startswith(("g3", "g4")):
        purpose = "Matched retrospective transfer losses, accuracy and held-out-unit bootstrap intervals."
    elif name.endswith("_contrasts.csv") and name.startswith(("g3", "g4")):
        purpose = "Paired transfer comparisons, including side-adjusted score-interaction increments."
    elif name == "learning_curves.csv":
        purpose = "Two-seed nested fitting-size learning curves on the common validation intersection."
    elif name == "learning_curve_support.csv":
        purpose = "Fitting-only demographic category support, reference and fallback across curve subsets."
    elif name == "penalty_path_refits.csv":
        purpose = "Reproduced original demographic development penalty candidates, fit/validation losses and coefficient norms."
    elif name == "temperature_candidates.csv":
        purpose = "Validation-only temperatures and selection flags for each new model."
    elif name == "temperature_results.csv":
        purpose = "Validation-selected temperature paired saved-test comparisons; 95% pointwise intervals."
    elif name == "abstention_utility.csv":
        purpose = "Fixed-cutoff coverage, selective accuracy and +1/-1/0 utility on saved G2 predictions."
    elif name == "DIAGNOSTIC_EVIDENCE.md":
        purpose = "Interpretive synthesis of controls, slices, curves, penalty/temperature, transfer and abstention."
    elif name.startswith("presentation_") and name.endswith(".pdf"):
        purpose = "Presentation-ready diagnostic visualizations based on the saved tables."
    elif "calibration" in name:
        purpose = "Overall or subgroup reliability-bin metrics, response/pair support and low-support flags."
    elif "slices" in name:
        purpose = "Development or saved-test subgroup/support/feature-quality performance records."
    elif name.endswith(".py"):
        purpose = "Diagnostic implementation, tests, validation, publication or inventory source."
    else:
        purpose = "Diagnostic protocol, access guard, freeze, outcome/provenance record or integrity verification."
    return "post-test demographic diagnostics", "exploratory all-demographic predictive performance", purpose, limitations


def run() -> None:
    verify_inputs()
    inventory = CATALOG / "demographic_diagnostics_inventory.csv"
    audit_path = CATALOG / "demographic_diagnostics_integrity_audit.json"
    if inventory.exists() or audit_path.exists():
        raise FileExistsError("Diagnostic catalog update already exists.")
    old_audit = json.loads((CATALOG / "integrity_audit.json").read_text(encoding="utf-8"))
    demographic_audit = json.loads((CATALOG / "demographic_integrity_audit.json").read_text(encoding="utf-8"))
    if (digest(CATALOG/"file_inventory.csv") != old_audit["inventory_sha256"] or
            digest(CATALOG/"demographic_inventory.csv") != demographic_audit["supplemental_inventory_sha256"]):
        raise ValueError("Earlier inventory lineage mismatch.")
    prior = {r["path"]: r for r in read(CATALOG/"file_inventory.csv")}
    prior.update({r["path"]: r for r in read(CATALOG/"demographic_inventory.csv")})
    before = {p.relative_to(ROOT).as_posix(): digest(p) for p in files()}
    changed = [path for path, row in prior.items() if before.get(path) != row["sha256"]]
    if changed != ["Presentation_Evidence/folder_tree.txt"]:
        raise ValueError(f"Unexpected previously inventoried evidence changes: {changed}")
    added = sorted(set(before) - set(prior))
    evidence = ROOT / "Presentation_Evidence"
    manifest = evidence / "demographic_diagnostics_selection_manifest.csv"
    publication = json.loads((evidence / "demographic_diagnostics_verification.json").read_text(encoding="utf-8"))
    if digest(manifest) != publication["selection_manifest_sha256"]:
        raise ValueError("Diagnostic presentation manifest hash mismatch.")
    copies = {}
    for row in read(manifest):
        source = row["source"].replace("\\", "/")
        if before.get(source) != row["sha256"]:
            raise ValueError("Diagnostic presentation source changed.")
        for field in ("by_stage_copy", "by_flow_copy"):
            copy = "Presentation_Evidence/" + row[field].replace("\\", "/")
            if copy in copies or before.get(copy) != row["sha256"]:
                raise ValueError(f"Diagnostic presentation copy mismatch: {copy}")
            copies[copy] = source
    if len(copies) != 2 * publication["selected_sources"]:
        raise ValueError("Diagnostic presentation copy count mismatch.")
    records = []
    for path in sorted(added + changed):
        source = copies.get(path, path)
        stage, claim, purpose, limitation = describe(source)
        candidate = ROOT.joinpath(*path.split("/"))
        row_count = columns = pages = ""
        if candidate.suffix.lower() == ".csv":
            with candidate.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader, [])
                columns, row_count = len(header), sum(1 for _ in reader)
        elif candidate.suffix.lower() == ".parquet":
            metadata = pq.read_metadata(candidate)
            row_count, columns = metadata.num_rows, metadata.num_columns
        elif candidate.suffix.lower() == ".pdf":
            pages = len(PdfReader(candidate).pages)
        records.append({"path": path, "canonical_source": source, "presentation_copy": path in copies,
                        "stage": stage, "supports_claim": claim, "what_it_is": purpose,
                        "interpretation_limit": limitation, "bytes": candidate.stat().st_size,
                        "sha256": before[path], "data_rows": row_count, "columns": columns, "pdf_pages": pages,
                        "update_type": "updated_historical_file" if path in prior else "new_file"})
    after = {p.relative_to(ROOT).as_posix(): digest(p) for p in files()}
    if before != after:
        raise RuntimeError("Project originals changed while creating diagnostic inventory.")
    with inventory.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    audit = {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
             "previously_inventoried_files": len(prior), "previous_files_changed": changed,
             "new_files_inventoried": len(added), "current_total_project_files_in_scope": len(before),
             "presentation_sources": publication["selected_sources"], "presentation_copies_verified": len(copies),
             "previous_catalog_inventory_sha256": digest(CATALOG/"file_inventory.csv"),
             "previous_demographic_inventory_sha256": digest(CATALOG/"demographic_inventory.csv"),
             "supplemental_inventory_sha256": digest(inventory),
             "project_files_unchanged_during_catalog_collection": True,
             "tests_or_evaluations_rerun": False, "raw_outcomes_reopened": False,
             "lineage": "Read the original and demographic inventories, then apply this supplemental inventory; the refreshed presentation tree replaces its prior row."}
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2))


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


if __name__ == "__main__":
    run()
