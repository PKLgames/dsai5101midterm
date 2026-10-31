"""Extend the evidence catalog inventory without rewriting historical records."""

import csv
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from pypdf import PdfReader

from study import HERE, ROOT, digest, write_json

CATALOG = ROOT/"evidence_catalog"
sys.path.insert(0, str(CATALOG))
from catalog import files  # noqa: E402


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def describe(path):
    name = Path(path).name
    if path == "Presentation_Evidence/folder_tree.txt":
        return "Updated presentation folder tree; navigation, not independent evidence."
    if path.startswith("Presentation_Evidence/"):
        return "Kernel evidence copy manifest or verification record; integrity, not new observations."
    if not path.startswith("kernel_experiment/"):
        raise ValueError(f"Unclassified new file: {path}")
    if "_interrupted_attempts.csv" in name:
        return "Preserved incomplete candidate log from an interrupted run; not a final candidate set."
    if name.startswith("curve_") and name.endswith("_attempts.csv"):
        return "Complete 21-candidate width/ridge fitting log for one nested subset."
    if name.startswith("curve_") and name.endswith("_model.json"):
        return "Selected fitting-only kernel transformation and coefficients for one nested subset."
    if name.endswith("_predictions.parquet"):
        return "Matched saved kernel and original comparator test predictions."
    if name.endswith("_results.csv"):
        return "Matched model response losses, uncertainty, Brier and accuracy."
    if name.endswith("_contrasts.csv"):
        return "Paired response-loss differences and conditional cluster-bootstrap intervals."
    if "calibration" in name:
        return "Reliability bins with counts, pair support and low-support flags."
    if "subgroup" in name or "slices" in name or "within" in name:
        return "Within/between-cat or fitting-support/feature-quality diagnostics."
    if "abstention" in name:
        return "Fixed confidence, coverage, selective accuracy and +1/-1/0 utility evidence."
    if "penalty" in name or "attempts" in name:
        return "Kernel width/ridge candidate path and independent refit evidence."
    if "fair_coin" in name:
        return "Fixed-settings independent fair-coin operational control."
    if "temperature" in name:
        return "Validation-only fixed-grid temperature candidates and selection freeze."
    if name.endswith(".pdf"):
        return "Presentation-ready retrospective kernel diagnostic figure."
    if name.endswith(".py"):
        return "Kernel diagnostic implementation, verification or publication source."
    if name == "RESULTS.md":
        return "Measured kernel results, bounded interpretations and limitations."
    if name == "learning_curves.csv":
        return "Nested fitting-size validation learning curves for two seeds."
    return "Kernel study scope, fitting-only model, freeze, provenance or verification record."


def run():
    output = CATALOG/"kernel_inventory.csv"
    audit = CATALOG/"kernel_integrity_audit.json"
    if output.exists() or audit.exists():
        raise FileExistsError("Kernel catalog inventory already exists.")
    previous_names = ("file_inventory.csv", "demographic_inventory.csv", "demographic_diagnostics_inventory.csv")
    proofs = ("integrity_audit.json", "demographic_integrity_audit.json", "demographic_diagnostics_integrity_audit.json")
    for i, (name, proof) in enumerate(zip(previous_names, proofs)):
        data = json.loads((CATALOG/proof).read_text(encoding="utf-8"))
        key = "inventory_sha256" if i == 0 else "supplemental_inventory_sha256"
        if digest(CATALOG/name) != data[key]:
            raise ValueError(f"Historical inventory hash mismatch: {name}")
    prior = {}
    for name in previous_names:
        prior.update({r["path"]: r for r in read(CATALOG/name)})
    before = {p.relative_to(ROOT).as_posix(): digest(p) for p in files()}
    changed = [name for name, row in prior.items() if before.get(name) != row["sha256"]]
    if changed != ["Presentation_Evidence/folder_tree.txt"]:
        raise ValueError(f"Unexpected historical changes: {changed}")
    added = sorted(set(before)-set(prior))
    evidence = ROOT/"Presentation_Evidence"
    manifest = evidence/"kernel_selection_manifest.csv"
    publication = json.loads((evidence/"kernel_verification.json").read_text(encoding="utf-8"))
    if digest(manifest) != publication["manifest_sha256"]:
        raise ValueError("Kernel presentation manifest mismatch.")
    copies = {}
    for row in read(manifest):
        source = row["source"].replace("\\", "/")
        if before.get(source) != row["sha256"]:
            raise ValueError(f"Kernel source mismatch: {source}")
        for field in ("by_stage_copy", "by_flow_copy"):
            copy = "Presentation_Evidence/"+row[field].replace("\\", "/")
            if before.get(copy) != row["sha256"] or copy in copies:
                raise ValueError(f"Kernel copy mismatch: {copy}")
            copies[copy] = source
    if len(copies) != 62:
        raise ValueError("Kernel copied asset count mismatch.")
    rows = []
    for path in sorted(added+changed):
        source = copies.get(path,path)
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
        rows.append({"path": path, "canonical_source": source, "presentation_copy": path in copies,
                     "stage": "post-test kernel extension", "supports_claim": "exploratory nonlinear visual prediction",
                     "what_it_is": describe(source),
                     "interpretation_limit": "Post-test retrospective; not independently confirmatory or causal.",
                     "bytes": candidate.stat().st_size, "sha256": before[path],
                     "data_rows": row_count, "columns": columns, "pdf_pages": pages,
                     "update_type": "updated_historical_file" if path in prior else "new_file"})
    after = {p.relative_to(ROOT).as_posix(): digest(p) for p in files()}
    if before != after:
        raise RuntimeError("Project originals changed while building inventory.")
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {"status": "passed", "previous_inventoried_files": len(prior), "previous_files_changed": changed,
              "new_files_inventoried": len(added), "current_total_project_files_in_scope": len(before),
              "kernel_presentation_sources": len(set(copies.values())), "kernel_presentation_copies_verified": len(copies),
              "historical_inventory_sha256": {name:digest(CATALOG/name) for name in previous_names},
              "kernel_inventory_sha256": digest(output), "original_project_files_unchanged_during_collection": True,
              "raw_test_outcomes_reopened": False, "tests_or_evaluations_rerun": False,
              "lineage": "Read prior inventories in order, then apply kernel inventory; refreshed presentation tree replaces prior row."}
    write_json(audit, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
