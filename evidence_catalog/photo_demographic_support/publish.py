"""Publish verified photo support evidence and extend the inventory lineage."""

import csv
import json
import shutil
import sys
from pathlib import Path

from pypdf import PdfReader

from analyze import HERE, ROOT, digest, snapshot, write_csv, write_json

CATALOG = HERE.parent
sys.path.insert(0, str(ROOT))
from evidence_catalog.catalog import files  # noqa: E402


def tree_lines(directory, prefix=""):
    children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold()))
    lines = []
    for index, path in enumerate(children):
        last = index == len(children)-1
        lines.append(prefix + ("`-- " if last else "|-- ") + path.name + ("\\" if path.is_dir() else ""))
        if path.is_dir():
            lines.extend(tree_lines(path, prefix + ("    " if last else "|   ")))
    return lines


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def run():
    evidence = ROOT/"Presentation_Evidence"
    stage = Path("By_Stage")/"18_Photo_Demographic_Support"
    flow = Path("By_Presentation_Flow")/"03_Diagnostics"/"12_Photo_Demographic_Support"
    manifest = evidence/"photo_demographic_support_selection_manifest.csv"
    audit = evidence/"photo_demographic_support_verification.json"
    inventory = CATALOG/"photo_demographic_support_inventory.csv"
    inventory_audit = CATALOG/"photo_demographic_support_integrity_audit.json"
    if any(p.exists() for p in (evidence/stage, evidence/flow, manifest, audit, inventory, inventory_audit)):
        raise FileExistsError("Publication or supplemental inventory already exists.")
    baseline = json.loads((HERE/"original_hashes_before.json").read_text(encoding="utf-8"))
    if snapshot() != baseline:
        raise RuntimeError("Pre-existing project files changed since the outcome-blind analysis.")
    verification = json.loads((HERE/"verification.json").read_text(encoding="utf-8"))
    if verification["status"] != "independent_pandas_reconciliation_passed":
        raise ValueError("Independent count reconciliation incomplete.")
    roles = {
        "REPORT.md": "Methods, group support table and bounded interpretation.",
        "per_photo_demographic_counts.csv": "Per-photo group counts and within/between-cat support.",
        "group_support_summary.csv": "Per-group distribution across fitting photos.",
        "matched_group_support.csv": "Shared-photo and shared-pair group coverage.",
        "support_summary.json": "Partition totals and demographic-category universe.",
        "presentation_photo_demographic_support.pdf": "Three-page structural support visualization.",
        "analysis_provenance.json": "Structural inputs, outputs, and original snapshot hashes.",
        "verification.json": "Independent pandas reconciliation record.",
    }
    source_hashes = {name: digest(HERE/name) for name in roles}
    for path in (stage, flow):
        (evidence/path).mkdir(parents=True, exist_ok=False)
    entries = []
    for name, role in roles.items():
        for path in (stage, flow):
            target = evidence/path/name
            shutil.copy2(HERE/name, target)
            if digest(target) != source_hashes[name]:
                raise RuntimeError(f"Presentation copy mismatch: {target}")
        entries.append({"source": (HERE/name).relative_to(ROOT).as_posix(), "by_stage_copy": str(stage/name),
                        "by_flow_copy": str(flow/name), "bytes": (HERE/name).stat().st_size,
                        "sha256": source_hashes[name], "presentation_role": role})
    for name, details in baseline.items():
        path = ROOT/name
        if not path.is_file() or digest(path) != details["sha256"]:
            raise RuntimeError(f"Pre-existing file changed during copying: {name}")
    if any(digest(HERE/name) != value for name, value in source_hashes.items()):
        raise RuntimeError("Analysis source changed during copying.")
    write_csv(manifest, entries, list(entries[0]))
    write_json(audit, {"status": "all_checks_passed", "previous_project_files_unchanged_before_tree_refresh": len(baseline),
                       "selected_sources": len(roles), "copies_verified": 2*len(roles), "manifest_sha256": digest(manifest),
                       "verification_sha256": digest(HERE/"verification.json"), "models_fitted": False,
                       "test_rows_read": False, "outcome_columns_read": False,
                       "note": "The presentation tree was intentionally refreshed; prior manifests were preserved."})
    (evidence/"folder_tree.txt").write_text("Presentation_Evidence\\\n" + "\n".join(tree_lines(evidence)) + "\n",
                                           encoding="utf-8")
    prior_names = ("file_inventory.csv", "demographic_inventory.csv", "demographic_diagnostics_inventory.csv",
                   "kernel_inventory.csv", "cat_cnn_inventory.csv", "cat_demographic_support_inventory.csv")
    old_audit = json.loads((CATALOG/"cat_demographic_support_integrity_audit.json").read_text(encoding="utf-8"))
    historical_hashes = old_audit["historical_inventory_sha256"]
    historical_hashes["cat_demographic_support_inventory.csv"] = old_audit["cat_demographic_support_inventory_sha256"]
    previous, previous_managed = {}, {}
    for name in prior_names:
        if digest(CATALOG/name) != historical_hashes[name]:
            raise ValueError(f"Historical inventory hash mismatch: {name}")
        for row in read(CATALOG/name):
            if row["path"].startswith("evidence_catalog/"):
                previous_managed[row["path"]] = row
            else:
                previous[row["path"]] = row
    for path, row in previous_managed.items():
        if digest(ROOT.joinpath(*path.split("/"))) != row["sha256"]:
            raise RuntimeError(f"Earlier catalog-managed study file changed: {path}")
    current = {path.relative_to(ROOT).as_posix(): digest(path) for path in files()}
    changed = sorted(name for name, record in previous.items() if current.get(name) != record["sha256"])
    if changed != ["Presentation_Evidence/folder_tree.txt"]:
        raise RuntimeError(f"Unexpected historical changes: {changed}")
    added = sorted(set(current)-set(previous))
    copies = {}
    for entry in entries:
        for field in ("by_stage_copy", "by_flow_copy"):
            copy = "Presentation_Evidence/" + entry[field].replace("\\", "/")
            if current.get(copy) != entry["sha256"]:
                raise RuntimeError(f"Inventory copy mismatch: {copy}")
            copies[copy] = entry["source"]
    if set(added) != set(copies) | {manifest.relative_to(ROOT).as_posix(), audit.relative_to(ROOT).as_posix()}:
        raise RuntimeError("Unexpected project additions.")
    managed = sorted(p.relative_to(ROOT).as_posix() for p in HERE.iterdir() if p.is_file())
    inventory_rows = []
    for path in sorted(added + changed + managed):
        candidate = ROOT.joinpath(*path.split("/"))
        count = columns = pages = ""
        if candidate.suffix.lower() == ".csv":
            with candidate.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                columns, count = len(next(reader, [])), sum(1 for _ in reader)
        elif candidate.suffix.lower() == ".pdf":
            pages = len(PdfReader(candidate).pages)
        inventory_rows.append({"path": path, "canonical_source": copies.get(path, path), "presentation_copy": path in copies,
                               "stage": "outcome-blind photo demographic structural support",
                               "supports_claim": "descriptive response and photo-pair support, not demographic preference",
                               "what_it_is": "Study source/output" if path in managed else "Presentation copy or navigation/integrity metadata",
                               "interpretation_limit": "Not unique respondents, an effect estimate, or a power guarantee.",
                               "bytes": candidate.stat().st_size, "sha256": digest(candidate), "data_rows": count,
                               "columns": columns, "pdf_pages": pages, "update_type": "updated_historical_file" if path in changed
                               else "catalog_managed" if path in managed else "new_file"})
    write_csv(inventory, inventory_rows, list(inventory_rows[0]))
    if current != {path.relative_to(ROOT).as_posix(): digest(path) for path in files()}:
        raise RuntimeError("Project changed during inventory collection.")
    write_json(inventory_audit, {"status": "passed", "previous_inventoried_files": len(previous),
                                "previous_catalog_managed_files_verified": len(previous_managed),
                                "previous_files_changed": changed, "new_project_files_inventoried": len(added),
                                "current_total_project_files_in_scope": len(current), "catalog_managed_study_files": len(managed),
                                "presentation_sources": len(roles), "presentation_copies_verified": len(copies),
                                "historical_inventory_sha256": historical_hashes, "photo_demographic_support_inventory_sha256": digest(inventory),
                                "models_fitted": False, "test_rows_read": False, "outcome_columns_read": False,
                                "scope_note": "Project scope excludes evidence_catalog; this supplement separately lists catalog-managed study files, but not itself or its audit."})
    print(json.dumps({"project_scope": len(current), "new_project_files": len(added), "managed_study_files": len(managed),
                      "previous_managed_verified": len(previous_managed), "copies_verified": len(copies)}, indent=2))


if __name__ == "__main__":
    run()
