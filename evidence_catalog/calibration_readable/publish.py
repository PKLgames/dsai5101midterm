"""Publish the readable calibration figure without replacing frozen evidence."""

import csv
import json
import shutil
import sys
from pathlib import Path

from pypdf import PdfReader

from plot import HERE, ROOT, SERIES, sha

sys.path.insert(0, str(ROOT))
from evidence_catalog.catalog import files  # noqa: E402


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def tree_lines(directory, prefix=""):
    paths = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold()))
    result = []
    for i, path in enumerate(paths):
        last = i == len(paths)-1
        result.append(prefix+("`-- " if last else "|-- ")+path.name+("\\" if path.is_dir() else ""))
        if path.is_dir():
            result.extend(tree_lines(path, prefix+("    " if last else "|   ")))
    return result


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run():
    catalog, evidence = ROOT/"evidence_catalog", ROOT/"Presentation_Evidence"
    manifest = evidence/"readable_calibration_selection_manifest.csv"
    inventory = catalog/"readable_calibration_inventory.csv"
    audit = catalog/"readable_calibration_integrity_audit.json"
    names = ("file_inventory.csv", "demographic_inventory.csv", "demographic_diagnostics_inventory.csv",
             "kernel_inventory.csv", "cat_cnn_inventory.csv", "cat_demographic_support_inventory.csv",
             "photo_demographic_support_inventory.csv", "heldout_agreement_inventory.csv")
    previous_audit = json.loads((catalog/"heldout_agreement_integrity_audit.json").read_text(encoding="utf-8"))
    hashes = previous_audit["historical_inventory_sha256"]
    hashes["heldout_agreement_inventory.csv"] = previous_audit["heldout_agreement_inventory_sha256"]
    previous, managed = {}, {}
    for name in names:
        if sha(catalog/name) != hashes[name]:
            raise ValueError(f"Historical inventory hash mismatch: {name}")
        for row in read(catalog/name):
            (managed if row["path"].startswith("evidence_catalog/") else previous)[row["path"]] = row
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in files()}
    if set(before) != set(previous) or any(before[name] != r["sha256"] for name, r in previous.items()):
        raise RuntimeError("Historical project inventory differs before publication.")
    for name, record in managed.items():
        if sha(ROOT.joinpath(*name.split("/"))) != record["sha256"]:
            raise RuntimeError(f"Historical managed source changed: {name}")
    provenance = json.loads((HERE/"provenance.json").read_text(encoding="utf-8"))
    for name, value in provenance["outputs_sha256"].items():
        if sha(HERE/name) != value:
            raise RuntimeError(f"Plot output changed: {name}")
    if sha(ROOT/"tables"/"test_calibration.csv") != provenance["input_sha256"]:
        raise RuntimeError("Frozen input changed.")
    stage = Path("By_Stage")/"09_Frozen_Test"
    flow = Path("By_Presentation_Flow")/"03_Diagnostics"/"03_Test_Calibration_and_Slices"
    copies = {}
    for filename in ("calibration_readable.pdf", "calibration_readable.png"):
        for directory in (stage, flow):
            target = evidence/directory/filename
            if target.exists():
                raise FileExistsError(f"Presentation copy already exists: {target}")
            copies[target] = HERE/filename
    if any(p.exists() for p in (manifest, inventory, audit)):
        raise FileExistsError("Publication metadata already exists.")
    for target, source in copies.items():
        shutil.copy2(source, target)
        if sha(target) != sha(source):
            raise RuntimeError("Calibration copy hash mismatch.")
    records = [{"source": (HERE/name).relative_to(ROOT).as_posix(), "by_stage_copy": str(stage/name),
                "by_flow_copy": str(flow/name), "sha256": sha(HERE/name), "bytes": (HERE/name).stat().st_size,
                "purpose": "Readable relabel of original supported-bin calibration; no new model or data."}
               for name in ("calibration_readable.pdf", "calibration_readable.png")]
    write_csv(manifest, records)
    if any(sha(ROOT.joinpath(*name.split("/"))) != value for name, value in before.items()):
        raise RuntimeError("Pre-existing project file changed during copy.")
    (evidence/"folder_tree.txt").write_text("Presentation_Evidence\\\n" + "\n".join(tree_lines(evidence)) + "\n",
                                           encoding="utf-8")
    current = {p.relative_to(ROOT).as_posix(): sha(p) for p in files()}
    added = set(current)-set(before)
    expected = {p.relative_to(ROOT).as_posix() for p in copies} | {manifest.relative_to(ROOT).as_posix()}
    changed = [name for name in before if current.get(name) != before[name]]
    if added != expected or changed != ["Presentation_Evidence/folder_tree.txt"]:
        raise RuntimeError(f"Unexpected additions or changes: {added ^ expected}, {changed}")
    sources = {p.relative_to(ROOT).as_posix(): s.relative_to(ROOT).as_posix() for p, s in copies.items()}
    study = [p.relative_to(ROOT).as_posix() for p in HERE.iterdir() if p.is_file()]
    rows = []
    for name in sorted(list(added)+changed+study):
        path = ROOT.joinpath(*name.split("/"))
        count = columns = pages = ""
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream)
                columns, count = len(next(reader, [])), sum(1 for _ in reader)
        elif path.suffix.lower() == ".pdf":
            pages = len(PdfReader(path).pages)
        rows.append({"path": name, "canonical_source": sources.get(name, name), "presentation_copy": name in sources,
                     "stage": "presentation-only readable calibration", "supports_claim": "same frozen calibration bins",
                     "what_it_is": "Relabelled plot, plotted data, source or navigation manifest",
                     "interpretation_limit": "No new model, data, or independent evaluation.",
                     "bytes": path.stat().st_size, "sha256": sha(path), "data_rows": count, "columns": columns,
                     "pdf_pages": pages, "update_type": "updated_historical_file" if name in changed else
                     "catalog_managed" if name in study else "new_file"})
    write_csv(inventory, rows)
    summary = {"status": "passed", "previous_inventoried_files": len(before), "previous_managed_verified": len(managed),
               "previous_files_changed": changed, "new_project_files_inventoried": len(added),
               "current_total_project_files_in_scope": len(current), "catalog_managed_study_files": len(study),
               "presentation_sources": 2, "presentation_copies_verified": len(copies),
               "historical_inventory_sha256": hashes, "readable_calibration_inventory_sha256": sha(inventory),
               "frozen_data_and_original_figure_unchanged": True, "models": [s[0] for s in SERIES],
               "note": "The only changed prior project file is the intentionally refreshed presentation tree."}
    with audit.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
