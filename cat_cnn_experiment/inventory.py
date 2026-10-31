"""Extend evidence inventory without rewriting historical snapshot lineage."""

import csv
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from pypdf import PdfReader

from study import HERE, ROOT, digest, write_json

CATALOG=ROOT/"evidence_catalog"
sys.path.insert(0,str(CATALOG))
from catalog import files  # noqa: E402


def read(path):
    with path.open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))


def describe(path):
    name=Path(path).name
    if path=="Presentation_Evidence/folder_tree.txt":
        return "Updated presentation folder tree; navigation, not independent evidence."
    if path.startswith("Presentation_Evidence/"):
        return "Cat + CNN evidence copy manifest or verification record; integrity, not new observations."
    if path in {"graph.py", "two_islands_graph.png"}:
        return "Pre-existing external addition observed during the study; not analyzed or attributed to this experiment."
    if not path.startswith("cat_cnn_experiment/"):
        raise ValueError(f"Unclassified new file: {path}")
    if name.startswith("curve_") and name.endswith("_attempts.csv"):
        return "Complete 49-candidate cat/CNN penalty fitting log for one nested subset."
    if name.startswith("curve_") and name.endswith("_model.json"):
        return "Selected fitting-only cat keys, CNN preprocessing and coefficients for one nested subset."
    if name.endswith("_predictions.parquet"):
        return "Matched saved hybrid and original comparator test predictions."
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
        return "Two-penalty candidate path and independent refit evidence."
    if "fair_coin" in name:
        return "Fixed-settings independent fair-coin operational control."
    if "temperature" in name:
        return "Validation-only fixed-grid temperature candidates and selection freeze."
    if name.endswith(".pdf"):
        return "Presentation-ready retrospective Cat + CNN diagnostic figure."
    if name.endswith(".py"):
        return "Cat + CNN diagnostic implementation, verification or publication source."
    if name=="RESULTS.md":
        return "Measured Cat + CNN results, bounded interpretations and limitations."
    if name=="learning_curves.csv":
        return "Nested fitting-size validation learning curves for two seeds."
    if name=="g4_unsupported.json":
        return "G4 unsupported by explicit user choice; no unseen cat score or test predictions."
    return "Cat + CNN study scope, fitting-only model, freeze, provenance or verification record."


def run():
    output=CATALOG/"cat_cnn_inventory.csv";audit=CATALOG/"cat_cnn_integrity_audit.json"
    if output.exists() or audit.exists():raise FileExistsError("Cat + CNN inventory already exists.")
    names=("file_inventory.csv","demographic_inventory.csv","demographic_diagnostics_inventory.csv","kernel_inventory.csv")
    proofs=("integrity_audit.json","demographic_integrity_audit.json","demographic_diagnostics_integrity_audit.json","kernel_integrity_audit.json")
    for i,(name,proof) in enumerate(zip(names,proofs)):
        data=json.loads((CATALOG/proof).read_text(encoding="utf-8"))
        key="inventory_sha256" if i==0 else "kernel_inventory_sha256" if i==3 else "supplemental_inventory_sha256"
        if digest(CATALOG/name)!=data[key]:raise ValueError(f"Historical inventory hash mismatch: {name}")
    prior={}
    for name in names:prior.update({r["path"]:r for r in read(CATALOG/name)})
    before={p.relative_to(ROOT).as_posix():digest(p) for p in files()}
    changed=[name for name,row in prior.items() if before.get(name)!=row["sha256"]]
    if changed!=["Presentation_Evidence/folder_tree.txt"]:
        raise ValueError(f"Unexpected historical changes: {changed}")
    added=sorted(set(before)-set(prior));evidence=ROOT/"Presentation_Evidence"
    manifest=evidence/"cat_cnn_selection_manifest.csv"
    publication=json.loads((evidence/"cat_cnn_verification.json").read_text(encoding="utf-8"))
    if digest(manifest)!=publication["manifest_sha256"]:raise ValueError("Presentation manifest mismatch.")
    copies={}
    for row in read(manifest):
        source=row["source"].replace("\\","/")
        if before.get(source)!=row["sha256"]:raise ValueError(f"Presentation source mismatch: {source}")
        for field in ("by_stage_copy","by_flow_copy"):
            copy="Presentation_Evidence/"+row[field].replace("\\","/")
            if before.get(copy)!=row["sha256"] or copy in copies:raise ValueError(f"Presentation copy mismatch: {copy}")
            copies[copy]=source
    if len(copies)!=2*publication["selected_sources"]:raise ValueError("Copied asset count mismatch.")
    rows=[]
    for path in sorted(added+changed):
        source=copies.get(path,path);candidate=ROOT.joinpath(*path.split("/"));count=columns=pages=""
        if candidate.suffix.lower()==".csv":
            with candidate.open(encoding="utf-8-sig",newline="") as stream:
                reader=csv.reader(stream);header=next(reader,[]);columns,count=len(header),sum(1 for _ in reader)
        elif candidate.suffix.lower()==".parquet":
            meta=pq.read_metadata(candidate);count,columns=meta.num_rows,meta.num_columns
        elif candidate.suffix.lower()==".pdf":pages=len(PdfReader(candidate).pages)
        external = source in {"graph.py", "two_islands_graph.png"}
        rows.append({"path":path,"canonical_source":source,"presentation_copy":path in copies,
                     "stage":"external unclassified" if external else "post-test Cat + CNN extension",
                     "supports_claim":"not established by this study" if external else "exploratory additive cat/CNN prediction",
                     "what_it_is":describe(source),"interpretation_limit":"Retrospective; not independently confirmatory or causal.",
                     "bytes":candidate.stat().st_size,"sha256":before[path],"data_rows":count,"columns":columns,
                     "pdf_pages":pages,"update_type":"updated_historical_file" if path in prior else "new_file"})
    after={p.relative_to(ROOT).as_posix():digest(p) for p in files()}
    if before!=after:raise RuntimeError("Original project files changed during inventory.")
    with output.open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    report={"status":"passed","previous_inventoried_files":len(prior),"previous_files_changed":changed,
            "new_files_inventoried":len(added),"current_total_project_files_in_scope":len(before),
            "presentation_sources":len(set(copies.values())),"presentation_copies_verified":len(copies),
            "historical_inventory_sha256":{name:digest(CATALOG/name) for name in names},
            "cat_cnn_inventory_sha256":digest(output),"original_project_files_unchanged_during_collection":True,
            "raw_test_outcomes_reopened":False,"tests_or_evaluations_rerun":False,
            "lineage":"Read prior inventories in order, then apply Cat + CNN inventory; refreshed presentation tree replaces prior row."}
    write_json(audit,report);print(json.dumps(report,indent=2))


if __name__=="__main__":
    run()
