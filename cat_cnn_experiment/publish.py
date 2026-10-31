"""Publish selected Cat + CNN evidence by copy, preserving original artifacts."""

import json
import shutil
from pathlib import Path

from study import EXCLUDED, HERE, ROOT, digest, snapshot, verify_base, write_csv, write_json


def tree_lines(directory,prefix=""):
    children=sorted(directory.iterdir(),key=lambda p:(not p.is_dir(),p.name.casefold()))
    lines=[]
    for i,path in enumerate(children):
        last=i==len(children)-1
        lines.append(prefix+("`-- " if last else "|-- ")+path.name+("\\" if path.is_dir() else ""))
        if path.is_dir():lines.extend(tree_lines(path,prefix+("    " if last else "|   ")))
    return lines


def run():
    verify_base()
    evidence=ROOT/"Presentation_Evidence"
    stage=Path("By_Stage")/"16_Retrospective_Cat_CNN"
    flow=Path("By_Presentation_Flow")/"03_Diagnostics"/"10_Retrospective_Cat_CNN"
    manifest,audit=evidence/"cat_cnn_selection_manifest.csv",evidence/"cat_cnn_verification.json"
    if any(p.exists() for p in (evidence/stage,evidence/flow,manifest,audit,HERE/"publication_verification.json")):
        raise FileExistsError("Study publication already started.")
    before=json.loads((HERE/"original_hashes_before.json").read_text(encoding="utf-8"))
    prior=json.loads((HERE/"evidence_hashes_before.json").read_text(encoding="utf-8"))
    additions=json.loads((HERE/"publication_baseline_additions.json").read_text(encoding="utf-8"))
    if additions["status"]!="external_additions_recorded_without_modification":
        raise ValueError("Invalid external additions record.")
    if any(name in before for name in additions["new_files"]):
        raise ValueError("An external addition unexpectedly overlaps the original snapshot.")
    expected={**before,**additions["new_files"]}
    if snapshot(ROOT,EXCLUDED)!=expected or snapshot(evidence,set())!=prior:
        raise RuntimeError("Pre-existing project or presentation evidence changed during study.")
    summary=json.loads((HERE/"summary.json").read_text(encoding="utf-8"))
    verification=json.loads((HERE/"verification.json").read_text(encoding="utf-8"))
    additional=json.loads((HERE/"additional_verification.json").read_text(encoding="utf-8"))
    subgroup=json.loads((HERE/"subgroup_verification.json").read_text(encoding="utf-8"))
    if (summary["status"]!="complete_retrospective_exploratory" or verification["status"]!="passed" or
            additional["status"]!="passed" or subgroup["status"]!="passed"):
        raise ValueError("Study/verification incomplete.")
    if summary["verification_sha256"]!=digest(HERE/"verification.json"):
        raise ValueError("Verification record differs from summary.")
    for name,value in summary["outputs_sha256"].items():
        if digest(HERE/name)!=value:raise ValueError(f"Summary output mismatch: {name}")
    roles={
        "RESULTS.md":"Evidence-linked synthesis and interpretation limitations.",
        "protocol.json":"Cat + CNN fitting, selection and G4 unsupported policy.",
        "presentation_cat_cnn.pdf":"Matched G2/G3, curves, penalty, calibration and utility figures.",
        "presentation_abstention.pdf":"Coverage, selective accuracy and fixed-cutoff counts.",
        "g2_results.csv":"Matched original G2 and hybrid metrics.",
        "g2_contrasts.csv":"Original G2 paired comparator differences and intervals.",
        "g2_subgroups.csv":"Within/between-cat G2 metrics.",
        "g2_subgroup_contrasts.csv":"G2 within/between-cat paired intervals.",
        "g2_calibration.csv":"Overall G2 calibration bins.",
        "test_support_slices.csv":"Original-style G2 support and feature-quality slices.",
        "test_support_calibration.csv":"Support and quality subgroup calibration bins.",
        "validation_support_slices.csv":"Development support and feature-quality slices.",
        "validation_support_calibration.csv":"Development supported-subgroup reliability bins.",
        "learning_curves.csv":"Two-seed nested fitting-size learning curves.",
        "g2_attempts.csv":"Complete two-penalty 49-candidate validation log.",
        "penalty_refits.csv":"Independent 49-candidate penalty-path refits.",
        "temperature_candidates.csv":"Validation-only fixed temperature grid.",
        "g2_abstention_utility.csv":"Fixed-confidence coverage, selective accuracy and utility grid.",
        "fair_coin_control.csv":"Twenty fixed-setting independent fair-coin controls.",
        "fair_coin_provenance.json":"Random-label control status, scope and hashes.",
        "g3_results.csv":"Retrospective known-cat G3 matched test metrics.",
        "g3_contrasts.csv":"G3 held-out-photo paired intervals.",
        "g3_within_results.csv":"G3 within-cat model metrics.",
        "g3_within_contrasts.csv":"G3 within-cat held-out-photo paired intervals.",
        "g4_unsupported.json":"G4 explicitly unsupported; no unseen cat score or test predictions.",
        "additional_paired_contrasts.csv":"Matched comparisons with kernel and all-demographic models; inputs differ.",
        "design_rank_and_protocol_erratum.json":"Outcome-blind rank check and explicit correction to frozen protocol interpretation.",
        "verification.json":"Independent selected-model and cluster-bootstrap replay.",
        "additional_verification.json":"Independent penalty, comparator and support-slice verification.",
        "subgroup_verification.json":"Independent G2/G3 subgroup paired-bootstrap replay.",
        "penalty_provenance.json":"Penalty-path reproduction status and hashes.",
        "summary.json":"Study scope and summary-output hashes."
    }
    hashes={name:digest(HERE/name) for name in roles}
    for directory in (stage,flow):(evidence/directory).mkdir(parents=True,exist_ok=False)
    entries=[]
    for name,role in roles.items():
        for directory in (stage,flow):
            target=evidence/directory/name;shutil.copy2(HERE/name,target)
            if digest(target)!=hashes[name]:raise RuntimeError("Presentation copy mismatch.")
        entries.append({"source":str(Path("cat_cnn_experiment")/name),"by_stage_copy":str(stage/name),
                        "by_flow_copy":str(flow/name),"bytes":(HERE/name).stat().st_size,"sha256":hashes[name],
                        "presentation_role":role})
    if snapshot(ROOT,EXCLUDED)!=expected:raise RuntimeError("Original project or recorded external files changed during copying.")
    for name,details in prior.items():
        if digest(evidence/name)!=details["sha256"]:raise RuntimeError("Prior presentation evidence changed.")
    if any(digest(HERE/name)!=value for name,value in hashes.items()):raise RuntimeError("Study source changed during copying.")
    write_csv(manifest,entries,list(entries[0]))
    report={"status":"all_checks_passed","original_project_files_unchanged":len(before),
            "external_new_project_files_preserved":len(additions["new_files"]),
            "external_additions_record_sha256":digest(HERE/"publication_baseline_additions.json"),
            "previous_evidence_files_unchanged_before_tree_refresh":len(prior),"selected_sources":len(roles),
            "copies_verified":2*len(roles),"manifest_sha256":digest(manifest),
            "study_summary_sha256":digest(HERE/"summary.json"),"primary_claims_models_and_deployment_changed":False,
            "note":"Historical manifests remain unchanged; only folder_tree.txt refreshed."}
    write_json(audit,report);write_json(HERE/"publication_verification.json",report)
    (evidence/"folder_tree.txt").write_text("Presentation_Evidence\\\n"+"\n".join(tree_lines(evidence))+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    run()
