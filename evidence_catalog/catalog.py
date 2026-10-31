"""Read-only evidence inventory and integrity audit; does not rerun analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXCLUDED = {".git", ".venv", ".cache", "evidence_catalog"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files() -> list[Path]:
    result = []
    for directory, subdirs, names in os.walk(ROOT):
        base = Path(directory)
        if base == ROOT:
            subdirs[:] = [name for name in subdirs if name not in EXCLUDED]
        subdirs[:] = [name for name in subdirs if name not in {"__pycache__", "matplotlib_config"}]
        result.extend(base / name for name in names if (base / name).is_file())
    return sorted(result)


PURPOSES = {
    "data_audit": ("1", "C01; all empirical claims", "Schema, missingness, outcome counts and raw integrity audit.", "Audit counts are not predictive performance."),
    "cleaning_log": ("1", "all empirical claims", "Explicit anomaly, retained-response and quarantine decisions.", "Identical-looking responses are not automatically duplicate observations."),
    "raw_hashes": ("1", "all empirical claims", "SHA-256 inventory of source CSVs and photos, including duplicate-content audit.", "Hash identity is not scientific validity."),
    "official_test_coverage": ("1", "scope; deployment", "Mutually exclusive official target overlap against all labeled training rows.", "Unlabeled target coverage; no official performance."),
    "official_test_coverage_stage4": ("4", "scope; deployment", "Official target structure and fitting graph/feature availability.", "Unlabeled target coverage; no official performance."),
    "rows_canonical": ("2", "C01-C06", "Audited response IDs, canonical pairs, orientation and original partition; original G2 test labels masked.", "Raw CSV still contains labels; masking is procedural."),
    "split_manifest": ("2", "C02-C06", "Original pair-disjoint 70/15/15 split and stratification.", "Not respondent-disjoint or historically pristine."),
    "common_population_manifest": ("3", "C02-C06", "Feature-matched fitting and structural held-out membership.", "Population comparability, not prediction quality."),
    "common_population_report": ("3", "C02-C06", "Frozen population counts and final admissions alongside historical QA statuses.", "Historical pending flags must not be confused with final admission."),
    "feature_quality": ("3", "C04; C06", "Pre-review automated feature coverage and successful extraction summary.", "Use admission decisions for final reviewed status."),
    "feature_admission_decisions": ("3", "C04; C06", "Recorded admission of H, C and Y after quality review.", "Does not validate attribute truth or preference usefulness."),
    "feature_review": ("3", "C04; C06", "128 outcome-blind individual review records for 64 fitting photos.", "Small quality sample, not full ground-truth annotation."),
    "feature_review_manifest": ("3", "C04; C06", "Fixed review sample, reviewers, flags and immutable hashes.", "Sample provenance, not causal or complete annotation evidence."),
    "graph_summary": ("4", "C02-C06", "Fitting photo/cat undirected and directed connectivity statistics.", "Penalized finiteness differs from unpenalized BT MLE existence."),
    "graph_support": ("4", "C02-C06", "Visualization of fitting graph support and strict-G2 coverage.", "Graph coverage is not prediction accuracy."),
    "graph_components": ("4", "C02-C06", "Node-level components, opponents, wins/losses and fitting support.", "Free-score comparison must respect fitting components."),
    "graph_edges": ("4", "C02-C06", "Fitting-only weighted undirected and winner-to-loser graph edges.", "Not evidence of real preference cycles."),
    "g2_eligibility": ("4", "C02-C06", "Frozen row-level full/common strict-G2 masks and reasons.", "Eligibility uses structure rather than held-out outcomes."),
    "split_coverage": ("4", "C02-C06", "Eligible response/pair counts by partition and comparison type.", "Coverage is operational, not a power guarantee."),
    "simulations": ("5", "C07", "Accepted fits and expected-risk results for BT, opposite-group and cycle simulations.", "Known-truth demonstration; not a real-data cycle finding."),
    "simulations_ABC": ("5", "C07", "Three known-truth recovery/pooling/scalar-ranking illustrations.", "Synthetic references are not real-data information limits."),
    "pair_summary": ("6", "C01; S02", "Development canonical pair proportions, support and Wilson intervals.", "Working-model intervals; tiny support; no respondent linkage."),
    "pair_support_summary": ("6", "C01; S02", "Pair support distributions by development subset and comparison type.", "Sparse support does not identify a human randomness mechanism."),
    "photo_support": ("6", "C01; C06", "Unique photo response/opponent support by development subset.", "Repeated photos induce dependence; counts are not respondents."),
    "context_support": ("6", "C05; S02", "Context response, pair, photo and cat support.", "Eligibility thresholds do not guarantee precise interactions."),
    "context_pair_support": ("6", "S02", "Pair-by-context response cells and observed proportions.", "Different pair compositions prevent broad preference attribution."),
    "context_overlap": ("6", "S02", "Shared pair counts and registered per-cell support gate.", "No adequate matched cells here; no adjusted context effect."),
    "pair_entropy": ("6", "C01", "In-sample empirical canonical pair entropy summaries.", "Not an oracle, information limit or held-out predictor."),
    "disagreement_examples": ("6", "C01", "Eligibility records for fixed high-support example selection.", "No eligible examples; do not lower thresholds after seeing results."),
    "disagreement": ("6", "C01", "Empirical pair-proportion/disagreement visualization.", "Development description, not a causal mechanism or information bound."),
    "support_distribution": ("6", "C01", "Development pair support distribution.", "Operational evidence of small per-pair sample sizes."),
    "all_fit_attempts": ("7", "C02-C06", "Complete candidate selection log including numerical acceptance and fallbacks.", "Validation search record, not independent test evidence."),
    "validation_results": ("7", "C02-C06", "Selected-model development losses and penalties.", "Selection evidence, not held-out test performance."),
    "validation_predictions": ("7", "C02-C06", "Saved matched validation predictions for reconstruction and diagnosis.", "Validation was used for selection."),
    "model_selection": ("7", "C02-C06", "Selected parameters, coefficients, context mapping and model aliases.", "Frozen choices do not remove earlier-analysis exposure."),
    "model_resources": ("7", "C06", "Stored coefficient/backbone counts and scoped fitting/inference timings.", "Counts are not effective degrees of freedom; timings exclude specified IO/extraction."),
    "diagnostic_results": ("8", "C06", "Development subgroup/support/feature-quality performance.", "Diagnostic, not independent discoveries."),
    "diagnostic_calibration": ("8", "C06", "Development reliability bins/plots.", "Calibration does not establish information completeness."),
    "diagnostic_confidence": ("8", "C06", "Development confidence and loss-distribution summaries.", "Confidence is not guaranteed correctness."),
    "diagnostic_interpretation": ("8", "C06", "Bounded diagnostic interpretation ledger.", "Interpretations must remain linked to quantitative checks."),
    "worst_validation_losses": ("8", "C06", "Recorded worst-loss validation cases and comparisons.", "Examples explain model errors, not representative causal effects."),
    "penalty_path_refits": ("8", "C06", "Reproduced registered penalty paths and fit/validation scores.", "No post-test penalty-grid expansion."),
    "penalty_paths": ("8", "C06", "Fit/validation loss and coefficient-norm penalty paths.", "Supports shrinkage/variance diagnoses, not a unique cause."),
    "learning_curves": ("8", "S04; C06", "Nested fitting-size results on one common eligible validation population.", "Not full validation population or an information-limit estimate."),
    "learning_curve_candidates": ("8", "S04; C06", "All learning-curve fitting/tuning candidates.", "Additional development tuning; not primary-test candidates."),
    "learning_curve_context_support": ("8", "S04", "Support/reference/fallback changes across nested fitting subsets.", "Effective interaction models change as sample support changes."),
    "learning_curve_preprocessing": ("8", "S04", "Fit-only preprocessing saved separately for learning-curve subsets.", "Method/provenance rather than new empirical preference evidence."),
    "synthetic_label_control": ("8", "C07", "Independent fair-coin fitting/validation controls at fixed procedures.", "Passing an operational control is not a universal leakage certificate."),
    "grouped_ablations": ("8", "C06", "Refitted H/Y feature-group-drop development comparisons.", "Conditional predictive value, not percentage of behavior caused by features."),
    "representation_recovery": ("8", "S05; C06", "Ridge recovery of estimated fitting-photo M2 scores by H/C/Y.", "Estimated, noisy in-sample targets; not independent human preference ground truth."),
    "cnn_pca_diagnostic": ("8", "S07; C06", "Development-only PCA64 candidate and selection log.", "Secondary probe, not promoted to primary representation."),
    "temperature_diagnostic": ("8", "C06", "Validation-selected fixed temperature grid diagnostic.", "Secondary; must not select using internal-test results."),
    "feature_contributions": ("8", "C06; model-output explanation", "Exact linear feature contributions for recorded validation examples.", "Explains a model prediction, not a causal human preference."),
    "development_decisions": ("8", "C06", "Explicit additional development and diagnostic decisions.", "Decision record, not a numerical result."),
    "model_freeze": ("9", "C02-C06", "Pre-test registry, selected settings, masks and source/artifact SHA-256 seal.", "Integrity and temporal provenance, not historical pristine-data certification."),
    "test_predictions": ("9", "C02-C06", "Saved frozen response outcomes, logits, probabilities and losses.", "Reanalysis uses previously exposed outcomes, not a new independent test."),
    "main_results": ("9", "C02-C06; S01", "Matched frozen test metrics, uncertainty, fit/validation and resource columns.", "Point-model ranking does not establish every paired difference."),
    "primary_contrasts": ("9", "C02-C05", "Four prespecified paired contrasts with approximate 98.75% intervals.", "Conditional unordered-pair bootstrap omits other dependence and training variability."),
    "secondary_results": ("9", "C05-C06", "Context/side/position/temperature secondary paired contrasts.", "95% pointwise intervals unadjusted for secondary multiplicity."),
    "subgroup_results": ("9", "S03; C06", "Frozen-test comparison-type, context, support and quality subgroup metrics.", "Low-support slices cannot support strong subgroup conclusions."),
    "test_calibration": ("9", "C06", "Frozen-test probability bins, observed frequencies, counts and support flags.", "Bin estimates are noisy; calibration is not information completeness."),
    "calibration": ("9", "C06", "Frozen-test reliability plot.", "Does not prove optimality, causal explanation or no missing information."),
    "diagnostic_synthesis": ("10/10.5", "C01-C07; S01-S10; X105-01-X105-06", "Evidence-linked claim/result/interpretation/alternatives/nonclaim/next-evidence ledger.", "Synthesis is not a new independent dataset."),
    "claims_final": ("10/10.5", "C01-C07; S01-S10; X105-01-X105-06", "Current 23-row final claims ledger, including appended exploratory demographics.", "Original 17-row hashes apply to preserved prefix, not current whole file."),
    "presentation_feature_coefficients": ("10 supplement", "model association", "Signed frozen H/Y coefficients selected for presentation.", "Scaling/coding affect magnitudes; not standalone importance or causality."),
    "presentation_context_interactions": ("10 supplement", "model association", "Selected CNN weight deviations by pet-care group against reference.", "CNN coordinates lack individual semantic labels."),
    "presentation_feature_importance_H": ("10 supplement", "model association", "Five positive and five negative frozen handcrafted coefficients.", "Predictive association, not causal effect or measured feature-ablation importance."),
    "presentation_feature_importance_Y": ("10 supplement", "model association", "Five positive and five negative frozen automated-attribute coefficients.", "CLIP labels are predictions, not verified breed or causal effects."),
    "exploratory_demographic_semantic_rates": ("10.5", "X105-01-X105-06", "Development demographic semantic-trait discriminating comparison rates.", "Unadjusted different-pair composition; no causal demographic preferences."),
    "exploratory_demographic_semantic_overlap": ("10.5", "X105-01-X105-06", "Shared semantic-discriminating pairs and matched per-group support.", "No adequately supported matched pair-group contrasts."),
    "exploratory_demographic_semantic_support": ("10.5", "X105-01-X105-06", "Demographic development response/pair support.", "Missing demographics separate; counts do not establish population representativeness."),
    "exploratory_demographic_semantics": ("10.5", "X105-01-X105-06", "Support-gated demographic semantic-rate heatmaps.", "Descriptive T2 correlation, not adjusted preference or causal effect."),
    "exploratory_demographic_semantic_support": ("10.5", "X105-01-X105-06", "Development demographic support table/figure.", "Support is operational, not a power guarantee."),
    "exploratory_abstention_utility": ("10 supplement", "exploratory utility", "Frozen-test confidence-grid coverage, accuracy, utility and cluster intervals.", "Post-test thresholds not optimized; utility includes abstentions in denominator."),
    "presentation_abstention_utility": ("10 supplement", "exploratory utility", "Four-page frozen-test selective prediction and utility dashboard.", "Greater selected accuracy is not necessarily higher total utility."),
    "stage11_contributions_draft": ("11", "administrative", "User-authorized individual contribution/presentation placeholders.", "Not certified actual completed contributions."),
    "third_party_resources": ("11", "reproducibility/licensing", "Software/checkpoint/dataset resources and distribution restrictions.", "Not blanket legal/license clearance."),
    "archive_reproduction": ("11", "reproducibility", "Clean-environment exact archive cached-statistical reconstruction record.", "Historical test run, not a newly rerun full suite; no full-image end-to-end rerun."),
    "reproduction_statistical": ("11", "reproducibility", "Fresh-process 16-model frozen-setting and saved-outcome reconstruction.", "No retuning or unseen official performance."),
    "feature_regeneration_H": ("11", "reproducibility", "Fixed 16-image handcrafted extraction comparison.", "Subset numerical reproducibility, not all-photo re-extraction."),
    "feature_regeneration_C": ("11", "reproducibility", "Fixed 16-image CNN extraction comparison.", "Subset numerical reproducibility, not all-photo re-extraction."),
    "feature_regeneration_Y": ("11", "reproducibility", "Fixed eight-image GPU YOLO-CLIP extraction/mask/label comparison.", "Observed tolerance, not a promise of cross-device bitwise equality."),
}


def describe(relative: Path) -> tuple[str, str, str, str]:
    parts, stem = relative.parts, relative.stem
    first = parts[0]
    if first == "dataset":
        return ("1", "all empirical claims", "Original input photo or response file; no independent model result.",
                "Training labels historically exposed; official labels blank; raw photos are research/educational assets.")
    if first == "legacy":
        return ("reference only", "implementation provenance", "Retained teammate pipeline/reference implementation.",
                "Not a new result and not the authoritative current evaluation.")
    if first == "src" or first == "tests":
        return ("implementation", "methods/tests", "Executable implementation or focused test supporting artifact generation and safeguards.",
                "Presence of a test file is not evidence a new test run passed.")
    if first == "models":
        return ("7", "C02-C06", "Selected frozen fitting-only identity/shared/contextual model coefficients.",
                "Full/common copies and pooled aliases are not independent models or evidence replications.")
    if first == "features":
        return ("3", "C04; C06", "Outcome-blind feature cache or extraction/schema/checkpoint manifest.",
                "Automated measurements, not verified semantic ground truth; use final admission records.")
    if first == "artifacts" and len(parts) > 1 and parts[1] == "review_panels":
        return ("3", "C04; C06", "One of 64 fixed outcome-blind fitting-photo QA panels.",
                "Small manual quality sample, not complete annotation or empirical preference evidence.")
    if first == "artifacts" and len(parts) > 1 and parts[1] == "preprocessing":
        return ("7", "C02-C06", "Learned preprocessing fit on unique original fitting photos.",
                "Must not reuse for a new split without fitting-only re-estimation.")
    if first in {"g3_experiment", "g4_experiment", "official_prediction"}:
        stage = "22 retrospective" if first == "g3_experiment" else "new retrospective G4" if first == "g4_experiment" else "23 deployment"
        claim = "retrospective G3" if first == "g3_experiment" else "retrospective G4" if first == "g4_experiment" else "prediction completeness, not accuracy"
        if "within_cat_subgroup" in parts:
            stage, claim = "G3 post-test subgroup", "retrospective within-cat G3"
        descriptions = {
            "coverage": "Outcome-blind reserved population coverage and eligibility gate.",
            "protocol": "Fixed scope, thresholds, selection/uncertainty rules and limitations.",
            "core_split": "Separate nonreserved pair-disjoint core split; only fitting rows train.",
            "row_allocation": "Per-response core/reserved/quarantine assignment and reason.",
            "photo_reservation": "Duplicate-aware reserved photo group assignment.",
            "cat_reservation": "Duplicate-aware held-out cat group assignment.",
            "selected_models": "Validation-selected model coefficients, context mappings and fit-only preprocessing.",
            "freeze": "Separate experiment freeze JSON or SHA-256 seal.",
            "all_fit_attempts": "Recorded fitting/tuning candidates and numerical acceptance.",
            "validation_results": "Validation-only selected-model losses and penalties.",
            "test_predictions": "Saved experiment response outcomes, logits, probabilities and losses.",
            "test_results": "Matched experiment test log loss, Brier, accuracy and cluster intervals.",
            "test_contrasts": "Paired response-loss contrasts and cluster bootstrap intervals.",
            "within_cat_test_results": "Saved-prediction within-cat subgroup model metrics and photo-cluster intervals.",
            "within_cat_contrasts": "Within-cat paired exploratory model differences and intervals.",
            "within_cat_summary": "Within-cat support, previous-outcome overlap and M1 score-cancellation explanation.",
            "presentation_g3_results": "One-page retrospective known-cat unseen-photo results.",
            "presentation_g4_results": "One-page retrospective one-unseen-cat results; M1-zero cold-start convention.",
            "presentation_within_cat_g3": "One-page within-cat saved-prediction subgroup model results.",
            "official_predictions": "All 6224 official rows with only choice/prob_a filled; no upload.",
            "routing": "Per-official-row deployment route, novelty count, logit and probability.",
            "deployment_policy": "Sealed validation-only model/routing rule and transfer limitations.",
            "deployment_models": "Separate all-labeled-response refit of M0/M1/M4-C and unique-labeled-photo preprocessing.",
            "deployment_provenance": "Official output hashes, route counts, scope and format verification.",
            "verification": "Recorded independent numerical/output/hash reconstruction.",
            "development_access": "Record of separate fitting/validation outcome access and historical overlap.",
            "evaluation_started": "Write-once test-access guard and prior partition overlap.",
            "deployment_started": "Write-once all-labeled deployment-refit access record.",
        }
        purpose = descriptions.get(stem)
        if purpose is None:
            purpose = ("Method/test/documentation supporting this isolated experiment." if relative.suffix in {".py", ".md"}
                       else "Experiment-specific provenance, preservation snapshot or verification record.")
        limit = ("All-labeled deployment, not a new evaluation; no official labels or performance claim."
                 if first == "official_prediction" else
                 "Retrospective outcome exposure; no independent confirmation or causal inference; G2 frozen.")
        return stage, claim, purpose, limit
    if stem in PURPOSES:
        stage, claims, purpose, limit = PURPOSES[stem]
        return stage, claims, purpose, limit
    if stem.endswith("provenance") or stem.endswith("protocol") or stem.startswith("stage"):
        return ("supporting provenance", "linked stages/claims", "Stage-specific source/output hashes, gate, status or decision registry.",
                "Verify historical scope and append lineage; a status file alone is not a new result.")
    if relative.suffix == ".py":
        return ("implementation", "methods", "Reproducible artifact-generation or validation source.", "Not an empirical result by itself.")
    if first == "artifacts":
        return ("supporting provenance", "linked stages/claims", "Recorded access, integrity, status or environment/device verification.",
                "Operational record, not an independently established empirical claim.")
    if first in {"processed", "outputs", "notebooks"}:
        return ("legacy/unclassified", "none until linked", "Working output or notebook; inspect lineage before citing.",
                "Not automatically an accepted frozen result.")
    return ("governance/report/environment", "project scope or reproduction",
            "Root governance, claim registry, documentation, report, archive or dependency specification.",
            "Report/contributions may be draft; dependency files are methods, not preference evidence.")


def run() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Set PYTHONDONTWRITEBYTECODE=1.")
    if any((HERE / name).exists() for name in ("file_inventory.csv", "integrity_audit.json")):
        raise FileExistsError("Catalog outputs already exist.")
    source_files = files()
    before = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    copies = {}
    evidence = ROOT / "Presentation_Evidence"
    for manifest in evidence.glob("*selection_manifest.csv"):
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                source = row["source"].replace("\\", "/")
                for key in ("by_stage_copy", "by_flow_copy"):
                    copy = "Presentation_Evidence/" + row[key].replace("\\", "/")
                    if copy in copies:
                        raise ValueError(f"Duplicate presentation selection: {copy}")
                    if sha(ROOT.joinpath(*source.split("/"))) != row["sha256"]:
                        raise ValueError(f"Presentation source hash changed: {source}")
                    if sha(ROOT.joinpath(*copy.split("/"))) != row["sha256"]:
                        raise ValueError(f"Presentation copy hash changed: {copy}")
                    copies[copy] = source
    rows = []
    for path in source_files:
        rel = path.relative_to(ROOT)
        posix = rel.as_posix()
        original = copies.get(posix, posix)
        info = describe(Path(original))
        row_count = page_count = ""
        columns = ""
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader, [])
                row_count = sum(1 for _ in reader)
                columns = len(header)
        elif path.suffix.lower() == ".parquet":
            metadata = pq.read_metadata(path)
            row_count, columns = metadata.num_rows, metadata.num_columns
        elif path.suffix.lower() == ".pdf":
            page_count = len(PdfReader(path).pages)
        rows.append({"path": posix, "canonical_source": original, "presentation_copy": posix in copies,
                     "stage": info[0], "supports_claim": info[1], "what_it_is": info[2],
                     "interpretation_limit": info[3], "bytes": path.stat().st_size,
                     "sha256": before[str(rel)], "data_rows": row_count, "columns": columns, "pdf_pages": page_count})
    checked = 0
    reports = ["artifacts/model_freeze.json", "artifacts/graph_provenance.json",
               "artifacts/stage7_verification.json", "artifacts/evaluation_provenance.json",
               "artifacts/stage11_report_provenance.json", "artifacts/stage11_package_provenance.json"]
    for folder in ("g3_experiment", "g4_experiment"):
        reports += [folder + "/freeze.json", folder + "/evaluation_provenance.json"]
    for name in reports:
        path = ROOT.joinpath(*name.split("/"))
        data = json.loads(path.read_text(encoding="utf-8"))
        if name.endswith("/freeze.json") and not name.startswith("artifacts/"):
            seal = path.with_suffix(".sha256")
        elif name == "artifacts/model_freeze.json":
            seal = ROOT / "artifacts" / "model_freeze.sha256"
        else:
            seal = None
        if seal is not None and seal.read_text(encoding="ascii").strip() != sha(path):
            raise ValueError(f"Freeze seal mismatch: {name}")
        groups = ("artifacts_sha256", "input_sha256", "inputs_sha256", "output_sha256",
                  "outputs_sha256", "source_sha256", "experiment_sha256", "root_input_sha256")
        for group in groups:
            entries = data.get(group, {})
            if not isinstance(entries, dict):
                continue
            for source, expected in entries.items():
                if not isinstance(expected, str) or len(expected) != 64:
                    continue
                base = path.parent if group == "experiment_sha256" or (
                    group == "output_sha256" and name.startswith(("g3_experiment/", "g4_experiment/"))) else ROOT
                candidate = base.joinpath(*source.replace("\\", "/").split("/"))
                if not candidate.is_file() or sha(candidate) != expected:
                    raise ValueError(f"Recorded provenance mismatch: {name} -> {source}")
                checked += 1
    stage10 = json.loads((ROOT / "artifacts" / "stage10_provenance.json").read_text(encoding="utf-8"))
    append = json.loads((ROOT / "artifacts" / "stage10_5_provenance.json").read_text(encoding="utf-8"))
    for name, lineage in append["append_lineage"].items():
        path = ROOT.joinpath(*name.split("/"))
        prefix = hashlib.sha256(path.read_bytes()[:lineage["parent_bytes"]]).hexdigest()
        if (sha(path) != lineage["current_sha256"] or prefix != lineage["parent_sha256"] or
                lineage["parent_sha256"] != stage10["output_sha256"][name]):
            raise ValueError(f"Synthesis append lineage mismatch: {name}")
    archive = ROOT / "Group_19_Code.zip"
    proof = json.loads((ROOT / "artifacts" / "archive_reproduction.json").read_text(encoding="utf-8"))
    if sha(archive) != proof["archive_sha256"]:
        raise ValueError("Archive differs from the historical clean reproduction.")
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("Archive CRC check failed.")
        archive_entry_count = len(zipped.infolist())
    after = {str(p.relative_to(ROOT)): sha(p) for p in files()}
    if before != after:
        raise RuntimeError("Original project files changed during evidence collection.")
    with (HERE / "file_inventory.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    audit = {"status": "passed", "collected_utc": datetime.now(timezone.utc).isoformat(),
             "project_files_inventoried": len(rows), "presentation_copies_verified": len(copies),
             "canonical_sources_in_presentation": len(set(copies.values())),
             "provenance_hash_references_verified": checked, "original_files_unchanged": True,
             "freeze_seals_verified": ["G2", "G3", "G4"],
             "synthesis_append_lineage_verified": True,
             "archive_matches_clean_reproduction_hash": True, "archive_crc_verified": True,
             "archive_entries": archive_entry_count,
             "by_root_directory": dict(sorted(Counter(Path(r["path"]).parts[0] for r in rows).items())),
             "by_extension": dict(sorted(Counter(Path(r["path"]).suffix.lower() for r in rows).items())),
             "excluded_root_directories": sorted(EXCLUDED),
             "excluded_runtime_directories": ["__pycache__", "matplotlib_config"],
             "no_models_refit": True, "no_tests_rerun": True, "no_evaluations_rerun": True,
             "raw_outcome_values_not_analyzed": True,
             "scope_note": "CSV metadata includes counting records in raw files but no outcome analysis. Saved table/PDF metadata and source hashes are read; historical tests are not rerun.",
             "inventory_sha256": sha(HERE / "file_inventory.csv")}
    (HERE / "integrity_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    run()
