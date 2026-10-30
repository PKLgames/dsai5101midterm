"""Stage 10 evidence-linked diagnostic synthesis; no fitting or evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .development import _hash
from .freeze import validate_freeze

OUTPUTS = ("claims_final.csv", "tables/diagnostic_synthesis.csv", "artifacts/stage10_provenance.json")
LIMITATIONS = (
    "Internal holdout is not certified pristine because of earlier analyses. "
    "Pair-bootstrap intervals are conditional on fitted models and development choices; they omit "
    "training variability, respondent linkage, and shared-image dependence. "
    "Secondary intervals are descriptive and not multiplicity-adjusted."
)


class Evidence:
    """Select exact saved cells or explicit aggregates and retain their provenance."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.tables: dict[str, list[dict[str, str]]] = {}
        self.facts: list[dict[str, Any]] = []
        self.source_hashes: dict[str, str] = {}

    def read(self, name: str) -> list[dict[str, str]]:
        if name not in self.tables:
            path = self.root.joinpath(*name.split("/"))
            before = _hash(path)
            if path.suffix == ".parquet":
                rows = [{key: "" if value is None else str(value) for key, value in row.items()}
                        for row in pq.read_table(path).to_pylist()]
            else:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                        raise ValueError(f"Missing or duplicate CSV headers in {name}")
                    rows = list(reader)
            if any(None in row or any(value is None for value in row.values()) for row in rows):
                raise ValueError(f"Malformed CSV rows in {name}")
            if _hash(path) != before:
                raise RuntimeError(f"Evidence changed during read: {name}")
            self.tables[name] = rows
            self.source_hashes[name] = before
        return self.tables[name]

    def value(self, name: str, selector: dict[str, str], column: str, operation: str = "cell") -> Any:
        rows = [row for row in self.read(name) if all(row.get(key) == value for key, value in selector.items())]
        if operation == "cell":
            if len(rows) != 1:
                raise ValueError(f"Expected exactly one evidence row in {name}: {selector}; got {len(rows)}")
            value: Any = rows[0][column]
        elif operation == "count":
            value = len(rows)
        elif operation == "count_mixed_choices":
            value = 0
            for row in rows:
                n, k = int(row["responses"]), int(row[column])
                if n <= 0 or not 0 <= k <= n:
                    raise ValueError(f"Invalid pair counts in {name}")
                value += 0 < k < n
        else:
            if not rows:
                raise ValueError(f"Cannot aggregate empty evidence in {name}: {selector}")
            values = [float(row[column]) for row in rows]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Nonfinite evidence in {name}/{column}")
            if operation == "sum":
                value = sum(values)
            elif operation == "max":
                value = max(values)
            elif operation == "mean":
                value = sum(values) / len(values)
            elif operation == "count_positive":
                value = sum(item > 0 for item in values)
            else:
                raise ValueError(f"Unknown evidence operation: {operation}")
        self.facts.append({
            "evidence_id": f"E{len(self.facts)+1:04d}", "source": name,
            "sha256": self.source_hashes[name], "selector": selector, "column": column,
            "operation": operation, "matched_rows": len(rows), "value": value,
        })
        return value

    def number(
        self, name: str, selector: dict[str, str], column: str, operation: str = "cell",
        digits: int | None = 6,
    ) -> str:
        value = float(self.value(name, selector, column, operation))
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite reported number: {name}/{column}")
        if digits is None:
            if not value.is_integer():
                raise ValueError(f"Expected integer evidence: {name}/{column}={value}")
            rendered = str(int(value))
        else:
            rendered = f"{value:.{digits}f}"
        self.facts[-1]["rendered"] = rendered
        return rendered

    def references_since(self, start: int) -> str:
        return json.dumps(self.facts[start:], ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def direction(low: float, high: float) -> str:
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("Invalid uncertainty interval.")
    if low > 0:
        return "candidate_improves_under_bootstrap"
    if high < 0:
        return "candidate_worsens_under_bootstrap"
    return "direction_uncertain"


def primary_claim(evidence: Evidence, contrast: str) -> tuple[str, str]:
    path = "tables/primary_contrasts.csv"
    selector = {"contrast": contrast}
    row = next((row for row in evidence.read(path) if row["contrast"] == contrast), None)
    if row is None:
        raise ValueError(f"Missing primary contrast: {contrast}")
    if float(row["interval_level"]) != .9875:
        raise ValueError(f"Primary interval is not the registered level: {contrast}")
    for model in (row["reference"], row["candidate"]):
        main = [item for item in evidence.read("tables/main_results.csv")
                if item["population"] == "common" and item["model_id"] == model]
        if len(main) != 1 or main[0]["responses"] != row["responses"] or main[0]["pairs"] != row["pairs"]:
            raise ValueError(f"Primary comparison population mismatch: {contrast}/{model}")
    level = evidence.number(path, selector, "interval_level")
    difference = evidence.number(path, selector, "response_loss_difference")
    low = evidence.number(path, selector, "response_interval_low")
    high = evidence.number(path, selector, "response_interval_high")
    responses = evidence.number(path, selector, "responses", digits=None)
    pairs = evidence.number(path, selector, "pairs", digits=None)
    pair_difference = evidence.number(path, selector, "pair_loss_difference")
    pair_low = evidence.number(path, selector, "pair_interval_low")
    pair_high = evidence.number(path, selector, "pair_interval_high")
    point = float(row["response_loss_difference"])
    source_metrics = evidence.read("tables/main_results.csv")
    reference = next(item for item in source_metrics if item["population"] == "common"
                     and item["model_id"] == row["reference"])
    candidate = next(item for item in source_metrics if item["population"] == "common"
                     and item["model_id"] == row["candidate"])
    if not math.isclose(float(reference["response_log_loss"]) - float(candidate["response_log_loss"]),
                        point, abs_tol=1e-12):
        raise ValueError(f"Primary contrast does not reconcile with model losses: {contrast}")
    message = (
        f"Frozen common strict-G2 test: {row['reference']} minus {row['candidate']} response-weighted "
        f"log loss = {difference} nats/response, interval [{low}, {high}] "
        f"at confidence level {level}; {responses} responses across {pairs} unordered pairs. "
        f"Pair-weighted difference = {pair_difference} nats/pair, interval [{pair_low}, {pair_high}]. "
        "Positive differences favor the candidate."
    )
    return message, direction(float(row["response_interval_low"]), float(row["response_interval_high"]))


def build_claims(e: Evidence, selection: dict[str, Any], planned: list[dict[str, str]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    original = {row["claim_id"]: row for row in planned}
    if set(original) != {f"C{i:02d}" for i in range(1, 8)} or len(original) != len(planned):
        raise ValueError("The original question registry is missing or contains unexpected/duplicate claims.")
    block = selection["selected_representation"]
    if block != "C":
        raise ValueError("This synthesis is written for the frozen C selection; do not silently relabel another run.")

    def add(
        claim_id: str, question: str, partition: str, result: str, interpretation: str,
        alternatives: str, nonclaim: str, next_evidence: str, status: str, start: int,
        evidence_status: str = "Real-data claim",
    ) -> None:
        registered = original.get(claim_id)
        claims.append({
            "claim_id": claim_id, "task_roles": registered["task_roles"] if registered else "T2 and T3",
            "target": "None", "analysis_target": "synthetic" if evidence_status == "Known-truth demonstration" else
                "strict_G2" if "test" in partition or "validation" in partition else "descriptive",
            "activity": "Clarify" if evidence_status == "Methodological rule" else "Explain",
            "evidence_status": evidence_status, "progress": "synthesized", "result_status": status,
            "question": registered["question"] if registered else question, "partition": partition,
            "measured_result": result, "supported_interpretation": interpretation,
            "alternative_explanations": alternatives, "nonclaim": nonclaim,
            "next_discriminating_evidence": next_evidence,
            "source_evidence_json": e.references_since(start), "limitations": LIMITATIONS,
        })

    start = len(e.facts)
    pair_path = "tables/pair_support_summary.csv"
    base = {"partition": "development", "comparison_type": "all"}
    singleton = e.number(pair_path, {**base, "response_band": "1"}, "pairs", digits=None)
    multi = e.number(pair_path, {**base, "response_band": "2-4"}, "pairs", digits=None)
    n = e.number(pair_path, base, "responses", "sum", digits=None)
    count = e.number(pair_path, base, "pairs", "sum", digits=None)
    maximum = e.number("tables/pair_summary.csv", {"partition": "development"}, "responses", "max", digits=None)
    mixed = e.number("tables/pair_summary.csv", {"partition": "development"},
                     "canonical_i_choices", "count_mixed_choices", digits=None)
    example_status = e.value("tables/disagreement_examples.csv",
                             {"partition": "development", "selection": "closest_to_half"}, "status")
    if example_status != "no_pair_with_at_least_20_responses":
        raise ValueError("Disagreement example availability differs from this synthesis.")
    add("C01", "", "development=fit+validation",
        f"The development set has {n} responses across {count} pairs: {singleton} singleton pairs and "
        f"{multi} pairs in the registered repeated-response band. Maximum pair support is {maximum}; "
        f"{mixed} pairs have responses choosing both canonical endpoints. No pair qualifies for the "
        "prespecified high-support disagreement examples.",
        "Observed disagreement coexists with very small per-pair support; unanimous empirical proportions "
        "should not be interpreted as precise underlying preferences. Wilson intervals use a binomial working model.",
        "Finite response counts, pair composition, and unavailable respondent linkage can all affect observed proportions.",
        "No individual-consistency, behavioral attribution, oracle entropy, or information-limit conclusion.",
        "Collect more responses to the same pairs and retain respondent linkage with appropriate consent.",
        "descriptive_low_pair_support", start)

    for claim_id, label, interpretation, alternatives, nonclaim, future in (
        ("C02", "cat_identity",
         "Cat identity improves prediction over the constant baseline for the frozen known-photo unseen-pair target; "
         "the response-weighted interval excludes no difference.",
         "The result is target-specific; residual dependence and earlier selection can affect precision.",
         "This is not objective attractiveness, causal cat appeal, or evidence of transfer to unseen cats.",
         "Replicate the frozen procedure with new pairs or independent respondents."),
        ("C03", "photo_detail",
         "The photo-identity point estimate is worse than cat identity, but the paired interval crosses no difference. "
         "Additional predictive value from photo identity is not established by this test.",
         "Photo-score estimation variance, regularization, and finite holdout variability remain plausible.",
         "An interval spanning no difference is not equivalence or evidence that photo-level signal is absent.",
         "Use independently collected comparisons and stronger support per photo; treat new remedies as exploratory."),
        ("C04", "visual_sharing",
         "The CNN shared scorer has a slightly better point estimate than photo identity, but the paired interval "
         "crosses no difference. The validation-selected advantage is not clearly resolved on the internal test.",
         "Validation selection optimism, finite holdout variation, feature geometry, and shrinkage remain plausible.",
         "Do not assert confirmed superiority to photo identity, semantic meaning for CNN coordinates, or causal visual effects.",
         "Replicate with an independent evaluation set; retain frozen H and Y results rather than discarding negative extensions."),
        ("C05", "recorded_context",
         "The matched CNN contextual procedure yields a small response-weighted improvement under the conditional "
         "pair-bootstrap approximation; its primary interval excludes no difference.",
         "Context measurement, uneven support, pair composition, reference coding, and residual dependence remain relevant.",
         "This is not a causal pet-care effect or a claim that every subgroup benefits.",
         "Replicate with independently sampled respondents and deliberately overlapping pair/context cells."),
    ):
        start = len(e.facts)
        result, status = primary_claim(e, label)
        expected = "candidate_improves_under_bootstrap" if claim_id in {"C02", "C05"} else "direction_uncertain"
        if status != expected:
            raise ValueError(f"Frozen evidence no longer supports the written interpretation of {claim_id}.")
        if claim_id == "C05":
            if selection["selected"]["M4*"]["pooled"]:
                raise ValueError("Context was described as selected despite a pooled primary fallback.")
            path = "tables/secondary_results.csv"
            selector = {"contrast": "context_with_group_side"}
            side = e.number(path, selector, "response_loss_difference")
            low = e.number(path, selector, "response_interval_low")
            high = e.number(path, selector, "response_interval_high")
            level = e.number(path, selector, "interval_level")
            result += (f" The side-adjusted secondary contrast is {side}, interval [{low}, {high}] "
                       f"at confidence level {level}; secondary multiplicity is not adjusted.")
        add(claim_id, "", "frozen_common_strict_G2_test", result, interpretation, alternatives,
            nonclaim, future, status, start)

    start = len(e.facts)
    path = "tables/validation_results.csv"
    train_m1 = e.number(path, {"model": "common:M1"}, "train_loss")
    train_m2 = e.number(path, {"model": "common:M2"}, "train_loss")
    val_m1 = e.number(path, {"model": "common:M1"}, "validation_loss")
    val_m2 = e.number(path, {"model": "common:M2"}, "validation_loss")
    temp = e.number("tables/temperature_diagnostic.csv", {"model": "M2", "selected": "True"}, "temperature", digits=1)
    delta = e.number("tables/secondary_results.csv", {"contrast": "temperature_M2"}, "response_loss_difference")
    low = e.number("tables/secondary_results.csv", {"contrast": "temperature_M2"}, "response_interval_low")
    high = e.number("tables/secondary_results.csv", {"contrast": "temperature_M2"}, "response_interval_high")
    level = e.number("tables/secondary_results.csv", {"contrast": "temperature_M2"}, "interval_level")
    add("C06", "", "fitting_and_validation; frozen_test_secondary",
        f"Fitting losses: cat identity {train_m1}, photo identity {train_m2}; validation losses: "
        f"cat identity {val_m1}, photo identity {val_m2}. Validation selected photo-score temperature "
        f"{temp}; the frozen secondary test improvement is {delta}, interval [{low}, {high}] "
        f"at confidence level {level}.",
        "The large photo-model training gain without a validation gain is consistent with estimation variance. "
        "The preselected temperature's secondary test improvement supports confidence adjustment as one opportunity, "
        "not a unique explanation.",
        "Representation restrictions, shrinkage, noisy estimated photo targets, and sampling variation remain possible.",
        "Do not attribute negative results to human randomness, assign percentages to causes, or claim calibration was the only error.",
        "Use the registered learning curves and score-recovery evidence alongside independent replication; new post-test repairs are exploratory.",
        "bounded_diagnostic_account", start)

    start = len(e.facts)
    count = e.number("tables/synthetic_label_control.csv", {"model": "M2"}, "repetition", "count", digits=None)
    control_parts = []
    for model in ("M2", "M3*", "M4*"):
        loss = e.number("tables/synthetic_label_control.csv", {"model": model}, "log_loss", "mean")
        improvements = e.number("tables/synthetic_label_control.csv", {"model": model},
                                "improvement", "count_positive", digits=None)
        control_parts.append(f"{model} mean fair-coin validation loss {loss}, baseline improvements {improvements}")
    baseline = e.number("tables/synthetic_label_control.csv", {"model": "M2", "repetition": "0"}, "baseline_loss")
    simulation_parts = []
    for simulation, model, column, label in (
        ("A", "BT", "reference_log_loss", "correct-BT oracle"),
        ("B", "pooled", "reference_log_loss", "opposite-groups pooled oracle"),
        ("B", "group_aware", "reference_log_loss", "opposite-groups contextual oracle"),
        ("C", "BT", "pair_oracle_log_loss", "cycle pair oracle"),
        ("C", "BT", "reference_log_loss", "cycle best-BT reference"),
    ):
        value = e.number("artifacts/simulations.parquet", {"simulation": simulation, "model": model},
                         column, "mean")
        simulation_parts.append(f"{label}: {value}")
    add("C07", "", "synthetic_known_truth",
        f"The fair-coin control used {count} independent repetitions per procedure; " +
        "; ".join(control_parts) + f", against baseline {baseline}. Saved known-truth simulation reference "
        "losses (nats): " + "; ".join(simulation_parts) + ".",
        "The synthetic control provides no repeated fair-coin improvement signal under its registered rule. "
        "Known-truth demonstrations clarify estimator behavior and model restrictions.",
        "Untested software paths, dependence, and analysis-induced selection are not exhausted by synthetic tests.",
        "Passing controls do not certify leakage freedom, and a synthetic cycle does not demonstrate real-data cycles.",
        "Independently reproduce software tests and evaluate future changes with new controls.",
        "known_truth_and_control_only", start, "Known-truth demonstration")

    start = len(e.facts)
    parts = []
    for model in ("common:M1", "common:M2", "M3-H", "M3-C", "M3-Y", "M4-C"):
        value = e.number("tables/main_results.csv", {"population": "common", "model_id": model}, "response_log_loss")
        parts.append(f"{model}: {value}")
    fallback = []
    for block_name in ("H", "Y"):
        state = e.value("tables/validation_results.csv", {"model": f"common:M4-{block_name}"}, "pooled")
        delta = e.number("tables/secondary_results.csv", {"contrast": f"{block_name}_context"}, "response_loss_difference")
        if state != "True" or float(delta) != 0:
            raise ValueError("The H/Y pooled-fallback interpretation does not match frozen evidence.")
        fallback.append(f"{block_name} contextual difference {delta}")
    add("S01", "How sensitive are results to the visual representation?", "frozen_common_strict_G2_test",
        "Test response-weighted losses (nats/response): " + "; ".join(parts) + ". " + "; ".join(fallback) + ".",
        "Cat identity has the lowest point loss among these frozen primary-ladder models. CNN is the strongest "
        "shared representation by point loss in this table. Validation selected the pooled fallback for H and Y, "
        "so their contextual procedures equal their noncontextual scorers by construction.",
        "Feature informativeness, estimation variance, transfer mismatch, and confidence errors can all affect these results.",
        "Point ordering is not proof of every pairwise difference, and pooled selection does not imply pet-care context is irrelevant. "
        "Do not reselect the deployment or primary model on the internal test.",
        "Use independent replication to resolve representation differences and retain the automated-attribute quality audit.",
        "negative_and_representation_sensitive_results", start)

    start = len(e.facts)
    rows = []
    for band in ("1", "2-4", "5-9", "10+"):
        eligible = e.number("tables/pair_support_summary.csv",
                            {"partition": "development", "comparison_type": "all", "response_band": band},
                            "pairs", digits=None)
        rows.append(f"{band}: {eligible} pairs")
    qualifying = e.number("tables/context_overlap.csv", {"partition": "development"},
                          "shared_pairs_with_5_responses_each", "sum", digits=None)
    add("S02", "What do disagreement and context-support limitations leave unresolved?", "development",
        "Development support bands: " + "; ".join(rows) +
        f". Across the context-pair comparisons, qualifying shared pairs under the per-group support rule sum to {qualifying}.",
        "There is pair overlap, but no pair-group comparison meets the registered support rule for plotted contrasts. "
        "Broad group-preference conclusions and high-support disagreement examples are not supported.",
        "Unequal group participation and different pair compositions can masquerade as group preference differences.",
        "The absence of supported pair-level plots does not prove equal preferences or absence of contextual signal.",
        "Collect repeated responses on deliberately shared pairs across context groups.",
        "insufficient_pair_group_support", start)

    start = len(e.facts)
    group_parts = []
    for name in ("M3-C", "M4-C"):
        selector = {"population": "common", "model_id": name, "dimension": "pet_care", "group": "missing"}
        n = e.number("tables/subgroup_results.csv", selector, "responses", digits=None)
        p = e.number("tables/subgroup_results.csv", selector, "pairs", digits=None)
        loss = e.number("tables/subgroup_results.csv", selector, "response_log_loss")
        state = e.value("tables/subgroup_results.csv", selector, "support_status")
        if state != "low_support":
            raise ValueError("Missing-context slice no longer has the documented low-support status.")
        group_parts.append(f"{name}: {loss} on {n} responses and {p} pairs, marked low support")
    add("S03", "Does the aggregate context gain imply improvement for every group?", "frozen_test_low_support_subgroup",
        "; ".join(group_parts) + ".",
        "The missing-context slice has a worse contextual point loss despite the small aggregate gain; "
        "its low support prevents a strong subgroup conclusion.",
        "Sampling variation, sparse missingness support, interaction estimation, and measurement can explain the slice.",
        "Do not infer universal group improvement, a harmful causal effect, or respondent-level behavior.",
        "Increase missing-context sample support and examine the data-collection process in a new study.",
        "unfavorable_low_support_slice", start)

    start = len(e.facts)
    descriptions = []
    for model in ("M2", "M3*", "M4*"):
        selector = {"model": model, "fraction": "1.0", "seed": "20261030"}
        loss = e.number("tables/learning_curves.csv", selector, "validation_loss")
        size = e.number("tables/learning_curves.csv", selector, "validation_responses", digits=None)
        pairs = e.number("tables/learning_curves.csv", selector, "validation_pairs", digits=None)
        descriptions.append(f"{model}: {loss} on the common subset of {size} validation responses and {pairs} pairs")
    for seed in ("20261030", "20261031"):
        state = e.value("tables/learning_curve_context_support.csv",
                        {"seed": seed, "fraction": "0.25", "pet_care": "missing"}, "behavior")
        if state != "reference_fallback":
            raise ValueError("The missing-context learning-curve fallback no longer matches the synthesis.")
    add("S04", "What do the nested learning curves say about sample support?", "common_validation_intersection",
        "At the full fitting fraction for the first registered learning-curve seed: " +
        "; ".join(descriptions) + ". The missing-context category uses reference fallback in both smallest subsets.",
        "The curves compare a fixed representation on the same eligible validation subset while refitting preprocessing, "
        "context support, and penalties. Effective interaction structure can change across sizes.",
        "Subset composition and validation retuning contribute to curve variation as well as sample size.",
        "These curves are not independent test performance, do not establish an information limit, and cannot be "
        "compared numerically with the larger headline validation population without noting the different rows.",
        "Replicate with further independent fitting samples and record actual response, pair, photo, and context support.",
        "development_learning_curve_evidence", start)

    start = len(e.facts)
    numbers = []
    for representation in ("H", "C", "Y"):
        selector = {"representation": representation, "partition": "common_fit_photos"}
        rmse = e.number("tables/representation_recovery.csv", selector, "rmse")
        corr = e.number("tables/representation_recovery.csv", selector, "correlation")
        numbers.append(f"{representation}: reconstruction RMSE {rmse}, correlation {corr}")
    add("S05", "Can the representations reproduce the estimated photo-score pattern?", "common_fitting_photos",
        "; ".join(numbers) + ".",
        "In this in-sample ridge probe CNN reproduces the estimated fitting-photo score pattern more closely than H or Y.",
        "Estimated photo-score noise, ridge choice, representation dimension, and in-sample fitting affect reconstruction.",
        "This is not an independent preference test; reconstruction does not prove recovery of true utility or causality.",
        "Use independently supported photo-score targets and predeclared photo-holdout checks in a new study.",
        "secondary_estimated_target_probe", start)

    start = len(e.facts)
    changes = []
    for representation, group in (("H", "color_brightness"), ("H", "contrast_sharpness_edges"), ("H", "size"),
                                  ("Y", "coat_measurements"), ("Y", "framing_view"), ("Y", "eyes_posture"),
                                  ("Y", "inferred_breed")):
        selector = {"block": representation, "group": group, "selected": "True"}
        loss = e.number("tables/grouped_ablations.csv", selector, "validation_loss")
        base_loss = e.number("tables/grouped_ablations.csv", selector, "baseline_validation_loss")
        changes.append(f"{representation} drop {group}: loss {loss} versus full-block {base_loss}")
    add("S06", "Which registered feature groups contributed conditional predictive value?", "common_strict_validation",
        "; ".join(changes) + ".",
        "Dropping each tested H group or Y coat measurements worsened the selected validation point loss. "
        "Removing Y framing/view or eyes/posture slightly improved its validation point loss; these diagnostics were not promoted.",
        "Correlated groups, regularization retuning, automated measurement error, and finite validation variability remain alternatives.",
        "Ablation does not attribute a fraction of human behavior to features; CLIP-inferred breed categories are not true pedigree.",
        "Validate automated measurements and replicate prespecified group ablations on new development data.",
        "secondary_mixed_ablation_findings", start)

    start = len(e.facts)
    selector = {"model": "PCA-C", "selected": "True"}
    dims = e.number("tables/cnn_pca_diagnostic.csv", selector, "dimensions", digits=None)
    loss = e.number("tables/cnn_pca_diagnostic.csv", selector, "validation_loss")
    baseline = e.number("tables/cnn_pca_diagnostic.csv", selector, "baseline_validation_loss")
    add("S07", "Did the registered PCA reduction improve the CNN scorer?", "common_strict_validation",
        f"The selected PCA diagnostic retained {dims} dimensions and had validation loss {loss}, "
        f"versus {baseline} for the original CNN scorer.",
        "This registered dimensionality-reduction diagnostic did not improve the validation point loss; it was not promoted.",
        "Discarded directions, representation geometry, and shrinkage interactions may explain the result.",
        "Do not infer that every dimension-reduction method fails or use test outcomes to promote another candidate.",
        "Predeclare any further representation changes and evaluate them on fresh development and evaluation data.",
        "unfavorable_optional_diagnostic", start)

    start = len(e.facts)
    quality_parts = []
    for b in ("H", "C", "Y"):
        selector = {"block": b}
        successful = e.number("tables/feature_quality.csv", selector, "successful_images", digits=None)
        coverage = e.number("tables/feature_quality.csv", selector, "validation_coverage")
        decision = e.value("artifacts/feature_admission_decisions.csv", selector, "decision")
        if decision != "admitted":
            raise ValueError(f"{b} feature pipeline is not admitted.")
        quality_parts.append(f"{b}: {successful} successful image records, validation coverage fraction {coverage}")
    add("S08", "Does operational feature quality establish measurement correctness?", "outcome_blind_feature_audit",
        "; ".join(quality_parts) + ". The separate review/admission records admit all retained pipelines.",
        "Integration and coverage passed the registered operational gates; weak predictive extensions remain legitimate results.",
        "Image crop loss, detector/attribute error, taxonomy mismatch, and transfer limitations remain possible.",
        "Automated success and visual review do not establish annotation truth, true pedigree, or causal feature effects. "
        "The automated quality table is a pre-review snapshot; admission is recorded separately.",
        "Collect independent attribute annotations and audit harder/no-detection images without preference outcomes.",
        "operational_quality_not_ground_truth", start)

    start = len(e.facts)
    comparisons = []
    for model in ("common:M1", "M3-C", "M4-C"):
        for comparison in ("within_cat", "between_cats"):
            selector = {"population": "common", "model_id": model,
                        "dimension": "comparison_type", "group": comparison}
            loss = e.number("tables/subgroup_results.csv", selector, "response_log_loss")
            n = e.number("tables/subgroup_results.csv", selector, "responses", digits=None)
            comparisons.append(f"{model}/{comparison}: loss {loss}, {n} responses")
    add("S09", "Are within-cat and between-cat results interchangeable?", "frozen_test_comparison_type_subgroups",
        "; ".join(comparisons) + ".",
        "The comparison types have different point losses. The aggregate target combines these populations; "
        "the shared scalar scorers do not eliminate within-cat difficulty.",
        "Different pair composition, support, latent distinctions, and measurement can contribute.",
        "These descriptive subgroup metrics are not paired subgroup significance tests, equivalence claims, or causal accounts of disagreement.",
        "Collect additional within-cat pair support and predeclare targeted replication rather than retuning on this test.",
        "descriptive_target_heterogeneity", start)
    start = len(e.facts)
    remaining = []
    for label in ("photo_detail", "visual_sharing"):
        selector = {"contrast": label}
        low = e.number("tables/primary_contrasts.csv", selector, "response_interval_low")
        high = e.number("tables/primary_contrasts.csv", selector, "response_interval_high")
        remaining.append(f"{label} paired interval [{low}, {high}]")
    add("S10", "What remains unresolved after the registered tests?", "frozen_common_strict_G2_test",
        "; ".join(remaining) + ". These intervals span no difference; the other primary claims and "
        "the separate development diagnostics do not identify a unique source of residual uncertainty.",
        "The evidence narrows which tested representations demonstrated predictive value for known-photo unseen "
        "matchups without isolating the causes of remaining loss.",
        "Limited representations, estimation variability, measurement error, unrecorded context, dependence, "
        "and earlier analysis exposure remain unresolved.",
        "No objective attractiveness ranking, causal behavioral decomposition, numerical information limit, "
        "dominant uncertainty source, respondent-disjoint generalization, or validated new-photo transfer claim.",
        "Acquire genuinely independent evaluation data with respondent linkage and independently validated image "
        "attributes; preregister future interventions before outcome access.",
        "unresolved_limits", start)
    return claims


def verify_inputs(root: Path) -> dict[str, Any]:
    frozen = validate_freeze(root)
    for name in ("artifacts/graph_provenance.json", "artifacts/stage6_provenance.json",
                 "artifacts/stage7_verification.json", "artifacts/stage8_core.json",
                 "artifacts/stage8_provenance.json", "artifacts/evaluation_provenance.json"):
        report = json.loads((root / name).read_text(encoding="utf-8"))
        for group in ("input_sha256", "inputs_sha256", "source_sha256", "output_sha256", "outputs_sha256"):
            for source, digest in report.get(group, {}).items():
                path = root.joinpath(*source.split("/"))
                if not path.is_file() or _hash(path) != digest:
                    raise ValueError(f"Synthesis prerequisite hash mismatch: {source}")
    evaluation = json.loads((root / "artifacts/evaluation_provenance.json").read_text(encoding="utf-8"))
    if evaluation.get("status") != "evaluated_once" or evaluation["freeze_sha256"] != _hash(root / "artifacts/model_freeze.json"):
        raise ValueError("The final frozen evaluation is missing or inconsistent.")
    return frozen


def validate_claims(claims: list[dict[str, Any]], evidence: Evidence) -> None:
    if len({row["claim_id"] for row in claims}) != len(claims):
        raise ValueError("Final claims contain duplicate IDs.")
    if not {f"C{i:02d}" for i in range(1, 8)} <= {row["claim_id"] for row in claims}:
        raise ValueError("A registered research question is absent from the synthesis.")
    for row in claims:
        for field in ("question", "measured_result", "supported_interpretation", "alternative_explanations",
                      "nonclaim", "next_discriminating_evidence", "partition", "limitations"):
            if not row[field]:
                raise ValueError(f"Missing synthesis field {field}: {row['claim_id']}")
        if row["target"] != "None" or row["evidence_status"] not in {
            "Real-data claim", "Known-truth demonstration", "Methodological rule"
        }:
            raise ValueError("Invalid synthesis labels.")
        citations = json.loads(row["source_evidence_json"])
        if not citations:
            raise ValueError(f"Claim lacks supporting evidence: {row['claim_id']}")
        for citation in citations:
            if evidence.source_hashes[citation["source"]] != citation["sha256"]:
                raise ValueError(f"Source reference hash mismatch: {row['claim_id']}")
            if "rendered" in citation and citation["rendered"] not in row["measured_result"]:
                raise ValueError(f"Cited numeric value is absent from the result: {row['claim_id']}")


def run_synthesis(root: Path) -> dict[str, Any]:
    paths = [root.joinpath(*name.split("/")) for name in OUTPUTS]
    if any(path.exists() for path in paths):
        raise FileExistsError("Stage 10 outputs already exist; refusing to overwrite.")
    frozen = verify_inputs(root)
    evidence = Evidence(root)
    planned = evidence.read("claims.csv")
    selection = json.loads((root / "artifacts/model_selection.json").read_text(encoding="utf-8"))
    claims = build_claims(evidence, selection, planned)
    validate_claims(claims, evidence)
    # Recheck source identity immediately before publishing synthesis.
    for name, digest in evidence.source_hashes.items():
        if _hash(root.joinpath(*name.split("/"))) != digest:
            raise RuntimeError(f"Evidence source changed while synthesizing: {name}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(claims[0])
    for path in paths[:2]:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(claims)
    provenance = {
        "stage": 10, "status": "synthesized", "claims": len(claims),
        "registered_questions_covered": [row["claim_id"] for row in claims if row["claim_id"].startswith("C")],
        "new_model_fits": False, "test_evaluation_rerun": False, "raw_outcomes_loaded": False,
        "selected_representation": frozen["selected_representation"],
        "freeze_sha256": _hash(root / "artifacts/model_freeze.json"),
        "evaluation_provenance_sha256": _hash(root / "artifacts/evaluation_provenance.json"),
        "empirical_number_trace": "Each result is built from cited CSV cells or explicit aggregates; selectors, operations, raw values, and source hashes are embedded per claim.",
        "input_sha256": {
            **evidence.source_hashes,
            "src/synthesis.py": _hash(root / "src/synthesis.py"),
            "artifacts/model_selection.json": _hash(root / "artifacts/model_selection.json"),
            "artifacts/model_freeze.json": _hash(root / "artifacts/model_freeze.json"),
            "artifacts/evaluation_provenance.json": _hash(root / "artifacts/evaluation_provenance.json"),
        },
        "output_sha256": {path.relative_to(root).as_posix(): _hash(path) for path in paths[:2]},
        "limitations": LIMITATIONS,
    }
    with paths[2].open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        result = run_synthesis(args.root.resolve())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Stage 10 synthesis failed: {exc}", file=sys.stderr)
        return 2
    print(f"Stage 10: {result['claims']} evidence-linked synthesis rows; no fits or evaluation rerun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
