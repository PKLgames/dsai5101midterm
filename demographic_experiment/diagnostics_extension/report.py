"""Write a bounded numerical synthesis of completed parity diagnostics."""

from __future__ import annotations

import json

from common import EXPERIMENT, HERE, digest, read_csv, verify_inputs


def run() -> None:
    verify_inputs()
    target = HERE / "DIAGNOSTIC_EVIDENCE.md"
    if target.exists():
        raise FileExistsError("Diagnostic evidence synthesis already exists.")
    summary = json.loads((HERE / "diagnostic_summary.json").read_text(encoding="utf-8"))
    verified = json.loads((HERE / "verification.json").read_text(encoding="utf-8"))
    if summary["status"] != "all_requested_additional_diagnostics_available" or verified["status"] != "independent_replay_passed":
        raise ValueError("Diagnostics or independent replay are incomplete.")
    temperatures = read_csv(HERE / "temperature_results.csv")
    abstention = read_csv(HERE / "abstention_utility.csv")
    curves = read_csv(HERE / "learning_curves.csv")
    control = json.loads((EXPERIMENT / "control_provenance.json").read_text(encoding="utf-8"))
    lines = [
        "# All-demographic diagnostic parity: retrospective evidence",
        "",
        "All numbers are exploratory after original G2/G3/G4 outcomes were exposed. "
        "Do not interpret them as independently validated, causal, or a new primary-model/deployment choice.",
        "",
        "## Existing controls, calibration, and within/between-cat slices",
        "",
        "The original demographic supplement already contains 20 fair-coin controls per requested model "
        "(80 accepted fits; no investigation trigger), four-model overall calibration, and comparison-type "
        "subgroup metrics. The control recorded improvements over log(2): "
        + ", ".join(f"{name} {value}/20" for name, value in control["improved_repetitions"].items()) + ".",
        "See [control results](../control_results.csv), [control provenance](../control_provenance.json), "
        "[existing calibration](../calibration.csv), and [existing subgroups](../subgroups.csv). "
        "Control passage does not certify leakage-free software or establish independent generalization.",
        "",
        "New [validation support metrics](validation_support_slices.csv) and "
        "[validation calibration](validation_support_calibration.csv) use original M4-C support/quality group definitions; "
        "[validation demographic slices](validation_demographic_slices.csv) and "
        "[validation demographic calibration](validation_demographic_calibration.csv) cover all six demographics. "
        "The new [test support slices](test_support_slices.csv), "
        "[test support calibration](test_support_calibration.csv), and "
        "[test demographic calibration](subgroup_calibration.csv) use saved G2 predictions. "
        "Overall slice support is 100 responses/30 pairs; bin support is 30 responses/10 pairs. "
        "Quality flags overlap; do not sum overlapping slices. Low-support estimates/bins remain flagged.",
        "",
        "## Nested-pair learning curves",
        "",
        "Two original seeds and four nested fractions; demographic support, references, category effects, "
        "Y/C preprocessing and registered penalty grids are reselected at each size. All models use the same "
        "1,836-response/589-pair strict-validation intersection. These development curves were motivated after "
        "the test was seen and are not an information-limit estimate. See "
        "[curves](learning_curves.csv), [fitting category support](learning_curve_support.csv), and individual "
        "curve attempt logs/models in this directory. The two 100% seeds have exactly the same ordered fitting "
        "and validation rows: one full-size fit was reused after identity checks, not treated as two independent fits. "
        "Partial logs from resource-management interruptions remain separately labeled.",
        "",
        "| Seed | Fraction | Fit responses / pairs / photos | D | Cat+D | Y+D | C+D |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for seed in (20261030, 20261031):
        for size in (.25, .5, .75, 1.):
            subset = {r["model"]: r for r in curves if int(r["seed"]) == seed and float(r["fraction"]) == size}
            d = subset["D"]
            lines.append(f"| {seed} | {size:.2f} | {d['fitting_responses']} / {d['fitting_pairs']} / {d['fitting_photos']} | "
                         + " | ".join(f"{float(subset[name]['validation_loss']):.6f}" for name in ("D", "Cat+D", "Y+D", "C+D")) + " |")
    pooled = [r for r in curves if r["pooled"] == "True"]
    lines.extend(["", f"Exact no-score-interaction fallbacks were selected in **{len(pooled)} of 24** "
                  "Cat/Y/C curve cells; see the curve table for the affected seed and size.",
                  "", "## Reproduced penalty paths and temperature",
                  "", "The [penalty-path refit table](penalty_path_refits.csv) reproduces the original "
                  "94 fitting/validation candidate records (91 numerical refits and three exact fallback copies) "
                  "and includes non-intercept coefficient norms. "
                  "A path diagnoses shrinkage and fit/validation patterns, not causation.",
                  "",
                  "| Model | Validation-selected temperature | Unscaled minus scaled G2 loss | 95% pair interval |",
                  "|---|---:|---:|---|"])
    for row in temperatures:
        lines.append(f"| {row['model']} | {float(row['temperature']):g} | "
                     f"{float(row['difference_unscaled_minus_temperature']):+.6f} | "
                     f"[{float(row['ci95_low']):.6f}, {float(row['ci95_high']):.6f}] |")
    lines.extend(["", "Positive means temperature scaling reduces log loss. Temperature candidates were chosen on "
                  "validation and sealed before this saved-test reanalysis, but this experiment was proposed after "
                  "the test had already been seen. T=1 implies exact identity, not a new improvement.",
                  "", "## Separately refitted retrospective G3/G4 transfer",
                  "", "These models were refit and retuned on the existing G3/G4 fitting and validation populations. "
                  "Existing test outcomes were previously exposed; no fresh independent test is claimed. "
                  "G3 uncertainty resamples held-out photos; G4 resamples held-out cats. G4 Cat+D assigns the "
                  "unseen cat zero for both pooled and all demographic-specific cat-score components. "
                  "This is a fixed cold-start convention, not a learned unseen-cat preference.",
                  "", "| Target | Model | Validation-selected pooled fallback | Test response loss | 95% cluster interval |",
                  "|---|---|---|---:|---|"])
    for target_name in ("g3", "g4"):
        models = json.loads((HERE / f"{target_name}_models.json").read_text(encoding="utf-8"))
        results = {r["model"]: r for r in read_csv(HERE / f"{target_name}_results.csv")}
        for name in ("D", "Cat+D", "Y+D", "C+D"):
            r = results[name]
            lines.append(f"| {target_name.upper()} | {name} | {models['selected'][name]['pooled']} | "
                         f"{float(r['log_loss']):.6f} | [{float(r['ci_low']):.6f}, {float(r['ci_high']):.6f}] |")
    lines.extend(["", "Interpret score-interaction gains only using the side-adjusted comparisons in "
                  "[G3 contrasts](g3_contrasts.csv) and [G4 contrasts](g4_contrasts.csv). "
                  "A gain versus the original base model can combine side effects, score effects and refitting. "
                  "G3/G4 output/model/attempt files remain isolated here; original records are unchanged.",
                  "", "## Abstention and utility",
                  "", "The [105-cell table](abstention_utility.csv) uses four demographic models and original "
                  "M4-C on the same saved G2 outcomes, thresholds 0.50–1.00 by 0.025, "
                  "correct +1, incorrect -1, abstain 0, and all original responses as the denominator. "
                  "Confidence ties remain together; no test-optimal cutoff or deployment decision is selected.",
                  "", "| Model | Cutoff | Retained | Coverage | Selective accuracy | Utility / original response |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name in ("D", "Cat+D", "Y+D", "C+D", "M4-C"):
        for cutoff in (.5, .6, .7):
            r = next(row for row in abstention if row["model"] == name and float(row["threshold"]) == cutoff)
            accuracy = f"{100*float(r['selective_accuracy']):.2f}%" if r["selective_accuracy"] else "undefined"
            lines.append(f"| {name} | {cutoff:.2f} | {r['retained_responses']} | {100*float(r['coverage']):.2f}% | "
                         f"{accuracy} | {float(r['utility_per_original_response']):.6f} |")
    lines.extend(["", "Higher accuracy among retained cases does not necessarily mean greater utility across all "
                  "original cases. Intervals are pointwise 95%, pair-clustered and unadjusted; low-support/empty "
                  "subsets are explicitly flagged. See the [four-page abstention dashboard](presentation_abstention_full.pdf), "
                  "[fixed diagnostics figure](presentation_fixed_diagnostics.pdf) "
                  "and [learning/penalty/transfer figure](presentation_learning_penalty_transfer.pdf).",
                  "", "## Boundaries",
                  "", "Displayed A is not established to mean left; demographics-only cannot rank cats/photos. "
                  "All six missing demographic columns coincide in the original fitting data, so individual "
                  "missing-field effects are not separately identifiable. CLIP categories are automated measurements, "
                  "not verified pedigree. No causal demographic, context or feature effect follows.",
                  "", f"Diagnostic summary SHA-256: `{digest(HERE / 'diagnostic_summary.json')}`; "
                  f"verification SHA-256: `{digest(HERE / 'verification.json')}`.",
                  ""])
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved evidence synthesis: {target}")


if __name__ == "__main__":
    run()
