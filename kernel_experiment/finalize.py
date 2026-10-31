"""Summarize verified kernel experiments and render presentation figures."""

import csv
import json
import os
import tempfile

from kernel import HERE, ROOT
from study import digest, load_frozen, write_csv, write_json


def read(name):
    with (HERE/name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def run():
    if any((HERE/name).exists() for name in ("learning_curves.csv", "presentation_kernel_study.pdf",
                                              "RESULTS.md", "summary.json")):
        raise FileExistsError("Kernel final outputs already exist.")
    verify = json.loads((HERE/"verification.json").read_text(encoding="utf-8"))
    if verify["status"] != "passed":
        raise ValueError("Kernel independent verification did not pass.")
    curve = []
    for seed in (20261030, 20261031):
        for size in (25, 50, 75, 100):
            name = f"curve_{seed}_{size:03d}"
            model = load_frozen(name)
            curve.append({"seed": seed, "fraction": size/100, "fitting_responses": model["fit_responses"],
                          "fitting_pairs": model["fit_pairs"], "fitting_photos": model["fit_photos"],
                          "validation_responses": model["validation_responses"], "validation_pairs": model["validation_pairs"],
                          "width_multiplier": model["selected"]["width_multiplier"],
                          "ridge": model["selected"]["ridge"], "validation_loss": model["selected"]["validation_loss"]})
    write_csv(HERE/"learning_curves.csv", curve, list(curve[0]))
    g2, g3, g4 = (read(f"{label}_results.csv") for label in ("g2", "g3", "g4"))
    contrasts = {name: read(f"{name}_contrasts.csv") for name in ("g2", "g3", "g4")}
    utility, penalties = read("g2_abstention_utility.csv"), read("g2_attempts.csv")
    make_plot(curve, penalties, {"g2": g2, "g3": g3, "g4": g4}, utility)
    chosen = {name: next(r for r in results if r["model"] == "kernel_M3-C")
              for name, results in (("g2", g2), ("g3", g3), ("g4", g4))}
    evidence = [
        "# Retrospective nonlinear CNN kernel scorer: evidence",
        "",
        "Pooled photo-level RBF Nyström M3-C extension. This is post-test exploratory and noncausal; "
        "prior G2/G3/G4 outcomes were exposed. It does not replace the original primary model or deployment.",
        "",
        "## Model and fitting",
        "",
        "The preference logit is a displayed-A intercept plus the difference between two nonlinear photo scores. "
        "Up to 256 landmarks are chosen by fixed hash from unique fitting photos only. CNN centering/block-RMS, "
        "bandwidth median, landmark map and Nyström center/RMS are estimated on fitting photos only, including "
        "each learning-curve and transfer split. Three width multipliers and seven ridge values are tuned on "
        "the matching validation set; all candidates and solver acceptance are retained.",
        "",
        "| Target | Responses / pairs | Held-out unit | Kernel log loss | Brier | Accuracy |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name in ("g2", "g3", "g4"):
        row = chosen[name]
        evidence.append(f"| {name.upper()} | {row['responses']} / {row['pairs']} | {row['clusters']} {row['cluster_unit']} | "
                        f"{float(row['log_loss']):.6f} | {float(row['brier']):.6f} | {100*float(row['accuracy']):.2f}% |")
    evidence += ["", "## Matched paired differences", "",
                 "Reference minus kernel: positive favors the kernel. All intervals are 95% pointwise, "
                 "unadjusted and conditional on selected fits; G2 resamples unordered pairs, "
                 "G3 held-out photos and G4 held-out cats.", "",
                 "| Target | Original reference | Difference | 95% interval |", "|---|---|---:|---|"]
    for name in ("g2", "g3", "g4"):
        for row in contrasts[name]:
            if row["reference"] in ("original:M3-C", "original:M4-C"):
                evidence.append(f"| {name.upper()} | {row['reference']} | {float(row['difference']):+.6f} | "
                                f"[{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}] |")
    evidence += [
        "", "The linear-versus-kernel comparisons above span zero; these data do not establish an "
        "advantage from this kernel. M4-C is contextual and the kernel scorer is pooled, so that comparison "
        "changes both model class and context; M3-C is the most direct like-for-like reference.",
        "",
        "The separately saved [paired demographic-CNN comparisons](additional_paired_contrasts.csv) use the "
        "same rows/clusters but C+D also uses demographics. They cannot isolate nonlinearity.",
        "",
        "## Controls, calibration and slices",
        "",
        "The fixed selected G2 kernel settings passed 20 fair-coin fits: 0 failures, 0/20 improvements "
        "over log(2). This is an operational check, not a leakage-free certificate. "
        "Original-style development and test support/quality slices and reliability bins are in "
        "[validation slices](validation_slices.csv), [validation bins](validation_calibration.csv), "
        "[G2 test slices](g2_slices.csv) and [G2 bins](g2_slice_calibration.csv). "
        "Slice support is 100 responses/30 pairs and bin support 30 responses/10 pairs; low support is flagged.",
        "",
        "G2 within-cat loss is **0.695371** on 455 responses/170 pairs; between-cat loss is "
        "**0.665649** on 3,278 responses/1,037 pairs. G3 within-cat loss is **0.706694** on "
        "286 responses/106 pairs; its between-cat loss is **0.668005** on 1,964 responses/612 pairs. "
        "G4's approved target has between-cat comparisons only. These are point estimates, "
        "not paired subgroup improvement claims.",
        "",
        "## Learning curves, penalty path and temperature",
        "",
        "The [learning curves](learning_curves.csv) use the original 1,836-response/589-pair validation "
        "intersection and refit all preprocessing, landmark selection, width and ridge at each nested subset. "
        "The [G2 candidate log](g2_attempts.csv) has all 21 accepted width/ridge combinations. "
        "The 100% seeds have the same fitting/validation data and are not independent replications.",
        "",
        "| Seed | Fraction | Fit responses / pairs / photos | Width multiplier | Ridge | Validation loss |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in curve:
        evidence.append(f"| {r['seed']} | {r['fraction']:.2f} | {r['fitting_responses']} / {r['fitting_pairs']} / "
                        f"{r['fitting_photos']} | {r['width_multiplier']:g} | {r['ridge']:g} | {r['validation_loss']:.6f} |")
    evidence += [
        "", "Validation selected temperature **T=1**, so the temperature-scaled scorer equals the unscaled "
        "scorer exactly; this is not a new improvement.",
        "",
        "## Abstention and utility",
        "",
        "Fixed confidence thresholds 0.50–1.00 in 0.025 steps; +1 correct, -1 incorrect, 0 abstain; "
        "denominator is all 3,733 G2 responses. No test threshold is selected.",
        "",
        "| Kernel cutoff | Retained responses / pairs | Coverage | Retained accuracy | Utility / original response |",
        "|---:|---|---:|---:|---:|",
    ]
    for t in (.5, .6, .7):
        row = next(r for r in utility if r["model"] == "kernel_M3-C" and float(r["threshold"]) == t)
        evidence.append(f"| {t:.2f} | {row['retained_responses']} / {row['retained_pairs']} | "
                        f"{100*float(row['coverage']):.2f}% | {100*float(row['selective_accuracy']):.2f}% | "
                        f"{float(row['utility_per_original_response']):.6f} |")
    evidence += [
        "", "Higher retained accuracy does not establish improved total utility. The fixed 0.60 cutoff reduces "
        "utility from 0.167961 to 0.127779 despite increasing selected accuracy.",
        "", "## Verification and limitations",
        "",
        "The [verification record](verification.json) independently reconstructs all 11 selected G2/G3/G4/"
        "curve fits, checks fitting-only landmarks and transforms, validates gradients and candidate selection, "
        "and replays 5,000 cluster-bootstrap draws per target and 63 abstention cells. Four focused kernel "
        "tests passed. The original model and all G2/G3/G4 outcomes were previously exposed; these "
        "post-hoc intervals are not primary 98.75% confirmatory contrasts.",
        "",
        "Changing both nonlinear representation and width/penalty search makes this a procedure comparison, "
        "not an isolated causal effect of nonlinearity. Repeated photos/cats/respondents and training/model "
        "selection uncertainty are not fully represented by the cluster bootstrap.",
        ""
    ]
    (HERE/"RESULTS.md").write_text("\n".join(evidence), encoding="utf-8")
    outputs = ["learning_curves.csv", "presentation_kernel_study.pdf", "RESULTS.md"]
    write_json(HERE/"summary.json", {"status": "complete_retrospective_exploratory",
               "verification_sha256": digest(HERE/"verification.json"),
               "outputs_sha256": {name: digest(HERE/name) for name in outputs},
               "model": "pooled photo-level RBF Nyström M3-C", "primary_claims_changed": False,
               "deployment_changed": False, "limitations": "Prior outcome exposure; conditional pointwise intervals, not causal."})
    print("Finalized kernel tables, synthesis and four-page presentation figure.")


def make_plot(curves, penalties, results, utility):
    with tempfile.TemporaryDirectory(prefix="mpl_", dir=HERE) as config:
        os.environ["MPLCONFIGDIR"] = config
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(HERE/"presentation_kernel_study.pdf") as pdf:
            fig, axes = plt.subplots(1, 3, figsize=(14, 7))
            for ax, target in zip(axes, ("g2", "g3", "g4")):
                data = [r for r in results[target] if r["model"] in ("kernel_M3-C", "original:M3-C", "original:M4-C")]
                for i, row in enumerate(data):
                    ax.hlines(i, float(row["ci_low"]), float(row["ci_high"]), color="#28677c")
                    ax.plot(float(row["log_loss"]), i, "o", color="#28677c")
                ax.set_yticks(range(len(data)), [r["model"] for r in data])
                ax.invert_yaxis()
                ax.set(title=target.upper(), xlabel="Response-weighted log loss")
            fig.suptitle("Pooled nonlinear CNN versus linear scorers", fontsize=15)
            fig.text(.05, .04, "Retrospective, post-test, not causal or independent; 95% pointwise pair/photo/cat bootstrap.", fontsize=9)
            fig.subplots_adjust(left=.11, right=.97, top=.86, bottom=.16, wspace=.55)
            pdf.savefig(fig)
            plt.close(fig)
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 7))
            for seed in (20261030, 20261031):
                group = [r for r in curves if r["seed"] == seed]
                axes[0].plot([r["fitting_responses"] for r in group], [r["validation_loss"] for r in group], "o-", label=str(seed))
            axes[0].set(title="Nested learning curves", xlabel="Fitting responses", ylabel="Common-subset validation loss")
            axes[0].legend()
            for width in (.25, 1., 4.):
                group = sorted((r for r in penalties if float(r["width_multiplier"]) == width), key=lambda r: float(r["ridge"]))
                axes[1].plot([float(r["ridge"]) for r in group], [float(r["validation_loss"]) for r in group], "o-", label=f"width {width:g}")
                axes[1].plot([float(r["ridge"]) for r in group], [float(r["training_loss"]) for r in group], "--", alpha=.5)
            axes[1].set_xscale("log")
            axes[1].set(title="Penalty paths: validation solid, fitting dashed", xlabel="Ridge", ylabel="Log loss")
            axes[1].legend()
            fig.suptitle("Fit-only transformations; 1,836 common validation responses", fontsize=14)
            fig.text(.05, .025, "All 21 width/ridge candidates refit per subset; the two 100% seed inputs coincide.", fontsize=9)
            fig.tight_layout(rect=(0, .06, 1, .95))
            pdf.savefig(fig)
            plt.close(fig)
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 7))
            bins = read("g2_calibration.csv")
            for row in bins:
                if row["mean_probability"]:
                    axes[0].scatter(float(row["mean_probability"]), float(row["observed_a_rate"]),
                                    facecolors="#28677c" if row["support"] == "supported" else "none", edgecolors="#28677c")
            axes[0].plot([0,1],[0,1],"--",color="gray")
            axes[0].set(title="G2 calibration (hollow = low support)", xlabel="Mean predicted A", ylabel="Observed A", xlim=(0,1), ylim=(0,1))
            for name in ("kernel_M3-C", "original:M3-C", "original:M4-C"):
                group = [r for r in utility if r["model"] == name]
                axes[1].plot([float(r["threshold"]) for r in group],
                             [float(r["utility_per_original_response"]) for r in group], label=name)
            axes[1].set(title="Fixed +1/-1/0 utility", xlabel="Confidence threshold", ylabel="Utility / original response")
            axes[1].legend(fontsize=8)
            fig.suptitle("Calibration and abstention: saved G2 outcomes", fontsize=14)
            fig.text(.05,.025,"Post-test descriptive only; no threshold or deployment model selected.",fontsize=9)
            fig.tight_layout(rect=(0,.06,1,.95))
            pdf.savefig(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.axis("off")
            lines = [
                "Predictive association, not causal effect.",
                "Original G2, G3 and G4 outcomes were already exposed: no independent confirmation.",
                "The scorer is one nonlinear scalar per photo, not an unrestricted pair model.",
                "G3 and G4 use separately refitted fitting-only landmarks and validation-tuned settings.",
                "Pointwise 95% intervals omit model-selection and some shared-image/respondent dependence.",
                "No post-test model repair, official-prediction change or abstention threshold selection.",
            ]
            fig.suptitle("Interpretation and safeguards", fontsize=15)
            for i, line in enumerate(lines):
                ax.text(.05, .85-i*.13, "- " + line, fontsize=11, transform=ax.transAxes)
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    run()
