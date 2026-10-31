"""Reconcile completed new diagnostics and create a presentation dashboard."""

from __future__ import annotations

import csv
import json
import os
import tempfile

from common import EXPERIMENT, HERE, digest, read_csv, verify_inputs, write_csv, write_json


def run() -> None:
    verify_inputs()
    outputs = ("learning_curves.csv", "learning_curve_support.csv", "presentation_learning_penalty_transfer.pdf",
               "diagnostic_summary.json")
    if any((HERE / name).exists() for name in outputs):
        raise FileExistsError("Diagnostic finalization already exists.")
    path_proof = json.loads((HERE / "penalty_path_provenance.json").read_text(encoding="utf-8"))
    if path_proof["status"] != "all_candidates_reproduced" or path_proof["output_sha256"] != digest(HERE / "penalty_path_refits.csv"):
        raise ValueError("Penalty-path reproduction is incomplete or changed.")
    curves, support, curve_hashes = [], [], {}
    for seed in (20261030, 20261031):
        for size in (25, 50, 75, 100):
            label = f"{seed}_{size:03d}"
            model_path = HERE / f"curve_{label}_models.json"
            log_path = HERE / f"curve_{label}_attempts.csv"
            model = json.loads(model_path.read_text(encoding="utf-8"))
            if (model["candidate_log_sha256"] != digest(log_path) or
                model["demographic_freeze_sha256"] != digest(EXPERIMENT / "freeze.json") or
                model["test_outcomes_read"]):
                raise ValueError(f"Learning-curve provenance mismatch: {label}")
            attempts = read_csv(log_path)
            if len(attempts) != 94 or any(r["accepted"] != "True" for r in attempts):
                raise ValueError(f"Learning-curve candidate log incomplete: {label}")
            for name in ("D", "Cat+D", "Y+D", "C+D"):
                selected = model["selected"][name]
                curves.append({"seed": seed, "fraction": size/100, "model": name,
                               "fitting_responses": model["fitting_responses"],
                               "fitting_pairs": model["fitting_pairs"],
                               "fitting_photos": len(model["fitting_photos"]),
                               "validation_responses": model["validation_responses"],
                               "validation_pairs": model["validation_pairs"],
                               "ridge": selected["ridge"], "rho": "exact_pooled" if selected["pooled"] else selected["rho"],
                               "pooled": selected["pooled"], "validation_loss": selected["validation_loss"]})
            for demographic, spec in model["encoding"].items():
                for category, count in spec["fitting_counts"].items():
                    support.append({"seed": seed, "fraction": size/100, "demographic": demographic,
                                    "category": category, "fitting_responses": count,
                                    "fitting_pairs": spec["fitting_pairs"][category],
                                    "reference": spec["reference"],
                                    "interaction_deviation": category in spec["deviations"],
                                    "status": "reference" if category == spec["reference"] else
                                              "estimated" if category in spec["deviations"] else "unsupported_zero_fallback"})
            curve_hashes[label] = digest(model_path)
    write_csv(HERE / "learning_curves.csv", curves, list(curves[0]))
    write_csv(HERE / "learning_curve_support.csv", support, list(support[0]))
    transfer_results = {}
    for target in ("g3", "g4"):
        proof = json.loads((HERE / f"{target}_provenance.json").read_text(encoding="utf-8"))
        if proof["status"] != "retrospective_transfer_completed":
            raise ValueError(f"{target} transfer incomplete.")
        if digest(HERE / f"{target}_freeze.json") != (HERE / f"{target}_freeze.sha256").read_text(encoding="ascii").strip():
            raise ValueError(f"{target} transfer seal mismatch.")
        for name, expected in proof["outputs_sha256"].items():
            if digest(HERE / name) != expected:
                raise ValueError(f"{target} transfer output mismatch: {name}")
        transfer_results[target] = (read_csv(HERE / f"{target}_results.csv"), read_csv(HERE / f"{target}_contrasts.csv"))
    reuse = json.loads((HERE / "full_size_reuse_provenance.json").read_text(encoding="utf-8"))
    if (reuse["status"] != "identical_full_size_inputs_verified" or
            reuse["source_models_sha256"] != digest(HERE / "curve_20261031_100_models.json") or
            reuse["derived_models_sha256"] != digest(HERE / "curve_20261030_100_models.json") or
            reuse["candidate_log_sha256"] != digest(HERE / "curve_20261031_100_attempts.csv") or
            reuse["candidate_log_sha256"] != digest(HERE / "curve_20261030_100_attempts.csv")):
        raise ValueError("Identical full-size curve reuse does not verify.")
    fixed = json.loads((HERE / "fixed_diagnostics_provenance.json").read_text(encoding="utf-8"))
    for name, expected in fixed["output_sha256"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"Fixed diagnostic output mismatch: {name}")
    plot(curves, read_csv(HERE / "penalty_path_refits.csv"), transfer_results)
    write_json(HERE / "diagnostic_summary.json", {
        "status": "all_requested_additional_diagnostics_available",
        "learning_curve_rows": len(curves), "learning_support_rows": len(support),
        "learning_candidate_logs": len(curve_hashes), "learning_distinct_fit_subsets": 7,
        "learning_full_size_reused_not_independent": True,
        "full_size_reuse_provenance_sha256": digest(HERE / "full_size_reuse_provenance.json"),
        "learning_model_sha256": curve_hashes,
        "penalty_candidates": path_proof["candidate_records"], "penalty_refits": path_proof["numerical_refits"],
        "penalty_fallback_copies": path_proof["exact_fallback_copies"],
        "g3_provenance_sha256": digest(HERE / "g3_provenance.json"),
        "g4_provenance_sha256": digest(HERE / "g4_provenance.json"),
        "fixed_provenance_sha256": digest(HERE / "fixed_diagnostics_provenance.json"),
        "outputs_sha256": {name: digest(HERE / name) for name in outputs[:-1]},
        "existing_control_and_calibration": "Original demographic experiment fair-coin controls, calibration, and within/between subgroup outputs are preserved and reused.",
        "limitations": "All new diagnostics are exploratory after prior test exposure, not causal or independently confirmatory. No deployment selection."})
    print(f"Finalized {len(curves)} learning-curve rows and G3/G4/penalty dashboard.")


def plot(curves: list[dict], paths: list[dict], transfers: dict) -> None:
    with tempfile.TemporaryDirectory(prefix="mpl_", dir=HERE) as config:
        os.environ["MPLCONFIGDIR"] = config
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(HERE / "presentation_learning_penalty_transfer.pdf") as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            for ax, name in zip(axes.ravel(), ("D", "Cat+D", "Y+D", "C+D")):
                for seed in (20261030, 20261031):
                    rows = sorted([r for r in curves if r["model"] == name and r["seed"] == seed],
                                  key=lambda r: r["fitting_responses"])
                    ax.plot([r["fitting_responses"] for r in rows], [r["validation_loss"] for r in rows],
                            marker="o", label=str(seed))
                ax.set(title=name, xlabel="Fitting responses", ylabel="Common-subset validation log loss")
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            fig.suptitle("Nested learning curves: 1,836 responses / 589 validation pairs", fontsize=14)
            fig.text(.05, .02, "Exploratory; preprocessing, demographics and tuning refit at each size; model fallback can change.", fontsize=9)
            fig.tight_layout(rect=(0, .05, 1, .95))
            pdf.savefig(fig)
            plt.close(fig)
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            for ax, name in zip(axes.ravel(), ("D", "Cat+D", "Y+D", "C+D")):
                subset = [r for r in paths if r["model"] == name and r["pooled"] == "False"]
                norm_axis = ax.twinx()
                for rho in sorted({float(r["rho"]) for r in subset}):
                    group = sorted((r for r in subset if float(r["rho"]) == rho), key=lambda r: float(r["ridge"]))
                    x = [float(r["ridge"]) for r in group]
                    ax.plot(x, [float(r["validation_loss"]) for r in group], marker=".", label=f"val rho={rho:g}")
                    ax.plot(x, [float(r["training_loss"]) for r in group], linestyle="--", alpha=.55)
                    norm_axis.plot(x, [float(r["coefficient_norm"]) for r in group], linestyle=":", color="gray", alpha=.45)
                ax.set_xscale("log")
                ax.set(title=name, xlabel="Ridge lambda", ylabel="Log loss")
                norm_axis.set_ylabel("Coefficient norm")
                ax.legend(fontsize=7)
            fig.suptitle("Reproduced development penalty paths: validation solid, fitting dashed", fontsize=14)
            fig.text(.05, .02, "Gray dotted: non-intercept coefficient norm; point fits are exploratory and regularized.", fontsize=9)
            fig.tight_layout(rect=(0, .05, 1, .95))
            pdf.savefig(fig)
            plt.close(fig)
            fig, axes = plt.subplots(1, 2, figsize=(13, 7))
            for ax, target in zip(axes, ("g3", "g4")):
                results, _ = transfers[target]
                cat_name = "original:M1" if target == "g3" else "original:M1-zero"
                names = ["original:M0", "D", cat_name, "Cat+D", "original:M3-Y", "Y+D", "original:M4-C", "C+D"]
                subset = [next(r for r in results if r["model"] == name) for name in names]
                for i, row in enumerate(subset):
                    ax.hlines(i, float(row["ci_low"]), float(row["ci_high"]), color="#28677c")
                    ax.plot(float(row["log_loss"]), i, "o", color="#28677c")
                ax.set_yticks(range(len(names)), names)
                ax.invert_yaxis()
                ax.set(title=f"{target.upper()}: {subset[0]['held_out_clusters']} held-out clusters",
                       xlabel="Response-weighted log loss")
            fig.suptitle("Separate G3/G4 refits: retrospective transfer comparison", fontsize=14)
            fig.text(.05, .05, "Not independent or causal. G3 resamples held-out photos; G4 resamples held-out cats.", fontsize=9)
            fig.text(.05, .025, "G4 Cat+D sets the unseen cat's pooled and demographic-specific score components to zero.", fontsize=9)
            fig.subplots_adjust(left=.12, right=.97, bottom=.14, top=.88, wspace=.45)
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    run()
