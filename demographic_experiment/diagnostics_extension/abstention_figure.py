"""Presentation-ready abstention coverage, accuracy and utility dashboard."""

from __future__ import annotations

import json
import os
import tempfile

from common import HERE, digest, read_csv, verify_inputs, write_json


def number(row: dict, key: str) -> float | None:
    return float(row[key]) if row[key] != "" else None


def run() -> None:
    verify_inputs()
    pdf_path, proof_path = HERE / "presentation_abstention_full.pdf", HERE / "abstention_figure_provenance.json"
    if pdf_path.exists() or proof_path.exists():
        raise FileExistsError("Abstention figure already exists.")
    provenance = json.loads((HERE / "fixed_diagnostics_provenance.json").read_text(encoding="utf-8"))
    if digest(HERE / "abstention_utility.csv") != provenance["output_sha256"]["abstention_utility.csv"]:
        raise ValueError("Saved abstention input hash mismatch.")
    rows = read_csv(HERE / "abstention_utility.csv")
    models = ("D", "Cat+D", "Y+D", "C+D", "M4-C")
    if len(rows) != len(models)*21 or set(r["model"] for r in rows) != set(models):
        raise ValueError("Unexpected abstention model/threshold coverage.")
    with tempfile.TemporaryDirectory(prefix="mpl_", dir=HERE) as config:
        os.environ["MPLCONFIGDIR"] = config
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(pdf_path) as pdf:
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.5))
            for name in models:
                subset = [r for r in rows if r["model"] == name]
                axes[0].plot([float(r["threshold"]) for r in subset], [float(r["coverage"]) for r in subset], label=name)
                available = [r for r in subset if r["selective_accuracy"]]
                axes[1].plot([float(r["coverage"]) for r in available],
                             [float(r["selective_accuracy"]) for r in available], label=name)
                for r in available:
                    axes[1].plot(float(r["coverage"]), float(r["selective_accuracy"]), "o", markersize=3,
                                 markerfacecolor="none" if r["support_status"] != "supported" else None,
                                 color=axes[1].lines[-1].get_color())
            axes[0].set(title="Confidence and coverage", xlabel="Minimum confidence", ylabel="Retained / all responses")
            axes[1].set(title="Selective accuracy versus coverage", xlabel="Coverage", ylabel="Accuracy among retained")
            for ax in axes:
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            fig.suptitle("Post-test exploratory abstention: no threshold selected", fontsize=15)
            fig.text(.05, .035, "Hollow points: <100 retained responses or <30 pairs. No retained cases: accuracy undefined.", fontsize=9)
            fig.tight_layout(rect=(0, .07, 1, .95))
            pdf.savefig(fig)
            plt.close(fig)
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.5))
            for name in models:
                subset = [r for r in rows if r["model"] == name]
                axes[0].plot([float(r["threshold"]) for r in subset],
                             [float(r["utility_per_original_response"]) for r in subset], label=name)
            chosen = [r for r in rows if r["model"] == "C+D"]
            x = [float(r["threshold"]) for r in chosen]
            axes[1].plot(x, [float(r["utility_per_original_response"]) for r in chosen], label="Observed C+D")
            axes[1].plot(x, [float(r["model_implied_utility_per_original_response"]) for r in chosen],
                         linestyle="--", label="Model-implied C+D")
            axes[1].fill_between(x, [float(r["utility_ci95_low"]) for r in chosen],
                                 [float(r["utility_ci95_high"]) for r in chosen], alpha=.2, label="95% pointwise pair CI")
            axes[0].set(title="Realized utility: +1 / -1 / 0", xlabel="Minimum confidence", ylabel="Utility / original response")
            axes[1].set(title="C+D observed and model-implied utility", xlabel="Minimum confidence", ylabel="Utility / original response")
            for ax in axes:
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            fig.suptitle("Utility includes abstentions in the denominator", fontsize=15)
            fig.text(.05, .035, "Implied utility is expected reward only if probabilities are calibrated; intervals are not multiplicity adjusted.", fontsize=9)
            fig.tight_layout(rect=(0, .07, 1, .95))
            pdf.savefig(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(13, 7.5))
            ax.axis("off")
            data = []
            for name in models:
                for cutoff in (.5, .6, .7):
                    r = next(r for r in rows if r["model"] == name and float(r["threshold"]) == cutoff)
                    accuracy = f"{100*float(r['selective_accuracy']):.1f}%" if r["selective_accuracy"] else "undefined"
                    data.append([name, f"{cutoff:.2f}", r["retained_responses"], r["retained_pairs"],
                                 f"{100*float(r['coverage']):.1f}%", accuracy,
                                 f"{float(r['utility_per_original_response']):.4f}", r["support_status"]])
            table = ax.table(cellText=data, colLabels=["Model", "Cutoff", "Retained", "Pairs", "Coverage",
                                                       "Accuracy", "Utility / all", "Support"],
                             loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.4)
            fig.suptitle("Fixed illustrative cutoffs: counts and support", fontsize=15)
            fig.text(.05, .045, "These cutoffs were fixed for description, not chosen to maximize test utility.", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.axis("off")
            safeguards = [
                "Post-test retrospective analysis; the saved G2 outcomes were already exposed.",
                "No model or deployment threshold is selected from these curves.",
                "Correct +1, incorrect -1, abstain 0; utility denominator is all original responses.",
                "Retained accuracy is undefined when no predictions remain.",
                "5,000 shared unordered-pair bootstrap draws; 95% pointwise intervals are unadjusted.",
                "Respondent linkage, shared-image dependence and model-selection uncertainty remain unaccounted for.",
                "Under calibrated probabilities and these payoffs, always predict is weakly optimal in expected total utility.",
                "Demographics-only predicts displayed A; A is not established to mean left.",
            ]
            fig.suptitle("Interpretation and safeguards", fontsize=15)
            for i, text in enumerate(safeguards):
                ax.text(.05, .9-i*.105, "- " + text, fontsize=11, transform=ax.transAxes)
            pdf.savefig(fig)
            plt.close(fig)
    write_json(proof_path, {"status": "completed", "pages": 4,
                            "input_sha256": digest(HERE/"abstention_utility.csv"),
                            "output_sha256": digest(pdf_path), "threshold_selected": False,
                            "interpretation": "Post-test descriptive, no independent or causal claim."})
    print("Created four-page coverage, accuracy, utility and safeguards dashboard.")


if __name__ == "__main__":
    run()
