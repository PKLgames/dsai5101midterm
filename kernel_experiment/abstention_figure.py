"""Plot selective accuracy, coverage and fixed-cutoff counts from verified saved results."""

import csv
import os
import tempfile

from kernel import HERE
from study import digest, write_json


def run():
    pdf, proof = HERE/"presentation_kernel_abstention.pdf", HERE/"abstention_figure_provenance.json"
    if pdf.exists() or proof.exists():
        raise FileExistsError("Abstention figure exists.")
    with (HERE/"g2_abstention_utility.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    models = ("kernel_M3-C", "original:M3-C", "original:M4-C")
    if len(rows) != 63 or set(r["model"] for r in rows) != set(models):
        raise ValueError("Incomplete abstention grid.")
    with tempfile.TemporaryDirectory(prefix="mpl_", dir=HERE) as config:
        os.environ["MPLCONFIGDIR"] = config
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(pdf) as writer:
            fig, axes = plt.subplots(1, 2, figsize=(13, 7))
            for name in models:
                group = [r for r in rows if r["model"] == name]
                axes[0].plot([float(r["threshold"]) for r in group], [float(r["coverage"]) for r in group], label=name)
                available = [r for r in group if r["selective_accuracy"]]
                line, = axes[1].plot([float(r["coverage"]) for r in available],
                                     [float(r["selective_accuracy"]) for r in available], label=name)
                for row in available:
                    axes[1].plot(float(row["coverage"]), float(row["selective_accuracy"]), "o", markersize=4,
                                 markeredgecolor=line.get_color(),
                                 markerfacecolor=line.get_color() if row["support_status"] == "supported" else "white")
            axes[0].set(title="Confidence and coverage", xlabel="Minimum confidence", ylabel="Retained / all")
            axes[1].set(title="Selective accuracy versus coverage", xlabel="Coverage", ylabel="Accuracy among retained")
            for ax in axes:
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            fig.suptitle("Post-test exploratory abstention; no cutoff selected", fontsize=15)
            fig.text(.05,.035,"Hollow points: low support (<100 retained responses or <30 pairs); empty accuracy is undefined.",fontsize=9)
            fig.tight_layout(rect=(0,.07,1,.95))
            writer.savefig(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axis("off")
            data = []
            for name in models:
                for cutoff in (.5,.6,.7):
                    r = next(r for r in rows if r["model"] == name and float(r["threshold"]) == cutoff)
                    data.append([name, f"{cutoff:.2f}", r["retained_responses"], r["retained_pairs"],
                                 f"{100*float(r['coverage']):.1f}%", f"{100*float(r['selective_accuracy']):.1f}%",
                                 f"{float(r['utility_per_original_response']):.4f}", r["support_status"]])
            table = ax.table(cellText=data, colLabels=["Model","Threshold","Retained","Pairs","Coverage","Accuracy","Utility/all","Support"],
                             loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1,1.5)
            fig.suptitle("Fixed illustrative thresholds (+1/-1/0)", fontsize=14)
            fig.text(.05,.05,"Utility denominator includes all original responses; no post-test threshold selection.",fontsize=9)
            writer.savefig(fig)
            plt.close(fig)
    write_json(proof, {"status": "completed", "pages": 2, "input_sha256": digest(HERE/"g2_abstention_utility.csv"),
                       "output_sha256": digest(pdf), "threshold_selected": False})
    print("Saved two-page abstention coverage/accuracy and fixed-count figure.")


if __name__ == "__main__":
    run()
