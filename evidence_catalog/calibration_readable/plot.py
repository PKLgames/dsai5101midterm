"""Re-label the original frozen G2 calibration figure without changing its data."""

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT/"tables"/"test_calibration.csv"
SERIES = (
    ("common:M0", "Baseline", "#171717", "D"),
    ("common:M1", "Cat ID", "#0072B2", "o"),
    ("common:M2", "Photo ID", "#D55E00", "s"),
    ("M3-C", "CNN", "#009E73", "^"),
    ("M4-C", "CNN + Context", "#CC79A7", "v"),
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read_points(source):
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for model, label, _, _ in SERIES:
        selected = [r for r in rows if r["population"] == "common" and r["model_id"] == model
                    and r["dimension"] == "overall" and r["group"] == "all" and r["support_status"] == "supported"]
        selected.sort(key=lambda r: int(r["bin"]))
        if not selected or len({r["bin"] for r in selected}) != len(selected):
            raise ValueError(f"No supported bins or duplicate bins for {model}.")
        for r in selected:
            count, pairs = int(r["responses"]), int(r["pairs"])
            x, y = float(r["mean_probability"]), float(r["observed_a_rate"])
            if count < 30 or pairs < 10 or not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(f"Invalid supported calibration bin for {model}.")
            result.append({"model_id": model, "display_name": label, "bin": int(r["bin"]),
                           "responses": count, "pairs": pairs, "mean_probability": x, "observed_a_rate": y})
    return result


def render(points, pdf, png):
    with tempfile.TemporaryDirectory(prefix="calibration_mpl_") as config:
        os.environ["MPLCONFIGDIR"] = config
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9.5, 8))
        ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1.5, label="Perfect calibration")
        for model, label, color, marker in SERIES:
            data = [r for r in points if r["model_id"] == model]
            ax.plot([r["mean_probability"] for r in data], [r["observed_a_rate"] for r in data],
                    marker=marker, markersize=6.5, linewidth=1.7, color=color, label=label)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability of choosing photo A",
               ylabel="Observed fraction choosing photo A", title="Model calibration plot")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#e2e2e2", linewidth=.7)
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=.95)
        fig.text(.10, .055, "Held-out test: 3,733 responses. Each point is a prediction-probability bin.",
                 fontsize=9.5)
        fig.text(.10, .025, "Bins: at least 30 responses and 10 photo pairs. Context = pet care; photo A is not necessarily left.",
                 fontsize=9)
        fig.subplots_adjust(left=.12, right=.95, top=.91, bottom=.15)
        fig.savefig(pdf)
        fig.savefig(png, dpi=220)
        plt.close(fig)


def run(output_dir=HERE):
    outputs = [output_dir/"calibration_readable.pdf", output_dir/"calibration_readable.png",
               output_dir/"plotted_bins.csv", output_dir/"provenance.json"]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Readable calibration output already exists.")
    source_hash = sha(SOURCE)
    points = read_points(SOURCE)
    render(points, outputs[0], outputs[1])
    with outputs[2].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(points)
    if sha(SOURCE) != source_hash:
        raise RuntimeError("Frozen calibration input changed during plotting.")
    provenance = {"status": "presentation_relabel_only", "input": "tables/test_calibration.csv",
                  "input_sha256": source_hash, "source_plot": "figures/calibration.pdf",
                  "source_plot_sha256": sha(ROOT/"figures"/"calibration.pdf"),
                  "models_in_original_order": [s[0] for s in SERIES],
                  "model_labels": {s[0]: s[1] for s in SERIES}, "bin_counts": {
                      s[0]: sum(r["model_id"] == s[0] for r in points) for s in SERIES},
                  "outputs_sha256": {p.name: sha(p) for p in outputs[:3]},
                  "frozen_inputs_modified": False, "model_fit_or_re_evaluation": False}
    with outputs[3].open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    run()
