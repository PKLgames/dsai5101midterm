"""Exploratory abstention curves from saved frozen common-test predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .audit import _digest

PROTOCOL = "artifacts/abstention_utility_protocol.json"
OUTPUTS = (
    "figures/presentation_abstention_utility.pdf",
    "tables/exploratory_abstention_utility.csv",
    "artifacts/abstention_utility_provenance.json",
)
LABEL = "Post-Test Exploratory Analysis | Not Threshold Selection"
MODEL_LABELS = {
    "M0": "M0: constant", "M1": "M1: cat identity", "M2": "M2: photo identity",
    "M3*": "M3*: frozen CNN", "M4*": "M4*: matched context",
}


def _hash(path: Path) -> str:
    return _digest(path)[0]


def verify_sources(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen_path = root / "artifacts/model_freeze.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    seal = (root / "artifacts/model_freeze.sha256").read_text(encoding="ascii").strip()
    evaluation = json.loads((root / "artifacts/evaluation_provenance.json").read_text(encoding="utf-8"))
    if _hash(frozen_path) != seal or evaluation["freeze_sha256"] != seal:
        raise ValueError("Frozen evaluation identity or freeze seal does not match.")
    if evaluation["status"] != "evaluated_once" or frozen["test_outcomes_loaded_at_freeze"] is not False:
        raise ValueError("A valid single frozen evaluation is required.")
    for name in ("artifacts/test_predictions.parquet", "tables/main_results.csv"):
        if _hash(root / name) != evaluation["output_sha256"][name]:
            raise ValueError(f"Frozen prediction source changed: {name}")
    return frozen, evaluation


def load_models(root: Path, frozen: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    block = frozen["selected_representation"]
    keys = {
        "M0": "common:M0", "M1": "common:M1", "M2": "common:M2",
        "M3*": f"M3-{block}", "M4*": f"M4-{block}",
    }
    if block != "C":
        raise ValueError("The user-approved CNN comparison differs from this frozen representation.")
    records = pq.read_table(
        root / "artifacts/test_predictions.parquet",
        columns=["population", "model_id", "train_row_id", "pair_id", "outcome", "probability", "logit", "loss"],
        filters=[("population", "=", "common"), ("model_id", "in", list(keys.values()))],
    ).to_pylist()
    models = {
        label: sorted([row for row in records if row["model_id"] == key], key=lambda row: row["train_row_id"])
        for label, key in keys.items()
    }
    reference = models["M0"]
    if not reference:
        raise ValueError("The frozen common-test prediction population is empty.")
    reference_ids = [(row["train_row_id"], row["pair_id"], row["outcome"]) for row in reference]
    with (root / "tables/main_results.csv").open(encoding="utf-8", newline="") as stream:
        main = {row["model_id"]: row for row in csv.DictReader(stream) if row["population"] == "common"}
    for label, rows in models.items():
        if [(row["train_row_id"], row["pair_id"], row["outcome"]) for row in rows] != reference_ids:
            raise ValueError(f"Matched frozen rows, pairs or outcomes differ: {label}")
        if len({row["train_row_id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate frozen response IDs: {label}")
        p = np.asarray([row["probability"] for row in rows], dtype=np.float64)
        eta = np.asarray([row["logit"] for row in rows], dtype=np.float64)
        y = np.asarray([row["outcome"] for row in rows], dtype=np.float64)
        loss = np.asarray([row["loss"] for row in rows], dtype=np.float64)
        if not (np.isfinite(p).all() and np.isfinite(eta).all() and np.isfinite(loss).all()
                and np.isin(y, [0, 1]).all() and ((0 <= p) & (p <= 1)).all()):
            raise ValueError(f"Invalid saved probabilities, logits, losses or outcomes: {label}")
        if not np.allclose(expit(eta), p, rtol=0, atol=1e-12):
            raise ValueError(f"Frozen probability/logit relationship differs: {label}")
        expected_loss = y*np.logaddexp(0., -eta)+(1-y)*np.logaddexp(0., eta)
        if not np.allclose(loss, expected_loss, rtol=0, atol=1e-12):
            raise ValueError(f"Frozen losses do not reproduce from saved outcomes: {label}")
        baseline = main[keys[label]]
        if int(baseline["responses"]) != len(rows) or int(baseline["pairs"]) != len({row["pair_id"] for row in rows}):
            raise ValueError("Saved test support counts do not match main results.")
        if not math.isclose(float(np.mean((p >= .5) == y)), float(baseline["accuracy"]), abs_tol=1e-12):
            raise ValueError(f"Frozen baseline accuracy differs: {label}")
    return models


def point_metrics(
    probabilities: Any, outcomes: Any, losses: Any, pair_ids: list[str], threshold: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    p, y, loss = [np.asarray(value, dtype=np.float64) for value in (probabilities, outcomes, losses)]
    if (p.ndim != 1 or not len(p) or p.shape != y.shape or p.shape != loss.shape or len(pair_ids) != len(p)
        or not all(np.isfinite(value).all() for value in (p, y, loss))
        or np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all()
        or not .5 <= threshold <= 1):
        raise ValueError("Invalid probabilities, binary outcomes, pair IDs, losses, or confidence threshold.")
    confidence = np.maximum(p, 1-p)
    retain = confidence >= threshold
    correct = ((p >= .5) == y)
    utilities = np.where(retain, np.where(correct, 1., -1.), 0.)
    implied = np.where(retain, 2*confidence-1, 0.)
    n = int(retain.sum())
    wins = int((retain & correct).sum())
    pairs = len({pair_ids[i] for i in np.flatnonzero(retain)})
    metrics = {
        "threshold": threshold, "original_responses": len(p), "original_pairs": len(set(pair_ids)),
        "retained_responses": n, "retained_pairs": pairs, "abstained_responses": len(p)-n,
        "correct_retained": wins, "incorrect_retained": n-wins, "coverage": n/len(p),
        "selective_accuracy": wins/n if n else None, "selective_error_rate": (n-wins)/n if n else None,
        "selective_log_loss": float(loss[retain].mean()) if n else None,
        "utility_per_original_response": float(utilities.mean()),
        "utility_per_retained_response": float(utilities[retain].mean()) if n else None,
        "model_implied_utility_per_original_response": float(implied.mean()),
        "always_predict_utility": float(np.mean(np.where(correct, 1., -1.))),
        "always_abstain_utility": 0.,
        "utility_difference_from_always_predict": float(utilities.mean()-np.mean(np.where(correct, 1., -1.))),
        "support_status": "no_retained_cases" if not n else "supported" if n >= 100 and pairs >= 30 else "low_support",
    }
    if n and not math.isclose(metrics["utility_per_original_response"],
                              metrics["coverage"]*(2*metrics["selective_accuracy"]-1), abs_tol=1e-14):
        raise RuntimeError("Utility and selective accuracy identity did not reconcile.")
    return metrics, retain, correct, utilities


def shared_pair_resamples(pair_ids: list[str], seed: int, replicates: int) -> tuple[np.ndarray, np.ndarray]:
    pairs = sorted(set(pair_ids))
    if not pairs or replicates < 2:
        raise ValueError("Pair bootstrap requires observations and at least two resamples.")
    index = {pair: i for i, pair in enumerate(pairs)}
    codes = np.asarray([index[pair] for pair in pair_ids], dtype=np.int64)
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, len(pairs)), dtype=np.float64)
    for i in range(replicates):
        draw = rng.integers(0, len(pairs), size=len(pairs))
        samples[i] = np.bincount(draw, minlength=len(pairs))
    return codes, samples


def interval(values: np.ndarray) -> tuple[float, float, float]:
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Exploratory interval values must be finite.")
    low, high = np.quantile(values, [.025, .975])
    return float(low), float(high), float(np.std(values, ddof=1))


def calculate(models: dict[str, list[dict[str, Any]]], thresholds: list[float],
              seed: int, replicates: int) -> list[dict[str, Any]]:
    if not models or not thresholds or thresholds != sorted(set(thresholds)) or thresholds[0] != .5:
        raise ValueError("A fixed ordered threshold grid beginning at 0.5 is required.")
    reference = next(iter(models.values()))
    pair_ids = [row["pair_id"] for row in reference]
    identity = [(row["train_row_id"], row["pair_id"], row["outcome"]) for row in reference]
    if len(set(row["train_row_id"] for row in reference)) != len(reference):
        raise ValueError("Duplicate response identities.")
    codes, samples = shared_pair_resamples(pair_ids, seed, replicates)
    k = samples.shape[1]
    pair_counts = np.bincount(codes, minlength=k)
    denominator = samples @ pair_counts
    records = []
    with threadpool_limits(limits=1):
        for model, rows in models.items():
            if [(row["train_row_id"], row["pair_id"], row["outcome"]) for row in rows] != identity:
                raise ValueError("All models must use identical response IDs, pairs, order and outcomes.")
            p = np.asarray([row["probability"] for row in rows])
            y = np.asarray([row["outcome"] for row in rows])
            losses = np.asarray([row["loss"] for row in rows])
            implied_history = []
            for threshold in thresholds:
                point, retained, correct, utility = point_metrics(p, y, losses, pair_ids, threshold)
                accepted = np.bincount(codes, weights=retained.astype(float), minlength=k)
                wins = np.bincount(codes, weights=(retained & correct).astype(float), minlength=k)
                total_utility = np.bincount(codes, weights=utility, minlength=k)
                retained_boot = samples @ accepted
                utility_boot = samples @ total_utility / denominator
                coverage_boot = retained_boot / denominator
                difference_boot = (samples @ (
                    total_utility-np.bincount(codes, weights=np.where(correct, 1., -1.), minlength=k)
                )) / denominator
                zero = retained_boot == 0
                conditional_ci = (point["support_status"] == "supported" and not zero.any())
                alow = ahigh = ase = None
                if conditional_ci:
                    alow, ahigh, ase = interval((samples @ wins)/retained_boot)
                ulo, uhi, use = interval(utility_boot)
                clo, chi, cse = interval(coverage_boot)
                dlo, dhi, dse = interval(difference_boot)
                pair_mean = total_utility/pair_counts
                records.append({
                    "model": model, "partition": "frozen_common_strict_G2_test", **point,
                    "utility_ci95_low": ulo, "utility_ci95_high": uhi, "utility_bootstrap_se": use,
                    "coverage_ci95_low": clo, "coverage_ci95_high": chi, "coverage_bootstrap_se": cse,
                    "selective_accuracy_ci95_low": alow, "selective_accuracy_ci95_high": ahigh,
                    "selective_accuracy_bootstrap_se": ase,
                    "undefined_selective_accuracy_bootstrap_replicates": int(zero.sum()),
                    "selective_accuracy_interval_status": "reported" if conditional_ci else "not_reported_no_or_low_support",
                    "utility_difference_ci95_low": dlo, "utility_difference_ci95_high": dhi,
                    "utility_difference_bootstrap_se": dse,
                    "pair_weighted_utility": float(pair_mean.mean()),
                    "bootstrap_replicates": replicates,
                    "interval_interpretation": "exploratory pointwise 95%; conditional pair bootstrap; not multiplicity-adjusted",
                    "deployment_threshold_selected": False,
                })
                implied_history.append(point["model_implied_utility_per_original_response"])
            if np.any(np.diff(implied_history) > 1e-14):
                raise RuntimeError("Model-implied total utility increased after discarding nonnegative expected rewards.")
    return records


def plot(path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import PercentFormatter

    colors = {"M0": "#6d6d6d", "M1": "#177a9a", "M2": "#cf7630", "M3*": "#427653", "M4*": "#88558f"}
    layout_checks = []

    def page_header(fig: Any, title: str) -> None:
        fig.suptitle(title, x=.06, y=.96, ha="left", fontsize=20, fontweight="bold", color="#16384c")
        fig.text(.06, .90, LABEL, fontsize=12, color="#8c4c26")

    def finish(pdf: Any, fig: Any, number: int) -> None:
        from matplotlib.text import Text
        fig.text(.96, .025, str(number), ha="right", fontsize=9, color="#666666")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        suppressed = set()
        for ax in fig.axes:
            lo, hi = sorted(ax.get_xlim())
            suppressed.update(text for value, text in zip(ax.get_xticks(), ax.get_xticklabels()) if not lo <= value <= hi)
            lo, hi = sorted(ax.get_ylim())
            suppressed.update(text for value, text in zip(ax.get_yticks(), ax.get_yticklabels()) if not lo <= value <= hi)
        outside = []
        for artist in fig.findobj(Text):
            if artist.get_visible() and artist.get_text() and artist not in suppressed:
                box = artist.get_window_extent(renderer)
                if box.x0 < -2 or box.y0 < -2 or box.x1 > fig.bbox.width+2 or box.y1 > fig.bbox.height+2:
                    outside.append(artist.get_text())
        if outside:
            raise ValueError(f"Abstention figure text extends outside page {number}: {outside}")
        layout_checks.append({"page": number, "outside_page_text": outside})
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        fig.subplots_adjust(left=.075, right=.97, top=.76, bottom=.25, wspace=.26)
        page_header(fig, "Abstention: confidence, coverage, and selective accuracy")
        for model, color in colors.items():
            rows = [row for row in records if row["model"] == model]
            axes[0].plot([r["threshold"] for r in rows], [r["coverage"] for r in rows],
                         color=color, label=MODEL_LABELS[model], linewidth=2)
            available = [row for row in rows if row["selective_accuracy"] is not None]
            axes[1].plot([r["coverage"] for r in available], [r["selective_accuracy"] for r in available],
                         color=color, linewidth=1.6)
            for row in available:
                supported = row["support_status"] == "supported"
                axes[1].plot(row["coverage"], row["selective_accuracy"], marker="o", markersize=4,
                             color=color, markerfacecolor=color if supported else "white")
        axes[0].set(xlabel="Minimum confidence retained", ylabel="Coverage (retained / all responses)",
                    xlim=(.5, 1), ylim=(-.02, 1.02))
        axes[1].set(xlabel="Coverage", ylabel="Accuracy among retained responses", xlim=(-.02, 1.02), ylim=(0, 1.02))
        axes[1].axhline(.5, linestyle="--", color="#999999", linewidth=.7)
        for ax in axes:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.2)
        axes[1].xaxis.set_major_formatter(PercentFormatter(1))
        axes[0].legend(fontsize=10, frameon=False)
        fig.text(.06, .16, "Confidence = max(P(A), 1-P(A)). Retain confidence >= threshold; tied confidence values stay together.", fontsize=11)
        fig.text(.06, .12, "Filled points: at least 100 retained responses and 30 pairs. Hollow points: low support; use caution.", fontsize=11)
        fig.text(.06, .08, "Different models retain different rows. Empty selections have undefined accuracy, not 0% or 100%.", fontsize=11)
        finish(pdf, fig, 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        fig.subplots_adjust(left=.075, right=.97, top=.76, bottom=.28, wspace=.27)
        page_header(fig, "Utility: correct +1, incorrect -1, abstention 0")
        for model, color in colors.items():
            rows = [row for row in records if row["model"] == model]
            axes[0].plot([r["threshold"] for r in rows], [r["utility_per_original_response"] for r in rows],
                         color=color, label=MODEL_LABELS[model], linewidth=2)
        context = [row for row in records if row["model"] == "M4*"]
        xs = [r["threshold"] for r in context]
        axes[1].plot(xs, [r["utility_per_original_response"] for r in context],
                     color=colors["M4*"], label="Observed utility", linewidth=2)
        axes[1].fill_between(xs, [r["utility_ci95_low"] for r in context],
                             [r["utility_ci95_high"] for r in context], color=colors["M4*"], alpha=.18,
                             label="Exploratory pointwise 95% pair CI")
        axes[1].plot(xs, [r["model_implied_utility_per_original_response"] for r in context],
                     color="#555555", linestyle="--", label="Model-implied utility (calibration assumption)")
        axes[1].set_title("M4*: model confidence versus observed outcomes", fontsize=11)
        for ax in axes:
            ax.axhline(0, linestyle=":", color="#555555", label="Always abstain: utility 0")
            ax.set_xlabel("Minimum confidence retained")
            ax.set_ylabel("Utility per original test response")
            ax.set_xlim(.5, 1)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, frameon=False)
        fig.text(.06, .18, "Utility = (correct retained - incorrect retained) / all original responses, not just retained cases.", fontsize=11)
        fig.text(.06, .14, "With these payoffs and calibrated probabilities, always predict is weakly optimal in expected total utility.", fontsize=11)
        fig.text(.06, .10, "Higher selective accuracy alone is not a utility gain. Model-implied utility is not guaranteed real-world utility.", fontsize=11)
        fig.text(.06, .06, "Thresholds are fixed for display; no test-optimal threshold or deployment policy is selected.", fontsize=11)
        finish(pdf, fig, 2)

        fig, ax = plt.subplots(figsize=(14, 9))
        fig.subplots_adjust(left=.05, right=.97, top=.79, bottom=.18)
        page_header(fig, "Fixed illustrative thresholds: counts and utility")
        ax.set_axis_off()
        chosen = [row for row in records if row["threshold"] in (.5, .6, .7)]
        table_rows = []
        for row in chosen:
            table_rows.append([
                row["model"], f"{row['threshold']:.2f}", str(row["retained_responses"]), str(row["retained_pairs"]),
                f"{row['coverage']:.1%}", f"{row['selective_accuracy']:.1%}" if row["selective_accuracy"] is not None else "N/A",
                f"{row['utility_per_original_response']:+.4f}", row["support_status"].replace("_", " "),
            ])
        table = ax.table(cellText=table_rows, colLabels=[
            "Model", "Threshold", "Retained N", "Retained pairs", "Coverage", "Retained accuracy", "Utility / all N", "Support",
        ], cellLoc="center", loc="center", colWidths=[.07, .09, .11, .13, .10, .16, .13, .21])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)
        for (i, _), cell in table.get_celld().items():
            cell.set_edgecolor("#d8e2e8")
            cell.set_facecolor("#16384c" if i == 0 else "#edf3f7" if i % 2 else "white")
            if i == 0:
                cell.get_text().set_color("white")
        total, pairs = records[0]["original_responses"], records[0]["original_pairs"]
        fig.text(.06, .13, f"Common frozen population: {total:,} responses, {pairs:,} unordered pairs. Values are descriptive.", fontsize=11)
        fig.text(.06, .09, "The complete fixed grid, pointwise intervals, and undefined-resample counts are saved in the accompanying CSV.", fontsize=11)
        fig.text(.06, .05, "Low-support rows are displayed transparently; do not interpret them as reliable deployment evidence.", fontsize=11)
        finish(pdf, fig, 3)

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.subplots_adjust(left=.06, right=.96, top=.80, bottom=.08)
        page_header(fig, "Interpretation, safeguards, and limitations")
        ax.set_axis_off()
        import textwrap
        bullets = [
            "Data: saved frozen common strict-G2 predictions only. No raw withheld choices are loaded; no refit or new model selection is performed.",
            "These are illustrative payoffs, not measured error costs or a claim that human decisions should use this utility.",
            "Selective accuracy conditions on accepted responses. Utility includes every original response, with abstentions contributing zero.",
            "A constant-confidence baseline retains all or none at a cutoff. No arbitrary tie-breaking or exact-coverage quota is used.",
            "Pair bootstrap keeps all responses and sampled-pair multiplicity. Conditional accuracy intervals require adequate retained support and no empty bootstrap selections.",
            "Intervals are exploratory and pointwise, not multiplicity-adjusted. They omit training variability, respondent linkage, shared-image dependence, and prior analysis-induced selection.",
            "No optimum is picked from the test curves. A deployment threshold requires explicit decision costs and an independent validation policy study.",
        ]
        y = .96
        for text in bullets:
            wrapped = textwrap.fill(text, 116)
            ax.text(0, y, "- " + wrapped, fontsize=13, va="top", linespacing=1.3, transform=ax.transAxes)
            y -= .060*(wrapped.count("\n")+1)+.025
        finish(pdf, fig, 4)
    return layout_checks


def run(root: Path) -> dict[str, Any]:
    outputs = [root / name for name in OUTPUTS]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Abstention outputs already exist; refusing to overwrite or rerun.")
    frozen, evaluation = verify_sources(root)
    protocol = json.loads((root / PROTOCOL).read_text(encoding="utf-8"))
    if protocol["payoffs"] != {"correct": 1, "incorrect": -1, "abstain": 0}:
        raise ValueError("Utility assumptions differ from the user-approved +1/-1/0 payoffs.")
    if protocol["threshold_grid"] != {"start": .5, "stop": 1., "step": .025}:
        raise ValueError("The documented fixed threshold grid differs from this implementation.")
    models = load_models(root, frozen)
    thresholds = [i/40 for i in range(20, 41)]
    boot = protocol["bootstrap"]
    if (boot["seed"] != frozen["bootstrap"]["seed"] or boot["replicates"] != 5000 or
        protocol["support_rule"] != {"retained_responses": 100, "retained_pairs": 30}):
        raise ValueError("Bootstrap or retained-support configuration changed.")
    protected_names = [
        "artifacts/model_freeze.json", "artifacts/model_freeze.sha256",
        "artifacts/evaluation_provenance.json", "artifacts/test_predictions.parquet",
        "artifacts/model_selection.json", "tables/main_results.csv", "tables/primary_contrasts.csv",
        "tables/secondary_results.csv", "claims_final.csv", "tables/diagnostic_synthesis.csv",
        "Group_19_Report.pdf", "Group_19_Code.zip",
        PROTOCOL, "src/abstention_utility.py",
    ]
    protected = {name: _hash(root / name) for name in protected_names}
    records = calculate(models, thresholds, boot["seed"], boot["replicates"])
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    with tempfile.TemporaryDirectory(prefix="abstention-figure-") as directory:
        staged_pdf = Path(directory) / "presentation_abstention_utility.pdf"
        layout = plot(staged_pdf, records)
        with outputs[1].open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        with outputs[0].open("xb") as stream:
            stream.write(staged_pdf.read_bytes())
    for name, digest in protected.items():
        if _hash(root / name) != digest:
            raise RuntimeError(f"Existing frozen/report artifact changed: {name}")
    verify_sources(root)
    provenance = {
        "status": "post_test_exploratory", "figure": OUTPUTS[0],
        "models": list(models), "common_responses": len(models["M0"]),
        "common_pairs": len({row["pair_id"] for row in models["M0"]}),
        "thresholds": thresholds, "payoffs": protocol["payoffs"],
        "threshold_selected": False, "deployment_changed": False, "models_refit": False,
        "raw_test_outcomes_read": False, "frozen_evaluation_rerun": False,
        "working_data": "Only already-saved outcomes and probabilities in the original frozen prediction artifact",
        "freeze_sha256": evaluation["freeze_sha256"], "bootstrap": boot, "layout_checks": layout,
        "input_sha256": {**protected, PROTOCOL: _hash(root / PROTOCOL),
                          "src/abstention_utility.py": _hash(root / "src/abstention_utility.py")},
        "output_sha256": {path.relative_to(root).as_posix(): _hash(path) for path in outputs[:-1]},
        "interpretation": protocol["decision_theory_note"],
        "limits": "Exploratory post-test thresholds and pointwise pair-bootstrap intervals do not establish an independently validated abstention policy.",
        "existing_deliverables": "The original report, code ZIP, claims, synthesis, and Presentation_Evidence copies were not modified to incorporate this supplementary extension.",
    }
    with outputs[2].open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        report = run(args.root.resolve())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Exploratory abstention analysis failed: {exc}", file=sys.stderr)
        return 2
    print(f"Generated exploratory abstention utility PDF for {len(report['models'])} frozen models; no threshold selected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
