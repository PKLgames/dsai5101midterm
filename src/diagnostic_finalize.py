"""Reconcile and summarize completed Stage 8 development diagnostics."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from .development import _hash
from .diagnostics import load_data, write_csv


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _plots(root: Path, paths: list[dict[str, str]], curves: list[dict[str, str]], curve_status: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for column, block in enumerate(("H", "C", "Y")):
        for row_index, model in enumerate((f"common:M3-{block}", f"common:M4-{block}")):
            ax = axes[row_index, column]
            norm_axis = ax.twinx()
            selected = [row for row in paths if row["model"] == model]
            if not selected:
                raise ValueError(f"No reproduced penalty path for {model}.")
            for rho in sorted({float(row["rho"]) for row in selected}):
                group = sorted([row for row in selected if float(row["rho"]) == rho],
                               key=lambda row: float(row["ridge"]))
                xs = [float(row["ridge"]) for row in group]
                suffix = "" if row_index == 0 else f" rho={rho:g}"
                ax.plot(xs, [float(row["train_loss"]) for row in group], linestyle="--", label="train" + suffix)
                ax.plot(xs, [float(row["validation_loss"]) for row in group], marker=".", label="validation" + suffix)
                norm_axis.plot(xs, [float(row["coefficient_norm"]) for row in group],
                               linestyle=":", alpha=.5, color="gray")
            ax.set_xscale("log")
            ax.set_title(model)
            ax.set_xlabel("Ridge lambda")
            ax.set_ylabel("Log loss")
            norm_axis.set_ylabel("Coefficient L2 norm (gray)")
            ax.legend(fontsize="x-small")
    fig.suptitle("Development penalty paths; gray dotted lines show coefficient norms")
    fig.tight_layout()
    fig.savefig(root / "figures/penalty_paths.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if curve_status["status"] != "completed":
        ax.text(.5, .5, "Learning curves omitted: inadequate common validation coverage",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        for model in ("M2", "M3*", "M4*"):
            for seed in sorted({row["seed"] for row in curves}):
                entries = sorted([row for row in curves if row["model"] == model
                                  and row["seed"] == seed and row["status"] == "ok"],
                                 key=lambda row: int(row["fitting_responses"]))
                if entries:
                    ax.plot([int(row["fitting_responses"]) for row in entries],
                            [float(row["validation_loss"]) for row in entries],
                            marker="o", label=f"{model}, seed {seed}")
        ax.set(xlabel="Actual fitting responses", ylabel="Common-subset validation log loss",
               title=f"Nested-pair curves; validation: {curve_status['responses']} responses, {curve_status['pairs']} pairs")
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(root / "figures/learning_curves.pdf")
    plt.close(fig)


def run_finalize(root: Path) -> dict[str, Any]:
    outputs = [root / "figures/penalty_paths.pdf", root / "figures/learning_curves.pdf",
               root / "artifacts/development_decisions.csv", root / "tables/diagnostic_interpretation.csv",
               root / "artifacts/stage8_provenance.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Stage 8 final artifacts already exist.")
    data = load_data(root)
    core_path = root / "artifacts/stage8_core.json"
    core = json.loads(core_path.read_text(encoding="utf-8"))
    if core["status"] != "passed_operational_control":
        raise ValueError("Synthetic-label control requires investigation before completion.")
    for group in ("input_sha256", "output_sha256"):
        for name, digest in core[group].items():
            path = root.joinpath(*name.split("/"))
            if not path.is_file() or _hash(path) != digest:
                raise ValueError(f"Stage 8 core provenance mismatch: {name}")
    curve_status = json.loads((root / "artifacts/learning_curve_status.json").read_text(encoding="utf-8"))
    representations = json.loads((root / "artifacts/representation_diagnostic_status.json").read_text(encoding="utf-8"))
    reports = json.loads((root / "artifacts/diagnostic_report_status.json").read_text(encoding="utf-8"))
    if curve_status["status"] not in {"completed", "omitted_inadequate_common_validation"}:
        raise ValueError("Learning-curve diagnostics are incomplete.")
    if representations["status"] != "completed" or reports["status"] != "completed":
        raise ValueError("Representation or confidence diagnostics are incomplete.")
    paths = _read(root / "tables/penalty_path_refits.csv")
    curves = _read(root / "tables/learning_curves.csv")
    candidates = _read(root / "tables/learning_curve_candidates.csv")
    control = _read(root / "tables/synthetic_label_control.csv")
    ablations = _read(root / "tables/grouped_ablations.csv")
    pca = _read(root / "tables/cnn_pca_diagnostic.csv")
    temperatures = _read(root / "tables/temperature_diagnostic.csv")
    curve_context = _read(root / "tables/learning_curve_context_support.csv")
    curve_preprocessing = json.loads((root / "artifacts/learning_curve_preprocessing.json").read_text(encoding="utf-8"))
    if len(control) != 60 or any(row["accepted"] != "True" for row in control):
        raise ValueError("Synthetic-label controls are incomplete or contain failed fits.")
    if not paths or any(row["accepted"] != "True" or abs(float(row["reproduction_delta"])) > 1e-6 for row in paths):
        raise ValueError("A penalty path refit failed or did not reproduce Stage 7.")
    if curve_status["status"] == "completed" and (not curves or not candidates or
                                                   any(row["accepted"] != "True" for row in candidates)):
        raise ValueError("Learning-curve fits are incomplete or failed.")
    if curve_preprocessing["status"] != curve_status["status"]:
        raise ValueError("Learning-curve preprocessing status does not match the curve run.")
    if curve_status["status"] == "completed" and (not curve_context or
                                                   len(curve_preprocessing["fitting_only_preprocessing"]) !=
                                                   len(curve_status["sizes"]) * len(data.protocol["learning_curve_seeds"])):
        raise ValueError("Learning-curve context support or fitting-only preprocessing is incomplete.")
    if any(row["status"] == "ok" and not row["validation_loss"] for row in ablations):
        raise ValueError("A registered ablation fit is missing its validation loss.")
    if len([row for row in pca if row["selected"] == "True"]) != 1:
        raise ValueError("CNN PCA diagnostic has no unique validation winner.")
    for model in ("M2", "M3*", "M4*"):
        if len([row for row in temperatures if row["model"] == model and row["selected"] == "True"]) != 1:
            raise ValueError(f"Temperature diagnostic lacks a unique choice for {model}.")
    contribution_rows = _read(root / "tables/feature_contributions.csv")
    if not contribution_rows:
        raise ValueError("Model-logit contribution examples are missing.")

    selected = data.selection["selected"]
    block = data.selection["selected_representation"]
    m1, m2 = selected["common:M1"], selected["common:M2"]
    m3, m4 = selected["M3*"], selected["M4*"]
    descriptions = [
        {
            "question": "Photo identity versus cat identity",
            "measured_result": f"Validation loss M1={m1['validation_loss']:.6f}; M2={m2['validation_loss']:.6f}; fitting losses M1={m1['train_loss']:.6f}, M2={m2['train_loss']:.6f}.",
            "supported_interpretation": "Photo-specific fitting gains did not transfer to validation in this comparison; the gap is consistent with estimation variance.",
            "alternatives": "Regularization, model form, population composition, and sampling variability remain possible.",
            "unestablished": "This does not prove photo differences are irrelevant or identify a unique cause.",
            "next_evidence": "Learning curves and paired internal-test uncertainty after freeze.",
        },
        {
            "question": "Selected visual representation versus photo identity",
            "measured_result": f"Validation loss M2={m2['validation_loss']:.6f}; M3-{block}={m3['validation_loss']:.6f}; M1={m1['validation_loss']:.6f}.",
            "supported_interpretation": "The selected shared representation improved on photo identity in development, but its validation advantage does not by itself establish test performance.",
            "alternatives": "Representation geometry, shrinkage, and selection variability may contribute.",
            "unestablished": "The result is not a causal interpretation of visual coordinates.",
            "next_evidence": "Inspect representation and crop quality; evaluate the frozen comparison with paired intervals.",
        },
        {
            "question": "Recorded pet-care context",
            "measured_result": f"Validation loss M3*={m3['validation_loss']:.6f}; M4*={m4['validation_loss']:.6f}; pooled fallback selected={m4['pooled']}; M3-side={selected['M3-side']['validation_loss']:.6f}; M4-side={selected['M4-side']['validation_loss']:.6f}.",
            "supported_interpretation": "The matched context extension's development difference is a validation selection result, not a held-out conclusion.",
            "alternatives": "Uneven group support, composition, shrinkage, and confidence errors remain possible.",
            "unestablished": "This does not establish a causal or universally generalizable pet-care effect.",
            "next_evidence": "Inspect shared-pair support and the side diagnostic; evaluate the frozen matched contrast with paired intervals.",
        },
        {
            "question": "YOLO-CLIP representation versus selected visual representation",
            "measured_result": f"Validation loss M3-Y={selected['M3-Y']['validation_loss']:.6f}; M3-{block}={m3['validation_loss']:.6f}.",
            "supported_interpretation": "The automated attributes were less effective than the selected shared representation in this development comparison.",
            "alternatives": "Measurement error, label taxonomy, transfer representations, and regularization remain possible explanations.",
            "unestablished": "This does not validate automated labels as ground truth or demonstrate a causal attribute effect.",
            "next_evidence": "Review detection-quality slices, fitting-only score reconstruction, and grouped ablations.",
        },
    ]
    for row in descriptions:
        row.update({"partition": "common_strict_validation", "evidence_status": "Real-data claim"})
    decisions = [
        {"decision": "primary_representation", "value": block, "status": "retained_stage7_selection",
         "scope": "development only; no test-driven reselection"},
        {"decision": "synthetic_label_control", "value": core["status"], "status": "completed",
         "scope": "20 independent repetitions; does not certify absence of leakage"},
        {"decision": "learning_curve_coverage", "value": json.dumps(curve_status), "status": curve_status["status"],
         "scope": "100 responses and 30 pairs are operational thresholds, not power guarantees"},
        {"decision": "cnn_pca", "value": "development-only diagnostic", "status": "not_promoted",
         "scope": "not part of the primary candidate registry"},
        {"decision": "temperature", "value": json.dumps({row["model"]: float(row["temperature"])
                                                            for row in temperatures if row["selected"] == "True"}),
         "status": "not_promoted",
         "scope": "must be fixed before any optional final test use"},
        {"decision": "ablation", "value": "registered H and Y groups", "status": "secondary",
         "scope": "conditional predictive value, not causal behavioral percentages"},
        {"decision": "score_recovery", "value": "fitting-only M2 scores; ridge 0.001", "status": "secondary",
         "scope": "in-sample reconstruction of estimated scores, not an independent preference test"},
        {"decision": "contribution_examples", "value": "first five validation response IDs", "status": "secondary",
         "scope": "model-logit explanations, not causal preference effects"},
    ]
    _plots(root, paths, curves, curve_status)
    write_csv(outputs[2], decisions)
    write_csv(outputs[3], descriptions)
    input_names = [
        "artifacts/stage8_protocol.json", "artifacts/stage8_probe_protocol.json",
        "artifacts/stage8_contribution_protocol.json", "artifacts/stage8_core.json",
        "artifacts/learning_curve_status.json", "artifacts/representation_diagnostic_status.json",
        "artifacts/learning_curve_preprocessing.json", "tables/learning_curve_context_support.csv",
        "artifacts/diagnostic_report_status.json", "tables/learning_curves.csv",
        "tables/learning_curve_candidates.csv", "tables/penalty_path_refits.csv",
        "tables/grouped_ablations.csv", "tables/cnn_pca_diagnostic.csv",
        "tables/temperature_diagnostic.csv", "tables/representation_recovery.csv",
        "tables/feature_contributions.csv", "tables/worst_validation_losses.csv",
        "tables/diagnostic_confidence.csv", "figures/diagnostic_calibration.pdf",
        "src/diagnostics.py", "src/learning_curves.py", "src/learning_curve_support.py", "src/representation_diagnostics.py",
        "src/diagnostic_reports.py", "src/contributions.py", "src/diagnostic_finalize.py",
    ]
    result = {
        "status": "completed", "internal_test_outcomes_read": False,
        "selected_representation": block, "synthetic_control": core["status"],
        "learning_curve_status": curve_status,
        "input_sha256": {name: _hash(root.joinpath(*name.split("/"))) for name in input_names},
        "output_sha256": {path.relative_to(root).as_posix(): _hash(path) for path in outputs[:-1]},
    }
    outputs[-1].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
