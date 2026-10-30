"""Descriptive coefficient charts from frozen models; no model refitting."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .development import _hash
from .freeze import validate_freeze

DISCLAIMER = "Predictive Association, Not Causal Effect"
OUTPUTS = (
    "figures/presentation_feature_importance_H.pdf",
    "figures/presentation_feature_importance_Y.pdf",
    "figures/presentation_context_interactions.pdf",
    "tables/presentation_feature_coefficients.csv",
    "tables/presentation_context_interactions.csv",
    "artifacts/presentation_feature_provenance.json",
)

H_LABELS = {
    "brightness": "Mean luminance",
    "contrast": "Luminance contrast",
    "saturation_mean": "Mean saturation",
    "saturation_std": "Saturation variation",
    "colorfulness": "Colorfulness",
    "sharpness_laplacian_var": "Laplacian variance (thumbnail)",
    "entropy_bits": "Luminance entropy",
    "edge_density": "Edge density",
    "warm_pixel_ratio": "Warm-pixel fraction",
    "hue_mean": "Circular mean hue",
    "hue_concentration": "Hue concentration",
    "mean_r": "Mean red channel",
    "mean_g": "Mean green channel",
    "mean_b": "Mean blue channel",
    "aspect_ratio": "Original width / height",
    "log_pixels": "Log original pixel count",
    "overexposed_frac": "High-value pixel fraction",
    "underexposed_frac": "Low-value pixel fraction",
    "is_effectively_gray": "Effectively grayscale indicator",
    "log_file_kb": "Log file size",
}
Y_FAMILIES = {
    "breed": "CLIP-inferred breed",
    "eyes": "Eyes",
    "eyes_colour": "Eye color",
    "colour_primary": "Primary coat color",
    "colour_secondary": "Secondary coat color",
    "colour_pattern": "Coat pattern",
    "framing": "Framing",
    "view": "View",
    "age": "CLIP-inferred age",
    "posture": "Posture",
}
GROUP_LABELS = {
    "never": "Never",
    "occasionally": "Occasional care",
    "past_regular": "Past regular care",
    "current_regular": "Current regular care",
    "missing": "Missing context",
}


def top_signed(names: list[str], coefficients: Any, count: int = 5) -> list[dict[str, Any]]:
    weights = np.asarray(coefficients, dtype=np.float64)
    if count < 1 or weights.shape != (len(names),) or len(set(names)) != len(names):
        raise ValueError("Coefficient names, dimensions, or requested count are invalid.")
    if not np.isfinite(weights).all():
        raise ValueError("Feature coefficients must be finite.")
    positive = sorted((i for i, value in enumerate(weights) if value > 0),
                      key=lambda i: (-weights[i], names[i]))
    negative = sorted((i for i, value in enumerate(weights) if value < 0),
                      key=lambda i: (weights[i], names[i]))
    if len(positive) < count or len(negative) < count:
        raise ValueError(f"Cannot show top {count} of each sign: insufficient nonzero coefficients.")
    return [
        {"feature": names[index], "coefficient": float(weights[index]), "direction": direction,
         "rank_within_direction": rank}
        for direction, indices in (("positive", positive[:count]), ("negative", negative[:count]))
        for rank, index in enumerate(indices, start=1)
    ]


def context_shifts(
    names: list[str], coefficients: Any, groups: list[str], reference: str, top: int = 8,
) -> tuple[list[str], list[dict[str, Any]]]:
    theta = np.asarray(coefficients, dtype=np.float64)
    dimension = len(names)
    if (not dimension or len(set(names)) != dimension or not groups or reference in groups
        or len(set(groups)) != len(groups) or top < 1 or top > dimension):
        raise ValueError("Invalid CNN dimensions, interaction groups, reference, or display count.")
    if theta.shape != (1 + dimension * (1 + len(groups)),) or not np.isfinite(theta).all():
        raise ValueError("Context weights do not match intercept/base/interaction block layout.")
    base = theta[1:1+dimension]
    deviations = theta[1+dimension:].reshape(len(groups), dimension)
    magnitudes = np.max(np.abs(deviations), axis=0)
    order = sorted(range(dimension), key=lambda index: (-magnitudes[index], names[index]))[:top]
    rows = []
    for rank, index in enumerate(order, start=1):
        for group_index, group in enumerate(groups):
            shift = float(deviations[group_index, index])
            rows.append({
                "model": "M4-C", "feature": names[index], "display_rank": rank,
                "pet_care": group, "reference_group": reference,
                "reference_weight": float(base[index]), "interaction_shift": shift,
                "group_weight": float(base[index] + shift),
                "max_abs_shift_across_groups": float(magnitudes[index]),
            })
    return [names[index] for index in order], rows


def feature_label(feature: str, block: str) -> str:
    if "=" in feature:
        family, level = feature.split("=", 1)
        display = {"<other>": "other (pooled)", "<unknown>": "unknown / unseen",
                   "unknown": "unknown (pipeline)", "no_detection": "no detection"}.get(
                       level, level.replace("_", " ")
                   )
        return f"{Y_FAMILIES.get(family, family.replace('_', ' '))}: {display}"
    if feature.endswith("__missing"):
        return feature_label(feature.removesuffix("__missing"), block) + " (missing)"
    if block == "H":
        if feature not in H_LABELS:
            raise ValueError(f"Missing presentation label for handcrafted feature: {feature}")
        return H_LABELS[feature]
    if feature.startswith("share_"):
        return f"{feature.removeprefix('share_').title()} coat-pixel share"
    return {"tabby_score": "Tabby-pattern score", "pointed_score": "Pointed-pattern score",
            "bbox_area_frac": "Bounding-box area fraction"}.get(feature, feature.replace("_", " "))


def _coefficient_chart(path: Path, block: str, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.subplots_adjust(left=.34, right=.93, top=.80, bottom=.22)
    title = "Handcrafted image features (M3-H)" if block == "H" else "Automated image attributes (M3-Y)"
    fig.suptitle(title, x=.05, y=.96, ha="left", fontsize=21, fontweight="bold")
    fig.text(.05, .90, DISCLAIMER, ha="left", fontsize=14, color="#70421b", fontweight="bold")
    fig.text(.05, .85, "Five largest positive and five largest negative fitted feature weights",
             ha="left", fontsize=12, color="#444444")
    positions = np.arange(len(rows))
    colors = ["#176b91" if row["coefficient"] > 0 else "#bd5229" for row in rows]
    values = [row["coefficient"] for row in rows]
    bars = ax.barh(positions, values, height=.67, color=colors)
    ax.set_yticks(positions, [feature_label(row["feature"], block) for row in rows], fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", linewidth=.9)
    ax.axhline(4.5, color="#bbbbbb", linewidth=.7, linestyle="--")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#dddddd", linewidth=.6)
    limit = max(abs(value) for value in values) * 1.24
    ax.set_xlim(-limit, limit)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.set_xlabel("Signed coefficient on block-normalized feature (log-odds scale)", fontsize=12, labelpad=12)
    ax.tick_params(axis="x", labelsize=10)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:+.3f}", (value, bar.get_y() + bar.get_height()/2),
                    xytext=(5 if value > 0 else -5, 0), textcoords="offset points",
                    va="center", ha="left" if value > 0 else "right", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(.05, .14,
             "Positive / negative denotes the fitted association with image score; the displayed-A intercept is excluded.",
             ha="left", fontsize=10)
    if block == "H":
        notes = [
            "Numeric predictors use fitting-photo standardization and block normalization.",
            "These are conditional model weights, not causal effects, effect-size uncertainty, or standalone feature importance.",
        ]
    else:
        notes = [
            "Category differences must be interpreted within the same attribute; indicator weights depend on model coding.",
            "Numeric and categorical scales differ. 'Other' pools rare levels; CLIP-inferred breed is not true pedigree.",
            "These are conditional model weights, not causal effects or standalone feature importance.",
        ]
    for i, note in enumerate(notes):
        fig.text(.05, .105 - i*.027, note, ha="left", fontsize=9.5, color="#444444")
    fig.savefig(path)
    plt.close(fig)


def _context_chart(path: Path, names: list[str], rows: list[dict[str, Any]], groups: list[str], reference: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(left=.10, right=.96, top=.77, bottom=.23)
    fig.suptitle("Pet-care context: shifts in CNN weights (M4-C)", x=.05, y=.96,
                 ha="left", fontsize=21, fontweight="bold")
    fig.text(.05, .90, DISCLAIMER, ha="left", fontsize=14, color="#70421b", fontweight="bold")
    reference_label = GROUP_LABELS.get(reference, reference)
    fig.text(.05, .85, f"Largest shifts across groups | Reference: {reference_label} | Reference shift = 0",
             ha="left", fontsize=12, color="#444444")
    index = {(row["feature"], row["pet_care"]): row for row in rows}
    x = np.arange(len(names))
    width = .78 / len(groups)
    colors = ["#176b91", "#d27b2c", "#43855d", "#8b5e9f"]
    for i, group in enumerate(groups):
        values = [index[(name, group)]["interaction_shift"] for name in names]
        ax.bar(x + (i-(len(groups)-1)/2)*width, values, width=width*.94,
               label=GROUP_LABELS.get(group, group), color=colors[i % len(colors)])
    ax.axhline(0, color="#444444", linewidth=.9)
    ax.set_xticks(x, [name.replace("cnn_", "CNN ") for name in names], fontsize=11)
    ax.set_ylabel("Weight shift from reference (group weight minus reference weight)", fontsize=11, labelpad=10)
    ax.set_xlabel("Unlabeled CNN embedding dimension", fontsize=12, labelpad=12)
    ax.grid(axis="y", color="#dddddd", linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.13), ncol=4, frameon=False, fontsize=11)
    fig.text(.05, .14, "Bars are fitted interaction deviations, not group intercepts or changes in observed preference.",
             ha="left", fontsize=10)
    fig.text(.05, .105,
             f"Shown: {len(names)} dimensions ranked by maximum absolute deviation across the non-reference groups.",
             ha="left", fontsize=10)
    fig.text(.05, .075, "CNN coordinates have no individual semantic labels. Correlated coordinates and shrinkage affect their weights.",
             ha="left", fontsize=10, color="#444444")
    fig.text(.05, .045, "Descriptive model visualization after evaluation; not a new confirmatory test or causal interpretation.",
             ha="left", fontsize=10, color="#444444")
    fig.savefig(path)
    plt.close(fig)


def generate(root: Path) -> dict[str, Any]:
    paths = [root.joinpath(*name.split("/")) for name in OUTPUTS]
    if any(path.exists() for path in paths):
        raise FileExistsError("A requested presentation output already exists; refusing to overwrite.")
    frozen = validate_freeze(root)
    selection = json.loads((root / "artifacts/model_selection.json").read_text(encoding="utf-8"))
    inputs = ["artifacts/model_freeze.json", "artifacts/model_freeze.sha256",
              "artifacts/model_selection.json", "artifacts/preprocessing/H.json",
              "artifacts/preprocessing/Y.json", "artifacts/preprocessing/C.json",
              "models/M3-H.json", "models/M3-Y.json", "models/M4-C.json", "src/presentation_features.py"]
    before = {name: _hash(root.joinpath(*name.split("/"))) for name in inputs}
    coefficient_rows = []
    for block in ("H", "Y"):
        model = json.loads((root / "models" / f"M3-{block}.json").read_text(encoding="utf-8"))
        spec = json.loads((root / "artifacts/preprocessing" / f"{block}.json").read_text(encoding="utf-8"))
        if model != selection["selected"][f"M3-{block}"] or not model["accepted"] or model["pooled"]:
            raise ValueError(f"M3-{block} model differs from the accepted frozen selection.")
        if len(model["coefficients"]) != len(spec["final_names"]) + 1:
            raise ValueError(f"M3-{block} coefficient and preprocessing dimensions differ.")
        rows = top_signed(spec["final_names"], model["coefficients"][1:])
        for row in rows:
            coefficient_rows.append({
                "model": f"M3-{block}", **row, "display_label": feature_label(row["feature"], block),
                "source_model": f"models/M3-{block}.json",
                "source_preprocessing": f"artifacts/preprocessing/{block}.json",
                "interpretation": DISCLAIMER,
            })
    model = json.loads((root / "models/M4-C.json").read_text(encoding="utf-8"))
    spec = json.loads((root / "artifacts/preprocessing/C.json").read_text(encoding="utf-8"))
    if model != selection["selected"]["M4-C"] or not model["accepted"] or model["pooled"]:
        raise ValueError("M4-C is not the accepted fitted interaction model.")
    groups, reference = selection["interaction_contexts"], selection["reference_context"]
    if reference != frozen["context_reference"] or groups != frozen["context_interactions"]:
        raise ValueError("Context mapping differs from the frozen registry.")
    if spec["final_names"] != [f"cnn_{index:03d}" for index in range(576)]:
        raise ValueError("CNN feature names do not match the saved embedding coordinates.")
    chosen, shifts = context_shifts(spec["final_names"], model["coefficients"], groups, reference, top=8)
    for row in shifts:
        row["interpretation"] = DISCLAIMER
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        for path, block in zip(paths[:2], ("H", "Y")):
            _coefficient_chart(path, block, [row for row in coefficient_rows if row["model"] == f"M3-{block}"])
        _context_chart(paths[2], chosen, shifts, groups, reference)
    for path, rows in zip(paths[3:5], (coefficient_rows, shifts)):
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    for name, digest in before.items():
        if _hash(root.joinpath(*name.split("/"))) != digest:
            raise RuntimeError(f"Frozen chart input changed during generation: {name}")
    validate_freeze(root)
    report = {
        "status": "generated_from_frozen_coefficients",
        "disclaimer": DISCLAIMER,
        "model_path_resolution": {
            "models/common_M3-H.json": "models/M3-H.json",
            "models/common_M3-Y.json": "models/M3-Y.json",
        },
        "ranking_H_Y": "Five strictly positive and five strictly negative raw fitted coefficients, ranked separately by magnitude; exclude intercept; break ties by feature name.",
        "ranking_C": "Eight CNN dimensions ranked by maximum absolute interaction deviation over all non-reference pet-care groups; break ties by dimension name.",
        "context_reference": reference, "context_groups": groups, "context_dimensions": chosen,
        "new_model_fits": False, "test_outcomes_loaded": False, "test_evaluation_rerun": False,
        "scope": "User-requested descriptive presentation visualization after evaluation; not a newly registered confirmatory analysis.",
        "limitations": [
            "Coefficient magnitude depends on normalization, model coding, shrinkage, and correlated predictors; it is not standalone feature importance.",
            "Y categorical weights should be interpreted via within-attribute contrasts, not universal standalone effects.",
            "CNN dimensions have no individual semantic interpretation.",
        ],
        "input_sha256": before,
        "output_sha256": {path.relative_to(root).as_posix(): _hash(path) for path in paths[:-1]},
    }
    with paths[-1].open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        report = generate(args.root.resolve())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Presentation feature charts failed: {exc}", file=sys.stderr)
        return 2
    print(f"Created three frozen-coefficient PDFs; context reference: {report['context_reference']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
