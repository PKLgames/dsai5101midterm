"""Fit-only photo-level feature transformations for registered Stage 7 blocks."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .feature_y import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .features import H_FEATURES

UNKNOWN = "<unknown>"
OTHER = "<other>"


def _numeric(values: list[Any]) -> np.ndarray:
    return np.asarray([float(value) if value is not None and value != "" else np.nan
                       for value in values], dtype=np.float64)


def _category(value: Any) -> str:
    if value is None or value == "":
        return UNKNOWN
    result = str(value)
    if result in {UNKNOWN, OTHER}:
        raise ValueError(f"Reserved category found in raw feature values: {result}")
    return result


def fit_preprocessing(
    block: str, feature_rows: list[dict[str, Any]], fitting_images: set[str]
) -> dict[str, Any]:
    records = {row["image_id"]: row for row in feature_rows}
    if len(records) != len(feature_rows):
        raise ValueError(f"Duplicate feature IDs in {block}.")
    if not fitting_images or not fitting_images <= set(records):
        raise ValueError(f"Missing fitting photo features for {block}.")
    ids = sorted(fitting_images)
    if any(records[image]["extraction_status"] != "ok" for image in ids):
        raise ValueError(f"Unavailable fitting feature record in {block}.")
    if block not in {"H", "C", "Y"}:
        raise ValueError(f"Unknown feature block: {block}")
    spec: dict[str, Any] = {
        "block": block, "fitting_photo_count": len(ids), "numeric": [],
        "categorical": [], "dropped_all_missing": [], "dropped_constant": [],
        "final_names": [], "center": [], "rms_norm": None,
    }
    columns = list(H_FEATURES) if block == "H" else NUMERIC_FEATURES if block == "Y" else []
    for name in columns:
        raw = _numeric([records[image][name] for image in ids])
        if np.isinf(raw).any():
            raise ValueError(f"Nonfinite fitting values in {block}/{name}.")
        observed = raw[~np.isnan(raw)]
        if not len(observed):
            spec["dropped_all_missing"].append(name)
            continue
        median = float(np.median(observed))
        missing = np.isnan(raw)
        filled = np.where(missing, median, raw)
        mean, sd = float(filled.mean()), float(filled.std())
        keep = sd >= 1e-12
        indicator_mean, indicator_sd = float(missing.mean()), float(missing.std())
        indicator_keep = indicator_sd >= 1e-12
        spec["numeric"].append({
            "name": name, "median": median, "mean": mean, "sd": sd,
            "keep": keep, "indicator_mean": indicator_mean, "indicator_sd": indicator_sd,
            "indicator_keep": indicator_keep,
        })
        if not keep:
            spec["dropped_constant"].append(name)
        if not indicator_keep:
            spec["dropped_constant"].append(name + "__missing")
    if block == "Y":
        for name in CATEGORICAL_FEATURES:
            values = [_category(records[image][name]) for image in ids]
            counts = Counter(values)
            special = {UNKNOWN, "unknown", "no_detection"}
            pooled = sorted(value for value, count in counts.items() if count < 10 and value not in special)
            mapped = [OTHER if value in pooled else value for value in values]
            levels = sorted(set(mapped) | {UNKNOWN, OTHER, "unknown", "no_detection"})
            spec["categorical"].append({
                "name": name, "observed_counts": dict(sorted(counts.items())),
                "pooled_categories": pooled, "levels": levels,
            })
    matrix, names = _apply_unscaled(spec, [records[image] for image in ids])
    if matrix.shape[1] == 0:
        raise ValueError(f"{block} is degenerate: no fitting-varying dimensions.")
    center = matrix.mean(axis=0)
    centered = matrix - center
    rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not np.isfinite(rms) or rms < 1e-12:
        raise ValueError(f"{block} is degenerate: fitting RMS vector norm below threshold.")
    spec.update({"final_names": names, "center": center.tolist(), "rms_norm": rms})
    return spec


def _apply_unscaled(spec: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    n = len(rows)
    if spec["block"] == "C":
        names = [f"cnn_{index:03d}" for index in range(576)]
        matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64)
        if matrix.shape != (n, 576) or not np.isfinite(matrix).all():
            raise ValueError("Invalid CNN feature values.")
        return matrix, names
    columns, names = [], []
    for item in spec["numeric"]:
        values = _numeric([row[item["name"]] for row in rows])
        if np.isinf(values).any():
            raise ValueError(f"Nonfinite {item['name']} values.")
        missing = np.isnan(values)
        if item["keep"]:
            columns.append((np.where(missing, item["median"], values) - item["mean"]) / item["sd"])
            names.append(item["name"])
        if item["indicator_keep"]:
            columns.append((missing.astype(float) - item["indicator_mean"]) / item["indicator_sd"])
            names.append(item["name"] + "__missing")
    for item in spec["categorical"]:
        observed = set(item["observed_counts"])
        pooled = set(item["pooled_categories"])
        mapped = []
        for row in rows:
            value = _category(row[item["name"]])
            mapped.append(OTHER if value in pooled else value if value in observed or value in {"unknown", "no_detection"} else UNKNOWN)
        for level in item["levels"]:
            columns.append(np.asarray([float(value == level) for value in mapped]))
            names.append(f"{item['name']}={level}")
    return np.column_stack(columns) if columns else np.empty((n, 0)), names


def transform_photos(spec: dict[str, Any], feature_rows: list[dict[str, Any]]) -> np.ndarray:
    matrix, names = _apply_unscaled(spec, feature_rows)
    if names != spec["final_names"] or matrix.shape[1] != len(spec["center"]):
        raise ValueError("Preprocessing feature order does not match the fitted specification.")
    result = (matrix - np.asarray(spec["center"], dtype=np.float64)) / spec["rms_norm"]
    if not np.isfinite(result).all():
        raise ValueError("Preprocessing produced nonfinite features.")
    return result
