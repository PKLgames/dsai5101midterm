"""Fit the four fixed model families on an isolated fitting/validation split."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from threadpoolctl import threadpool_limits

from common import ROOT
from experiment import (DEMOGRAPHICS, category, design, fit_encoding, penalties,
                        safe, side_matrix)
from src.development import choose, feature_differences, score
from src.estimator import fit_logistic
from src.preprocessing import fit_preprocessing, transform_photos


def cat_differences(rows: list[dict[str, Any]], keys: list[str], allow_unknown: bool) -> sparse.csr_matrix:
    lookup = {name: i for i, name in enumerate(keys)}
    if len(lookup) != len(keys):
        raise ValueError("Duplicate fitting cat IDs.")
    rr, cc, vv = [], [], []
    for i, row in enumerate(rows):
        unknown = []
        for side, sign in (("a", 1.), ("b", -1.)):
            cat = row[f"{side}_cat"]
            if cat in lookup:
                rr.append(i)
                cc.append(lookup[cat])
                vv.append(sign)
            else:
                unknown.append(cat)
        if unknown and (not allow_unknown or len(unknown) != 1 or unknown[0] != row.get("held_out_cat")):
            raise ValueError("Unseen cat is not the single reserved G4 endpoint.")
        if allow_unknown and (len(unknown) != 1 or unknown[0] != row.get("held_out_cat")):
            raise ValueError("G4 rows must contain exactly one unseen cat.")
    block = sparse.coo_matrix((vv, (rr, cc)), shape=(len(rows), len(keys))).tocsr()
    block.eliminate_zeros()
    return block


def feature_diffs(train: list[dict[str, Any]], val: list[dict[str, Any]]) -> tuple[dict, dict, dict, list[str]]:
    ids = sorted({row[f"{side}_image"] for row in train + val for side in ("a", "b")})
    fit_ids = {row[f"{side}_image"] for row in train for side in ("a", "b")}
    specs, dx, dv = {}, {}, {}
    for block in ("Y", "C"):
        records = pq.read_table(ROOT / "features" / f"{block}.parquet").to_pylist()
        index = {r["image_id"]: r for r in records}
        if len(index) != len(records) or not set(ids) <= set(index):
            raise ValueError(f"{block} photo feature coverage mismatch.")
        if any(index[name]["extraction_status"] != "ok" for name in ids):
            raise ValueError(f"{block} unavailable on required photos.")
        spec = fit_preprocessing(block, records, fit_ids)
        values = transform_photos(spec, [index[image] for image in ids])
        specs[block] = spec
        dx[block] = feature_differences(train, ids, values)
        dv[block] = feature_differences(val, ids, values)
    return specs, dx, dv, sorted(fit_ids)


def train_models(train: list[dict[str, Any]], val: list[dict[str, Any]], output: Path,
                 *, g4: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Candidate log exists: {output}")
    if not train or not val or any(r["y_displayed"] not in (0, 1) for r in train+val):
        raise ValueError("Fitting/validation outcomes are invalid.")
    keys = sorted({r[f"{side}_cat"] for r in train for side in ("a", "b")})
    encoding = fit_encoding(train)
    sx, sv = side_matrix(train, encoding), side_matrix(val, encoding)
    specs, dx, dv, fitting_photos = feature_diffs(train, val)
    dx["Cat"], dv["Cat"] = cat_differences(train, keys, False), cat_differences(val, keys, g4)
    y = np.asarray([r["y_displayed"] for r in train], dtype=float)
    vy = np.asarray([r["y_displayed"] for r in val], dtype=float)
    fields = ["model", "branch", "ridge", "rho", "pooled", "pooled_from", "interacting", "accepted",
              "validation_loss", "training_loss", "coefficient_norm", "objective", "gradient_max",
              "coefficient_count", "iterations", "seconds", "fit_attempts"]
    selected = {}
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        stream.flush()

        def log(candidate: dict[str, Any]) -> None:
            row = {name: safe(candidate.get(name)) for name in fields}
            row["fit_attempts"] = json.dumps(safe(candidate["fit_attempts"]), allow_nan=False)
            writer.writerow(row)
            stream.flush()

        def sweep(name: str, branch: str, interactions: bool, fallback: dict | None = None) -> dict:
            d = 0 if branch == "D" else dx[branch].shape[1]
            x = design(sx, None if branch == "D" else dx[branch], interactions)
            vx = design(sv, None if branch == "D" else dv[branch], interactions)
            candidates = []
            if fallback is not None:
                pooled = {**fallback, "model": name, "pooled": True, "pooled_from": fallback["model"],
                          "rho": 1e99, "interacting": False, "seconds": 0., "fit_attempts": []}
                candidates.append(pooled)
                log(pooled)
            for ridge in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.):
                for rho in ((1, 10, 100) if interactions else (1,)):
                    started = time.monotonic()
                    result = fit_logistic(x, y, penalties(sx.shape[1], d, interactions, ridge, rho))
                    v = np.asarray(vx @ result.coefficients).ravel() if result.accepted else None
                    accepted = bool(result.accepted and v is not None and np.isfinite(v).all())
                    candidate = {
                        "model": name, "branch": branch, "ridge": ridge, "rho": rho, "pooled": False,
                        "pooled_from": "", "interacting": interactions, "accepted": accepted,
                        "validation_loss": score(vy, v)[0] if accepted else None,
                        "training_loss": score(y, np.asarray(x @ result.coefficients).ravel())[0] if accepted else None,
                        "coefficient_norm": float(np.linalg.norm(result.coefficients[1:])) if accepted else None,
                        "objective": safe(result.objective), "gradient_max": safe(result.gradient_max),
                        "coefficient_count": x.shape[1], "iterations": result.iterations,
                        "seconds": time.monotonic()-started, "fit_attempts": list(result.attempts),
                        "coefficients": result.coefficients.tolist(),
                    }
                    candidates.append(candidate)
                    log(candidate)
            winner = choose(candidates, contextual=fallback is not None)
            print(f"{output.name}: {name} val={winner['validation_loss']:.6f} pooled={winner['pooled']}", flush=True)
            return winner

        with threadpool_limits(limits=1):
            selected["D"] = sweep("D", "D", False)
            for branch in ("Cat", "Y", "C"):
                base = branch + "+side"
                selected[base] = sweep(base, branch, False)
                selected[branch + "+D"] = sweep(branch + "+D", branch, True, selected[base])
    return {"selected": selected, "encoding": encoding, "preprocessing": specs, "cat_keys": keys,
            "fitting_photos": fitting_photos, "fitting_responses": len(train), "validation_responses": len(val),
            "demographics": list(DEMOGRAPHICS), "g4_zero_unseen_cat": g4}
