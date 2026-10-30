"""Re-extract a fixed, outcome-blind image subset against the frozen caches."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .features import _file_hash as _hash


def _inventory(root: Path, block: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((root / "features/manifests" / f"{block}.json").read_text(encoding="utf-8"))
    path = root / "features" / f"{block}.parquet"
    if _hash(path) != manifest["feature_file_sha256"]:
        raise ValueError(f"{block} cached feature table changed.")
    count = 8 if block == "Y" else 16
    records = sorted(pq.read_table(path).to_pylist(), key=lambda row: row["image_id"])[:count]
    for row in records:
        image = root.joinpath("dataset", "images", *row["image_id"].split("/"))
        if not image.is_file() or _hash(image) != row["source_sha256"]:
            raise ValueError(f"Fixed-subset source image missing or changed: {row['image_id']}")
    return manifest, records


def check_features(root: Path, block: str, tolerance: float = 1e-5) -> dict[str, Any]:
    output = root / "artifacts" / f"feature_regeneration_{block}.json"
    if output.exists():
        raise FileExistsError(f"{block} feature-regeneration log already exists.")
    os.environ.setdefault("TORCH_HOME", str(root / ".cache/torch"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".cache/ultralytics"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    manifest, rows = _inventory(root, block)
    started = time.monotonic()
    observed: list[dict[str, Any]] = []
    if block in {"H", "C"}:
        from PIL import Image, ImageOps
        from legacy.src.model import handcrafted
        images = []
        for row in rows:
            path = root.joinpath("dataset", "images", *row["image_id"].split("/"))
            with Image.open(path) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB"))
        if block == "H":
            for row, image in zip(rows, images):
                path = root.joinpath("dataset", "images", *row["image_id"].split("/"))
                vector = handcrafted.compute(image, path.stat().st_size)
                observed.append(dict(zip(manifest["feature_names"], vector)))
        else:
            import torch
            from legacy.src.model import backbones, preprocess
            torch.set_num_threads(8)
            torch.use_deterministic_algorithms(True)
            model, _ = backbones.build_backbone("mobilenet_v3_small", pretrained=True)
            checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(manifest["checkpoint"]).name
            if _hash(checkpoint) != manifest["weight_sha256"]:
                raise ValueError("CNN checkpoint differs from the frozen source.")
            transform = preprocess.build_transform()
            with torch.inference_mode():
                vectors = model(torch.stack([transform(image) for image in images]))
                vectors = vectors / torch.linalg.vector_norm(vectors, dim=1)[:, None]
            observed = [dict(zip(manifest["feature_names"], vector)) for vector in vectors.cpu().numpy()]
    elif block == "Y":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("The frozen Y subset check requires CUDA; use the GPU environment.")
        import open_clip
        from .feature_y import (
            _load_legacy, _apply_legacy_postprocessing, coat_for_detection, decode_mask,
            NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        )
        torch.set_num_threads(8)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg, legacy = _load_legacy(root)
        detector, _ = legacy.load_yolo()
        detector.to("cuda:0")
        clip_cfg = open_clip.get_pretrained_cfg("ViT-B-32", "openai")
        clip_cache = root / ".cache/open_clip"
        clip_checkpoint = Path(open_clip.download_pretrained(clip_cfg, cache_dir=str(clip_cache)))
        if (_hash(cfg.WEIGHTS) != manifest["weight_sha256"]["detector"] or
            _hash(clip_checkpoint) != manifest["weight_sha256"]["clip"]):
            raise ValueError("Y checkpoint hashes differ from the frozen manifest.")
        clip, _, transform = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", force_quick_gelu=True,
            cache_dir=str(clip_cache), device="cuda:0",
        )
        clip.eval()
        for parameter in clip.parameters():
            parameter.requires_grad_(False)
        text = legacy.prepare_text(clip, open_clip.get_tokenizer("ViT-B-32"))
        paths = [root.joinpath("dataset", "images", *row["image_id"].split("/")) for row in rows]
        results = detector.predict([str(path) for path in paths], imgsz=512, conf=cfg.YOLO_CONF,
                                   iou=.7, max_det=300, agnostic_nms=False, augment=False,
                                   verbose=False, device="cuda:0", half=False, stream=False)
        if len(results) != len(rows):
            raise ValueError("Detector output count differs from the fixed subset.")
        crops, instances = [], []
        for row, path, result in zip(rows, paths, results):
            if Path(result.path).resolve() != path.resolve():
                raise ValueError("Detector output image identity mismatch.")
            instance = legacy.animal_instance(result)
            status = "no_detection" if instance is None else (
                "cat_detected" if instance["name"] == "cat" else "dog_fallback"
            )
            if status != row["detection_status"]:
                raise ValueError("Detector status changed on the fixed subset.")
            if instance is not None:
                original_mask = decode_mask(row["mask_bitmap_zlib"], row["mask_height"], row["mask_width"])
                if not np.array_equal(instance["mask"], original_mask):
                    raise ValueError("Frozen detector mask did not regenerate exactly on the fixed subset.")
            instances.append(instance)
            crops.append(legacy.crop_rgb(result.orig_img, None if instance is None else instance["xyxy"]))
        labels = legacy.classify_crops(clip, transform, text, crops)
        for result, instance, label in zip(results, instances, labels):
            record = {
                "cat_detected": instance is not None,
                "bbox_area_frac": None if instance is None else float(instance["area_frac"]),
                "edges_touched": 0 if instance is None else legacy.edges_touched(instance["mask"]),
                **{name: None for name in cfg.BOX_FIELDS},
            }
            if instance is not None:
                record.update(legacy.box_pixels(instance["xyxy"], result.orig_img.shape[0], result.orig_img.shape[1]))
            record.update(coat_for_detection(result.orig_img, instance, legacy))
            record.update(label)
            _apply_legacy_postprocessing(record, result.orig_img,
                                        None if instance is None else instance["mask"], cfg, legacy)
            observed.append({name: record[name] for name in NUMERIC_FEATURES + CATEGORICAL_FEATURES})
    else:
        raise ValueError(f"Unknown feature block: {block}")
    if len(observed) != len(rows):
        raise ValueError("Feature regeneration did not cover the fixed subset.")
    max_difference = 0.
    numeric_count = categorical_count = 0
    for row, regenerated in zip(rows, observed):
        for name in manifest["feature_names"]:
            original, current = row[name], regenerated[name]
            if isinstance(original, str):
                categorical_count += 1
                if original != current:
                    raise ValueError(f"Categorical Y label differs: {row['image_id']}/{name}")
            elif original is None:
                if current is not None:
                    raise ValueError(f"Missingness differs: {row['image_id']}/{name}")
            else:
                numeric_count += 1
                delta = abs(float(original)-float(current))
                if not np.isfinite(delta) or delta > tolerance:
                    raise ValueError(f"Feature difference exceeds tolerance: {row['image_id']}/{name}")
                max_difference = max(max_difference, delta)
    report = {
        "status": "passed", "block": block, "selection": "first sorted image IDs; same initial extraction batch",
        "image_ids": [row["image_id"] for row in rows], "numeric_values_compared": numeric_count,
        "categorical_values_compared": categorical_count, "tolerance": tolerance,
        "maximum_observed_numeric_difference": max_difference,
        "elapsed_seconds": time.monotonic()-started,
        "device": "cuda:0" if block == "Y" else "cpu",
        "raw_preference_outcomes_read": False, "frozen_feature_tables_modified": False,
        "scope": "Fixed subset only; observed tolerance on this environment, not a cross-device guarantee.",
        "cache_sha256": _hash(root / "features" / f"{block}.parquet"),
        "source_sha256": _hash(root / "src/feature_regeneration.py"),
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--block", required=True, choices=["H", "C", "Y"])
    args = parser.parse_args(argv)
    try:
        report = check_features(args.root.resolve(), args.block)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Feature regeneration failed: {exc}", file=sys.stderr)
        return 2
    print(f"{args.block} fixed-subset feature regeneration passed; maximum numeric difference "
          f"{report['maximum_observed_numeric_difference']:.3g}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
