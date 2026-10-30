"""Outcome-blind adapter for the existing YOLO-CLIP photo-label pipeline."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
import zlib
from importlib import metadata
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .features import _base_manifest, _check_source, _file_hash, _input_inventory

NUMERIC_FEATURES = [
    "share_black", "share_white", "share_gray", "share_orange", "share_cream",
    "share_brown", "tabby_score", "pointed_score", "bbox_area_frac",
]
CATEGORICAL_FEATURES = [
    "colour_primary", "colour_secondary", "colour_pattern", "breed", "framing",
    "view", "eyes", "eyes_colour", "age", "posture",
]
DIAGNOSTIC_FLOATS = [
    "det_conf", "mask_fraction", "breed_score", "breed_margin", "framing_score",
    "view_score", "eyes_score", "eyes_colour_score", "age_score", "posture_score",
]
DIAGNOSTIC_INTS = ["candidate_cat_count", "candidate_dog_count", "edges_touched"]
DETECTOR_SOURCE = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-seg.pt"
DETECTOR_SHA256 = "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152"
Y_PIPELINE_VERSION = "stage3-legacy-adapter-2"


def encode_mask(mask: Any) -> bytes:
    import numpy as np

    return zlib.compress(np.packbits(mask.reshape(-1), bitorder="big").tobytes())


def decode_mask(data: bytes, height: int, width: int) -> Any:
    import numpy as np

    if height <= 0 or width <= 0:
        raise ValueError("Mask dimensions must be positive.")
    packed = zlib.decompress(data)
    if len(packed) != (height * width + 7) // 8:
        raise ValueError("Stored mask length does not match its dimensions.")
    return np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big",
                         count=height * width).reshape(height, width).astype(bool)


def resolve_device(torch: Any, requested: str) -> str:
    if requested not in {"cpu", "cuda"}:
        raise ValueError("Y device must be cpu or cuda.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch environment.")
    return "cuda:0" if requested == "cuda" else "cpu"


def coat_for_detection(bgr: Any, instance: Any, legacy: Any) -> dict[str, Any]:
    if instance is None:
        return {
            **{name: None for name in NUMERIC_FEATURES if name != "bbox_area_frac"},
            "colour_primary": "no_detection", "colour_secondary": "no_detection",
            "colour_pattern": "no_detection",
        }
    return legacy.coat_features(bgr, instance["mask"])


def _apply_legacy_postprocessing(record: dict[str, Any], bgr: Any, mask: Any, cfg: Any, legacy: Any) -> None:
    if record["cat_detected"]:
        record["breed"] = legacy.reconcile_breed(record)
    elif record["breed"] not in {"domestic_shorthair", "domestic_longhair"}:
        if record["breed_margin"] < cfg.BREED_MARGIN:
            record["breed"] = (
                record["breed_second"]
                if record["breed_second"] in {"domestic_shorthair", "domestic_longhair"}
                else "domestic_shorthair"
            )
    legacy.apply_framing(record)
    single, pair = legacy.open_eye_score(legacy.crop_box(bgr, record))
    face = record["view"] in cfg.FACE_VIEWS and record["framing"] in cfg.FACE_FRAMING
    if single >= cfg.OPEN_EYE_SINGLE or pair >= cfg.OPEN_EYE_PAIR or (record["eyes"] == "half_closed" and face):
        record["eyes"] = "both_open"
        record["eyes_score"] = max(float(record["eyes_score"]), cfg.OPEN_EYE_SCORE)
        if not record["eyes_colour"] and record["eyes_colour_clip_score"] >= cfg.EYE_COLOUR_MIN:
            record["eyes_colour"] = record["eyes_colour_clip"]
            record["eyes_colour_score"] = record["eyes_colour_clip_score"]
    elif record["eyes"] == "closed" and record["view"] in cfg.AWAY_VIEWS:
        record["eyes"], record["eyes_colour"], record["eyes_colour_score"] = "not_visible", "", 0.0
    elif record["eyes"] == "closed":
        record["eyes"] = "both_open"
        record["eyes_score"] = max(float(record["eyes_score"]), cfg.CLOSED_TO_OPEN_SCORE)
    if record["eyes"] in {"closed", "not_visible"}:
        blob = legacy.eye_colour_from_mask(bgr, mask)
        if blob:
            record["eyes"] = "both_open"
            if record["eyes_colour_clip_score"] >= cfg.EYE_COLOUR_BLOB:
                record["eyes_colour"] = record["eyes_colour_clip"]
                record["eyes_colour_score"] = record["eyes_colour_clip_score"]
            else:
                record["eyes_colour"], record["eyes_colour_score"] = blob, 0.5


def _load_legacy(root: Path) -> tuple[Any, Any]:
    location = str(root / "legacy" / "photo-labelling")
    if location not in sys.path:
        sys.path.insert(0, location)
    cfg = importlib.import_module("label_photos")
    cfg.WEIGHTS = root / ".cache" / "yolo" / "yolo11n-seg.pt"
    legacy = importlib.import_module("yolo_clip_labels")
    return cfg, legacy


def run_y(root: Path, batch_size: int = 8, device: str = "cpu") -> None:
    if batch_size < 1:
        raise ValueError("YOLO-CLIP batch size must be positive.")
    feature_path = root / "features" / "Y.parquet"
    manifest_path = root / "features" / "manifests" / "Y.json"
    if feature_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite existing Y feature artifacts.")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".cache" / "ultralytics"))
    os.environ.setdefault("TORCH_HOME", str(root / ".cache" / "torch"))
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA.")
    inventory = _input_inventory(root)

    import torch
    actual_device = resolve_device(torch, device)

    import numpy as np
    import open_clip
    from PIL import Image, UnidentifiedImageError

    cfg, legacy = _load_legacy(root)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    detector, _ = legacy.load_yolo()
    if _file_hash(cfg.WEIGHTS) != DETECTOR_SHA256:
        raise ValueError("Detector checkpoint does not match the recorded Ultralytics release hash.")
    detector.to(actual_device)
    clip_cache = root / ".cache" / "open_clip"
    clip_cfg = open_clip.get_pretrained_cfg(cfg.CLIP_MODEL, cfg.CLIP_PRETRAINED)
    checkpoint = Path(open_clip.download_pretrained(clip_cfg, cache_dir=str(clip_cache)))
    clip, _, preprocess = open_clip.create_model_and_transforms(
        cfg.CLIP_MODEL, pretrained=cfg.CLIP_PRETRAINED,
        force_quick_gelu=cfg.CLIP_QUICK_GELU, cache_dir=str(clip_cache), device=actual_device,
    )
    clip.to(actual_device)
    clip.eval()
    for parameter in clip.parameters():
        parameter.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(cfg.CLIP_MODEL)
    text = legacy.prepare_text(clip, tokenizer)
    if clip.training or any(parameter.requires_grad for parameter in clip.parameters()):
        raise ValueError("CLIP must be frozen and in evaluation mode.")
    if next(clip.parameters()).device.type != device or next(detector.model.parameters()).device.type != device:
        raise RuntimeError(f"YOLO or CLIP failed to move to requested device: {actual_device}")
    if next(clip.parameters()).dtype != torch.float32:
        raise RuntimeError("CLIP must use float32 inference.")
    if not cfg.WEIGHTS.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("YOLO or CLIP checkpoint is unavailable for provenance.")

    names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    manifest = _base_manifest(root, "Y", names, [
        "src/feature_y.py", "legacy/photo-labelling/label_photos.py",
        "legacy/photo-labelling/yolo_clip_labels.py", "legacy/photo-labelling/save_photo_labels.py",
    ])
    manifest.update({
        "pipeline_version": Y_PIPELINE_VERSION,
        "feature_dtype": {"numeric": "float32", "categorical": "string"},
        "device": actual_device,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)) if device == "cuda" else None,
        "gpu_driver": (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True, capture_output=True, text=True,
            ).stdout.strip().splitlines()[0] if device == "cuda" else None
        ),
        "feature_definitions": {
            "numeric": "Legacy OpenCV HSV mask measurements; bbox area is detector box area / original image area.",
            "categorical": "Legacy HSV hard labels and CLIP hard labels after recorded postprocessing.",
        },
        "architecture": {"detector": "YOLO11n segmentation", "clip": cfg.CLIP_MODEL},
        "checkpoint": {"detector": str(cfg.WEIGHTS), "clip": str(checkpoint)},
        "checkpoint_source": {
            "detector": DETECTOR_SOURCE,
            "clip_pretrained_config": clip_cfg,
        },
        "weight_sha256": {"detector": _file_hash(cfg.WEIGHTS), "clip": _file_hash(checkpoint)},
        "exif_orientation": "Ultralytics file loader / OpenCV behavior; manual orientation audit required",
        "color_conversion": "OpenCV BGR to HSV for coat; BGR to RGB for CLIP",
        "resize_crop_interpolation": {
            "detector": "Ultralytics imgsz=512 with library defaults",
            "clip": repr(preprocess), "crop": "selected box padded 6%; square pad 127; full frame if no detection",
        },
        "detector": {
            "inference_size": 512, "confidence_threshold": cfg.YOLO_CONF, "nms_iou": 0.7,
            "max_detections": 300, "agnostic_nms": False,
            "class_mapping": detector.names, "cat_threshold": cfg.CAT_CONF, "dog_threshold": cfg.DOG_CONF,
            "selection": "Prefer cat candidates; otherwise dog fallback; maximize confidence times square root of box area.",
            "mask_coordinates": "Ultralytics masks.xy polygons in original-image coordinates",
            "minimum_mask_pixels": 50,
            "small_mask_fallback": "bounding box mask; quality flag retained",
            "no_detection": "full-frame CLIP; coat measurements null and coat labels no_detection",
        },
        "clip": {
            "model": cfg.CLIP_MODEL, "pretrained": cfg.CLIP_PRETRAINED,
            "quick_gelu": cfg.CLIP_QUICK_GELU, "evaluation_mode": True, "inference_mode": True,
            "templates": list(cfg.TEMPLATES), "taxonomy_and_prompts": cfg.PROMPT_GROUPS,
            "prompt_aggregation": "L2-normalize text embeddings; average template embeddings; L2-normalize",
            "score_normalization": "softmax over each fixed prompt family; not calibrated probabilities",
            "hard_label_selection": "argmax then legacy breed/framing/eye postprocessing",
            "breed_margin_threshold": cfg.BREED_MARGIN,
            "no_detection_breed_rule": "Apply confidence margin only; skip coat-dependent Siamese override.",
        },
        "hsv_and_pattern_rules": "Exact thresholds and pattern rules are in hashed legacy/photo-labelling/yolo_clip_labels.py.",
        "diagnostic_fields": DIAGNOSTIC_FLOATS + DIAGNOSTIC_INTS + [
            "detection_status", "mask_source", "mask_height", "mask_width",
            "mask_bitmap_zlib", *cfg.BOX_FIELDS,
        ],
        "mask_storage": "zlib-compressed big-endian np.packbits of selected original-image mask, row-major",
        "randomness": "no augmentation; torch deterministic algorithms enabled; CUDA cuDNN benchmark and TF32 disabled",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device == "cuda" else None,
        "batch_size": batch_size, "threads": 8,
        "library_versions": {name: metadata.version(name) for name in (
            "numpy", "pillow", "torch", "torchvision", "ultralytics",
            "open-clip-torch", "opencv-python", "pandas", "pyarrow",
        )},
    })
    schema = pa.schema([
        ("image_id", pa.string()), ("source_sha256", pa.string()),
        ("pipeline_version", pa.string()), ("extraction_status", pa.string()),
        ("failure_reason", pa.string()), ("detection_status", pa.string()), ("mask_source", pa.string()),
        *[(name, pa.float32()) for name in NUMERIC_FEATURES],
        *[(name, pa.string()) for name in CATEGORICAL_FEATURES],
        *[(name, pa.float32()) for name in DIAGNOSTIC_FLOATS],
        *[(name, pa.int32()) for name in DIAGNOSTIC_INTS],
        *[(name, pa.int32()) for name in cfg.BOX_FIELDS],
        ("mask_height", pa.int32()), ("mask_width", pa.int32()), ("mask_bitmap_zlib", pa.binary()),
    ])
    rows: list[dict[str, Any]] = []
    for start in range(0, len(inventory), batch_size):
        chunk = inventory[start:start + batch_size]
        usable, failed = [], {}
        for image_id, path, source_hash in chunk:
            if not source_hash:
                failed[image_id] = "Missing source SHA-256 from audit"
                continue
            try:
                _check_source(path, source_hash)
                with Image.open(path) as image:
                    image.verify()
                usable.append((image_id, path, source_hash))
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                failed[image_id] = f"{type(exc).__name__}: {exc}"
        computed: dict[str, dict[str, Any]] = {}
        if usable:
            results = detector.predict(
                [str(path) for _, path, _ in usable], imgsz=512, conf=cfg.YOLO_CONF,
                iou=0.7, max_det=300, agnostic_nms=False, augment=False,
                verbose=False, device=actual_device, half=False, stream=False,
            )
            if len(results) != len(usable):
                raise RuntimeError("YOLO returned a different number of results than input images.")
            records, crops, regions = [], [], []
            for (image_id, path, _), result in zip(usable, results):
                if Path(result.path).resolve() != path.resolve():
                    raise RuntimeError(f"YOLO returned a result for the wrong image: {image_id}")
                bgr = result.orig_img
                instance = legacy.animal_instance(result)
                candidate_counts = {"cat": 0, "dog": 0}
                if result.boxes is not None:
                    for class_id, confidence in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                        label = result.names[int(class_id)]
                        if label in candidate_counts and confidence >= (
                            cfg.CAT_CONF if label == "cat" else cfg.DOG_CONF
                        ):
                            candidate_counts[label] += 1
                record = {
                    "image_id": image_id, "cat_detected": instance is not None,
                    "detection_status": "no_detection" if instance is None else (
                        "cat_detected" if instance["name"] == "cat" else "dog_fallback"
                    ),
                    "mask_source": "none" if instance is None else instance["source"],
                    "det_conf": 0.0 if instance is None else float(instance["conf"]),
                    "mask_fraction": None if instance is None else float(instance["mask"].mean()),
                    "bbox_area_frac": None if instance is None else float(instance["area_frac"]),
                    "edges_touched": 0 if instance is None else legacy.edges_touched(instance["mask"]),
                    "candidate_cat_count": candidate_counts["cat"], "candidate_dog_count": candidate_counts["dog"],
                    "mask_height": None if instance is None else int(instance["mask"].shape[0]),
                    "mask_width": None if instance is None else int(instance["mask"].shape[1]),
                    "mask_bitmap_zlib": None if instance is None else encode_mask(instance["mask"]),
                    **{name: None for name in cfg.BOX_FIELDS},
                }
                if instance is not None:
                    record.update(legacy.box_pixels(instance["xyxy"], bgr.shape[0], bgr.shape[1]))
                record.update(coat_for_detection(bgr, instance, legacy))
                records.append(record)
                crops.append(legacy.crop_rgb(bgr, None if instance is None else instance["xyxy"]))
                regions.append((bgr, None if instance is None else instance["mask"]))
            labels = legacy.classify_crops(clip, preprocess, text, crops)
            if len(labels) != len(records):
                raise RuntimeError("CLIP returned a different number of results than input images.")
            for record, label, (bgr, mask) in zip(records, labels, regions):
                record.update(label)
                _apply_legacy_postprocessing(record, bgr, mask, cfg, legacy)
                for name in NUMERIC_FEATURES + DIAGNOSTIC_FLOATS:
                    if record.get(name) is not None and not np.isfinite(record[name]):
                        raise ValueError(f"Non-finite Y value in {record['image_id']}: {name}")
                computed[record["image_id"]] = record
        for image_id, _, source_hash in chunk:
            reason = failed.get(image_id, "")
            record = computed.get(image_id)
            if not reason and record is None:
                raise RuntimeError(f"No Y result was produced for {image_id}")
            output = {
                "image_id": image_id, "source_sha256": source_hash,
                "pipeline_version": Y_PIPELINE_VERSION,
                "extraction_status": "failed" if reason else "ok", "failure_reason": reason,
            }
            if record is not None:
                output.update({field.name: record.get(field.name) for field in schema
                               if field.name not in output})
            rows.append(output)
        print(f"Y extraction: {min(start + batch_size, len(inventory))}/{len(inventory)} images", flush=True)

    if len({row["image_id"] for row in rows}) != len(inventory):
        raise ValueError("Y output does not contain exactly one row per image.")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), feature_path)
    if device == "cuda":
        torch.cuda.synchronize()
    manifest.update({
        "image_count": len(rows),
        "successful_images": sum(row["extraction_status"] == "ok" for row in rows),
        "failed_images": sum(row["extraction_status"] != "ok" for row in rows),
        "no_detection_images": sum(row.get("detection_status") == "no_detection" for row in rows),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "feature_file_sha256": _file_hash(feature_path),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
