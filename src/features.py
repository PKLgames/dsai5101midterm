"""Outcome-blind extraction of the registered handcrafted and CNN features."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .audit import _digest
from .split import _verify_stage1, validate_manifest, collect_pairs


H_FEATURES = {
    "brightness": "mean luminance 0.299 R + 0.587 G + 0.114 B; RGB in [0,1]",
    "contrast": "standard deviation of the same luminance",
    "saturation_mean": "mean Pillow HSV saturation divided by 255",
    "saturation_std": "standard deviation of Pillow HSV saturation divided by 255",
    "colorfulness": "Hasler-Susstrunk rg/yb colorfulness on RGB in [0,1]",
    "sharpness_laplacian_var": "variance of four-neighbor luminance Laplacian on interior pixels",
    "entropy_bits": "base-2 entropy of 32 luminance bins over [0,1]",
    "edge_density": "fraction of interior central-difference luminance magnitudes above 0.12",
    "warm_pixel_ratio": "fraction of pixels with normalized R minus B above 0.04",
    "hue_mean": "saturation-weighted circular Pillow HSV hue mean divided by 2 pi",
    "hue_concentration": "saturation-weighted circular hue resultant length",
    "mean_r": "mean normalized red channel",
    "mean_g": "mean normalized green channel",
    "mean_b": "mean normalized blue channel",
    "aspect_ratio": "original oriented image width divided by height",
    "log_pixels": "natural logarithm of oriented image width times height",
    "overexposed_frac": "fraction with Pillow HSV value divided by 255 above 0.98",
    "underexposed_frac": "fraction with Pillow HSV value divided by 255 below 0.02",
    "is_effectively_gray": "1 if mean absolute R-G plus G-B is below 0.01; otherwise 0",
    "log_file_kb": "natural log of 1 plus max(file bytes,1) divided by 1024",
}


def _file_hash(path: Path) -> str:
    return _digest(path)[0]


def _sources(root: Path, names: list[str]) -> dict[str, str]:
    return {name: _file_hash(root.joinpath(*name.split("/"))) for name in names}


def _input_inventory(root: Path) -> list[tuple[str, Path, str]]:
    _verify_stage1(root)
    split_path = root / "artifacts" / "split_manifest.parquet"
    rows_path = root / "artifacts" / "rows_canonical.parquet"
    manifest = pq.read_table(split_path).to_pylist()
    structural = pq.read_table(
        rows_path,
        columns=["train_row_id", "a_image", "b_image", "comparison_type", "partition"],
    ).to_pylist()
    pairs = collect_pairs(structural)
    assignments = validate_manifest(pairs, manifest)
    if any(row["partition"] != assignments[
        json.dumps(sorted((row["a_image"], row["b_image"])), ensure_ascii=False, separators=(",", ":"))
    ] for row in structural):
        raise ValueError("Canonical rows do not agree with the split manifest.")
    hashes = json.loads((root / "artifacts" / "raw_hashes.json").read_text(encoding="utf-8"))
    prefix = "dataset/images/"
    images = [
        (name[len(prefix):], root.joinpath(*name.split("/")), entry["sha256"])
        for name, entry in hashes["files"].items() if name.startswith(prefix)
    ]
    if hashes["unreadable_images"]:
        for name in hashes["unreadable_images"]:
            images.append((name, root.joinpath("dataset", "images", *name.split("/")), ""))
    if not images or len({image_id for image_id, _, _ in images}) != len(images):
        raise ValueError("Image inventory is empty or contains duplicate image IDs.")
    return sorted(images)


def _base_manifest(root: Path, block: str, names: list[str], sources: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "block": block,
        "pipeline_version": "stage3-legacy-adapter-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": _sources(root, ["src/features.py", *sources]),
        "source_image_hashes": "sha256 per image in feature table; Stage 1 artifacts/raw_hashes.json",
        "feature_names": names,
        "feature_dimension": len(names),
        "feature_dtype": "float32",
        "image_id": "path relative to dataset/images",
        "extraction_status": "one row per image; status and failure_reason are explicit",
        "preference_outcomes_used": False,
        "device": "cpu",
        "precision": "float32",
        "library_versions": {},
        "admission": "pending_quality_review",
        "missing_source_policy": "failed extraction; feature values null",
    }


def _write_block(root: Path, block: str, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    feature_path = root / "features" / f"{block}.parquet"
    manifest_path = root / "features" / "manifests" / f"{block}.json"
    if feature_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing {block} feature artifacts.")
    names = manifest["feature_names"]
    schema = pa.schema([
        ("image_id", pa.string()), ("source_sha256", pa.string()),
        ("pipeline_version", pa.string()), ("extraction_status", pa.string()),
        ("failure_reason", pa.string()),
        *[(name, pa.float32()) for name in names],
    ])
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError(f"{block} output has duplicate image IDs.")
    manifest["image_count"] = len(rows)
    manifest["successful_images"] = sum(row["extraction_status"] == "ok" for row in rows)
    manifest["failed_images"] = len(rows) - manifest["successful_images"]
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), feature_path)
    manifest["feature_file_sha256"] = _file_hash(feature_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _result(image_id: str, source_hash: str, names: list[str], values: Any, reason: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image_id": image_id, "source_sha256": source_hash,
        "pipeline_version": "stage3-legacy-adapter-1",
        "extraction_status": "failed" if reason else "ok", "failure_reason": reason,
    }
    if reason:
        row.update({name: None for name in names})
    else:
        if len(values) != len(names):
            raise ValueError(f"Successful extraction has the wrong feature count for {image_id}.")
        row.update({name: float(value) for name, value in zip(names, values)})
    return row


def _check_source(path: Path, expected_hash: str) -> None:
    if _file_hash(path) != expected_hash:
        raise RuntimeError(f"Image changed since the Stage 1 audit: {path}")


def extract_handcrafted(root: Path, inventory: list[tuple[str, Path, str]]) -> None:
    import numpy as np
    from PIL import Image, ImageOps, UnidentifiedImageError
    from legacy.src.model import handcrafted

    names = list(handcrafted.HAND_NAMES)
    if names != list(H_FEATURES):
        raise ValueError("Legacy handcrafted feature order differs from the registered schema.")
    manifest = _base_manifest(root, "H", names, ["legacy/src/model/handcrafted.py"])
    manifest.update({
        "feature_definitions_and_units": H_FEATURES,
        "architecture": None, "checkpoint": None, "weight_sha256": None,
        "exif_orientation": "Pillow ImageOps.exif_transpose",
        "color_conversion": "Pillow RGB; HSV via Pillow",
        "resize_crop_interpolation": "256x256 maximum thumbnail; bicubic; aspect preserved; no crop",
        "size_features": "original EXIF-oriented dimensions; bytes from source file",
        "randomness": "no random operations",
        "library_versions": {name: metadata.version(name) for name in ("numpy", "pillow", "pyarrow")},
    })
    rows = []
    for image_id, path, source_hash in inventory:
        if not source_hash:
            rows.append(_result(image_id, source_hash, names, (), "Missing source SHA-256 from audit"))
            continue
        try:
            _check_source(path, source_hash)
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
            values = handcrafted.compute(rgb, path.stat().st_size)
            if values.shape != (len(names),) or not np.isfinite(values).all():
                raise ValueError("Invalid handcrafted feature shape or non-finite value.")
            rows.append(_result(image_id, source_hash, names, values, ""))
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            rows.append(_result(image_id, source_hash, names, (), f"{type(exc).__name__}: {exc}"))
    _write_block(root, "H", rows, manifest)


def extract_cnn(root: Path, inventory: list[tuple[str, Path, str]], batch_size: int = 16) -> None:
    import numpy as np
    from PIL import Image, ImageOps, UnidentifiedImageError
    import torch
    from torchvision import models
    from legacy.src.model import backbones, preprocess

    if batch_size < 1:
        raise ValueError("CNN batch size must be positive.")
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    model, dimension = backbones.build_backbone("mobilenet_v3_small", pretrained=True)
    if dimension != 576 or model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("CNN backbone does not meet the frozen 576-dimensional contract.")
    model.to("cpu")
    transform = preprocess.build_transform()
    names = [f"cnn_{index:03d}" for index in range(dimension)]
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Cannot locate CNN checkpoint for provenance: {checkpoint}")
    manifest = _base_manifest(root, "C", names, [
        "legacy/src/model/backbones.py", "legacy/src/model/preprocess.py",
    ])
    manifest.update({
        "architecture": "torchvision MobileNetV3-Small features plus adaptive average pool; classifier excluded",
        "checkpoint": str(checkpoint),
        "checkpoint_source": weights.url,
        "weight_sha256": _file_hash(checkpoint),
        "exif_orientation": "Pillow ImageOps.exif_transpose",
        "color_conversion": "Pillow RGB",
        "resize_crop_interpolation": "short side 232 bicubic; center crop 224; ImageNet channel normalization",
        "normalization": "L2 per pooled 576-dimensional embedding",
        "randomness": "no augmentation; torch deterministic algorithms enabled",
        "inference_mode": True, "evaluation_mode": True, "fine_tuning": False,
        "batch_size": batch_size, "threads": 8,
        "library_versions": {name: metadata.version(name) for name in ("numpy", "pillow", "torch", "torchvision", "pyarrow")},
    })
    rows = []
    for start in range(0, len(inventory), batch_size):
        chunk = inventory[start:start + batch_size]
        tensors, usable, failed = [], [], {}
        for image_id, path, source_hash in chunk:
            if not source_hash:
                failed[image_id] = "Missing source SHA-256 from audit"
                continue
            try:
                _check_source(path, source_hash)
                with Image.open(path) as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                tensors.append(transform(rgb))
                usable.append((image_id, source_hash))
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                failed[image_id] = f"{type(exc).__name__}: {exc}"
        computed: dict[str, Any] = {}
        if tensors:
            with torch.inference_mode():
                embeddings = model(torch.stack(tensors))
                if tuple(embeddings.shape) != (len(usable), dimension):
                    raise ValueError(f"Unexpected CNN output shape: {tuple(embeddings.shape)}")
                lengths = torch.linalg.vector_norm(embeddings, dim=1)
                safe_lengths = torch.where(lengths > 0, lengths, torch.ones_like(lengths))
                normalized = embeddings / safe_lengths[:, None]
            for index, (image_id, _) in enumerate(usable):
                vector = normalized[index].cpu().numpy()
                if not np.isfinite(vector).all() or not np.isfinite(float(lengths[index])) or float(lengths[index]) <= 0:
                    failed[image_id] = "Non-finite or zero-norm CNN embedding"
                else:
                    computed[image_id] = vector
        for image_id, _, source_hash in chunk:
            reason = failed.get(image_id, "")
            rows.append(_result(image_id, source_hash, names, computed.get(image_id, ()), reason))
    _write_block(root, "C", rows, manifest)


def run_features(root: Path, block: str) -> None:
    if block not in {"H", "C"}:
        raise ValueError(f"Feature block not implemented: {block}")
    if (root / "features" / f"{block}.parquet").exists() or (
        root / "features" / "manifests" / f"{block}.json"
    ).exists():
        raise FileExistsError(f"Refusing to overwrite existing {block} feature artifacts.")
    os.environ.setdefault("TORCH_HOME", str(root / ".cache" / "torch"))
    inventory = _input_inventory(root)
    if block == "H":
        extract_handcrafted(root, inventory)
    else:
        extract_cnn(root, inventory)
