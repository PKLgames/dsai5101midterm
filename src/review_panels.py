"""Create outcome-blind visual panels for the assigned feature reviewers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageOps

from .feature_y import decode_mask
from .features import _check_source, _file_hash


def _panel_image(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    image = image.copy()
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def run_review_panels(root: Path) -> int:
    destination = root / "artifacts" / "review_panels"
    if destination.exists():
        raise FileExistsError("Review-panel directory already exists; refusing to overwrite it.")
    with (root / "artifacts" / "feature_review.csv").open(encoding="utf-8", newline="") as stream:
        review_rows = list(csv.DictReader(stream))
    images = sorted({row["image_id"] for row in review_rows})
    if not images:
        raise ValueError("The feature review queue contains no images.")
    y_path = root / "features" / "Y.parquet"
    manifest_path = root / "features" / "manifests" / "Y.json"
    if y_path.exists() != manifest_path.exists():
        raise ValueError("Y table and manifest must either both exist or both be unavailable.")
    y = {row["image_id"]: row for row in pq.read_table(y_path).to_pylist()} if y_path.exists() else {}
    raw = json.loads((root / "artifacts" / "raw_hashes.json").read_text(encoding="utf-8"))
    from legacy.src.model import preprocess

    if y:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _file_hash(y_path) != manifest["feature_file_sha256"]:
            raise ValueError("Y feature table does not match its provenance manifest.")
    transform = preprocess.build_transform()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="review-panels-", dir=destination.parent) as temp:
        target = Path(temp) / "panels"
        target.mkdir()
        for image_id in images:
            relative = "dataset/images/" + image_id
            if relative not in raw["files"]:
                raise ValueError(f"Review image is absent from the audited image inventory: {image_id}")
            path = root.joinpath(*relative.split("/"))
            _check_source(path, raw["files"][relative]["sha256"])
            with Image.open(path) as source:
                original = ImageOps.exif_transpose(source).convert("RGB")
            crop = transform.transforms[1](transform.transforms[0](original))
            entry = y.get(image_id)
            if y and entry is None:
                raise ValueError(f"Y table does not contain review image {image_id}.")
            detector_panel = original.copy()
            label = "Detector unavailable"
            if entry is not None and entry["extraction_status"] == "ok":
                status = entry["detection_status"]
                rgb = np.asarray(original).copy()
                if status != "no_detection":
                    if (entry["mask_width"], entry["mask_height"]) != original.size:
                        raise ValueError(f"Detector and EXIF-corrected image orientations differ for {image_id}")
                    if entry["mask_bitmap_zlib"] is None:
                        raise ValueError(f"Cached detector mask is missing for {image_id}")
                    mask = decode_mask(entry["mask_bitmap_zlib"], entry["mask_height"], entry["mask_width"])
                    if entry["mask_fraction"] is None or abs(float(mask.mean()) - entry["mask_fraction"]) > 1e-5:
                        raise ValueError(f"Cached mask size is inconsistent for {image_id}")
                    rgb[mask] = (rgb[mask].astype(np.float32) * 0.6
                                 + np.array([255, 40, 40]) * 0.4).astype(np.uint8)
                detector_panel = Image.fromarray(rgb)
                label = f"{status}; mask: {entry['mask_source']}"
            canvas = Image.new("RGB", (1200, 440), "white")
            for index, image in enumerate((original, crop, detector_panel)):
                canvas.paste(_panel_image(image, 400, 400), (400 * index, 40))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), "EXIF-corrected original", fill="black")
            draw.text((408, 8), "CNN center crop", fill="black")
            draw.text((808, 8), label, fill="black")
            name = hashlib.sha256(image_id.encode("utf-8")).hexdigest() + ".jpg"
            canvas.save(target / name, quality=90)
        os.rename(target, destination)
    return len(images)
