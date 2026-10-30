import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from src.feature_y import encode_mask
from src.review_panels import run_review_panels


class ReviewPanelTests(unittest.TestCase):
    def test_panels_use_cached_mask_without_detector_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts").mkdir()
            (root / "features" / "manifests").mkdir(parents=True)
            image = root / "dataset" / "images" / "cat" / "a.png"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (8, 6), (50, 100, 150)).save(image)
            image_id = "cat/a.png"
            with (root / "artifacts" / "feature_review.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image_id"])
                writer.writeheader()
                writer.writerow({"image_id": image_id})
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            (root / "artifacts" / "raw_hashes.json").write_text(json.dumps({
                "files": {"dataset/images/cat/a.png": {"sha256": digest(image)}},
            }), encoding="utf-8")
            mask = np.zeros((6, 8), dtype=bool)
            mask[1:4, 2:6] = True
            row = {
                "image_id": image_id, "extraction_status": "ok",
                "detection_status": "cat_detected", "mask_source": "mask",
                "mask_width": 8, "mask_height": 6, "mask_fraction": float(mask.mean()),
                "mask_bitmap_zlib": encode_mask(mask),
            }
            path = root / "features" / "Y.parquet"
            pq.write_table(pa.Table.from_pylist([row]), path)
            (root / "features" / "manifests" / "Y.json").write_text(json.dumps({
                "feature_file_sha256": digest(path),
            }), encoding="utf-8")
            self.assertEqual(run_review_panels(root), 1)
            output = root / "artifacts" / "review_panels" / (hashlib.sha256(image_id.encode()).hexdigest() + ".jpg")
            self.assertTrue(output.is_file())
            with Image.open(output) as panel:
                self.assertEqual(panel.size, (1200, 440))


if __name__ == "__main__":
    unittest.main()
