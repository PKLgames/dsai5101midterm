import hashlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from legacy.src.model import handcrafted
from src.features import H_FEATURES, _result, extract_handcrafted


ROOT = Path(__file__).resolve().parents[1]


class HandcraftedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("src/features.py", "legacy/src/model/handcrafted.py"):
            target = self.root.joinpath(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT.joinpath(*name.split("/")), target)

    def test_feature_definitions_match_implemented_order(self):
        self.assertEqual(list(H_FEATURES), handcrafted.HAND_NAMES)
        self.assertEqual(len(H_FEATURES), 20)

    def test_red_image_rgb_range_color_and_original_dimensions(self):
        image = Image.new("RGB", (4, 2), (255, 0, 0))
        values = dict(zip(handcrafted.HAND_NAMES, handcrafted.compute(image, 2048)))
        self.assertAlmostEqual(values["brightness"], 0.299, places=5)
        self.assertAlmostEqual(values["mean_r"], 1.0)
        self.assertAlmostEqual(values["mean_g"], 0.0)
        self.assertAlmostEqual(values["mean_b"], 0.0)
        self.assertAlmostEqual(values["saturation_mean"], 1.0)
        self.assertAlmostEqual(values["warm_pixel_ratio"], 1.0)
        self.assertAlmostEqual(values["aspect_ratio"], 2.0)
        self.assertAlmostEqual(values["log_pixels"], math.log(8), places=5)
        self.assertAlmostEqual(values["log_file_kb"], math.log1p(2), places=5)
        self.assertAlmostEqual(values["sharpness_laplacian_var"], 0.0)
        self.assertAlmostEqual(values["edge_density"], 0.0)

    def test_boundary_and_laplacian_behavior(self):
        small = np.ones((2, 2), dtype=np.float32)
        self.assertEqual(handcrafted._edge_density(small), 0.0)
        self.assertEqual(handcrafted._laplacian_var(small), 0.0)
        constant = np.ones((4, 4), dtype=np.float32)
        self.assertEqual(handcrafted._edge_density(constant), 0.0)
        self.assertEqual(handcrafted._laplacian_var(constant), 0.0)
        constant[1, 1] = 0.0
        self.assertGreater(handcrafted._laplacian_var(constant), 0.0)

    def test_extract_records_orientation_and_failure_explicitly(self):
        image_path = self.root / "image.jpg"
        image = Image.new("RGB", (2, 3), (255, 0, 0))
        exif = image.getexif()
        exif[274] = 6
        image.save(image_path, exif=exif)
        bad_path = self.root / "bad.jpg"
        bad_path.write_bytes(b"not an image")
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        inventory = [
            ("cat_a/image.jpg", image_path, digest(image_path)),
            ("cat_b/bad.jpg", bad_path, digest(bad_path)),
        ]
        extract_handcrafted(self.root, inventory)
        rows = pq.read_table(self.root / "features/H.parquet").to_pylist()
        self.assertEqual([row["extraction_status"] for row in rows], ["ok", "failed"])
        self.assertAlmostEqual(rows[0]["aspect_ratio"], 1.5)
        self.assertTrue(rows[1]["failure_reason"])
        self.assertTrue(all(rows[1][name] is None for name in H_FEATURES))
        manifest = json.loads((self.root / "features/manifests/H.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["feature_names"], list(H_FEATURES))
        self.assertEqual((manifest["successful_images"], manifest["failed_images"]), (1, 1))

    def test_success_cannot_have_missing_features(self):
        with self.assertRaises(ValueError):
            _result("cat/a.jpg", "hash", ["a", "b"], [1], "")


if __name__ == "__main__":
    unittest.main()
