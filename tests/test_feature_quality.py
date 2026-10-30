import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.feature_quality import audit_block, build_components, select_review_images
from src.feature_y import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class FeatureQualityTests(unittest.TestCase):
    def test_strict_graph_uses_fitting_pairs_only(self):
        manifest = [
            {"image_i": "a", "image_j": "b", "partition": "fit"},
            {"image_i": "b", "image_j": "c", "partition": "validation"},
            {"image_i": "c", "image_j": "d", "partition": "fit"},
        ]
        graph = build_components(manifest)
        self.assertTrue(graph.supports("a", "b"))
        self.assertFalse(graph.supports("a", "c"))
        self.assertFalse(graph.supports("a", "missing"))
        restricted = build_components(manifest, {"a", "b", "c"})
        self.assertFalse(restricted.supports("c", "d"))

    def test_review_selection_is_deterministic_unique_and_fitting_only(self):
        images = {f"cat/{index}.jpg" for index in range(100)}
        flags = {image: {"no_detection"} for image in images}
        selected = select_review_images(images, flags, 20261103)
        self.assertEqual(selected, select_review_images(images, flags, 20261103))
        self.assertEqual(len(selected), 64)
        self.assertEqual(len({record[0] for record in selected}), 64)
        self.assertTrue({record[0] for record in selected} <= images)
        self.assertEqual(sum(record[1] == "all_fitting" for record in selected), 48)
        self.assertEqual(sum(record[1] == "flagged" for record in selected), 16)

    def test_flagged_selection_covers_rare_no_detection_when_available(self):
        images = {f"cat/{index}.jpg" for index in range(100)}
        initial = select_review_images(images, {}, 20261103)
        base = {record[0] for record in initial}
        rare = next(iter(images - base))
        flags = {image: {"low_clip_breed_margin"} for image in images}
        flags[rare].add("no_detection")
        selected = select_review_images(images, flags, 20261103)
        self.assertIn(rare, {record[0] for record in selected})
        self.assertEqual(len(selected), 64)

    def test_y_full_frame_coat_values_are_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "features" / "manifests").mkdir(parents=True)
            (root / "src").mkdir()
            source = root / "src" / "features.py"
            source.write_text("test", encoding="utf-8")
            checkpoint = root / "weights.pt"
            checkpoint.write_bytes(b"weights")
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            fields = [
                ("image_id", pa.string()), ("source_sha256", pa.string()),
                ("extraction_status", pa.string()), ("failure_reason", pa.string()),
                ("detection_status", pa.string()), ("mask_source", pa.string()),
                ("candidate_cat_count", pa.int32()), ("mask_fraction", pa.float32()),
                ("breed_margin", pa.float32()), ("mask_height", pa.int32()),
                ("mask_width", pa.int32()), ("mask_bitmap_zlib", pa.binary()),
                *[(name, pa.float32()) for name in NUMERIC_FEATURES],
                *[(name, pa.string()) for name in CATEGORICAL_FEATURES],
            ]
            row = {
                "image_id": "cat/a.jpg", "source_sha256": "source",
                "extraction_status": "ok", "failure_reason": "",
                "detection_status": "no_detection", "share_black": 0.5,
                "colour_primary": "no_detection", "colour_pattern": "no_detection",
            }
            path = root / "features" / "Y.parquet"
            pq.write_table(pa.Table.from_pylist([row], schema=pa.schema(fields)), path)
            names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
            info = {
                "feature_file_sha256": digest(path), "source_sha256": {"src/features.py": digest(source)},
                "feature_names": names, "feature_dimension": len(names),
                "checkpoint": {"detector": str(checkpoint), "clip": str(checkpoint)},
                "weight_sha256": {"detector": digest(checkpoint), "clip": digest(checkpoint)},
            }
            (root / "features" / "manifests" / "Y.json").write_text(json.dumps(info), encoding="utf-8")
            report, _ = audit_block(root, "Y", {"cat/a.jpg": "source"}, [
                {"a_image": "cat/a.jpg", "b_image": "cat/a.jpg"},
            ])
            self.assertEqual(report["status"], "quarantined")
            self.assertIn("no_detection_has_fabricated_measurement", report["details"])


if __name__ == "__main__":
    unittest.main()
