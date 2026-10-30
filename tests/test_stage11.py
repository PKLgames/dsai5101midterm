import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.feature_regeneration import _inventory
from src.reproduce import reproduce_statistics
from src.reproduction_check import run_check
from src.submission_package import included


class ReproductionGateTests(unittest.TestCase):
    def test_archive_excludes_photos_weights_environments_and_secrets(self):
        for name in (".git/config", ".venv/pyvenv.cfg", ".cache/clip/weights",
                     "artifacts/review_panels/example.jpg", "weights.pt",
                     "dataset/images/cat/a.jpg", ".env", "src/__pycache__/x.pyc"):
            self.assertFalse(included(Path(name)), name)
        for name in ("src/report.py", "features/Y.parquet", "dataset/train.csv",
                     "artifacts/model_freeze.json", "figures/calibration.pdf"):
            self.assertTrue(included(Path(name)), name)

    def test_working_copy_check_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "existing.json"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_check(root, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")

    def test_existing_reproduction_log_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts").mkdir()
            log = root / "artifacts/reproduction_statistical.json"
            log.write_text("existing", encoding="utf-8")
            with patch("src.reproduce.verify_history") as gate:
                with self.assertRaises(FileExistsError):
                    reproduce_statistics(root)
            gate.assert_not_called()
            self.assertEqual(log.read_text(encoding="utf-8"), "existing")

    def test_missing_freeze_blocks_reproduction_before_data_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("src.reproduce.load_development") as loader:
                with self.assertRaises(FileNotFoundError):
                    reproduce_statistics(root)
            loader.assert_not_called()

    def test_subset_inventory_requires_matching_images_and_cache_hash(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "features/manifests").mkdir(parents=True)
            path = root / "features/H.parquet"
            pq.write_table(pa.Table.from_pylist([
                {"image_id": "cat/a.jpg", "source_sha256": "missing"},
            ]), path)
            manifest = root / "features/manifests/H.json"
            manifest.write_text(json.dumps({
                "feature_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source image missing"):
                _inventory(root, "H")
            manifest.write_text(json.dumps({"feature_file_sha256": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cache.*changed"):
                _inventory(root, "H")


if __name__ == "__main__":
    unittest.main()
