import tempfile
import unittest
from pathlib import Path

from src.diagnostic_finalize import _plots


class FinalizeTests(unittest.TestCase):
    def test_plots_cover_all_representation_paths_and_omitted_curves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "figures").mkdir()
            paths = [{
                "model": f"common:{model}-{block}", "rho": "1" if model == "M4" else "0",
                "ridge": str(ridge), "train_loss": ".6", "validation_loss": ".7",
                "coefficient_norm": "1",
            } for block in ("H", "C", "Y") for model in ("M3", "M4") for ridge in (0.01, 0.1)]
            _plots(root, paths, [], {"status": "omitted_inadequate_common_validation"})
            for name in ("penalty_paths.pdf", "learning_curves.pdf"):
                self.assertEqual((root / "figures" / name).read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
