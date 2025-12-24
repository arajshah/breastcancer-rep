from __future__ import annotations

import unittest
import sys
from pathlib import Path

# Allow running tests without installing the src-layout package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.splitting import SplitFractions, assert_no_patient_leakage, assign_patient_splits


class TestSplitting(unittest.TestCase):
    def test_patient_level_no_leakage(self) -> None:
        rows = [
            {"sample_id": "s1", "patient_id": "p1", "label": "0"},
            {"sample_id": "s2", "patient_id": "p1", "label": "0"},
            {"sample_id": "s3", "patient_id": "p2", "label": "1"},
            {"sample_id": "s4", "patient_id": "p3", "label": "1"},
        ]
        out = assign_patient_splits(rows, seed=123, fractions=SplitFractions(val=0.25, test=0.25))
        splits = {r["split"] for r in out}
        self.assertTrue(splits.issubset({"train", "val", "test"}))
        assert_no_patient_leakage(out)


if __name__ == "__main__":
    unittest.main()


