from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow running tests without installing the src-layout package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.manifest import (  # noqa: E402
    assert_manifest_contract,
    assert_manifest_schema,
    build_manifest_from_records,
)


class TestManifestContract(unittest.TestCase):
    def test_valid_minimal_rows_pass_contract(self) -> None:
        rows = build_manifest_from_records(
            [
                {
                    "sample_id": "S0001",
                    "patient_id": "P0001",
                    "laterality": "LEFT",
                    "view": "CC",
                    "pathology": "BENIGN",
                    "label": "0",
                    "image_path": "/tmp/a.png",
                    "split": "train",
                }
            ]
        )
        assert_manifest_schema(rows)
        assert_manifest_contract(
            rows,
            require_labels=True,
            require_patient_ids=True,
            require_image_paths=True,
            require_splits=True,
        )

    def test_invalid_label_fails_contract(self) -> None:
        rows = build_manifest_from_records(
            [
                {
                    "sample_id": "S0001",
                    "patient_id": "P0001",
                    "label": "2",
                    "image_path": "/tmp/a.png",
                    "split": "train",
                }
            ]
        )
        with self.assertRaises(ValueError):
            assert_manifest_contract(
                rows,
                require_labels=True,
                require_patient_ids=True,
                require_image_paths=False,
                require_splits=False,
            )

    def test_missing_image_path_fails_when_required(self) -> None:
        rows = build_manifest_from_records(
            [
                {
                    "sample_id": "S0001",
                    "patient_id": "P0001",
                    "label": "1",
                    "image_path": "",
                    "split": "",
                }
            ]
        )
        with self.assertRaises(ValueError):
            assert_manifest_contract(
                rows,
                require_labels=True,
                require_patient_ids=True,
                require_image_paths=True,
                require_splits=False,
            )

    def test_invalid_split_fails_when_present(self) -> None:
        rows = build_manifest_from_records(
            [
                {
                    "sample_id": "S0001",
                    "patient_id": "P0001",
                    "label": "0",
                    "image_path": "/tmp/a.png",
                    "split": "dev",
                }
            ]
        )
        with self.assertRaises(ValueError):
            assert_manifest_contract(
                rows,
                require_labels=True,
                require_patient_ids=True,
                require_image_paths=False,
                require_splits=False,
            )


if __name__ == "__main__":
    unittest.main()

