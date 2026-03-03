from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow running tests without installing the src-layout package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.attach_images import attach_image_paths  # noqa: E402
from breastcancer_rep.manifest import build_manifest_from_records  # noqa: E402
from breastcancer_rep.toydata import ToyDataSpec, generate_toy_dataset  # noqa: E402


class TestAttachImages(unittest.TestCase):
    def test_attach_by_existing_image_filename(self) -> None:
        workdir = REPO_ROOT / "runs" / "test" / "attach_images_contract"
        if workdir.exists():
            for p in sorted(workdir.rglob("*"), reverse=True):
                if p.is_file() or p.is_symlink():
                    p.unlink()
                else:
                    p.rmdir()
            workdir.rmdir()

        manifest_path = generate_toy_dataset(workdir, seed=7, spec=ToyDataSpec(n_patients=2, images_per_patient=1))
        images_dir = workdir / "images"
        rows = []
        for i, img in enumerate(sorted(images_dir.glob("*.png"))):
            rows.append(
                {
                    "sample_id": f"S{i:04d}",
                    "patient_id": f"P{i:04d}",
                    "label": "0" if i % 2 == 0 else "1",
                    "image_path": "",
                    "source_image_file_path": str(img),
                }
            )
        manifest_rows = build_manifest_from_records(rows)

        out_rows, n_attached, n_missing = attach_image_paths(
            manifest_rows, image_roots=[images_dir], overwrite=False, strict=False
        )
        self.assertEqual(n_missing, 0)
        self.assertEqual(n_attached, len(out_rows))
        self.assertTrue(all((r.get("image_path") or "").strip() != "" for r in out_rows))
        self.assertTrue(manifest_path.exists())

    def test_attach_strict_raises_on_missing(self) -> None:
        rows = build_manifest_from_records(
            [
                {
                    "sample_id": "S0001",
                    "patient_id": "P0001",
                    "label": "1",
                    "image_path": "",
                    "source_image_file_path": "does_not_exist.png",
                }
            ]
        )
        with self.assertRaises(RuntimeError):
            attach_image_paths(rows, image_roots=[REPO_ROOT], strict=True)


if __name__ == "__main__":
    unittest.main()

