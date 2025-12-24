from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image

# Allow running tests without installing the src-layout package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.attach_images import attach_image_paths  # noqa: E402


class TestAttachImagePaths(unittest.TestCase):
    def test_matches_by_patient_id_prefix_before_uid(self) -> None:
        workdir = REPO_ROOT / ".test_attach_images"
        if workdir.exists():
            for p in sorted(workdir.rglob("*"), reverse=True):
                if p.is_file() or p.is_symlink():
                    p.unlink()
                else:
                    p.rmdir()
            workdir.rmdir()
        workdir.mkdir(parents=True, exist_ok=True)

        images_dir = workdir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        patient_id = "Mass-Training_P_00001_LEFT_CC_1"
        img_path = images_dir / f"{patient_id}_1.3.6.1.4.1.14519.5.2.1.1234_000.png"
        Image.new("L", (10, 10), color=0).save(img_path)

        rows = [
            {"sample_id": "s1", "patient_id": patient_id, "image_path": "", "label": "0", "split": ""},
        ]
        updated, stats = attach_image_paths(rows, images_dir=images_dir)
        self.assertEqual(stats.matched_rows, 1)
        self.assertTrue(updated[0]["image_path"].endswith(".png"))


if __name__ == "__main__":
    unittest.main()



