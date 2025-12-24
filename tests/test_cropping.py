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

from breastcancer_rep.cropping import crop_image_path  # noqa: E402
from breastcancer_rep.toydata import ToyDataSpec, generate_toy_dataset  # noqa: E402


class TestCropping(unittest.TestCase):
    def test_crop_outputs_fixed_size(self) -> None:
        workdir = REPO_ROOT / ".test_run_crop"
        if workdir.exists():
            for p in sorted(workdir.rglob("*"), reverse=True):
                if p.is_file() or p.is_symlink():
                    p.unlink()
                else:
                    p.rmdir()
            workdir.rmdir()

        manifest_path = generate_toy_dataset(workdir, seed=2, spec=ToyDataSpec(n_patients=4, images_per_patient=1, image_size=64))
        img_dir = workdir / "images"
        out_dir = workdir / "cropped"
        out_dir.mkdir(parents=True, exist_ok=True)

        img_path = next(img_dir.glob("*.png"))
        out_path = out_dir / img_path.name
        res = crop_image_path(img_path, out_path, size=32, fill=0)
        self.assertTrue(out_path.exists())
        with Image.open(out_path) as im:
            self.assertEqual(im.size, (32, 32))
        # bbox may be None or non-None, but should not crash
        _ = res.bbox


if __name__ == "__main__":
    unittest.main()



