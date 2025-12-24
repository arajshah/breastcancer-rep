from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

# Allow running tests without installing the src-layout package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.augment import AugmentConfig, augment_once  # noqa: E402
from breastcancer_rep.cleanup import remove_white_pixels  # noqa: E402


class TestCleanupAugment(unittest.TestCase):
    def test_remove_white_pixels(self) -> None:
        img = Image.new("L", (16, 16), color=0)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 3, 15], fill=255)  # white strip
        out, _replaced, used_white = remove_white_pixels(img, white_value=None, replacement_value=0)
        self.assertIn(used_white, (255, 65535))
        self.assertEqual(out.getpixel((1, 1)), 0)

    def test_augment_preserves_size(self) -> None:
        img = Image.new("L", (64, 64), color=10)
        cfg = AugmentConfig(seed=123)
        out = augment_once(img, cfg=cfg, salt=1)
        self.assertEqual(out.size, img.size)


if __name__ == "__main__":
    unittest.main()




