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

from breastcancer_rep.roi_overlay import overlay_roi_on_full  # noqa: E402


class TestRoiOverlay(unittest.TestCase):
    def test_resizes_mask_and_overlays(self) -> None:
        # Full image 64x64
        full = Image.new("L", (64, 64), color=10).convert("RGB")
        # ROI mask 32x32 with black lesion on white background
        roi = Image.new("L", (32, 32), color=255)
        draw = ImageDraw.Draw(roi)
        draw.rectangle([10, 10, 20, 20], fill=0)

        out, full_size, roi_before = overlay_roi_on_full(
            full,
            roi,
            resize_mask=True,
            transparent_value=255,
            overlay_rgb=(255, 0, 0),
            alpha=200,
        )
        self.assertEqual(full_size, (64, 64))
        self.assertEqual(roi_before, (32, 32))
        self.assertEqual(out.size, (64, 64))
        # Should have some red pixels in output (not all gray)
        px = out.getpixel((32, 32))
        self.assertTrue(px[0] >= px[1] and px[0] >= px[2])


if __name__ == "__main__":
    unittest.main()



