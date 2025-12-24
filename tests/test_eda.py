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

from breastcancer_rep.eda import nonzero_percent  # noqa: E402


class TestEda(unittest.TestCase):
    def test_nonzero_percent_histogram(self) -> None:
        img = Image.new("L", (10, 10), color=0)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 4, 9], fill=10)  # 5 columns * 10 rows = 50 nonzero
        pct = nonzero_percent(img)
        self.assertAlmostEqual(pct, 50.0, places=3)


if __name__ == "__main__":
    unittest.main()



