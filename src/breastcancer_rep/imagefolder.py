from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .io_utils import ensure_dir, safe_symlink


CopyMode = Literal["copy", "symlink"]


@dataclass(frozen=True)
class ImageFolderLayout:
    """
    Torchvision ImageFolder-style layout:
      root/{split}/{class_name}/*.png
    """

    root: Path
    class_for_label0: str = "BENIGN"
    class_for_label1: str = "MALIGNANT"

    def class_name(self, label: str) -> str:
        if str(label) == "0":
            return self.class_for_label0
        if str(label) == "1":
            return self.class_for_label1
        raise ValueError(f"Unsupported label for ImageFolder layout: {label!r}")


def materialize_imagefolder(
    manifest_rows: list[dict[str, str]],
    *,
    layout: ImageFolderLayout,
    mode: CopyMode = "symlink",
    require_exists: bool = True,
) -> dict[str, int]:
    """
    Materialize images into an ImageFolder directory tree according to split + label.

    Expects each row to have:
      - split in {"train","val","test"}
      - label in {"0","1"}
      - image_path set (absolute or relative) and pointing to a file
    """
    ensure_dir(layout.root)
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for r in manifest_rows:
        split = (r.get("split") or "").strip()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Row missing/invalid split: {split!r}")
        label = (r.get("label") or "").strip()
        class_name = layout.class_name(label)
        img_path = (r.get("image_path") or "").strip()
        if img_path == "":
            raise ValueError("Row missing image_path; cannot materialize ImageFolder.")
        src = Path(img_path)
        if require_exists and not src.exists():
            raise FileNotFoundError(f"Missing image file: {src}")

        dst_dir = layout.root / split / class_name
        ensure_dir(dst_dir)
        dst = dst_dir / src.name

        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            safe_symlink(src.resolve(), dst)

        counts[split] += 1
    return counts


