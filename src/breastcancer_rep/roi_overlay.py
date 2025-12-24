from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


_NAME_RE = re.compile(r"^(?P<prefix>Mass-.*?_(?:CC|MLO))(?:_.*)?\.png$", flags=re.IGNORECASE)


def key_from_filename(filename: str) -> str | None:
    """
    Extract a stable matching key from filenames like:
      Mass-Training_P_00038_LEFT_CC_1.png
      Mass-Training_P_00038_LEFT_CC_2.png

    Returns e.g. "Mass-Training_P_00038_LEFT_CC".
    """
    m = _NAME_RE.match(filename)
    if not m:
        return None
    return m.group("prefix")


@dataclass(frozen=True)
class MismatchRow:
    key: str
    full_path: Path
    roi_path: Path
    full_size: tuple[int, int]
    roi_size: tuple[int, int]


def build_index(folder: Path) -> dict[str, Path]:
    """
    Map key -> path. If multiple files map to same key, keeps the first in lexical order.
    """
    files = sorted(folder.glob("*.png"))
    out: dict[str, Path] = {}
    for p in files:
        k = key_from_filename(p.name)
        if k is None:
            continue
        out.setdefault(k, p)
    return out


def mask_to_rgba(
    mask_img: Image.Image,
    *,
    transparent_value: int = 255,
    overlay_rgb: tuple[int, int, int] = (255, 0, 0),
    alpha: int = 160,
) -> Image.Image:
    """
    Convert a grayscale-ish mask to an RGBA overlay:
      - pixels equal to transparent_value become fully transparent
      - other pixels become overlay_rgb with fixed alpha
    """
    if not (0 <= alpha <= 255):
        raise ValueError("alpha must be in [0,255]")

    m = mask_img.convert("L")
    # Build an alpha channel: 0 for background, alpha for foreground
    a = m.point(lambda p: 0 if p == transparent_value else alpha, mode="L")
    r, g, b = overlay_rgb
    color = Image.new("RGB", m.size, color=(r, g, b))
    rgba = color.convert("RGBA")
    rgba.putalpha(a)
    return rgba


def overlay_roi_on_full(
    full_img: Image.Image,
    roi_mask_img: Image.Image,
    *,
    resize_mask: bool = True,
    transparent_value: int = 255,
    overlay_rgb: tuple[int, int, int] = (255, 0, 0),
    alpha: int = 160,
) -> tuple[Image.Image, tuple[int, int], tuple[int, int]]:
    """
    Returns (overlayed_image, full_size, roi_size_before_resize)
    """
    full = full_img.convert("RGB")
    roi = roi_mask_img
    roi_size_before = roi.size
    if resize_mask and roi.size != full.size:
        roi = roi.resize(full.size)
    overlay = mask_to_rgba(roi, transparent_value=transparent_value, overlay_rgb=overlay_rgb, alpha=alpha)
    full_rgba = full.convert("RGBA")
    full_rgba.alpha_composite(overlay, dest=(0, 0))
    return full_rgba.convert("RGB"), full.size, roi_size_before


def write_mismatch_csv(path: Path, rows: list[MismatchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "full_path", "roi_path", "full_width", "full_height", "roi_width", "roi_height"])
        for r in rows:
            w.writerow([r.key, str(r.full_path), str(r.roi_path), r.full_size[0], r.full_size[1], r.roi_size[0], r.roi_size[1]])


def iter_pairs_from_folders(roi_dir: Path, full_dir: Path) -> Iterable[tuple[str, Path, Path]]:
    roi_idx = build_index(roi_dir)
    full_idx = build_index(full_dir)
    common = sorted(set(roi_idx.keys()) & set(full_idx.keys()))
    for k in common:
        yield k, full_idx[k], roi_idx[k]




