from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class CropResult:
    input_path: Path
    output_path: Path
    bbox: tuple[int, int, int, int] | None


def _nonzero_bbox(img: Image.Image) -> tuple[int, int, int, int] | None:
    """
    Bounding box of non-zero pixels.

    Uses a fast binary threshold and PIL's getbbox().
    Returns None if the image is all zeros.
    """
    gray = img.convert("L")
    mask = gray.point(lambda p: 255 if p != 0 else 0, mode="L")
    return mask.getbbox()


def _center_from_bbox(bbox: tuple[int, int, int, int], *, width: int, height: int) -> tuple[int, int]:
    left, top, right, bottom = bbox
    cx = int((left + right) / 2)
    cy = int((top + bottom) / 2)
    # clamp
    cx = max(0, min(width - 1, cx))
    cy = max(0, min(height - 1, cy))
    return cx, cy


def crop_fixed_size(
    img: Image.Image,
    *,
    center_xy: tuple[int, int],
    size: int,
    fill: int = 0,
) -> Image.Image:
    """
    Crop a fixed-size square around center_xy. Pads with `fill` if crop goes out of bounds.
    """
    if size <= 0:
        raise ValueError("size must be > 0")

    w, h = img.size
    cx, cy = center_xy
    half = size // 2
    left = cx - half
    top = cy - half
    right = left + size
    bottom = top + size

    # If fully inside, just crop.
    if left >= 0 and top >= 0 and right <= w and bottom <= h:
        return img.crop((left, top, right, bottom))

    # Otherwise, pad onto a new canvas and paste the intersecting region.
    canvas = Image.new(img.mode, (size, size), color=fill)
    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(w, right)
    src_bottom = min(h, bottom)
    patch = img.crop((src_left, src_top, src_right, src_bottom))

    dst_left = src_left - left
    dst_top = src_top - top
    canvas.paste(patch, (dst_left, dst_top))
    return canvas


def crop_image_path(
    input_path: Path,
    output_path: Path,
    *,
    size: int,
    fill: int = 0,
) -> CropResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as img:
        bbox = _nonzero_bbox(img)
        if bbox is None:
            center = (img.size[0] // 2, img.size[1] // 2)
        else:
            center = _center_from_bbox(bbox, width=img.size[0], height=img.size[1])
        cropped = crop_fixed_size(img, center_xy=center, size=size, fill=fill)
        cropped.save(output_path)
        return CropResult(input_path=input_path, output_path=output_path, bbox=bbox)


def iter_pngs(folder: Path) -> Iterable[Path]:
    yield from folder.glob("*.png")



