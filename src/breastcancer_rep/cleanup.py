from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class CleanupResult:
    input_path: Path
    output_path: Path
    replaced_pixels: int | None
    white_value: int


def infer_white_value(img: Image.Image) -> int:
    """
    Infer the "white" background value used in mask-like images.
    - For 8-bit grayscale: 255
    - For 16-bit grayscale: 65535
    """
    try:
        extrema = img.getextrema()
    except Exception:
        extrema = None

    # Common grayscale modes:
    # - "L": 8-bit
    # - "I;16": 16-bit
    if img.mode == "I;16":
        return 65535
    if img.mode == "L":
        return 255
    if extrema is not None:
        # extrema may be a tuple for single-channel, or tuples per band for RGB etc.
        if isinstance(extrema, tuple) and len(extrema) == 2 and all(isinstance(x, int) for x in extrema):
            maxv = extrema[1]
            return 65535 if maxv > 255 else 255
    # default conservative
    return 255


def remove_white_pixels(
    img: Image.Image,
    *,
    white_value: int | None = None,
    replacement_value: int = 0,
) -> tuple[Image.Image, int | None, int]:
    """
    Replace pixels equal to white_value with replacement_value.

    Returns: (new_image, replaced_pixels_count_or_None, used_white_value)
    """
    if white_value is None:
        white_value = infer_white_value(img)

    # Work in grayscale; mask-like images in this repo are typically grayscale.
    gray = img.convert("I") if img.mode == "I;16" else img.convert("L")
    mask = gray.point(lambda p: 255 if p == white_value else 0, mode="L")

    # Create replacement image in same mode as gray
    repl = Image.new(gray.mode, gray.size, color=replacement_value)
    out = Image.composite(repl, gray, mask)

    # Counting replaced pixels without numpy is non-trivial; skip unless needed.
    return out, None, white_value


def remove_white_edges_file(
    input_path: Path,
    output_path: Path,
    *,
    white_value: int | None = None,
    replacement_value: int = 0,
    overwrite: bool = False,
) -> CleanupResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return CleanupResult(input_path=input_path, output_path=output_path, replaced_pixels=None, white_value=white_value or 255)

    with Image.open(input_path) as img:
        out, replaced, used_white = remove_white_pixels(
            img, white_value=white_value, replacement_value=replacement_value
        )
        out.save(output_path)
        return CleanupResult(input_path=input_path, output_path=output_path, replaced_pixels=replaced, white_value=used_white)




