from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class ImageStatRow:
    sample_id: str
    image_path: str
    width: int
    height: int
    nonzero_percent: float
    pathology: str
    label: str
    split: str


def nonzero_percent(img: Image.Image) -> float:
    """
    Percent of pixels that are non-zero (computed on 8-bit grayscale).
    Uses histogram to avoid numpy dependency.
    """
    gray = img.convert("L")
    w, h = gray.size
    total = w * h
    if total == 0:
        return 0.0
    hist = gray.histogram()  # 256 bins for L
    zeros = hist[0] if hist else 0
    nz = total - zeros
    return (nz / total) * 100.0


def iter_png_paths(folder: Path) -> Iterable[Path]:
    yield from folder.glob("*.png")


def compute_stats_for_paths(
    paths: Iterable[Path],
    *,
    default_pathology: str = "",
    default_label: str = "",
    default_split: str = "",
) -> list[ImageStatRow]:
    rows: list[ImageStatRow] = []
    for p in paths:
        with Image.open(p) as img:
            w, h = img.size
            nz = nonzero_percent(img)
        rows.append(
            ImageStatRow(
                sample_id=p.stem,
                image_path=str(p),
                width=int(w),
                height=int(h),
                nonzero_percent=float(nz),
                pathology=default_pathology,
                label=default_label,
                split=default_split,
            )
        )
    return rows


def compute_stats_from_manifest_rows(
    manifest_rows: list[dict[str, str]],
    *,
    image_path_col: str = "image_path",
) -> list[ImageStatRow]:
    out: list[ImageStatRow] = []
    for r in manifest_rows:
        img_path = (r.get(image_path_col) or "").strip()
        if img_path == "":
            continue
        p = Path(img_path)
        with Image.open(p) as img:
            w, h = img.size
            nz = nonzero_percent(img)
        out.append(
            ImageStatRow(
                sample_id=(r.get("sample_id") or p.stem),
                image_path=str(p),
                width=int(w),
                height=int(h),
                nonzero_percent=float(nz),
                pathology=(r.get("pathology") or ""),
                label=(r.get("label") or ""),
                split=(r.get("split") or ""),
            )
        )
    return out


def write_stats_csv(path: Path, rows: list[ImageStatRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "image_path", "width", "height", "nonzero_percent", "pathology", "label", "split"])
        for r in rows:
            w.writerow([r.sample_id, r.image_path, r.width, r.height, f"{r.nonzero_percent:.6f}", r.pathology, r.label, r.split])



