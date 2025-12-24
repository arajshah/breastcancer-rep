"""
Convert CBIS-DDSM DICOM files to PNG files (portable CLI).

Typical usage:

  python Convert_DICOM_to_PNG.py \
    --source-root /path/to/CBIS-DDSM \
    --output-root /path/to/processed_png \
    --mode preserve \
    --dtype uint8 \
    --mapping-csv /path/to/dicom_to_png.csv

Notes:
- DICOM pixel values can be 16-bit for full mammograms; ROI/masks are often 8-bit-ish.
- This script does NOT download/unzip datasets. Unzip externally if needed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

from PIL import Image

try:
    import pydicom  # type: ignore
except Exception as e:  # pragma: no cover
    pydicom = None
    _PYDICOM_IMPORT_ERR = e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert DICOM tree to PNGs.")
    p.add_argument("--source-root", type=Path, required=True, help="Root directory to recursively scan for *.dcm.")
    p.add_argument("--output-root", type=Path, required=True, help="Directory to write PNG outputs.")
    p.add_argument(
        "--mode",
        choices=["preserve", "flatten"],
        default="preserve",
        help="preserve: keep relative folder structure under output-root; flatten: write all PNGs into one folder.",
    )
    p.add_argument(
        "--dtype",
        choices=["uint8", "uint16"],
        default="uint8",
        help="Pixel dtype to store in PNG. Use uint16 for full mammograms if you want to preserve dynamic range.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files instead of skipping.",
    )
    p.add_argument(
        "--mapping-csv",
        type=Path,
        default=None,
        help="Optional CSV to write mapping rows: source_path,png_path.",
    )
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def iter_dicom_files(root: Path):
    yield from root.rglob("*.dcm")


def dicom_to_array(dcm_path: Path, dtype: str):
    if pydicom is None:  # pragma: no cover
        raise RuntimeError(f"pydicom is required but failed to import: {_PYDICOM_IMPORT_ERR}")
    ds = pydicom.dcmread(str(dcm_path), force=True)
    arr = ds.pixel_array
    if dtype == "uint16":
        return arr.astype("uint16", copy=False)
    return arr.astype("uint8", copy=False)


def unique_dest_path(dest: Path, overwrite: bool) -> Path:
    if overwrite or not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def convert_tree(
    source_root: Path,
    output_root: Path,
    *,
    mode: str,
    dtype: str,
    overwrite: bool,
) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    ensure_dir(output_root)

    failed = 0
    converted = 0
    skipped = 0

    for dcm_path in iter_dicom_files(source_root):
        try:
            rel = dcm_path.relative_to(source_root)
        except ValueError:
            rel = dcm_path.name

        if mode == "preserve":
            out_dir = output_root / rel.parent
            ensure_dir(out_dir)
            dest = out_dir / (dcm_path.stem + ".png")
        else:
            out_dir = output_root
            ensure_dir(out_dir)
            # flatten name includes relative path components to reduce collisions
            safe_name = "__".join(rel.parts).replace(".dcm", ".png")
            dest = out_dir / safe_name

        dest = unique_dest_path(dest, overwrite=overwrite)
        if dest.exists() and not overwrite:
            skipped += 1
            continue

        arr = dicom_to_array(dcm_path, dtype=dtype)
        img = Image.fromarray(arr)
        img.save(dest)
        mappings.append((str(dcm_path), str(dest)))
        converted += 1

    print(f"Done. converted={converted} skipped={skipped} failed={failed}")
    return mappings


def write_mapping_csv(path: Path, mappings: list[tuple[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_path", "png_path"])
        w.writerows(mappings)


def main() -> None:
    args = parse_args()
    if not args.source_root.exists():
        raise FileNotFoundError(f"Missing --source-root: {args.source_root}")
    mappings = convert_tree(
        args.source_root,
        args.output_root,
        mode=args.mode,
        dtype=args.dtype,
        overwrite=args.overwrite,
    )
    if args.mapping_csv is not None:
        write_mapping_csv(args.mapping_csv, mappings)
        print(f"Wrote mapping CSV: {args.mapping_csv}")


if __name__ == "__main__":
    main()
