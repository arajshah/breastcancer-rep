"""
Utility script to adapt the Kaggle CBIS-DDSM JPEG mirror to the PNG-based
pipeline provided in this repository.

The original code expects DICOM files. The Kaggle release already converted
those DICOM objects to JPEGs and ships helpful metadata (dicom_info.csv). This
script reads that metadata, finds the JPEG counterparts for each series, and
exports them as PNG files grouped by series type (full, cropped ROI, ROI mask).

Usage (from the repository root):

    python prepare_kaggle_jpegs.py \
        --csv-dir ../images/csv \
        --jpeg-dir ../images/jpeg \
        --output-root ../processed_png

After running the script you can point the downstream preprocessing scripts to:
  - ../processed_png/full_png        (full-resolution mammograms)
  - ../processed_png/cropped_png     (cropped lesion patches)
  - ../processed_png/roi_mask_png    (ROI masks, used for overlay/mapping)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


SERIES_MAP = {
    "full mammogram images": "full_png",
    "cropped images": "cropped_png",
    "ROI mask images": "roi_mask_png",
}

CBIS_JPEG_PREFIX = Path("CBIS-DDSM") / "jpeg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Kaggle CBIS-DDSM JPEG files back into the PNG layout "
        "expected by the original pipeline."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("../images/csv"),
        help="Directory that contains dicom_info.csv (default: ../images/csv)",
    )
    parser.add_argument(
        "--jpeg-dir",
        type=Path,
        default=Path("../images/jpeg"),
        help="Root directory that contains the Kaggle JPEG folders "
        "(default: ../images/jpeg)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../processed_png"),
        help="Where to write the PNG copies (default: ../processed_png)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files instead of skipping them.",
    )
    return parser.parse_args()


def load_metadata(csv_dir: Path) -> pd.DataFrame:
    dicom_info_path = csv_dir / "dicom_info.csv"
    if not dicom_info_path.exists():
        raise FileNotFoundError(f"Missing {dicom_info_path}")

    df = pd.read_csv(dicom_info_path)
    df = df[df["PatientID"].str.startswith("Mass-", na=False)]
    df = df[df["SeriesDescription"].isin(SERIES_MAP)]
    # Keep only the columns we actually need so the DataFrame stays lean.
    return df[
        [
            "PatientID",
            "SeriesInstanceUID",
            "SeriesDescription",
            "InstanceNumber",
            "image_path",
        ]
    ].copy()


def resolve_jpeg_path(jpeg_dir: Path, image_path: str) -> Path:
    rel_path = Path(image_path)
    try:
        rel_path = rel_path.relative_to(CBIS_JPEG_PREFIX)
    except ValueError:
        # Some Kaggle dumps omit the CBIS-DDSM/jpeg prefix; fall back to raw path.
        if rel_path.is_absolute():
            return rel_path
    return jpeg_dir / rel_path


def export_series(df: pd.DataFrame, jpeg_dir: Path, out_root: Path, overwrite: bool) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    exported = 0
    skipped = 0
    missing = 0

    for row in df.itertuples(index=False):
        series_slug = SERIES_MAP[row.SeriesDescription]
        dest_dir = out_root / series_slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        instance_suffix = f"{int(row.InstanceNumber):03d}" if pd.notna(row.InstanceNumber) else "000"
        dest_name = f"{row.PatientID}_{row.SeriesInstanceUID}_{instance_suffix}.png"
        dest_path = dest_dir / dest_name

        if dest_path.exists() and not overwrite:
            skipped += 1
            continue

        jpg_path = resolve_jpeg_path(jpeg_dir, row.image_path)
        if not jpg_path.exists():
            missing += 1
            print(f"[WARN] Missing JPEG file: {jpg_path}", file=sys.stderr)
            continue

        with Image.open(jpg_path) as img:
            # Ensure single-channel grayscale to match the original PNG exports.
            grayscale = img.convert("L")
            grayscale.save(dest_path)

        exported += 1

    print(
        f"Finished exporting PNGs -> saved: {exported}, skipped: {skipped}, missing: {missing}"
    )


def main() -> None:
    args = parse_args()
    df = load_metadata(args.csv_dir)
    export_series(df, args.jpeg_dir, args.output_root, args.overwrite)


if __name__ == "__main__":
    main()
