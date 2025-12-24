from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MANIFEST_COLUMNS = [
    # identifiers
    "sample_id",  # unique per row
    "patient_id",  # used for anti-leakage splitting
    "participant_id",  # CBIS participant ID (e.g. 00038); may be empty
    # optional clinical-ish attributes
    "laterality",  # LEFT/RIGHT/UNKNOWN
    "view",  # CC/MLO/UNKNOWN
    # labels
    "pathology",  # original string label (e.g. BENIGN, MALIGNANT, BENIGN_WITHOUT_CALLBACK)
    "label",  # standardized numeric label (0/1)
    # data pointers
    "image_path",  # path to the image file on disk
    # optional upstream pointers (useful even when you don't download images yet)
    "source_image_file_path",
    "source_cropped_image_file_path",
    "source_roi_mask_file_path",
    # split assignment (train/val/test) – optional until assigned
    "split",
]


def normalize_pathology(pathology: str) -> tuple[str, int | None]:
    """
    Normalize CBIS-DDSM pathology strings.

    Returns (normalized_pathology, label) where label is:
      - 0 for benign-like
      - 1 for malignant
      - None if unknown
    """
    if pathology is None:
        return "UNKNOWN", None
    p = str(pathology).strip().upper()
    if p in {"MALIGNANT"}:
        return "MALIGNANT", 1
    if p in {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}:
        # Many papers collapse BENIGN_WITHOUT_CALLBACK into BENIGN.
        return p, 0
    return p, None


ManifestRow = dict[str, str]


def _coerce_row(row: dict) -> ManifestRow:
    out: ManifestRow = {}
    for col in MANIFEST_COLUMNS:
        val = row.get(col, None)
        if val is None:
            out[col] = ""
        else:
            out[col] = str(val)
    return out


def write_manifest_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(_coerce_row(r))


def read_manifest_csv(path: Path) -> list[ManifestRow]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows: list[ManifestRow] = []
        for row in r:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
        return rows


def assert_manifest_schema(rows: list[ManifestRow]) -> None:
    if not rows:
        raise ValueError("Manifest is empty.")
    cols = set(rows[0].keys())
    missing = [c for c in MANIFEST_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")


@dataclass(frozen=True)
class ManifestPaths:
    image_root: Path
    manifest_csv: Path


def build_manifest_from_records(records: Iterable[dict]) -> list[ManifestRow]:
    return [_coerce_row(r) for r in records]


