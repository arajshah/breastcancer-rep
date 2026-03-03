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

ALLOWED_LABELS = {"0", "1"}
ALLOWED_SPLITS = {"train", "val", "test"}
ALLOWED_LATERALITY = {"LEFT", "RIGHT", "UNKNOWN"}
ALLOWED_VIEW = {"CC", "MLO", "UNKNOWN"}


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
    for i, row in enumerate(rows):
        cols = set(row.keys())
        missing = [c for c in MANIFEST_COLUMNS if c not in cols]
        if missing:
            raise ValueError(f"Manifest row {i} missing required columns: {missing}")
        extras = [c for c in row.keys() if c not in MANIFEST_COLUMNS]
        if extras:
            raise ValueError(f"Manifest row {i} has unsupported extra columns: {extras}")


def _fmt_row_err(i: int, message: str) -> str:
    return f"row {i}: {message}"


def assert_manifest_contract(
    rows: list[ManifestRow],
    *,
    require_labels: bool = True,
    require_patient_ids: bool = True,
    require_image_paths: bool = False,
    require_splits: bool = False,
    require_unique_sample_ids: bool = True,
) -> None:
    """
    Validate manifest row values used by pipeline stages.

    The contract is strict but stage-aware:
    - labels/patient IDs are required by default
    - image_path and split are optional unless explicitly required by stage
    """
    assert_manifest_schema(rows)

    errors: list[str] = []
    sample_id_seen: set[str] = set()
    for i, row in enumerate(rows):
        sample_id = (row.get("sample_id") or "").strip()
        patient_id = (row.get("patient_id") or "").strip()
        label = (row.get("label") or "").strip()
        split = (row.get("split") or "").strip()
        image_path = (row.get("image_path") or "").strip()
        laterality = (row.get("laterality") or "").strip()
        view = (row.get("view") or "").strip()

        if sample_id == "":
            errors.append(_fmt_row_err(i, "sample_id is required."))
        elif require_unique_sample_ids:
            if sample_id in sample_id_seen:
                errors.append(_fmt_row_err(i, f"duplicate sample_id: {sample_id!r}"))
            sample_id_seen.add(sample_id)

        if require_patient_ids and patient_id == "":
            errors.append(_fmt_row_err(i, "patient_id is required."))

        if require_labels:
            if label == "":
                errors.append(_fmt_row_err(i, "label is required."))
            elif label not in ALLOWED_LABELS:
                errors.append(_fmt_row_err(i, f"label must be one of {sorted(ALLOWED_LABELS)}, got {label!r}"))
        elif label != "" and label not in ALLOWED_LABELS:
            errors.append(_fmt_row_err(i, f"label must be one of {sorted(ALLOWED_LABELS)}, got {label!r}"))

        if require_splits:
            if split == "":
                errors.append(_fmt_row_err(i, "split is required."))
            elif split not in ALLOWED_SPLITS:
                errors.append(_fmt_row_err(i, f"split must be one of {sorted(ALLOWED_SPLITS)}, got {split!r}"))
        elif split != "" and split not in ALLOWED_SPLITS:
            errors.append(_fmt_row_err(i, f"split must be one of {sorted(ALLOWED_SPLITS)}, got {split!r}"))

        if require_image_paths and image_path == "":
            errors.append(_fmt_row_err(i, "image_path is required."))

        if laterality != "" and laterality not in ALLOWED_LATERALITY:
            errors.append(
                _fmt_row_err(i, f"laterality must be one of {sorted(ALLOWED_LATERALITY)} or empty, got {laterality!r}")
            )
        if view != "" and view not in ALLOWED_VIEW:
            errors.append(_fmt_row_err(i, f"view must be one of {sorted(ALLOWED_VIEW)} or empty, got {view!r}"))

    if errors:
        preview = "; ".join(errors[:10])
        if len(errors) > 10:
            preview += f"; ... ({len(errors) - 10} more)"
        raise ValueError(f"Manifest contract validation failed: {preview}")


@dataclass(frozen=True)
class ManifestPaths:
    image_root: Path
    manifest_csv: Path


def build_manifest_from_records(records: Iterable[dict]) -> list[ManifestRow]:
    return [_coerce_row(r) for r in records]


