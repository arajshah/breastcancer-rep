from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttachStats:
    total_rows: int
    matched_rows: int
    unmatched_rows: int
    ambiguous_rows: int


def index_pngs_by_patient_id(images_dir: Path) -> dict[str, list[Path]]:
    """
    Build index patient_id -> list[png paths].

    Supports the naming scheme produced by prepare_kaggle_jpegs.py:
      {PatientID}_{SeriesInstanceUID}_{InstanceNumber}.png
    where SeriesInstanceUID typically begins with "1.3...", making "_1.3." a stable delimiter.
    """
    idx: dict[str, list[Path]] = {}
    for p in images_dir.glob("*.png"):
        stem = p.stem
        if "_1.3." in stem:
            patient_id = stem.split("_1.3.", 1)[0]
        else:
            # fallback: no UID delimiter; use full stem (useful for toy/small cases)
            patient_id = stem
        idx.setdefault(patient_id, []).append(p)
    return idx


def attach_image_paths(
    manifest_rows: list[dict[str, str]],
    *,
    images_dir: Path,
    patient_id_col: str = "patient_id",
    out_col: str = "image_path",
    choose: str = "first",
) -> tuple[list[dict[str, str]], AttachStats]:
    """
    Attach image paths to manifest rows by matching patient_id to PNG filenames.

    - choose="first": if multiple PNGs match a patient_id key, pick the lexicographically first.
    """
    if choose not in {"first"}:
        raise ValueError("choose must be 'first'")

    idx = index_pngs_by_patient_id(images_dir)
    updated: list[dict[str, str]] = []
    matched = 0
    unmatched = 0
    ambiguous = 0

    for r in manifest_rows:
        rr = dict(r)
        pid = (rr.get(patient_id_col) or "").strip()
        if pid == "":
            updated.append(rr)
            unmatched += 1
            continue

        matches = idx.get(pid, [])
        if not matches:
            updated.append(rr)
            unmatched += 1
            continue

        if len(matches) > 1:
            ambiguous += 1
        if choose == "first":
            chosen = sorted(matches)[0]
        rr[out_col] = str(chosen)
        updated.append(rr)
        matched += 1

    stats = AttachStats(
        total_rows=len(manifest_rows),
        matched_rows=matched,
        unmatched_rows=unmatched,
        ambiguous_rows=ambiguous,
    )
    return updated, stats



