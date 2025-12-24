from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.cbis import parse_cbis_patient_id  # noqa: E402
from breastcancer_rep.manifest import normalize_pathology, write_manifest_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a canonical manifest.csv from CBIS-DDSM Mass case description CSVs "
            "(train/test) without requiring the images to be downloaded."
        )
    )
    p.add_argument("--mass-train-csv", type=Path, required=True)
    p.add_argument("--mass-test-csv", type=Path, required=True)
    p.add_argument("--out-manifest", type=Path, required=True)
    p.add_argument(
        "--include-benign-without-callback",
        action="store_true",
        help="Keep BENIGN_WITHOUT_CALLBACK rows (label=0).",
    )
    return p.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [{k: (v if v is not None else "") for k, v in row.items()} for row in r]


def _case_rows_to_manifest(rows: list[dict[str, str]], *, allow_bwc: bool) -> list[dict]:
    """
    Convert CBIS-DDSM mass_case_description rows -> manifest rows.

    Expected columns (per TCIA distribution):
      - patient_id
      - pathology
      - image file path
      - cropped image file path
      - ROI mask file path
    """
    out: list[dict] = []
    for i, row in enumerate(rows):
        patient_id = row.get("patient_id", "").strip()
        pathology_raw = row.get("pathology", "").strip()
        pathology_norm, label = normalize_pathology(pathology_raw)

        if label is None:
            # Unknown label; skip.
            continue
        if (not allow_bwc) and pathology_norm == "BENIGN_WITHOUT_CALLBACK":
            continue

        parsed = parse_cbis_patient_id(patient_id)
        participant_id = parsed.participant_id
        laterality = parsed.laterality
        view = parsed.view

        # A stable-ish sample id: patient_id + row index suffix (there can be multiple ROIs per view)
        sample_id = f"{patient_id}__r{i:04d}" if patient_id else f"row__{i:06d}"

        out.append(
            dict(
                sample_id=sample_id,
                patient_id=patient_id,
                participant_id=participant_id,
                laterality=laterality,
                view=view,
                pathology=pathology_norm,
                label=str(label),
                image_path="",  # images not required at this stage
                source_image_file_path=row.get("image file path", ""),
                source_cropped_image_file_path=row.get("cropped image file path", ""),
                source_roi_mask_file_path=row.get("ROI mask file path", ""),
                split="",
            )
        )
    return out


def main() -> None:
    args = parse_args()
    train_rows = _read_rows(args.mass_train_csv)
    test_rows = _read_rows(args.mass_test_csv)
    manifest_rows = _case_rows_to_manifest(
        train_rows, allow_bwc=args.include_benign_without_callback
    ) + _case_rows_to_manifest(test_rows, allow_bwc=args.include_benign_without_callback)

    if not manifest_rows:
        raise RuntimeError("No labeled rows found. Check input CSVs and column names.")

    write_manifest_csv(manifest_rows, args.out_manifest)
    print("OK: wrote manifest")
    print(f"- rows: {len(manifest_rows)}")
    print(f"- path: {args.out_manifest}")


if __name__ == "__main__":
    main()


