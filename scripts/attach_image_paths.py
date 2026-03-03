from __future__ import annotations

import argparse
from pathlib import Path

from breastcancer_rep.attach_images import attach_image_paths
from breastcancer_rep.manifest import assert_manifest_contract, read_manifest_csv, write_manifest_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attach image_path values to a manifest by scanning image roots.")
    p.add_argument("--in-manifest", type=Path, required=True)
    p.add_argument("--out-manifest", type=Path, required=True)
    p.add_argument(
        "--image-root",
        dest="image_roots",
        type=Path,
        action="append",
        required=True,
        help="Root directory to scan for *.png files. Repeat for multiple roots.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing image_path values.")
    p.add_argument("--strict", action="store_true", help="Fail if any row cannot be attached.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest_csv(args.in_manifest)
    assert_manifest_contract(
        rows,
        require_labels=True,
        require_patient_ids=True,
        require_image_paths=False,
        require_splits=False,
    )
    out_rows, n_attached, n_missing = attach_image_paths(
        rows,
        image_roots=args.image_roots,
        overwrite=args.overwrite,
        strict=args.strict,
    )
    assert_manifest_contract(
        out_rows,
        require_labels=True,
        require_patient_ids=True,
        require_image_paths=False,
        require_splits=False,
    )
    write_manifest_csv(out_rows, args.out_manifest)
    print("OK: wrote manifest with attached image paths")
    print(f"- path: {args.out_manifest}")
    print(f"- attached: {n_attached}")
    print(f"- missing: {n_missing}")


if __name__ == "__main__":
    main()

