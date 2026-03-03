from __future__ import annotations

import argparse
from pathlib import Path

from breastcancer_rep.manifest import assert_manifest_contract, read_manifest_csv, write_manifest_csv
from breastcancer_rep.splitting import SplitFractions, assert_no_patient_leakage, assign_patient_splits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assign patient-level train/val/test splits to a manifest CSV.")
    p.add_argument("--in-manifest", type=Path, required=True)
    p.add_argument("--out-manifest", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
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
    rows = assign_patient_splits(rows, seed=args.seed, fractions=SplitFractions(val=args.val_frac, test=args.test_frac))
    assert_no_patient_leakage(rows)
    assert_manifest_contract(
        rows,
        require_labels=True,
        require_patient_ids=True,
        require_image_paths=False,
        require_splits=True,
    )
    write_manifest_csv(rows, args.out_manifest)
    print("OK: wrote split manifest")
    print(f"- path: {args.out_manifest}")


if __name__ == "__main__":
    main()


