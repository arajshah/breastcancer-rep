from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.manifest import read_manifest_csv, write_manifest_csv  # noqa: E402
from breastcancer_rep.splitting import SplitFractions, assert_no_patient_leakage, assign_patient_splits  # noqa: E402


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
    rows = assign_patient_splits(rows, seed=args.seed, fractions=SplitFractions(val=args.val_frac, test=args.test_frac))
    assert_no_patient_leakage(rows)
    write_manifest_csv(rows, args.out_manifest)
    print("OK: wrote split manifest")
    print(f"- path: {args.out_manifest}")


if __name__ == "__main__":
    main()


