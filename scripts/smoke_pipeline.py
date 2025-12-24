from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script without installing the package:
#   python scripts/smoke_pipeline.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.manifest import assert_manifest_schema, read_manifest_csv, write_manifest_csv  # noqa: E402
from breastcancer_rep.splitting import (  # noqa: E402
    SplitFractions,
    assert_no_patient_leakage,
    assign_patient_splits,
)
from breastcancer_rep.toydata import ToyDataSpec, generate_toy_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a dataset-free smoke pipeline (toy data + manifest + splits).")
    p.add_argument("--workdir", type=Path, required=True, help="Where to write toy images/manifest/splits.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-patients", type=int, default=12)
    p.add_argument("--images-per-patient", type=int, default=2)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = ToyDataSpec(
        n_patients=args.n_patients,
        images_per_patient=args.images_per_patient,
        image_size=args.image_size,
    )

    manifest_path = generate_toy_dataset(args.workdir, seed=args.seed, spec=spec)
    df = read_manifest_csv(manifest_path)
    assert_manifest_schema(df)

    df = assign_patient_splits(
        df, seed=args.seed, fractions=SplitFractions(val=args.val_frac, test=args.test_frac)
    )
    assert_no_patient_leakage(df)

    # Basic integrity checks
    missing_path_rows = [r for r in df if r.get("image_path", "") == ""]
    if missing_path_rows:
        raise RuntimeError("Manifest has missing image_path values.")
    missing = [r["image_path"] for r in df if not Path(r["image_path"]).exists()]
    if missing:
        raise RuntimeError(f"Manifest references missing image files. Example: {missing[0]}")

    # Write split manifest
    split_manifest = args.workdir / "manifest_with_splits.csv"
    write_manifest_csv(df, split_manifest)

    print("OK: smoke pipeline complete")
    print(f"- wrote: {manifest_path}")
    print(f"- wrote: {split_manifest}")
    # Print split counts without pandas
    counts: dict[tuple[str, str], int] = {}
    for r in df:
        key = (r["split"], r["label"])
        counts[key] = counts.get(key, 0) + 1
    print("- split counts (split,label -> n):")
    for key in sorted(counts.keys()):
        print(f"  {key[0]},{key[1]} -> {counts[key]}")


if __name__ == "__main__":
    main()


