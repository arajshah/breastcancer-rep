from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.attach_images import attach_image_paths  # noqa: E402
from breastcancer_rep.manifest import read_manifest_csv, write_manifest_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attach local PNG image paths to a manifest by matching patient_id.")
    p.add_argument("--in-manifest", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True, help="Directory containing *.png (e.g., processed PNGs).")
    p.add_argument("--out-manifest", type=Path, required=True)
    p.add_argument("--patient-id-col", default="patient_id")
    p.add_argument("--out-col", default="image_path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest_csv(args.in_manifest)
    updated, stats = attach_image_paths(
        rows,
        images_dir=args.images_dir,
        patient_id_col=args.patient_id_col,
        out_col=args.out_col,
        choose="first",
    )
    write_manifest_csv(updated, args.out_manifest)
    print("OK: wrote manifest with image paths")
    print(f"- out: {args.out_manifest}")
    print(f"- matched: {stats.matched_rows}/{stats.total_rows} (ambiguous={stats.ambiguous_rows}, unmatched={stats.unmatched_rows})")


if __name__ == "__main__":
    main()



