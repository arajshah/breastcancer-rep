from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.eda import compute_stats_for_paths, compute_stats_from_manifest_rows, iter_png_paths, write_stats_csv  # noqa: E402
from breastcancer_rep.manifest import read_manifest_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute image size and nonzero% stats and write CSV report.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-dir", type=Path, help="Directory of PNGs to analyze.")
    g.add_argument("--in-manifest", type=Path, help="Manifest CSV containing image_path (and optional labels).")
    p.add_argument("--image-path-col", default="image_path", help="Manifest image path column name.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to write report CSVs.")
    p.add_argument("--output-name", default="image_stats.csv", help="CSV filename (default: image_stats.csv).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / args.output_name

    if args.input_dir is not None:
        rows = compute_stats_for_paths(iter_png_paths(args.input_dir))
    else:
        manifest_rows = read_manifest_csv(args.in_manifest)
        rows = compute_stats_from_manifest_rows(manifest_rows, image_path_col=args.image_path_col)

    write_stats_csv(out_csv, rows)
    print(f"OK: wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()



