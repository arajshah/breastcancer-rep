from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.cleanup import remove_white_edges_file  # noqa: E402
from breastcancer_rep.manifest import read_manifest_csv, write_manifest_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remove pure-white pixels (edge artifacts) from PNGs (Pillow-only).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-dir", type=Path, help="Directory containing *.png to clean.")
    g.add_argument("--in-manifest", type=Path, help="Manifest CSV with image_path to clean.")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to write cleaned PNGs.")
    p.add_argument("--white-value", type=int, default=-1, help="White value to replace. -1 = auto.")
    p.add_argument("--replacement-value", type=int, default=0, help="Value to write in place of white pixels.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out-manifest", type=Path, default=None, help="Write updated manifest (manifest mode only).")
    return p.parse_args()


def iter_pngs(folder: Path):
    yield from folder.glob("*.png")


def main() -> None:
    args = parse_args()
    white_value = None if args.white_value == -1 else args.white_value
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir is not None:
        n = 0
        for p in iter_pngs(args.input_dir):
            out = out_dir / p.name
            remove_white_edges_file(
                p,
                out,
                white_value=white_value,
                replacement_value=args.replacement_value,
                overwrite=args.overwrite,
            )
            n += 1
        print(f"OK: cleaned {n} images -> {out_dir}")
        return

    rows = read_manifest_csv(args.in_manifest)
    updated = []
    n = 0
    for r in rows:
        img_path = (r.get("image_path") or "").strip()
        if img_path == "":
            updated.append(dict(r))
            continue
        src = Path(img_path)
        dst = out_dir / src.name
        remove_white_edges_file(
            src,
            dst,
            white_value=white_value,
            replacement_value=args.replacement_value,
            overwrite=args.overwrite,
        )
        rr = dict(r)
        rr["image_path"] = str(dst)
        updated.append(rr)
        n += 1

    if args.out_manifest is not None:
        write_manifest_csv(updated, args.out_manifest)
        print(f"OK: wrote updated manifest -> {args.out_manifest}")
    print(f"OK: cleaned {n} images -> {out_dir}")


if __name__ == "__main__":
    main()



