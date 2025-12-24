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
from breastcancer_rep.roi_overlay import (  # noqa: E402
    MismatchRow,
    iter_pairs_from_folders,
    overlay_roi_on_full,
    write_mismatch_csv,
)
from PIL import Image  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay ROI mask on full mammogram PNGs (resizes mask if needed).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--roi-dir", type=Path, help="Folder with ROI mask PNGs.")
    g.add_argument("--in-manifest", type=Path, help="Manifest CSV containing full/mask paths.")

    p.add_argument("--full-dir", type=Path, help="Folder with full mammogram PNGs (required with --roi-dir).")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to write overlay PNGs.")

    # manifest mode controls
    p.add_argument("--full-col", default="full_image_path", help="Manifest column containing full image path.")
    p.add_argument("--mask-col", default="roi_mask_path", help="Manifest column containing ROI mask path.")
    p.add_argument("--out-col", default="overlay_path", help="Manifest column to write overlay path into.")
    p.add_argument("--out-manifest", type=Path, default=None, help="Write updated manifest (manifest mode only).")

    # overlay config
    p.add_argument("--transparent-value", type=int, default=255, help="Mask pixel value treated as background.")
    p.add_argument("--alpha", type=int, default=160, help="Overlay alpha in [0,255].")
    p.add_argument("--overlay-color", default="255,0,0", help="Overlay color as R,G,B (default red).")
    p.add_argument("--no-resize", action="store_true", help="Do not resize mask to match full image.")

    # reporting
    p.add_argument("--mismatch-csv", type=Path, default=None, help="Write CSV describing size mismatches.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing overlays.")
    return p.parse_args()


def parse_rgb(s: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("overlay-color must be R,G,B")
    r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
    for v in (r, g, b):
        if v < 0 or v > 255:
            raise ValueError("overlay-color values must be in [0,255]")
    return r, g, b


def run_folder_mode(args: argparse.Namespace) -> None:
    if args.full_dir is None:
        raise ValueError("--full-dir is required when using --roi-dir")
    roi_dir = args.roi_dir
    full_dir = args.full_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_rgb = parse_rgb(args.overlay_color)
    mismatches: list[MismatchRow] = []
    n = 0
    for key, full_path, roi_path in iter_pairs_from_folders(roi_dir, full_dir):
        out_path = out_dir / f"{key}.png"
        if out_path.exists() and not args.overwrite:
            continue
        with Image.open(full_path) as full_img, Image.open(roi_path) as roi_img:
            overlayed, full_size, roi_size_before = overlay_roi_on_full(
                full_img,
                roi_img,
                resize_mask=not args.no_resize,
                transparent_value=args.transparent_value,
                overlay_rgb=overlay_rgb,
                alpha=args.alpha,
            )
            overlayed.save(out_path)
            n += 1
            if roi_size_before != full_size:
                mismatches.append(
                    MismatchRow(
                        key=key,
                        full_path=full_path,
                        roi_path=roi_path,
                        full_size=full_size,
                        roi_size=roi_size_before,
                    )
                )

    if args.mismatch_csv is not None:
        write_mismatch_csv(args.mismatch_csv, mismatches)
        print(f"OK: wrote mismatch CSV -> {args.mismatch_csv} (rows={len(mismatches)})")
    print(f"OK: wrote {n} overlays -> {out_dir}")


def run_manifest_mode(args: argparse.Namespace) -> None:
    rows = read_manifest_csv(args.in_manifest)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_rgb = parse_rgb(args.overlay_color)
    mismatches: list[MismatchRow] = []

    updated = []
    n = 0
    for r in rows:
        full_path_s = (r.get(args.full_col) or "").strip()
        mask_path_s = (r.get(args.mask_col) or "").strip()
        if full_path_s == "" or mask_path_s == "":
            updated.append(dict(r))
            continue
        full_path = Path(full_path_s)
        mask_path = Path(mask_path_s)
        key = full_path.stem
        out_path = out_dir / f"{key}__overlay.png"
        if out_path.exists() and not args.overwrite:
            rr = dict(r)
            rr[args.out_col] = str(out_path)
            updated.append(rr)
            continue
        with Image.open(full_path) as full_img, Image.open(mask_path) as roi_img:
            overlayed, full_size, roi_size_before = overlay_roi_on_full(
                full_img,
                roi_img,
                resize_mask=not args.no_resize,
                transparent_value=args.transparent_value,
                overlay_rgb=overlay_rgb,
                alpha=args.alpha,
            )
            overlayed.save(out_path)
            rr = dict(r)
            rr[args.out_col] = str(out_path)
            updated.append(rr)
            n += 1
            if roi_size_before != full_size:
                mismatches.append(
                    MismatchRow(
                        key=key,
                        full_path=full_path,
                        roi_path=mask_path,
                        full_size=full_size,
                        roi_size=roi_size_before,
                    )
                )

    if args.out_manifest is not None:
        write_manifest_csv(updated, args.out_manifest)
        print(f"OK: wrote updated manifest -> {args.out_manifest}")
    if args.mismatch_csv is not None:
        write_mismatch_csv(args.mismatch_csv, mismatches)
        print(f"OK: wrote mismatch CSV -> {args.mismatch_csv} (rows={len(mismatches)})")
    print(f"OK: wrote {n} overlays -> {out_dir}")


def main() -> None:
    args = parse_args()
    if args.roi_dir is not None:
        run_folder_mode(args)
        return
    run_manifest_mode(args)


if __name__ == "__main__":
    main()




