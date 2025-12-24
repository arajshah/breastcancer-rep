from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.cropping import crop_image_path, iter_pngs  # noqa: E402
from breastcancer_rep.manifest import read_manifest_csv, write_manifest_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crop fixed-size patches around nonzero region center (Pillow-only, pads at borders)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-dir", type=Path, help="Directory containing *.png images to crop.")
    g.add_argument("--in-manifest", type=Path, help="Manifest CSV with image_path column to crop.")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to write cropped PNGs.")
    p.add_argument("--size", type=int, default=598, help="Crop size (square). Default: 598.")
    p.add_argument("--fill", type=int, default=0, help="Padding fill value (default: 0).")
    p.add_argument(
        "--out-manifest",
        type=Path,
        default=None,
        help="If --in-manifest is used, write an updated manifest with image_path pointing to cropped outputs.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return p.parse_args()


def crop_folder(input_dir: Path, output_dir: Path, *, size: int, fill: int, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in iter_pngs(input_dir):
        out_path = output_dir / img_path.name
        if out_path.exists() and not overwrite:
            continue
        crop_image_path(img_path, out_path, size=size, fill=fill)
        n += 1
    print(f"OK: cropped {n} images -> {output_dir}")


def crop_manifest(
    in_manifest: Path,
    output_dir: Path,
    *,
    size: int,
    fill: int,
    overwrite: bool,
    out_manifest: Path | None,
) -> None:
    rows = read_manifest_csv(in_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    updated = []
    n = 0
    for r in rows:
        img_path = (r.get("image_path") or "").strip()
        if img_path == "":
            raise ValueError("Manifest row missing image_path; cannot crop.")
        src = Path(img_path)
        dst = output_dir / src.name
        if dst.exists() and not overwrite:
            # still update manifest pointer
            rr = dict(r)
            rr["image_path"] = str(dst)
            updated.append(rr)
            continue
        crop_image_path(src, dst, size=size, fill=fill)
        rr = dict(r)
        rr["image_path"] = str(dst)
        updated.append(rr)
        n += 1

    print(f"OK: cropped {n} images -> {output_dir}")
    if out_manifest is not None:
        write_manifest_csv(updated, out_manifest)
        print(f"OK: wrote updated manifest -> {out_manifest}")


def main() -> None:
    args = parse_args()
    if args.input_dir is not None:
        crop_folder(args.input_dir, args.output_dir, size=args.size, fill=args.fill, overwrite=args.overwrite)
        return
    crop_manifest(
        args.in_manifest,
        args.output_dir,
        size=args.size,
        fill=args.fill,
        overwrite=args.overwrite,
        out_manifest=args.out_manifest,
    )


if __name__ == "__main__":
    main()


