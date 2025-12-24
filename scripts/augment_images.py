from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.augment import AugmentConfig, augment_file  # noqa: E402
from breastcancer_rep.manifest import read_manifest_csv, write_manifest_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Pillow-only augmented images (deterministic).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-dir", type=Path, help="Directory containing *.png to augment.")
    g.add_argument("--in-manifest", type=Path, help="Manifest CSV with image_path to augment.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=5, help="Augmented images per input image.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rotation-deg", type=float, default=20.0)
    p.add_argument("--no-hflip", action="store_true")
    p.add_argument("--no-vflip", action="store_true")
    p.add_argument("--brightness-jitter", type=float, default=0.15)
    p.add_argument("--contrast-jitter", type=float, default=0.15)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--out-manifest",
        type=Path,
        default=None,
        help="If set (manifest mode), write an augmented manifest containing duplicated rows with new sample_id/image_path.",
    )
    return p.parse_args()


def iter_pngs(folder: Path):
    yield from folder.glob("*.png")


def main() -> None:
    args = parse_args()
    cfg = AugmentConfig(
        seed=args.seed,
        max_rotation_deg=args.max_rotation_deg,
        hflip=not args.no_hflip,
        vflip=not args.no_vflip,
        brightness_jitter=args.brightness_jitter,
        contrast_jitter=args.contrast_jitter,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir is not None:
        n_written = 0
        for p in iter_pngs(args.input_dir):
            written = augment_file(p, out_dir, n=args.n, cfg=cfg, overwrite=args.overwrite)
            n_written += len(written)
        print(f"OK: wrote {n_written} augmented images -> {out_dir}")
        return

    # manifest mode: duplicate rows (keeps same patient/label/split) but new sample_id and image_path
    rows = read_manifest_csv(args.in_manifest)
    augmented_rows = []
    n_written = 0
    for r in rows:
        img_path = (r.get("image_path") or "").strip()
        if img_path == "":
            augmented_rows.append(dict(r))
            continue
        src = Path(img_path)
        base_sample_id = (r.get("sample_id") or src.stem).strip() or src.stem
        written = augment_file(src, out_dir, n=args.n, cfg=cfg, prefix=base_sample_id, overwrite=args.overwrite)
        for i, out_path in enumerate(written, start=1):
            rr = dict(r)
            rr["sample_id"] = f"{base_sample_id}__aug{i:02d}"
            rr["image_path"] = str(out_path)
            augmented_rows.append(rr)
        n_written += len(written)

    if args.out_manifest is not None:
        write_manifest_csv(augmented_rows, args.out_manifest)
        print(f"OK: wrote augmented manifest -> {args.out_manifest} (rows={len(augmented_rows)})")
    print(f"OK: wrote {n_written} augmented images -> {out_dir}")


if __name__ == "__main__":
    main()




