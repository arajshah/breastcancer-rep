from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

# Allow running without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.toydata import ToyDataSpec, generate_toy_dataset  # noqa: E402
from breastcancer_rep.manifest import read_manifest_csv, write_manifest_csv  # noqa: E402
from breastcancer_rep.splitting import SplitFractions, assign_patient_splits, assert_no_patient_leakage  # noqa: E402
from breastcancer_rep.cropping import crop_image_path  # noqa: E402
from breastcancer_rep.cleanup import remove_white_edges_file  # noqa: E402
from breastcancer_rep.augment import AugmentConfig, augment_file  # noqa: E402
from breastcancer_rep.eda import compute_stats_from_manifest_rows, write_stats_csv  # noqa: E402
from breastcancer_rep.imagefolder import ImageFolderLayout, materialize_imagefolder  # noqa: E402


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    data_dir: Path
    reports_dir: Path
    images_dir: Path
    cropped_dir: Path
    cleaned_dir: Path
    augmented_dir: Path
    manifest_base: Path
    manifest_processed: Path
    manifest_splits: Path
    manifest_aug: Path
    imagefolder_root: Path


def make_run_dir(root: Path, name: str | None) -> Path:
    if name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"pipeline_{ts}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_paths(run_dir: Path) -> RunPaths:
    data_dir = run_dir / "data"
    reports_dir = run_dir / "reports"
    images_dir = data_dir / "images"
    cropped_dir = data_dir / "cropped"
    cleaned_dir = data_dir / "cleaned"
    augmented_dir = data_dir / "augmented"
    imagefolder_root = data_dir / "dataset_splits"
    return RunPaths(
        run_dir=run_dir,
        data_dir=data_dir,
        reports_dir=reports_dir,
        images_dir=images_dir,
        cropped_dir=cropped_dir,
        cleaned_dir=cleaned_dir,
        augmented_dir=augmented_dir,
        manifest_base=data_dir / "manifest_base.csv",
        manifest_processed=data_dir / "manifest_processed.csv",
        manifest_splits=data_dir / "manifest_splits.csv",
        manifest_aug=data_dir / "manifest_aug.csv",
        imagefolder_root=imagefolder_root,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an end-to-end pipeline into a single run directory.")
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--run-name", type=str, default=None)

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--toy", action="store_true", help="Run on generated toy data (no dataset download required).")
    mode.add_argument("--in-manifest", type=Path, help="Run pipeline starting from an existing manifest.csv.")

    # toy config
    p.add_argument("--toy-patients", type=int, default=20)
    p.add_argument("--toy-images-per-patient", type=int, default=2)
    p.add_argument("--toy-image-size", type=int, default=128)

    # preprocess
    p.add_argument("--crop-size", type=int, default=128, help="Crop size for this runner (toy default 128).")
    p.add_argument("--augment-n", type=int, default=2, help="Augmented images per original.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--materialize-mode", choices=["symlink", "copy"], default="symlink")

    # optional training hooks
    p.add_argument("--train-resnet", action="store_true", help="Attempt to run ResNet training script after materialization.")
    p.add_argument("--train-convnext", action="store_true", help="Attempt to run ConvNeXt training script after materialization.")
    return p.parse_args()


def try_run_training(script: Path, args: list[str]) -> None:
    """
    Run a python script if possible. Kept minimal: avoids subprocess flags beyond sys.executable.
    """
    import subprocess

    cmd = [sys.executable, str(script)] + args
    print(f"[train] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.runs_root, args.run_name)
    paths = build_paths(run_dir)

    # Ensure dirs
    for p in [paths.data_dir, paths.reports_dir, paths.images_dir, paths.cropped_dir, paths.cleaned_dir, paths.augmented_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Save run config snapshot for reproducibility
    run_cfg_path = paths.run_dir / "run_config.json"
    with run_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # 1) Get starting manifest (with image_path)
    if args.toy:
        base_manifest = generate_toy_dataset(
            paths.data_dir,
            seed=args.seed,
            spec=ToyDataSpec(
                n_patients=args.toy_patients,
                images_per_patient=args.toy_images_per_patient,
                image_size=args.toy_image_size,
            ),
        )
        rows = read_manifest_csv(base_manifest)
        # normalize to our expected runner file names
        write_manifest_csv(rows, paths.manifest_base)
        rows = read_manifest_csv(paths.manifest_base)
    else:
        rows = read_manifest_csv(args.in_manifest)
        write_manifest_csv(rows, paths.manifest_base)
        rows = read_manifest_csv(paths.manifest_base)
        # Guardrail: manifest mode must have some images, otherwise downstream steps are meaningless.
        has_any = any((r.get("image_path") or "").strip() != "" for r in rows)
        if not has_any:
            raise RuntimeError(
                "Manifest mode requires image_path values. "
                "Use scripts/attach_image_paths.py first, or provide a manifest that already has image_path."
            )

    # 2) Crop -> cleaned -> augment (update manifest pointers)
    processed_rows = []
    for r in rows:
        img_path = (r.get("image_path") or "").strip()
        if img_path == "":
            processed_rows.append(dict(r))
            continue
        src = Path(img_path)

        cropped_path = paths.cropped_dir / src.name
        crop_image_path(src, cropped_path, size=args.crop_size, fill=0)

        cleaned_path = paths.cleaned_dir / src.name
        remove_white_edges_file(cropped_path, cleaned_path, white_value=None, replacement_value=0, overwrite=True)

        rr = dict(r)
        rr["image_path"] = str(cleaned_path)
        processed_rows.append(rr)

    write_manifest_csv(processed_rows, paths.manifest_processed)

    # 3) Augment -> new manifest (duplicated rows)
    aug_cfg = AugmentConfig(seed=args.seed)
    augmented_rows = []
    for r in processed_rows:
        img_path = (r.get("image_path") or "").strip()
        if img_path == "":
            continue
        src = Path(img_path)
        base_id = (r.get("sample_id") or src.stem).strip() or src.stem
        outs = augment_file(src, paths.augmented_dir, n=args.augment_n, cfg=aug_cfg, prefix=base_id, overwrite=True)
        for i, outp in enumerate(outs, start=1):
            rr = dict(r)
            rr["sample_id"] = f"{base_id}__aug{i:02d}"
            rr["image_path"] = str(outp)
            augmented_rows.append(rr)

    # keep also original rows (common practice)
    all_for_split = processed_rows + augmented_rows
    write_manifest_csv(all_for_split, paths.manifest_aug)

    # 4) Assign patient-level splits
    split_rows = assign_patient_splits(
        all_for_split, seed=args.seed, fractions=SplitFractions(val=args.val_frac, test=args.test_frac)
    )
    assert_no_patient_leakage(split_rows)
    write_manifest_csv(split_rows, paths.manifest_splits)

    # 5) Materialize ImageFolder layout
    counts = materialize_imagefolder(
        split_rows,
        layout=ImageFolderLayout(root=paths.imagefolder_root),
        mode=args.materialize_mode,  # type: ignore[arg-type]
    )
    print(f"OK: materialized ImageFolder counts: {counts}")

    # 6) EDA report (CSV)
    stats = compute_stats_from_manifest_rows(split_rows, image_path_col="image_path")
    stats_csv = paths.reports_dir / "image_stats.csv"
    write_stats_csv(stats_csv, stats)
    print(f"OK: wrote EDA stats -> {stats_csv}")

    # 7) Optional training
    if args.train_resnet:
        try_run_training(
            REPO_ROOT / "model_development_and_evaluation.py",
            ["--data-root", str(paths.imagefolder_root), "--output-dir", str(paths.run_dir / "train_resnet")],
        )
    if args.train_convnext:
        try_run_training(
            REPO_ROOT / "model_convnext_absolute.py",
            ["--data-root", str(paths.imagefolder_root), "--output-dir", str(paths.run_dir / "train_convnext")],
        )

    print(f"OK: pipeline complete -> {paths.run_dir}")
    print(f"Run config saved -> {run_cfg_path}")


if __name__ == "__main__":
    main()


