from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import platform
import random as pyrandom
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]

from breastcancer_rep.toydata import ToyDataSpec, generate_toy_dataset
from breastcancer_rep.attach_images import attach_image_paths
from breastcancer_rep.manifest import assert_manifest_contract, read_manifest_csv, write_manifest_csv
from breastcancer_rep.splitting import SplitFractions, assign_patient_splits, assert_no_patient_leakage
from breastcancer_rep.cropping import crop_image_path
from breastcancer_rep.cleanup import remove_white_edges_file
from breastcancer_rep.augment import AugmentConfig, augment_file
from breastcancer_rep.eda import compute_stats_from_manifest_rows, write_stats_csv
from breastcancer_rep.imagefolder import ImageFolderLayout, materialize_imagefolder


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


@dataclass(frozen=True)
class StageResult:
    name: str
    elapsed_s: float
    details: dict[str, int | str | bool]


CONFIG_ALLOWED_KEYS = {
    "runs_root",
    "run_name",
    "toy",
    "in_manifest",
    "image_roots",
    "toy_patients",
    "toy_images_per_patient",
    "toy_image_size",
    "crop_size",
    "augment_n",
    "seed",
    "val_frac",
    "test_frac",
    "materialize_mode",
    "max_patients",
    "max_images",
    "train_resnet",
    "train_convnext",
    "skip_attach",
    "skip_crop",
    "skip_cleanup",
    "skip_augment",
    "skip_materialize",
    "skip_eda",
    "strict_splits",
    "fail_on_train_error",
}


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


def _build_parser(cfg: dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run an end-to-end pipeline into a single run directory.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(cfg["config"]) if cfg.get("config") else None,
        help="Optional JSON config file. CLI args override config values.",
    )
    p.add_argument("--runs-root", type=Path, default=Path(cfg.get("runs_root", "runs")))
    p.add_argument("--run-name", type=str, default=cfg.get("run_name", None))

    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--toy",
        action="store_true",
        default=bool(cfg.get("toy", False)),
        help="Run on generated toy data (no dataset download required).",
    )
    mode.add_argument(
        "--in-manifest",
        type=Path,
        default=(Path(cfg["in_manifest"]) if cfg.get("in_manifest") else None),
        help="Run pipeline starting from an existing manifest.csv.",
    )
    p.add_argument(
        "--image-root",
        dest="image_roots",
        type=Path,
        action="append",
        default=[Path(p) for p in cfg.get("image_roots", [])],
        help="Optional image roots used to attach image_path values when --in-manifest lacks them. Repeatable.",
    )

    # toy config
    p.add_argument("--toy-patients", type=int, default=int(cfg.get("toy_patients", 20)))
    p.add_argument("--toy-images-per-patient", type=int, default=int(cfg.get("toy_images_per_patient", 2)))
    p.add_argument("--toy-image-size", type=int, default=int(cfg.get("toy_image_size", 128)))

    # preprocess
    p.add_argument(
        "--crop-size", type=int, default=int(cfg.get("crop_size", 128)), help="Crop size for this runner."
    )
    p.add_argument("--augment-n", type=int, default=int(cfg.get("augment_n", 2)), help="Augmented images per original.")
    p.add_argument("--seed", type=int, default=int(cfg.get("seed", 42)))
    p.add_argument("--val-frac", type=float, default=float(cfg.get("val_frac", 0.1)))
    p.add_argument("--test-frac", type=float, default=float(cfg.get("test_frac", 0.1)))
    p.add_argument(
        "--materialize-mode",
        choices=["symlink", "copy"],
        default=str(cfg.get("materialize_mode", "symlink")),
    )
    p.add_argument(
        "--max-patients",
        type=int,
        default=int(cfg.get("max_patients", 0)),
        help="0 = no limit; otherwise sample N patient_ids.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=int(cfg.get("max_images", 0)),
        help="0 = no limit; otherwise take first N rows after filtering.",
    )

    # optional training hooks
    p.add_argument(
        "--train-resnet",
        action="store_true",
        default=bool(cfg.get("train_resnet", False)),
        help="Attempt to run ResNet training script after materialization.",
    )
    p.add_argument(
        "--train-convnext",
        action="store_true",
        default=bool(cfg.get("train_convnext", False)),
        help="Attempt to run ConvNeXt training script after materialization.",
    )

    # stage controls / strictness
    p.add_argument("--skip-attach", action="store_true", default=bool(cfg.get("skip_attach", False)))
    p.add_argument("--skip-crop", action="store_true", default=bool(cfg.get("skip_crop", False)))
    p.add_argument("--skip-cleanup", action="store_true", default=bool(cfg.get("skip_cleanup", False)))
    p.add_argument("--skip-augment", action="store_true", default=bool(cfg.get("skip_augment", False)))
    p.add_argument("--skip-materialize", action="store_true", default=bool(cfg.get("skip_materialize", False)))
    p.add_argument("--skip-eda", action="store_true", default=bool(cfg.get("skip_eda", False)))
    p.add_argument(
        "--strict-splits",
        action="store_true",
        default=bool(cfg.get("strict_splits", False)),
        help="Fail when any split is empty after assignment.",
    )
    p.add_argument(
        "--fail-on-train-error",
        action="store_true",
        default=bool(cfg.get("fail_on_train_error", False)),
        help="Fail pipeline if requested training subprocess fails.",
    )
    return p


def _load_json_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config JSON must be an object.")
    unknown = sorted(set(cfg.keys()) - CONFIG_ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return cfg


def parse_args_with_config() -> argparse.Namespace:
    """
    Two-pass parsing:
    - parse --config (if any)
    - apply config defaults
    - parse full CLI (CLI overrides)
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = _load_json_config(pre_args.config)
    cfg["config"] = str(pre_args.config) if pre_args.config else None
    p = _build_parser(cfg)
    args = p.parse_args()
    _validate_args(p, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.toy and args.in_manifest is None:
        parser.error("one of --toy or --in-manifest is required (can be provided via --config).")
    if args.toy and args.in_manifest is not None:
        parser.error("--toy and --in-manifest are mutually exclusive.")
    if args.crop_size <= 0:
        parser.error("--crop-size must be > 0.")
    if args.augment_n < 0:
        parser.error("--augment-n must be >= 0.")
    if args.toy_patients <= 0:
        parser.error("--toy-patients must be > 0.")
    if args.toy_images_per_patient <= 0:
        parser.error("--toy-images-per-patient must be > 0.")
    if args.toy_image_size <= 0:
        parser.error("--toy-image-size must be > 0.")
    if args.max_patients < 0 or args.max_images < 0:
        parser.error("--max-patients and --max-images must be >= 0.")
    if args.skip_materialize and (args.train_resnet or args.train_convnext):
        parser.error("Training requires materialization; remove --skip-materialize or disable training flags.")
    if args.skip_crop and not args.skip_cleanup:
        parser.error("--skip-cleanup must be set when --skip-crop is set (cleanup expects cropped files).")


def subset_rows(rows: list[dict[str, str]], *, seed: int, max_patients: int, max_images: int) -> list[dict[str, str]]:
    out = [dict(r) for r in rows]
    if max_patients and max_patients > 0:
        # sample patient_ids deterministically
        patient_ids = sorted({(r.get("patient_id") or "").strip() for r in out if (r.get("patient_id") or "").strip() != ""})
        rng = pyrandom.Random(seed)
        rng.shuffle(patient_ids)
        keep = set(patient_ids[: max_patients])
        out = [r for r in out if (r.get("patient_id") or "").strip() in keep]
    if max_images and max_images > 0:
        out = out[: max_images]
    return out


def try_run_training(script: Path, args: list[str]) -> int:
    import subprocess

    if not script.exists():
        print(f"[train] script not found, skipping: {script}")
        return 127
    cmd = [sys.executable, str(script)] + args
    print(f"[train] running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def _run_timed(
    stage_name: str,
    fn,
) -> StageResult:
    start = perf_counter()
    details = fn()
    elapsed = perf_counter() - start
    return StageResult(name=stage_name, elapsed_s=elapsed, details=details)


def main() -> None:
    args = parse_args_with_config()
    run_dir = make_run_dir(args.runs_root, args.run_name)
    paths = build_paths(run_dir)
    stage_results: list[StageResult] = []

    # Ensure dirs
    for p in [paths.data_dir, paths.reports_dir, paths.images_dir, paths.cropped_dir, paths.cleaned_dir, paths.augmented_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Save run config snapshot for reproducibility
    run_cfg_path = paths.run_dir / "run_config.json"
    with run_cfg_path.open("w", encoding="utf-8") as f:
        payload = dict(vars(args))
        payload["python_version"] = sys.version
        payload["platform"] = platform.platform()
        json.dump(payload, f, indent=2, default=str)

    # 1) Get starting manifest
    rows: list[dict[str, str]]
    if args.toy:
        def _stage_toy() -> dict[str, int | str | bool]:
            base_manifest = generate_toy_dataset(
                paths.data_dir,
                seed=args.seed,
                spec=ToyDataSpec(
                    n_patients=args.toy_patients,
                    images_per_patient=args.toy_images_per_patient,
                    image_size=args.toy_image_size,
                ),
            )
            loaded = read_manifest_csv(base_manifest)
            write_manifest_csv(loaded, paths.manifest_base)
            return {"rows": len(loaded), "manifest": str(paths.manifest_base)}

        stage_results.append(_run_timed("toy_manifest", _stage_toy))
        rows = read_manifest_csv(paths.manifest_base)
    else:
        rows = read_manifest_csv(args.in_manifest)
        write_manifest_csv(rows, paths.manifest_base)

        if args.image_roots and not args.skip_attach:
            def _stage_attach() -> dict[str, int | str | bool]:
                nonlocal rows
                rows, n_attached, n_missing = attach_image_paths(
                    rows,
                    image_roots=args.image_roots,
                    overwrite=False,
                    strict=False,
                )
                return {"attached": n_attached, "missing": n_missing, "skipped": False}

            stage_results.append(_run_timed("attach_image_paths", _stage_attach))
        elif args.image_roots and args.skip_attach:
            stage_results.append(
                StageResult(
                    name="attach_image_paths",
                    elapsed_s=0.0,
                    details={"attached": 0, "missing": 0, "skipped": True},
                )
            )
        write_manifest_csv(rows, paths.manifest_base)

        has_any = any((r.get("image_path") or "").strip() != "" for r in rows)
        if not has_any:
            raise RuntimeError(
                "Manifest mode requires image_path values. "
                "Provide a manifest with image_path or pass --image-root (and do not skip attach)."
            )

    assert_manifest_contract(
        rows,
        require_labels=True,
        require_patient_ids=True,
        require_image_paths=True,
        require_splits=False,
    )

    # Optional subsetting for fast iteration
    rows = subset_rows(rows, seed=args.seed, max_patients=args.max_patients, max_images=args.max_images)
    write_manifest_csv(rows, paths.manifest_base)
    assert_manifest_contract(
        rows,
        require_labels=True,
        require_patient_ids=True,
        require_image_paths=True,
        require_splits=False,
    )

    # 2) Crop / cleanup
    processed_rows: list[dict[str, str]] = []

    def _stage_preprocess() -> dict[str, int | str | bool]:
        nonlocal processed_rows
        n_cropped = 0
        n_cleaned = 0
        for r in rows:
            img_path = (r.get("image_path") or "").strip()
            if img_path == "":
                processed_rows.append(dict(r))
                continue
            src = Path(img_path)
            rr = dict(r)

            if not args.skip_crop:
                cropped_path = paths.cropped_dir / src.name
                crop_image_path(src, cropped_path, size=args.crop_size, fill=0)
                src = cropped_path
                n_cropped += 1

            if not args.skip_cleanup:
                cleaned_path = paths.cleaned_dir / src.name
                remove_white_edges_file(src, cleaned_path, white_value=None, replacement_value=0, overwrite=True)
                src = cleaned_path
                n_cleaned += 1

            rr["image_path"] = str(src)
            processed_rows.append(rr)
        return {"rows": len(processed_rows), "cropped": n_cropped, "cleaned": n_cleaned}

    stage_results.append(_run_timed("preprocess", _stage_preprocess))

    write_manifest_csv(processed_rows, paths.manifest_processed)
    assert_manifest_contract(
        processed_rows,
        require_labels=True,
        require_patient_ids=True,
        require_image_paths=True,
        require_splits=False,
    )

    # 3) Augment -> new manifest (duplicated rows)
    augmented_rows: list[dict[str, str]] = []

    def _stage_augment() -> dict[str, int | str | bool]:
        if args.skip_augment or args.augment_n == 0:
            return {"augmented_rows": 0, "skipped": True}
        aug_cfg = AugmentConfig(seed=args.seed)
        for r in processed_rows:
            img_path = (r.get("image_path") or "").strip()
            if img_path == "":
                continue
            src = Path(img_path)
            base_id = (r.get("sample_id") or src.stem).strip() or src.stem
            outs = augment_file(
                src,
                paths.augmented_dir,
                n=args.augment_n,
                cfg=aug_cfg,
                prefix=base_id,
                overwrite=True,
            )
            for i, outp in enumerate(outs, start=1):
                rr = dict(r)
                rr["sample_id"] = f"{base_id}__aug{i:02d}"
                rr["image_path"] = str(outp)
                augmented_rows.append(rr)
        return {"augmented_rows": len(augmented_rows), "skipped": False}

    stage_results.append(_run_timed("augment", _stage_augment))

    # keep also original rows (common practice)
    all_for_split = processed_rows + augmented_rows
    write_manifest_csv(all_for_split, paths.manifest_aug)
    assert_manifest_contract(
        all_for_split,
        require_labels=True,
        require_patient_ids=True,
        require_image_paths=True,
        require_splits=False,
    )

    # 4) Assign patient-level splits
    split_rows = assign_patient_splits(
        all_for_split, seed=args.seed, fractions=SplitFractions(val=args.val_frac, test=args.test_frac)
    )
    assert_no_patient_leakage(split_rows)
    assert_manifest_contract(
        split_rows,
        require_labels=True,
        require_patient_ids=True,
        require_image_paths=True,
        require_splits=True,
    )
    write_manifest_csv(split_rows, paths.manifest_splits)
    split_counts = {"train": 0, "val": 0, "test": 0}
    for row in split_rows:
        split_counts[row["split"]] += 1
    if args.strict_splits and any(v == 0 for v in split_counts.values()):
        raise RuntimeError(f"Strict split policy failed: empty split detected. counts={split_counts}")
    if any(v == 0 for v in split_counts.values()):
        print(f"[warn] some splits are empty for this run: {split_counts}")
    stage_results.append(
        StageResult(
            name="split_assign",
            elapsed_s=0.0,
            details={"train": split_counts["train"], "val": split_counts["val"], "test": split_counts["test"]},
        )
    )

    # 5) Materialize ImageFolder layout
    if not args.skip_materialize:
        def _stage_materialize() -> dict[str, int | str | bool]:
            counts = materialize_imagefolder(
                split_rows,
                layout=ImageFolderLayout(root=paths.imagefolder_root),
                mode=args.materialize_mode,  # type: ignore[arg-type]
            )
            return {"train": counts["train"], "val": counts["val"], "test": counts["test"], "skipped": False}

        materialize_result = _run_timed("materialize", _stage_materialize)
        stage_results.append(materialize_result)
        print(
            "OK: materialized ImageFolder counts: "
            f"{materialize_result.details.get('train')},"
            f"{materialize_result.details.get('val')},"
            f"{materialize_result.details.get('test')}"
        )
    else:
        stage_results.append(
            StageResult(
                name="materialize",
                elapsed_s=0.0,
                details={"train": 0, "val": 0, "test": 0, "skipped": True},
            )
        )

    # 6) EDA report (CSV)
    if not args.skip_eda:
        def _stage_eda() -> dict[str, int | str | bool]:
            stats = compute_stats_from_manifest_rows(split_rows, image_path_col="image_path")
            stats_csv = paths.reports_dir / "image_stats.csv"
            write_stats_csv(stats_csv, stats)
            return {"rows": len(stats), "csv": str(stats_csv), "skipped": False}

        eda_result = _run_timed("eda", _stage_eda)
        stage_results.append(eda_result)
        print(f"OK: wrote EDA stats -> {eda_result.details.get('csv')}")
    else:
        stage_results.append(
            StageResult(name="eda", elapsed_s=0.0, details={"rows": 0, "csv": "", "skipped": True})
        )

    # 7) Optional training
    train_returncodes: dict[str, int] = {}
    if args.train_resnet:
        rc = try_run_training(
            REPO_ROOT / "training" / "model_development_and_evaluation.py",
            ["--data-root", str(paths.imagefolder_root), "--output-dir", str(paths.run_dir / "train_resnet")],
        )
        train_returncodes["resnet"] = rc
    if args.train_convnext:
        rc = try_run_training(
            REPO_ROOT / "training" / "model_convnext_absolute.py",
            ["--data-root", str(paths.imagefolder_root), "--output-dir", str(paths.run_dir / "train_convnext")],
        )
        train_returncodes["convnext"] = rc
    if args.fail_on_train_error and any(v != 0 for v in train_returncodes.values()):
        raise RuntimeError(f"Training failed under strict mode: returncodes={train_returncodes}")
    stage_results.append(
        StageResult(
            name="training",
            elapsed_s=0.0,
            details={"requested": bool(args.train_resnet or args.train_convnext), **train_returncodes},
        )
    )

    summary = {
        "run_dir": str(paths.run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage_results": [
            {"name": s.name, "elapsed_s": round(s.elapsed_s, 6), "details": s.details} for s in stage_results
        ],
    }
    summary_path = paths.reports_dir / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"OK: pipeline complete -> {paths.run_dir}")
    print(f"Run config saved -> {run_cfg_path}")
    print(f"Run summary saved -> {summary_path}")


if __name__ == "__main__":
    main()


