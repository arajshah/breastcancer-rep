## breastcancer-rep

This repository provides a clear, reproducible workflow for building a **breast cancer classification** dataset and training pipeline from mammography images.

It is designed around the **CBIS-DDSM Mass** subset and a binary target:
- **BENIGN** (`0`)
- **MALIGNANT** (`1`)

Dataset reference: [CBIS-DDSM (TCIA collection)](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)

## What This Repo Does

With this repo, you can:
- build a canonical manifest from official CBIS metadata
- attach your local image files to that manifest
- preprocess images (crop, cleanup, augmentation)
- create patient-level train/val/test splits to avoid leakage
- materialize a PyTorch-compatible `ImageFolder` dataset
- run training/evaluation and generate run reports

The manifest is the central data contract across all stages.

## Repository Structure

- `src/breastcancer_rep/` - reusable pipeline library code
- `scripts/` - CLI entrypoints for data/pipeline stages
- `training/` - training and evaluation scripts
- `tests/` - unit and integration tests
- `runs/smoke/` - smoke/demo outputs
- `runs/test/` - test-run outputs

## Install

From repo root:

```bash
python -m pip install -e ".[dev,analysis,ml]"
```

## Official Data Setup

1. Download CBIS-DDSM Mass case description CSVs from TCIA:
   - `mass_case_description_train_set.csv`
   - `mass_case_description_test_set.csv`
2. Prepare your image files as PNGs in one or more local folders.
3. Recommended local layout:

```text
data/
  cbis/
    csv/
      mass_case_description_train_set.csv
      mass_case_description_test_set.csv
    images/
      ... png files ...
  manifests/
```

## Run The Pipeline (Official Data)

### 1) Build canonical manifest from official CSVs

```bash
python scripts/build_manifest_from_cbis_csv.py \
  --mass-train-csv data/cbis/csv/mass_case_description_train_set.csv \
  --mass-test-csv data/cbis/csv/mass_case_description_test_set.csv \
  --out-manifest data/manifests/manifest_cbis_mass.csv
```

### 2) Attach local image paths

```bash
python scripts/attach_image_paths.py \
  --in-manifest data/manifests/manifest_cbis_mass.csv \
  --out-manifest data/manifests/manifest_cbis_mass_attached.csv \
  --image-root data/cbis/images \
  --strict
```

### 3) Run full processing pipeline

```bash
python scripts/run_pipeline.py \
  --in-manifest data/manifests/manifest_cbis_mass_attached.csv \
  --runs-root runs/smoke \
  --run-name cbis_real \
  --crop-size 128 \
  --augment-n 2 \
  --val-frac 0.1 \
  --test-frac 0.1 \
  --materialize-mode symlink
```

### 4) (Optional) Train and evaluate

Train:

```bash
python training/model_development_and_evaluation.py \
  --data-root runs/smoke/cbis_real/data/dataset_splits \
  --output-dir runs/smoke/cbis_real/train_resnet
```

Evaluate checkpoint:

```bash
python training/model_evaluation.py \
  --data-root runs/smoke/cbis_real/data/dataset_splits \
  --checkpoint runs/smoke/cbis_real/train_resnet/model_best.pth \
  --output-dir runs/smoke/cbis_real/eval_resnet
```

## Pipeline Outputs

Each pipeline run writes a structured run directory:
- `run_config.json` - resolved runtime configuration
- `reports/pipeline_summary.json` - per-stage status and timing
- `reports/image_stats.csv` - image-level stats
- `data/dataset_splits/` - train/val/test ImageFolder layout
- stage manifests:
  - `data/manifest_base.csv`
  - `data/manifest_processed.csv`
  - `data/manifest_aug.csv`
  - `data/manifest_splits.csv`

## Manifest Contract (Enforced)

Core fields:
- `sample_id` (unique row id)
- `patient_id` (used for patient-level splitting)
- `pathology` (original string label)
- `label` (`0` benign-like, `1` malignant)
- `image_path` (resolved local image path)
- `split` (`train`, `val`, `test` once assigned)

Validation is enforced by CLI stages (schema + value checks + split leakage checks).

## Operational Flags (Runner)

`scripts/run_pipeline.py` supports:
- stage toggles: `--skip-attach`, `--skip-crop`, `--skip-cleanup`, `--skip-augment`, `--skip-materialize`, `--skip-eda`
- strictness controls: `--strict-splits`, `--fail-on-train-error`

## Quick Verification (Toy Data)

If you want to verify installation quickly without official data:

```bash
python scripts/smoke_pipeline.py --workdir runs/smoke/quickstart --seed 42
```


