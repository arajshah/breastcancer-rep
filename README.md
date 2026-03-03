## breastcancer-rep

**Manifest-driven** ML pipeline for **breast cancer classification from mammography** using the **CBIS-DDSM Mass** subset (binary: **BENIGN vs MALIGNANT**).

Dataset reference: [CBIS-DDSM (TCIA collection)](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)

### What “revival complete” means in this repo

After the revival is complete, the **supported workflow** is:
- **One canonical `manifest.csv`** is the source of truth for every sample (ids, label, image path, split).
- **Splits are assigned at the patient level** (anti-leakage).
- **CLIs are reproducible** (no Colab/Drive hard-coded paths).
- **Smoke tests run without downloading CBIS-DDSM** (toy images + manifest).

Legacy Colab-style scripts are kept for reference but are **not the recommended path**.

### Install (recommended)

From repo root:

```bash
python -m pip install -e ".[dev,analysis,ml]"
```

After installing, run CLIs directly with `python scripts/<name>.py ...`.

### Quickstart (no dataset download required)

This generates a tiny synthetic dataset under `--workdir` and validates:
- manifest schema
- patient-level split integrity (no leakage)
- that every `image_path` exists

```bash
python scripts/smoke_pipeline.py --workdir ./runs/smoke/quickstart --seed 42
```

Outputs:
- `./runs/smoke/quickstart/images/*.png`
- `./runs/smoke/quickstart/manifest.csv`
- `./runs/smoke/quickstart/manifest_with_splits.csv`

### Build a manifest from CBIS-DDSM metadata (no images required)

Download the CBIS-DDSM **Mass** case description CSVs from TCIA and run:

```bash
python scripts/build_manifest_from_cbis_csv.py \
  --mass-train-csv ./mass_case_description_train_set.csv \
  --mass-test-csv ./mass_case_description_test_set.csv \
  --out-manifest ./manifest_cbis_mass.csv
```

At this stage, `image_path` is intentionally left empty; you can still validate labels and splits.

### Attach image paths (when images exist)

If your manifest has empty `image_path`, attach file paths by scanning one or more roots:

```bash
python scripts/attach_image_paths.py \
  --in-manifest ./manifest_cbis_mass.csv \
  --out-manifest ./manifest_cbis_mass_attached.csv \
  --image-root ./processed_png/full_png \
  --image-root ./processed_png/cropped_png
```

### Assign splits + materialize an ImageFolder layout

1) Assign patient-level splits into the manifest:

```bash
python scripts/assign_splits.py \
  --in-manifest ./manifest_cbis_mass_attached.csv \
  --out-manifest ./manifest_cbis_mass_splits.csv \
  --seed 42 --val-frac 0.1 --test-frac 0.1
```

2) Materialize a torchvision `ImageFolder` layout (symlinks by default):

```bash
python scripts/materialize_imagefolder.py \
  --manifest ./manifest_cbis_mass_splits.csv \
  --output-root ./dataset_splits \
  --mode symlink
```

Resulting layout:

```
dataset_splits/
  train/{BENIGN,MALIGNANT}/*.png
  val/{BENIGN,MALIGNANT}/*.png
  test/{BENIGN,MALIGNANT}/*.png
```

### End-to-end pipeline runner

For one-command orchestration:

```bash
python scripts/run_pipeline.py --toy --runs-root ./runs/smoke --run-name demo_toy
```

Or from an existing manifest, with optional attach stage:

```bash
python scripts/run_pipeline.py \
  --in-manifest ./manifest_cbis_mass.csv \
  --image-root ./processed_png/full_png \
  --runs-root ./runs/smoke \
  --run-name demo_real
```

Useful operational flags:
- `--skip-attach`, `--skip-crop`, `--skip-cleanup`, `--skip-augment`, `--skip-materialize`, `--skip-eda`
- `--strict-splits` (fail if any split is empty)
- `--fail-on-train-error` (fail pipeline when training subprocess fails)

Each run now writes:
- `run_config.json` (resolved runtime config)
- `reports/pipeline_summary.json` (stage timings + stage outputs)
- `reports/image_stats.csv` (unless `--skip-eda`)

### Train + evaluate models (on ImageFolder `dataset_splits/`)

ResNet50 baseline (trains + evaluates + writes to `runs/`):

```bash
python training/model_development_and_evaluation.py --data-root ./dataset_splits
```

Evaluate an existing ResNet checkpoint:

```bash
python training/model_evaluation.py \
  --data-root ./dataset_splits \
  --checkpoint ./runs/resnet50_*/model_best.pth \
  --output-dir ./runs/eval_resnet
```

ConvNeXt staged fine-tuning:

```bash
python training/model_convnext_absolute.py \
  --data-root ./dataset_splits \
  --output-dir ./runs/convnext_stage
```

### Manifest conventions (source of truth)

The canonical columns live in `src/breastcancer_rep/manifest.py` and include:
- **`sample_id`**: unique row id
- **`patient_id`**: used for patient-level splitting (anti-leakage)
- **`pathology`**: original string label
- **`label`**: standardized numeric label (`0` benign-like, `1` malignant)
- **`image_path`**: path to the image used for training/eval
- **`split`**: `train` / `val` / `test` (optional until assigned)

Contract checks are enforced by CLIs:
- labels must be `0` or `1`
- `sample_id` must be present (and unique per manifest)
- `patient_id` must be present for split-related stages
- `image_path` is required for processing/materialization stages
- `split` is required for ImageFolder materialization

### Repository layout

- **`src/breastcancer_rep/`**: revival library (manifest IO, splitting, ImageFolder materialization, toy data).
- **`scripts/`**: revival CLIs (smoke pipeline, build/attach manifest, assign splits, materialize ImageFolder, pipeline runner).
- **`training/`**: training/evaluation scripts (ResNet, ConvNeXt).
- **`legacy/colab/`**: legacy Colab preprocessing scripts (kept for provenance and comparison).
- **`tests/`**: unit/smoke tests for splitting + ImageFolder materialization.
- **`runs/smoke/`**: smoke/demo run outputs.
- **`runs/test/`**: test-run outputs and test temporary artifacts.


