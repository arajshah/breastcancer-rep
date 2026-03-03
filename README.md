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

If you want to run scripts without installing, you can use `PYTHONPATH=./src` (examples below).

### Quickstart (no dataset download required)

This generates a tiny synthetic dataset under `--workdir` and validates:
- manifest schema
- patient-level split integrity (no leakage)
- that every `image_path` exists

```bash
PYTHONPATH=./src python src/scripts/smoke_pipeline.py --workdir ./.smoke_run --seed 42
```

Outputs:
- `./.smoke_run/images/*.png`
- `./.smoke_run/manifest.csv`
- `./.smoke_run/manifest_with_splits.csv`

### Build a manifest from CBIS-DDSM metadata (no images required)

Download the CBIS-DDSM **Mass** case description CSVs from TCIA and run:

```bash
PYTHONPATH=./src python src/scripts/build_manifest_from_cbis_csv.py \
  --mass-train-csv ./mass_case_description_train_set.csv \
  --mass-test-csv ./mass_case_description_test_set.csv \
  --out-manifest ./manifest_cbis_mass.csv
```

At this stage, `image_path` is intentionally left empty; you can still validate labels and splits.

### Assign splits + materialize an ImageFolder layout (when images exist)

1) Assign patient-level splits into the manifest:

```bash
PYTHONPATH=./src python src/scripts/assign_splits.py \
  --in-manifest ./manifest_cbis_mass.csv \
  --out-manifest ./manifest_cbis_mass_splits.csv \
  --seed 42 --val-frac 0.1 --test-frac 0.1
```

2) Materialize a torchvision `ImageFolder` layout (symlinks by default):

```bash
PYTHONPATH=./src python src/scripts/materialize_imagefolder.py \
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

### Train + evaluate models (on ImageFolder `dataset_splits/`)

ResNet50 baseline (trains + evaluates + writes to `runs/`):

```bash
python src/training/model_development_and_evaluation.py --data-root ./dataset_splits
```

Evaluate an existing ResNet checkpoint:

```bash
python src/training/model_evaluation.py \
  --data-root ./dataset_splits \
  --checkpoint ./runs/resnet50_*/model_best.pth \
  --output-dir ./runs/eval_resnet
```

ConvNeXt staged fine-tuning:

```bash
python src/training/model_convnext_absolute.py \
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

### Repository layout

- **`src/breastcancer_rep/`**: revival library (manifest IO, splitting, ImageFolder materialization, toy data).
- **`src/scripts/`**: revival CLIs (smoke pipeline, build manifest, assign splits, materialize ImageFolder).
- **`src/training/`**: training/evaluation scripts (ResNet, ConvNeXt).
- **`src/pre-processing/`**: **legacy preprocessing scripts** (Colab-style; kept for reference).
- **`tests/`**: unit/smoke tests for splitting + ImageFolder materialization.
- **`runs*/`, `./.smoke_run*`, `./.test_*`**: generated outputs from local runs/tests (should not be committed).


