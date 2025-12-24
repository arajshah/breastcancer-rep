## breastcancer-rep (revival in progress)

This repository is an ML pipeline for **breast cancer classification from mammography** using the **CBIS-DDSM Mass** subset (binary: **BENIGN vs MALIGNANT**).

The original work includes:
- **Metadata → label extraction** (CBIS-DDSM case description CSVs)
- **Image conversion/prep** (DICOM/JPEG → PNG, ROI mapping, cropping, augmentation, cleanup)
- **Dataset splits** (train/val/test folder layouts for Torch `ImageFolder`)
- **Model training/evaluation** (ResNet/ConvNeXt variants) and figure export (ROC/CM PDFs)

Dataset reference: [CBIS-DDSM (TCIA collection)](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)

### What’s changing (revitalization goals)

The repo is being upgraded to be:
- **Reproducible**: no hard-coded local/Colab paths
- **Manifest-driven**: one canonical `manifest.csv` describing every sample and split
- **Testable**: smoke tests that run without downloading the full dataset
- **Showcaseable**: clean CLI entrypoints and structured outputs (`reports/`, `runs/`)

### Quickstart (no dataset download required)

Run the smoke pipeline which generates a tiny synthetic dataset + manifest and validates:
- file layout invariants
- split integrity
- label encoding

```bash
python scripts/smoke_pipeline.py --workdir ./.smoke_run --seed 42
```

### Build a manifest from CBIS-DDSM metadata (no images required)

Download the **Mass** case description CSVs from TCIA (see dataset page) and run:

```bash
python scripts/build_manifest_from_cbis_csv.py \
  --mass-train-csv ./mass_case_description_train_set.csv \
  --mass-test-csv ./mass_case_description_test_set.csv \
  --out-manifest ./manifest_cbis_mass.csv
```

### Assign splits + materialize an ImageFolder layout (when images exist)

```bash
python scripts/assign_splits.py \
  --in-manifest ./manifest_cbis_mass.csv \
  --out-manifest ./manifest_cbis_mass_splits.csv \
  --seed 42 --val-frac 0.1 --test-frac 0.1

python scripts/materialize_imagefolder.py \
  --manifest ./manifest_cbis_mass_splits.csv \
  --output-root ./dataset_splits \
  --mode symlink
```

### Train + evaluate models (ImageFolder `dataset_splits/`)

ResNet50-based baseline (trains + evaluates + saves PDFs under `runs/`):

```bash
python model_development_and_evaluation.py \
  --data-root ./dataset_splits \
  --epochs 20 \
  --batch-size 16 \
  --image-size 224
```

Evaluate an existing ResNet checkpoint:

```bash
python model_evaluation.py \
  --data-root ./dataset_splits \
  --checkpoint ./runs/resnet50_*/model_best.pth \
  --output-dir ./runs/eval_resnet
```

ConvNeXt staged fine-tuning (saves under `runs/convnext_stage/` by default):

```bash
python model_convnext_absolute.py \
  --data-root ./dataset_splits \
  --output-dir ./runs/convnext_stage \
  --image-size 384 \
  --batch-size 64
```

### Repo structure (new)

The upgrade introduces a real Python package:
- `src/breastcancer_rep/`: reusable library code (manifest, splitting, validation)
- `scripts/`: CLI-style entrypoints that call the library
- `tests/`: unit/smoke tests for correctness and anti-leakage checks


