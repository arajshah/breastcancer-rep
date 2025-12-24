## breastcancer-rep

This repository is an ML pipeline for **breast cancer classification from mammography** using the **CBIS-DDSM Mass** subset (binary: **BENIGN vs MALIGNANT**).

Dataset reference: [CBIS-DDSM (TCIA collection)](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)

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

### Crop lesion-centered patches (folder or manifest)

```bash
python scripts/crop_centered.py \
  --input-dir ./some_pngs \
  --output-dir ./cropped_pngs \
  --size 598
```

Or update a manifest to point at cropped images:

```bash
python scripts/crop_centered.py \
  --in-manifest ./manifest_with_images.csv \
  --output-dir ./cropped_pngs \
  --out-manifest ./manifest_with_images_cropped.csv \
  --size 598
```

### Cleanup + augmentation (Pillow-only)

Remove pure-white pixels (common “white edge” artifact) from a folder:

```bash
python scripts/remove_white_edges.py \
  --input-dir ./cropped_pngs \
  --output-dir ./cropped_clean \
  --white-value -1
```

Generate deterministic augmentations (no keras/numpy required):

```bash
python scripts/augment_images.py \
  --input-dir ./cropped_clean \
  --output-dir ./augmented \
  --n 5 \
  --seed 42
```

### EDA / dataset stats (subset-friendly)

Compute per-image width/height and nonzero pixel % (useful for plotting and QC):

```bash
python scripts/eda_image_stats.py \
  --input-dir ./cropped_pngs \
  --output-dir ./reports
```

Or run from a manifest (uses label/pathology/split if present):

```bash
python scripts/eda_image_stats.py \
  --in-manifest ./manifest_with_images.csv \
  --output-dir ./reports
```

### R plots (optional)

If you have R with `ggplot2` and `dplyr` installed, you can generate publication-quality PDFs from `image_stats.csv`:

```bash
Rscript R_visualization.R --stats-csv ./reports/image_stats.csv --output-dir ./reports
```

### One-command pipeline (toy mode)

Run the full pipeline on a small synthetic dataset (no CBIS-DDSM download required). This produces:
- `runs/<run>/data/manifest_splits.csv`
- `runs/<run>/data/dataset_splits/` (ImageFolder layout)
- `runs/<run>/reports/image_stats.csv`

```bash
python scripts/run_pipeline.py --toy
```

### Manifest mode

If you have CBIS-DDSM metadata CSVs and a local folder of PNGs (e.g., produced by `prepare_kaggle_jpegs.py`), you can:

```bash
# 1) Build a label manifest from TCIA mass CSVs
python scripts/build_manifest_from_cbis_csv.py \
  --mass-train-csv ./mass_case_description_train_set.csv \
  --mass-test-csv ./mass_case_description_test_set.csv \
  --out-manifest ./manifest_cbis_mass.csv

# 2) Attach local PNG paths by matching patient_id prefixes
python scripts/attach_image_paths.py \
  --in-manifest ./manifest_cbis_mass.csv \
  --images-dir ./processed_png/cropped_png \
  --out-manifest ./manifest_cbis_mass_with_images.csv

# 3) Run the pipeline starting from your manifest
python scripts/run_pipeline.py --in-manifest ./manifest_cbis_mass_with_images.csv
```

### Repo structure

- `src/breastcancer_rep/`: reusable library code (manifest, splitting, validation)
- `scripts/`: CLI-style entrypoints that call the library
- `tests/`: unit/smoke tests for correctness and anti-leakage checks


