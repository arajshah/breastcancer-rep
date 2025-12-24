## breastcancer-rep (revival in progress)

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

### Repo structure (new)

The upgrade introduces a real Python package:
- `src/breastcancer_rep/`: reusable library code (manifest, splitting, validation)
- `scripts/`: CLI-style entrypoints that call the library
- `tests/`: unit/smoke tests for correctness and anti-leakage checks


