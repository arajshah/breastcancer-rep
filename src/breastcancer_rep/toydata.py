from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image, ImageDraw

from .manifest import build_manifest_from_records, normalize_pathology, write_manifest_csv


@dataclass(frozen=True)
class ToyDataSpec:
    n_patients: int = 12
    images_per_patient: int = 2
    image_size: int = 128
    malignant_fraction: float = 0.5


def _make_toy_image(seed: int, size: int, malignant: bool) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("L", (size, size), color=rng.randint(30, 70))
    draw = ImageDraw.Draw(img)
    # add a few faint rectangles (texture)
    for _ in range(6):
        x0 = rng.randint(0, size - 1)
        y0 = rng.randint(0, size - 1)
        x1 = min(size - 1, x0 + rng.randint(5, size // 3))
        y1 = min(size - 1, y0 + rng.randint(5, size // 3))
        shade = rng.randint(20, 40)
        draw.rectangle([x0, y0, x1, y1], fill=shade)
    # malignant cases get a bright circular blob
    if malignant:
        cx = rng.randint(size // 3, 2 * size // 3)
        cy = rng.randint(size // 3, 2 * size // 3)
        rad = rng.randint(size // 10, size // 6)
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=220)
    return img


def generate_toy_dataset(workdir: Path, *, seed: int = 42, spec: ToyDataSpec = ToyDataSpec()):
    """
    Generate a tiny dataset + manifest that can exercise the pipeline locally.

    Layout:
      workdir/
        images/*.png
        manifest.csv
    """
    workdir.mkdir(parents=True, exist_ok=True)
    img_dir = workdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    n_malignant = int(round(spec.n_patients * spec.malignant_fraction))
    patient_ids = [f"P{idx:04d}" for idx in range(spec.n_patients)]
    rng = random.Random(seed)
    malignant_patients = set(rng.sample(patient_ids, k=min(n_malignant, len(patient_ids))))

    records = []
    sample_idx = 0
    for pid in patient_ids:
        is_m = pid in malignant_patients
        pathology = "MALIGNANT" if is_m else "BENIGN"
        pathology_norm, label = normalize_pathology(pathology)
        for j in range(spec.images_per_patient):
            sample_id = f"S{sample_idx:06d}"
            sample_idx += 1
            laterality = "LEFT" if (j % 2 == 0) else "RIGHT"
            view = "CC" if (j % 2 == 0) else "MLO"
            img = _make_toy_image(seed=seed + sample_idx, size=spec.image_size, malignant=is_m)
            img_path = img_dir / f"{sample_id}_{pid}_{laterality}_{view}.png"
            img.save(img_path)
            records.append(
                dict(
                    sample_id=sample_id,
                    patient_id=pid,
                    laterality=laterality,
                    view=view,
                    pathology=pathology_norm,
                    label=label,
                    image_path=str(img_path),
                    split=None,
                )
            )

    manifest = build_manifest_from_records(records)
    manifest_path = workdir / "manifest.csv"
    write_manifest_csv(manifest, manifest_path)
    return manifest_path


