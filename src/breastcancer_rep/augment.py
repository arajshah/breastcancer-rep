from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance


@dataclass(frozen=True)
class AugmentConfig:
    seed: int = 42
    max_rotation_deg: float = 20.0
    hflip: bool = True
    vflip: bool = True
    brightness_jitter: float = 0.15  # +/- fraction
    contrast_jitter: float = 0.15  # +/- fraction


def _jitter_factor(rng: random.Random, jitter: float) -> float:
    if jitter <= 0:
        return 1.0
    lo = 1.0 - jitter
    hi = 1.0 + jitter
    return rng.uniform(lo, hi)


def augment_once(img: Image.Image, *, cfg: AugmentConfig, salt: int) -> Image.Image:
    """
    Deterministic single augmentation derived from cfg.seed + salt.
    Uses Pillow-only transforms.
    """
    rng = random.Random(cfg.seed + salt)

    out = img.copy()
    # Ensure consistent mode for transforms (keep grayscale if possible)
    if out.mode not in {"L", "RGB"}:
        out = out.convert("L")

    # flips
    if cfg.hflip and rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if cfg.vflip and rng.random() < 0.25:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)

    # rotation (expand=False keeps size)
    if cfg.max_rotation_deg > 0:
        angle = rng.uniform(-cfg.max_rotation_deg, cfg.max_rotation_deg)
        out = out.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)

    # brightness/contrast jitter
    b = _jitter_factor(rng, cfg.brightness_jitter)
    c = _jitter_factor(rng, cfg.contrast_jitter)
    out = ImageEnhance.Brightness(out).enhance(b)
    out = ImageEnhance.Contrast(out).enhance(c)

    return out


def augment_file(
    input_path: Path,
    output_dir: Path,
    *,
    n: int,
    cfg: AugmentConfig,
    prefix: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """
    Write N augmented variants of input_path into output_dir.
    Returns list of written paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with Image.open(input_path) as img:
        base = prefix or input_path.stem
        for i in range(1, n + 1):
            out_path = output_dir / f"{base}_aug{i:02d}.png"
            if out_path.exists() and not overwrite:
                written.append(out_path)
                continue
            aug = augment_once(img, cfg=cfg, salt=i)
            aug.save(out_path)
            written.append(out_path)
    return written



