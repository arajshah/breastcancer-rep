from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(src: Path, dst: Path) -> None:
    """
    Create a symlink, overwriting existing destination if present.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


