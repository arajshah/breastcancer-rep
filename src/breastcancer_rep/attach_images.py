from __future__ import annotations

from pathlib import Path


def _candidate_names(row: dict[str, str]) -> list[str]:
    """
    Build filename candidates from manifest pointers.
    Priority:
    1) existing image_path filename
    2) source_image_file_path filename
    3) sample_id + .png
    """
    out: list[str] = []
    image_path = (row.get("image_path") or "").strip()
    source_image = (row.get("source_image_file_path") or "").strip()
    sample_id = (row.get("sample_id") or "").strip()

    if image_path:
        out.append(Path(image_path).name)
    if source_image:
        out.append(Path(source_image).name)
    if sample_id:
        out.append(f"{sample_id}.png")

    # Preserve order but de-duplicate.
    seen: set[str] = set()
    deduped: list[str] = []
    for name in out:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def _build_file_index(image_roots: list[Path]) -> dict[str, Path]:
    """
    Index png files by basename across provided roots.
    If duplicates exist, first encountered path wins.
    """
    idx: dict[str, Path] = {}
    for root in image_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.png"):
            idx.setdefault(p.name, p)
    return idx


def attach_image_paths(
    rows: list[dict[str, str]],
    *,
    image_roots: list[Path],
    overwrite: bool = False,
    strict: bool = False,
) -> tuple[list[dict[str, str]], int, int]:
    """
    Attach image_path values by matching candidate filenames to image roots.

    Returns: (updated_rows, n_attached, n_missing)
    """
    if not image_roots:
        raise ValueError("image_roots must be non-empty.")
    idx = _build_file_index(image_roots)

    out: list[dict[str, str]] = []
    n_attached = 0
    n_missing = 0
    for row in rows:
        rr = dict(row)
        current = (rr.get("image_path") or "").strip()
        if current and not overwrite:
            out.append(rr)
            continue

        chosen: Path | None = None
        for name in _candidate_names(rr):
            chosen = idx.get(name)
            if chosen is not None:
                break

        if chosen is None:
            n_missing += 1
        else:
            rr["image_path"] = str(chosen.resolve())
            n_attached += 1
        out.append(rr)

    if strict and n_missing > 0:
        raise RuntimeError(f"Failed to attach {n_missing} manifest rows to image files.")
    return out, n_attached, n_missing

