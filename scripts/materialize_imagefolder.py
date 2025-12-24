from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.imagefolder import ImageFolderLayout, materialize_imagefolder  # noqa: E402
from breastcancer_rep.manifest import read_manifest_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Materialize a torchvision ImageFolder directory tree from a manifest with splits."
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--mode", choices=["copy", "symlink"], default="symlink")
    p.add_argument("--label0-name", default="BENIGN")
    p.add_argument("--label1-name", default="MALIGNANT")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest_csv(args.manifest)
    layout = ImageFolderLayout(
        root=args.output_root, class_for_label0=args.label0_name, class_for_label1=args.label1_name
    )
    counts = materialize_imagefolder(rows, layout=layout, mode=args.mode)
    print("OK: materialized ImageFolder")
    print(f"- root: {args.output_root}")
    print(f"- counts: {counts}")


if __name__ == "__main__":
    main()


