from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow running tests without installing the src-layout package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breastcancer_rep.imagefolder import ImageFolderLayout, materialize_imagefolder  # noqa: E402
from breastcancer_rep.manifest import read_manifest_csv  # noqa: E402
from breastcancer_rep.splitting import SplitFractions, assert_no_patient_leakage, assign_patient_splits  # noqa: E402
from breastcancer_rep.toydata import ToyDataSpec, generate_toy_dataset  # noqa: E402


class TestImageFolderMaterialize(unittest.TestCase):
    def test_materialize_symlinks(self) -> None:
        workdir = REPO_ROOT / ".test_run_materialize"
        if workdir.exists():
            # simple cleanup
            for p in sorted(workdir.rglob("*"), reverse=True):
                if p.is_file() or p.is_symlink():
                    p.unlink()
                else:
                    p.rmdir()
            workdir.rmdir()

        manifest_path = generate_toy_dataset(workdir, seed=1, spec=ToyDataSpec(n_patients=10, images_per_patient=2))
        rows = read_manifest_csv(manifest_path)
        rows = assign_patient_splits(rows, seed=1, fractions=SplitFractions(val=0.2, test=0.2))
        assert_no_patient_leakage(rows)

        out_root = workdir / "dataset_splits"
        counts = materialize_imagefolder(rows, layout=ImageFolderLayout(root=out_root), mode="symlink")
        self.assertEqual(sum(counts.values()), len(rows))

        # Basic structure checks
        self.assertTrue((out_root / "train").exists())
        self.assertTrue((out_root / "val").exists())
        self.assertTrue((out_root / "test").exists())
        self.assertTrue((out_root / "train" / "BENIGN").exists())
        self.assertTrue((out_root / "train" / "MALIGNANT").exists())

        # Ensure files are present
        any_file = next((out_root / "train").rglob("*.png"))
        self.assertTrue(any_file.exists())
        self.assertTrue(any_file.is_symlink())


if __name__ == "__main__":
    unittest.main()


