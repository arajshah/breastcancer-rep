from __future__ import annotations

import sys
import unittest
from pathlib import Path

# We call the script as a subprocess so we don't rely on imports working inside tests.
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPipelineRunner(unittest.TestCase):
    def test_toy_pipeline_writes_artifacts(self) -> None:
        run_name = ".test_pipeline_run"
        runs_root = REPO_ROOT / "runs_test_tmp"
        if runs_root.exists():
            for p in sorted(runs_root.rglob("*"), reverse=True):
                if p.is_file() or p.is_symlink():
                    p.unlink()
                else:
                    p.rmdir()
            runs_root.rmdir()

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_pipeline.py"),
            "--toy",
            "--runs-root",
            str(runs_root),
            "--run-name",
            run_name,
            "--toy-patients",
            "6",
            "--toy-images-per-patient",
            "1",
            "--toy-image-size",
            "64",
            "--crop-size",
            "64",
            "--augment-n",
            "1",
            "--seed",
            "7",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, msg=res.stderr)

        run_dir = runs_root / run_name
        self.assertTrue((run_dir / "data" / "manifest_splits.csv").exists())
        self.assertTrue((run_dir / "data" / "dataset_splits" / "train").exists())
        self.assertTrue((run_dir / "reports" / "image_stats.csv").exists())


if __name__ == "__main__":
    unittest.main()


