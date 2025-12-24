from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPipelineConfigSubset(unittest.TestCase):
    def test_config_and_subsetting(self) -> None:
        tmp = REPO_ROOT / "runs_test_tmp_cfg"
        if tmp.exists():
            for p in sorted(tmp.rglob("*"), reverse=True):
                if p.is_file() or p.is_symlink():
                    p.unlink()
                else:
                    p.rmdir()
            tmp.rmdir()
        tmp.mkdir(parents=True, exist_ok=True)

        cfg_path = tmp / "cfg.json"
        cfg = {
            "toy": True,
            "runs_root": str(tmp),
            "run_name": "cfg_run",
            "toy_patients": 10,
            "toy_images_per_patient": 1,
            "toy_image_size": 64,
            "crop_size": 64,
            "augment_n": 1,
            "seed": 5,
            "max_patients": 3,
            "max_images": 0,
        }
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_pipeline.py"), "--config", str(cfg_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, msg=res.stderr)

        run_dir = tmp / "cfg_run"
        manifest = run_dir / "data" / "manifest_base.csv"
        self.assertTrue(manifest.exists())
        # Ensure run_config snapshot exists
        self.assertTrue((run_dir / "run_config.json").exists())


if __name__ == "__main__":
    unittest.main()



