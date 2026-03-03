from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPipelineRunner(unittest.TestCase):
    def test_rejects_unknown_config_keys(self) -> None:
        cfg_path = REPO_ROOT / "runs" / "test" / "run_pipeline_bad_cfg.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps({"toy": True, "unknown_key": 1}), encoding="utf-8")
        cmd = [sys.executable, "scripts/run_pipeline.py", "--config", str(cfg_path)]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Unknown config keys", proc.stderr + proc.stdout)

    def test_writes_summary_with_stage_toggles(self) -> None:
        run_name = "phase4_test_summary"
        cmd = [
            sys.executable,
            "scripts/run_pipeline.py",
            "--toy",
            "--run-name",
            run_name,
            "--runs-root",
            "./runs/test",
            "--skip-eda",
            "--skip-materialize",
            "--augment-n",
            "0",
            "--toy-patients",
            "6",
            "--toy-images-per-patient",
            "1",
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        summary = REPO_ROOT / "runs" / "test" / run_name / "reports" / "pipeline_summary.json"
        self.assertTrue(summary.exists())
        payload = json.loads(summary.read_text(encoding="utf-8"))
        stage_names = {s["name"] for s in payload["stage_results"]}
        self.assertIn("preprocess", stage_names)
        self.assertIn("eda", stage_names)
        self.assertIn("materialize", stage_names)

    def test_strict_splits_fails_on_tiny_data(self) -> None:
        cmd = [
            sys.executable,
            "scripts/run_pipeline.py",
            "--toy",
            "--run-name",
            "phase4_test_strict_split",
            "--runs-root",
            "./runs/test",
            "--toy-patients",
            "4",
            "--toy-images-per-patient",
            "1",
            "--augment-n",
            "0",
            "--strict-splits",
            "--val-frac",
            "0.34",
            "--test-frac",
            "0.34",
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Strict split policy failed", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()

