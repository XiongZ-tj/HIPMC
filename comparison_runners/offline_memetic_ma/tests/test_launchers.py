"""Short end-to-end CLI tests; all outputs live in an auto-cleaned temp folder."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="ma_launcher_tests_")
        cls.base = Path(cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def execute(self, script, name, threads=1, refinement=3):
        output = self.base / name
        command = [sys.executable, "-B", str(ROOT / script), "--population", "10",
                   "--generations", "5", "--refinement-generations", str(refinement),
                   "--local-search-evaluations", "20", "--local-search-interval", "1",
                   "--threads", str(threads), "--instance-limit", "2",
                   "--output-dir", str(output), "--build-dir", str(self.base / (name + "-build"))]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        with (output / "plan_manifest.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        for row in rows:
            with (output / row["convergence_file"]).open(newline="", encoding="utf-8") as handle:
                convergence = list(csv.DictReader(handle))
            baseline = float(convergence[-1]["baseline_best_ttd"])
            self.assertLessEqual(float(row["objective_total_tardiness"]), baseline + 1e-8)
            self.assertEqual(sum(r["search_phase"] == "joint_neighborhood_refinement" for r in convergence), refinement)
        return command, output, rows

    def test_generated_resume_and_parallelism(self):
        command, output, rows = self.execute("comparison_runners/run_offline_memetic_planning.py", "generated")
        hashes = {row["plan_file"]: hashlib.sha256((output / row["plan_file"]).read_bytes()).hexdigest() for row in rows}
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        for name, digest in hashes.items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
        incompatible = list(command)
        incompatible[incompatible.index("--refinement-generations") + 1] = "4"
        rejected = subprocess.run(incompatible, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(rejected.returncode, 0, rejected.stderr)
        for name, digest in hashes.items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
        # Reuse the compiled executable while changing only instance-level threads.
        other = self.base / "parallel"
        parallel = list(command)
        parallel[parallel.index("--threads") + 1] = "2"
        parallel[parallel.index("--output-dir") + 1] = str(other)
        subprocess.run(parallel, cwd=ROOT, check=True, capture_output=True, text=True)
        with (other / "plan_manifest.csv").open(newline="", encoding="utf-8") as handle:
            parallel_rows = list(csv.DictReader(handle))
        for serial, concurrent in zip(rows, parallel_rows, strict=True):
            for key in ("objective_total_tardiness", "search_evaluations", "generations", "optimizer_seed"):
                self.assertEqual(serial[key], concurrent[key])

    def test_industrial_source_adaptation(self):
        self.execute("comparison_runners/industrial_case/run_offline_memetic_planning.py", "industrial")

    def test_zero_refinement(self):
        self.execute("comparison_runners/run_offline_memetic_planning.py", "no-refinement", refinement=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
