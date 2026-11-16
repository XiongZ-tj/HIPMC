"Batch-solve the frozen small industrial instances with resumable outputs."
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
SOLVER = Path(__file__).with_name("milp_exact_solver.py")
SUPPORTED_JOB_TIERS = (10, 20)


def _read_result(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def validate_manifest_tiers(rows: list[dict]) -> tuple[int, ...]:
    "Reject manifests before launching any Gurobi processes."

    observed_tiers = {int(row["n_jobs"]) for row in rows}
    unsupported_tiers = sorted(observed_tiers - set(SUPPORTED_JOB_TIERS))
    if unsupported_tiers:
        raise ValueError(
            "The small-instance manifest contains retired or unsupported "
            f"job-count tiers {unsupported_tiers}. Regenerate it with "
            "MILP/generate_small_industrial_instances.py; the formal MILP "
            "benchmark now retains only 10- and 20-job instances."
        )
    return tuple(sorted(observed_tiers))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=WORKSPACE / "datasets/industrial_case_small/manifest.csv")
    parser.add_argument(
        "--dataset-root", type=Path,
        default=WORKSPACE / "datasets/industrial_case_small")
    parser.add_argument(
        "--output", type=Path,
        default=WORKSPACE / "runs/milp_small_industrial")
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument(
        "--scene", default="aerospace_heat_treatment_case",
        choices=("synthetic_reference", "aerospace_heat_treatment_case"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet-solver", action="store_true")
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(
            f"Small-instance manifest not found: {args.manifest}. "
            "Run MILP/generate_small_industrial_instances.py first.")
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"The manifest is empty: {args.manifest}")
    validate_manifest_tiers(rows)

    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"instances={len(rows)} time_limit={args.time_limit:.0f}s "
        f"scene={args.scene} python={sys.executable}", flush=True)
    summary: list[dict] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        job_file = args.dataset_root / row["job_file"]
        if not job_file.is_file():
            raise FileNotFoundError(f"Manifest job file not found: {job_file}")
        command = [
            sys.executable, str(SOLVER), "--job-file", str(job_file),
            "--time-limit", str(args.time_limit), "--scene", args.scene,
            "--output", str(args.output), "--mip-gap", str(args.mip_gap),
            "--solver-seed", str(args.solver_seed),
        ]
        if args.threads is not None:
            command.extend(("--threads", str(args.threads)))
        if args.force:
            command.append("--force")
        if args.quiet_solver:
            command.append("--quiet")

        label = f"jobs-{row['n_jobs']}/instance-{int(row['instance_number']):03d}"
        print(f"[{index:02d}/{len(rows)}] START {label}", flush=True)
        instance_started = time.perf_counter()
        process = subprocess.run(
            command, cwd=WORKSPACE, text=True, capture_output=True,
            timeout=max(300.0, args.time_limit + 300.0))
        elapsed = time.perf_counter() - instance_started
        stem = f"{job_file.parent.name}_{job_file.stem}"
        result_path = args.output / f"{stem}_milp.json"
        log_path = args.output / f"{stem}_solver.log"
        log_path.write_text(
            process.stdout + ("\n[stderr]\n" + process.stderr if process.stderr else ""),
            encoding="utf-8")
        result = _read_result(result_path)
        entry = {
            "n_jobs": int(row["n_jobs"]),
            "instance_number": int(row["instance_number"]),
            "job_file": row["job_file"],
            "process_return_code": process.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "solver_log": str(log_path),
        }
        if process.returncode == 0 and result is not None:
            entry.update({
                "status": result.get("status_name"),
                "has_incumbent": result.get("has_incumbent"),
                "is_proven_global_production_optimum": result.get(
                    "is_proven_global_production_optimum"),
                "objective_TT": result.get("objective"),
                "objective_bound": result.get("obj_bound"),
                "mip_gap": result.get("mip_gap"),
                "node_count": result.get("node_count"),
            })
        else:
            entry.update({
                "status": "PROCESS_ERROR",
                "error": (process.stderr or process.stdout)[-1000:],
            })
        summary.append(entry)
        objective = entry.get("objective_TT")
        objective_text = "-" if objective is None else f"{float(objective):.3f}"
        proof = "PROVEN" if entry.get(
            "is_proven_global_production_optimum") else "NOT_PROVEN"
        print(
            f"[{index:02d}/{len(rows)}] END   {label} "
            f"status={entry['status']} proof={proof} TT={objective_text} "
            f"gap={entry.get('mip_gap')} t={elapsed:.1f}s",
            flush=True)

    fields = tuple(dict.fromkeys(key for row in summary for key in row))
    summary_path = args.output / "summary_all.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    total = time.perf_counter() - started
    print(f"completed={len(summary)} elapsed_minutes={total / 60:.2f}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
