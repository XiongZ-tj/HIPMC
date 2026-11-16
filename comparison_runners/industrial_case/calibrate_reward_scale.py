"""Calibrate the industrial-case reward scale on disjoint SDR simulations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from comparison_runners.industrial_case.case_common import (
    WORKSPACE,
    atomic_csv,
    atomic_json,
    build_static_simulator,
    canonical_sha256,
    file_sha256,
    generate_records,
    validate_dataset,
)
from methods.baselines.sdr.shared import RuleController


DEFAULT_OUTPUT = (
    WORKSPACE / "reward_scales"
    / "offline_calibrated_sdr_reward_scale_aerospace_heat_treatment_industrial_case.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-per-load", type=int, default=60)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--arrival-seed-base", type=int, default=91_000_000)
    parser.add_argument("--simulation-seed-base", type=int, default=92_000_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.instances_per_load <= 0 or args.replications <= 0 or args.log_every <= 0:
        raise ValueError("Calibration budgets and --log-every must be positive.")
    dataset = validate_dataset()
    rows = []
    loads = (0.9, 1.0, 1.1)
    total = len(loads) * args.instances_per_load * args.replications
    progress = 0
    for load_index, load in enumerate(loads):
        controller = RuleController("CH", "MMU", "Threshold", 293)
        simulator = build_static_simulator(
            controller, target_load=load,
            arrival_seed=args.arrival_seed_base + load_index * 100_000,
        )
        for instance_index in range(args.instances_per_load):
            arrival_seed = args.arrival_seed_base + load_index * 100_000 + instance_index
            records = generate_records(target_load=load, arrival_seed=arrival_seed)
            for replication in range(args.replications):
                simulation_seed = args.simulation_seed_base + load_index * 100_000 + instance_index * args.replications + replication
                simulator.target_load = load
                simulator.reset(
                    instance_seed=arrival_seed,
                    simulation_seed=simulation_seed,
                    policy_seed=simulation_seed,
                    job_records=records,
                )
                metrics = simulator.run_episode(training=False)
                rows.append({
                    "target_load": load,
                    "instance_index": instance_index + 1,
                    "replication_index": replication + 1,
                    "arrival_seed": arrival_seed,
                    "simulation_seed": simulation_seed,
                    "total_tardiness": metrics["total_tardiness"],
                    "mean_tardiness": metrics["mean_tardiness"],
                })
                progress += 1
                if progress % args.log_every == 0 or progress == total:
                    print(f"[INDUSTRIAL-CALIBRATION] {progress}/{total} load={load:.2f} mean_tardiness={metrics['mean_tardiness']:.3f}", flush=True)
    per_instance = {}
    for row in rows:
        per_instance.setdefault((row["target_load"], row["instance_index"]), []).append(float(row["mean_tardiness"]))
    values = np.asarray([np.mean(group) for group in per_instance.values()], dtype=float)
    low, high = np.quantile(values, [0.05, 0.95])
    scale = max(1.0, float(np.mean(np.clip(values, low, high))))
    contract = {
        "schema_version": 1,
        "normalization_name": "offline-calibrated SDR reward normalization",
        "reference_policy": "urgency-aware batch formation with maximum-utilization batch selection and health-state-threshold maintenance",
        "reference_policy_code": "CH+MMU+Threshold",
        "dataset_contract_sha256": dataset["dataset_contract_sha256"],
        "loads": list(loads),
        "instances_per_load": args.instances_per_load,
        "replications": args.replications,
        "arrival_seed_base": args.arrival_seed_base,
        "simulation_seed_base": args.simulation_seed_base,
        "independence": "arrival and simulation seeds are disjoint from training, validation, and formal test",
        "estimator": "mean of per-instance replication means after cross-instance 5th/95th percentile winsorization",
        "metric": "mean_tardiness",
        "floor": 1.0,
    }
    artifact = {
        **contract,
        "contract_sha256": canonical_sha256(contract),
        "value": scale,
        "winsorization_bounds": [float(low), float(high)],
        "raw_instance_mean_count": len(values),
    }
    target = args.output.resolve()
    detail = target.with_name(target.stem + "_runs.csv")
    atomic_csv(detail, rows)
    artifact["calibration_runs_file"] = detail.name
    artifact["calibration_runs_sha256"] = file_sha256(detail)
    atomic_json(target, artifact)
    print(f"[INDUSTRIAL-CALIBRATION] scale={scale:.6f} output={target}", flush=True)


if __name__ == "__main__":
    main()
