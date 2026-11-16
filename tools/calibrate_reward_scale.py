"""Create an independently calibrated SDR reward-scale artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from experiment.config import ConditionConfig, SCENARIOS
from experiment.reward_scale import (
    calibrate_offline_sdr_scale,
    save_reward_scale,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="base")
    parser.add_argument("--target-load", type=float, default=0.95)
    parser.add_argument("--initial-jobs", type=int, default=30)
    parser.add_argument("--total-jobs", type=int, default=150)
    parser.add_argument("--due-date-factor", type=float, default=2.5)
    parser.add_argument("--degradation-multiplier", type=float, default=1.0)
    parser.add_argument("--instances", type=int, default=200)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--instance-seed-start", type=int, default=6000000)
    parser.add_argument("--simulation-seed-start", type=int, default=7000000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    condition = ConditionConfig(
        target_load=args.target_load,
        initial_jobs=args.initial_jobs,
        total_jobs=args.total_jobs,
        due_date_factor=args.due_date_factor,
        degradation_multiplier=args.degradation_multiplier,
        seed=42,
    )
    artifact = calibrate_offline_sdr_scale(
        workspace=WORKSPACE,
        scenario_name=args.scenario,
        condition=condition,
        instances=args.instances,
        replications=args.replications,
        instance_seed_start=args.instance_seed_start,
        simulation_seed_start=args.simulation_seed_start,
    )
    output = args.output.resolve()
    save_reward_scale(artifact, output)
    print(json.dumps({
        "output": str(output),
        "offline_calibrated_sdr_scale": artifact["scale"]["value"],
        "scale_summary": artifact["scale"],
        "calibration": artifact["calibration"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
