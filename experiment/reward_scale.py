"""Independent offline-calibrated SDR reward scales."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from experiment.artifacts import atomic_json_save
from experiment.config import ConditionConfig, SCENARIOS, serialize_config
from experiment.instances import INSTANCE_GENERATION_PROTOCOL
from experiment.provenance import source_fingerprint
from experiment.reward_reference import (
    REWARD_REFERENCE_ALLOCATION_RULE,
    REWARD_REFERENCE_BATCH_RULE,
    REWARD_REFERENCE_MAINTENANCE_RULE,
    REWARD_REFERENCE_POLICY,
)


REWARD_SCALE_SCHEMA_VERSION = 3
NORMALIZATION_PROTOCOL = "offline_calibrated_sdr_v2"
REFERENCE_POLICY = REWARD_REFERENCE_POLICY
REFERENCE_METRIC = "mean_tardiness"
ESTIMATOR = (
    "mean_of_per_instance_replication_means_after_cross_instance_"
    "5_95_percent_winsorization"
)


def _canonical_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_contract(
        *, scenario_name: str, condition: ConditionConfig) -> dict:
    scenario = SCENARIOS[scenario_name]
    condition_data = serialize_config(condition)
    # The training master seed selects policy/episode streams; it is not an
    # instance-distribution parameter and must not prevent multi-seed reuse.
    condition_data.pop("seed", None)
    contract = {
        "instance_generation_protocol": INSTANCE_GENERATION_PROTOCOL,
        "scenario": serialize_config(scenario),
        "condition": condition_data,
        "reference_policy": REFERENCE_POLICY,
        "reference_metric": REFERENCE_METRIC,
    }
    # Canonical JSON types make an in-memory contract identical to the same
    # contract after a JSON write/read round trip (notably tuple -> list).
    return json.loads(json.dumps(contract, ensure_ascii=False))


def robust_calibrated_scale(per_instance_means: list[float]) -> dict:
    """Return the offline 5--95% winsorized cross-instance scale."""

    values = np.asarray(per_instance_means, dtype=float)
    if np.any(values < 0):
        raise ValueError("Reference mean tardiness cannot be negative.")
    q05, median, q95 = np.quantile(values, (0.05, 0.50, 0.95))
    scale = float(np.clip(values, q05, q95).mean())
    if scale <= 0:
        raise ValueError(
            "The offline-calibrated SDR scale must be positive.")
    return {
        "value": scale,
        "estimator": ESTIMATOR,
        "raw_mean": float(values.mean()),
        "q05": float(q05),
        "median": float(median),
        "q95": float(q95),
        "zero_rate": float(np.mean(values == 0.0)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def calibrate_offline_sdr_scale(
        *, workspace: Path, scenario_name: str, condition: ConditionConfig,
        instances: int, replications: int, instance_seed_start: int,
        simulation_seed_start: int) -> dict:
    """Evaluate the frozen SDR policy and build one auditable scale artifact."""

    if instances < 2 or replications <= 0:
        raise ValueError("Require instances >= 2 and replications > 0.")
    scenario = SCENARIOS[scenario_name]
    from methods.baselines.sdr import shared

    shared.configure_scenario(scenario.job_processing_times)
    simulator = shared.env(
        condition.as_job_parameters(),
        list(scenario.machine_counts),
        sdrs=(
            REWARD_REFERENCE_BATCH_RULE,
            REWARD_REFERENCE_ALLOCATION_RULE,
        ),
        maint_policy=REWARD_REFERENCE_MAINTENANCE_RULE,
        degradation_multiplier=condition.degradation_multiplier,
    ).core
    per_instance_means = []
    for instance_index in range(instances):
        scores = []
        instance_seed = instance_seed_start + instance_index
        for replication_index in range(replications):
            simulation_seed = (
                simulation_seed_start
                + instance_index * replications
                + replication_index)
            simulator.reset(
                instance_seed=instance_seed,
                simulation_seed=simulation_seed,
                policy_seed=simulation_seed,
            )
            scores.append(float(
                simulator.run_episode(training=False)[REFERENCE_METRIC]))
        per_instance_means.append(float(np.mean(scores)))

    contract = calibration_contract(
        scenario_name=scenario_name, condition=condition)
    return {
        "schema_version": REWARD_SCALE_SCHEMA_VERSION,
        "created_local": datetime.now().isoformat(),
        "contract": contract,
        "contract_sha256": _canonical_sha256(contract),
        "calibration": {
            "instances": instances,
            "replications_per_instance": replications,
            "instance_seed_start": instance_seed_start,
            "instance_seed_end": instance_seed_start + instances - 1,
            "simulation_seed_start": simulation_seed_start,
            "simulation_seed_end": (
                simulation_seed_start + instances * replications - 1),
            "replication_aggregation": "arithmetic_mean_per_instance",
            "outlier_handling": "cross_instance_5_95_percent_winsorization",
        },
        "normalization_protocol": NORMALIZATION_PROTOCOL,
        "scale": robust_calibrated_scale(per_instance_means),
        "source_fingerprint_sha256": source_fingerprint(workspace),
    }


def save_reward_scale(artifact: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_save(artifact, path)


def load_reward_scale(
        path: Path, *, workspace: Path, scenario_name: str,
        condition: ConditionConfig) -> tuple[dict, str]:
    "Read the independently calibrated reward scale from an artifact."

    path = path.resolve()
    artifact = json.loads(path.read_text(encoding="utf-8"))
    return artifact, artifact_sha256(path)
