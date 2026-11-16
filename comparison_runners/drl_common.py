"""Shared contracts for formal E1 learned-method experiments."""

from __future__ import annotations

import csv
import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from experiment.artifacts import atomic_csv_save
from experiment.config import SCENARIOS, ConditionConfig
from experiment.device import resolve_torch_device
from experiment.seeds import TrainingSeedProtocol


WORKSPACE = Path(__file__).resolve().parents[1]
DRL_METHODS = ("ddqnb", "psa_mappo", "hdrl_option")
LEARNED_METHODS = ("ours", "ours_categorical_baa", *DRL_METHODS)
DEFAULT_TRAINING_MANIFEST = (
    WORKSPACE / "experiment_manifests"
    / "e1_learned_method_training_cases.csv")
DEFAULT_VALIDATION_MANIFEST = (
    WORKSPACE / "experiment_manifests"
    / "e1_learned_method_validation_cases.csv")


@dataclass(frozen=True)
class FrozenCSVManifest:
    path: Path
    sha256: str
    rows: tuple[dict[str, str], ...]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_artifact_names(method: str, *, machine_count: int = 5) -> tuple[str, ...]:
    if method in ("ours", "ours_categorical_baa"):
        return ("hierarchical_inference.pt",)
    if method == "ddqnb":
        return ("ddqnb_model.pth",)
    if method == "psa_mappo":
        return ("psa_mappo_model.pth",)
    if method == "hdrl_option":
        return ("hdrl_option_model.pth",)
    raise ValueError(f"Unsupported DRL baseline: {method}")


def checkpoint_name(method: str) -> str:
    if method in ("ours", "ours_categorical_baa"):
        return "hierarchical_checkpoint.pt"
    if method not in LEARNED_METHODS:
        raise ValueError(f"Unsupported DRL baseline: {method}")
    return f"{method}_checkpoint.pt"

def _ensure_csv_manifest(
        path: str | Path,
        rows: list[dict],
        fields: tuple[str, ...],
        *,
        label: str) -> FrozenCSVManifest:
    target = Path(path).resolve()
    normalized = [
        {field: str(row[field]) for field in fields}
        for row in rows
    ]
    if target.is_file():
        with target.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            observed = [dict(row) for row in reader]
    else:
        atomic_csv_save(rows, target, fieldnames=fields)
        observed = normalized
    return FrozenCSVManifest(
        path=target,
        sha256=file_sha256(target),
        rows=tuple(observed),
    )


def ensure_training_manifest(
        path: str | Path,
        training_seeds: Iterable[int],
        episodes: int) -> FrozenCSVManifest:
    fields = (
        "schema_version", "training_seed", "episode",
        "instance_seed", "simulation_seed", "policy_seed",
    )
    rows = []
    for training_seed in training_seeds:
        protocol = TrainingSeedProtocol(int(training_seed))
        for episode in range(episodes):
            seeds = protocol.for_episode(episode)
            rows.append({
                "schema_version": 1,
                "training_seed": int(training_seed),
                "episode": episode + 1,
                "instance_seed": seeds.instance_seed,
                "simulation_seed": seeds.simulation_seed,
                "policy_seed": seeds.policy_seed,
            })
    target=Path(path).resolve()
    if target.is_file():
        with target.open(newline='',encoding='utf-8') as stream:
            existing=list(csv.DictReader(stream))
        preserved={(int(row['training_seed']),int(row['episode'])):row for row in existing}
        for row in rows:
            preserved.setdefault((int(row['training_seed']),int(row['episode'])),row)
        rows=[preserved[key] for key in sorted(preserved)]
    atomic_csv_save(rows,target,fieldnames=fields)
    return FrozenCSVManifest(target,file_sha256(target),tuple(rows))


def ensure_validation_manifest(
        path: str | Path,
        instance_count: int) -> FrozenCSVManifest:
    fields = (
        "schema_version", "validation_index", "instance_seed",
        "simulation_seed", "policy_seed",
    )
    rows = [{
        "schema_version": 1,
        "validation_index": index,
        "instance_seed": 3_000_000 + index,
        "simulation_seed": 4_000_000 + index,
        "policy_seed": 4_000_000 + index,
    } for index in range(instance_count)]
    manifest = _ensure_csv_manifest(
        path, rows, fields, label="E1 learned-method validation")
    return FrozenCSVManifest(manifest.path, manifest.sha256,
                             manifest.rows[:instance_count])


def training_condition(training_seed: int) -> ConditionConfig:
    """Return the representative centre condition of the frozen E1 grid."""

    return ConditionConfig(
        target_load=1.0,
        initial_jobs=30,
        total_jobs=150,
        due_date_factor=2.5,
        degradation_multiplier=1.0,
        seed=int(training_seed),
    )


def build_drl_environment(
        method: str,
        condition: ConditionConfig,
        *,
        device: str,
        calibrated_reward_scale: float | None = None,
        reward_scale_artifact_sha256: str | None = None):
    if method not in LEARNED_METHODS:
        raise ValueError(f"Unsupported learned method: {method}")
    module = importlib.import_module(
        "methods.hierarchical"
        if method in ("ours", "ours_categorical_baa")
        else f"methods.baselines.{method}.shared")
    scenario = SCENARIOS["base"]
    module.configure_scenario(scenario.job_processing_times)
    resolved_device = str(resolve_torch_device(device))
    environment = module.env(
        job_parameters=condition.as_job_parameters(),
        machine_parameters=list(scenario.machine_counts),
        degradation_multiplier=condition.degradation_multiplier,
        device=resolved_device,
        calibrated_reward_scale=calibrated_reward_scale,
        reward_scale_artifact_sha256=reward_scale_artifact_sha256,
    )
    return environment, resolved_device


def simulation_core(environment):
    """Return the event-driven simulator behind either learned-method API.

    Baseline environments are training wrappers exposing their simulator as
    ``core``.  The proposed method is itself the simulator.  Keeping this
    distinction here prevents evaluation code from depending on either
    method-specific object layout.
    """

    return getattr(environment, "core", environment)


def _physical_machine_count(environment) -> int:
    core = simulation_core(environment)
    machines = getattr(core, "machines", ())
    if machines:
        return len(machines)
    machine_parameters = getattr(environment, "machine_parameters", None)
    if machine_parameters is not None:
        return sum(machine_parameters)
    scenario = getattr(core, "scenario", None)
    machine_counts = getattr(scenario, "machine_counts", None)
    if machine_counts is not None:
        return sum(machine_counts)
    raise AttributeError(
        "Cannot determine the physical-machine count from the learned-method "
        "environment.")


def load_model(environment, method: str, model_dir: str | Path) -> dict[str, str]:
    directory = Path(model_dir).resolve()
    names = model_artifact_names(
        method,
        machine_count=_physical_machine_count(environment),
    )
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {method.upper()} model artifacts in {directory}: "
            + ", ".join(missing))
    if method in ("ours", "ours_categorical_baa"):
        environment.load_inference(directory / "hierarchical_inference.pt")
    else:
        environment.load_inference_model(directory)
    return {name: file_sha256(directory / name) for name in names}
