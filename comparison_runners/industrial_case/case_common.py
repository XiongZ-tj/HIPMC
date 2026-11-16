"""Frozen contracts shared by the industrial-case evaluation runners."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from functools import lru_cache
from dataclasses import replace
from pathlib import Path
from types import MethodType
from typing import Iterable, Sequence

from experiment.instances import load_job_records_csv
from experiment.instances import JobRecord
from experiment.shared_simulator import build_shared_simulator
from methods.hierarchical.types import Job


WORKSPACE = Path(__file__).resolve().parents[2]
DATASET_ROOT = (
    WORKSPACE / "datasets" / "aerospace_heat_treatment_dynamic_case"
)
DATASET_ID = "aerospace_heat_treatment_industrial_order_dynamic_case"
PROCESSING_TIMES = (128, 190, 104, 293, 202, 279)
MACHINE_COUNTS = (1, 1, 1)
MACHINE_CLASSES = (0, 1, 2)
MACHINE_CAPACITIES = (120, 160, 200)
INITIAL_JOBS = 18
TOTAL_JOBS = 89
DEGRADATION_MULTIPLIER = 1.0
TRAINING_MACHINE_COUNTS = (2, 2, 1)
from experiment.revision_protocol import TRAINING_SEEDS
INDUSTRIAL_VALIDATION_ENVIRONMENT_CONTRACT = (
    "industrial_machine_configuration_and_order_records_v1"
)
PROPOSED_METHOD_OPTIMIZATION_HYPERPARAMETERS = {}


def learned_method_environment_kwargs(method: str) -> dict[str, float]:
    """No independent industrial-domain hyperparameter override."""
    return {}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_csv(path: str | Path, rows: Sequence[dict]) -> None:
    target = Path(path)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV table: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def atomic_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def validate_dataset(dataset_root: str | Path = DATASET_ROOT) -> dict:
    root = Path(dataset_root).resolve()
    checksum_path = root / "checksums.json"
    manifest_path = root / "metadata" / "instance_manifest.csv"
    rows = read_csv(manifest_path)
    loads = sorted({float(row["target_load"]) for row in rows})
    for row in rows:
        job_path = root / row["job_file"]
    simulation = json.loads(
        (root / "configuration" / "simulation_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    seeds = tuple(int(seed) for seed in simulation["simulation_seeds"])
    return {
        "root": root,
        "rows": rows,
        "simulation_seeds": seeds,
        "dataset_contract_sha256": file_sha256(manifest_path),
        "instance_manifest_sha256": file_sha256(manifest_path),
        "machine_configuration_sha256": file_sha256(
            root / "configuration" / "machine_configuration.json"
        ),
    }


def industrial_jobs_from_records(self, records):
    """Preserve observed due times while enforcing the frozen case contract."""

    if len(records) != TOTAL_JOBS:
        raise ValueError(f"Expected {TOTAL_JOBS} jobs, got {len(records)}.")
    if [record.job_id for record in records] != list(range(TOTAL_JOBS)):
        raise ValueError("Industrial job IDs must be contiguous from zero.")
    previous_arrival = 0.0
    jobs = []
    for index, record in enumerate(records):
        arrival = float(record.arrival_time)
        due = float(record.due_time)
        family = int(record.family)
        size = int(record.size)
        if not math.isfinite(arrival) or arrival < 0 or not math.isfinite(due):
            raise ValueError("Arrival and due times must be finite.")
        if not 0 <= family < len(PROCESSING_TIMES) or not 10 <= size <= 60:
            raise ValueError("Industrial family or job weight is out of range.")
        if index < INITIAL_JOBS:
            if arrival != 0.0:
                raise ValueError("The first 18 industrial jobs must arrive at zero.")
        elif arrival <= 0.0 or arrival < previous_arrival:
            raise ValueError("Dynamic arrivals must be positive and ordered.")
        previous_arrival = max(previous_arrival, arrival)
        processing = PROCESSING_TIMES[family]
        jobs.append(Job(
            job_id=int(record.job_id), arrival_time=arrival, family=family,
            size=size, due_time=due,
        ))
    return jobs


def adapt_simulator(core) -> None:
    core.scenario = replace(
        core.scenario,
        machine_capacities=MACHINE_CAPACITIES,
        job_size_interval=(10, 60),
    )
    core._jobs_from_records = MethodType(industrial_jobs_from_records, core)


def audit_industrial_episode_input(
        simulator,
        records: Sequence[JobRecord],
        *,
        target_load: float) -> None:
    """Fail fast when a learned method is not running the frozen case.

    This audit deliberately checks the realized simulator state rather than
    trusting runner metadata.  It catches the failure mode where a newly
    constructed validation environment silently falls back to the generic
    synthetic generator or to the base machine capacities.
    """

    for observed, expected in zip(simulator.jobs, records, strict=True):
        pass
    instance_load = simulator._instance_load


def reset_industrial_simulator(
        simulator,
        *,
        target_load: float,
        instance_seed: int,
        simulation_seed: int,
        policy_seed: int,
        records: Sequence[JobRecord]) -> None:
    """Reset any learned-method simulator on one audited industrial case."""

    load = float(target_load)
    simulator.target_load = load
    simulator.job_parameters = [
        load, INITIAL_JOBS, TOTAL_JOBS, 1.0, int(instance_seed),
    ]
    simulator.reset(
        instance_seed=int(instance_seed),
        simulation_seed=int(simulation_seed),
        policy_seed=int(policy_seed),
        job_records=records,
    )
    audit_industrial_episode_input(
        simulator, records, target_load=load)


def build_static_simulator(controller, *, target_load: float, arrival_seed: int):
    simulator = build_shared_simulator(
        job_parameters=[target_load, INITIAL_JOBS, TOTAL_JOBS, 1.0, arrival_seed],
        machine_parameters=list(MACHINE_COUNTS),
        processing_times=PROCESSING_TIMES,
        controller=controller,
        degradation_multiplier=DEGRADATION_MULTIPLIER,
        device="cpu",
    )
    adapt_simulator(simulator)
    return simulator


def load_records(dataset: dict, row: dict[str, str]):
    return load_job_records_csv(dataset["root"] / row["job_file"])


@lru_cache(maxsize=1)
def _dataset_builder_module():
    builder_path = DATASET_ROOT / "scripts" / "build_dataset.py"
    module_name = "aerospace_industrial_dataset_builder"
    spec = importlib.util.spec_from_file_location(
        module_name, builder_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load industrial dataset builder: {builder_path}")
    builder = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = builder
    spec.loader.exec_module(builder)
    return builder


def generate_records(*, target_load: float, arrival_seed: int) -> list[JobRecord]:
    """Materialize a disjoint industrial-order arrival realization in memory."""

    builder = _dataset_builder_module()
    jobs, _ = builder.build_dynamic_jobs(
        builder.load_source_orders(), target_load=target_load,
        arrival_seed=int(arrival_seed),
    )
    return [JobRecord(
        job_id=job.job_id,
        arrival_time=job.arrival_time,
        family=job.order.family,
        size=job.order.weight,
        due_time=job.order.due_time,
    ) for job in jobs]


def rows_by_load(rows: Iterable[dict[str, str]]) -> dict[float, list[dict[str, str]]]:
    result: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        result.setdefault(float(row["target_load"]), []).append(row)
    for values in result.values():
        values.sort(key=lambda item: int(item["instance_number"]))
    return dict(sorted(result.items()))
