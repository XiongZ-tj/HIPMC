"""Method-independent dynamic job-instance generation."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


INSTANCE_GENERATION_PROTOCOL = "fixed_count_nominal_finite_load_v2"
FROZEN_JOB_RECORD_SCHEMA_VERSION = 1
FROZEN_JOB_RECORD_FIELDS = (
    "job_id",
    "arrival_time",
    "family",
    "size",
    "due_time",
)


@dataclass(frozen=True)
class JobRecord:
    job_id: int
    arrival_time: float
    family: int
    size: int
    due_time: float


@dataclass(frozen=True)
class InstanceLoad:
    """Auditable workload quantities for one finite dynamic instance."""

    target_load: float
    arrival_span: float
    dynamic_nominal_workload: float
    initial_nominal_workload: float
    total_nominal_workload: float
    plant_capacity: float
    realized_dynamic_load: float
    finite_instance_load: float


def validate_job_records(
        records: Sequence[JobRecord],
        *,
        initial_jobs: int,
        total_jobs: int,
        due_date_factor: float,
        processing_times: tuple[int, ...] | list[int],
        size_interval: tuple[int, int]) -> None:
    """Validate one materialized instance against its generation contract."""

    low, high = size_interval
    expected_ids = list(range(total_jobs))
    previous_dynamic_arrival = 0.0
    for index, record in enumerate(records):
        if index < initial_jobs:
            pass
        else:
            previous_dynamic_arrival = record.arrival_time
        expected_due = (
            record.arrival_time
            + float(processing_times[record.family]) * due_date_factor
        )


def save_job_records_csv(
        records: Sequence[JobRecord], path: str | Path) -> None:
    """Atomically save round-trip-safe frozen job records for C++/Python."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FROZEN_JOB_RECORD_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "job_id": record.job_id,
                "arrival_time": format(record.arrival_time, ".17g"),
                "family": record.family,
                "size": record.size,
                "due_time": format(record.due_time, ".17g"),
            })
    os.replace(temporary, target)


def load_job_records_csv(path: str | Path) -> list[JobRecord]:
    """Load one frozen C++/Python-neutral job-record table strictly."""

    source = Path(path)
    with source.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        records = [
            JobRecord(
                job_id=int(row["job_id"]),
                arrival_time=float(row["arrival_time"]),
                family=int(row["family"]),
                size=int(row["size"]),
                due_time=float(row["due_time"]),
            )
            for row in reader
        ]
    return records


def _balanced_integers(
        count: int,
        values: tuple[int, ...],
        rng: np.random.Generator) -> np.ndarray:
    """Return a shuffled, near-exactly balanced categorical sample."""

    if count <= 0:
        return np.empty(0, dtype=np.int64)
    repeats, remainder = divmod(count, len(values))
    result = np.tile(np.asarray(values, dtype=np.int64), repeats)
    if remainder:
        result = np.concatenate((
            result,
            rng.choice(np.asarray(values), size=remainder, replace=False),
        ))
    rng.shuffle(result)
    return result


def _job_pool(
        count: int,
        family_count: int,
        size_interval: tuple[int, int],
        rng: np.random.Generator) -> list[tuple[int, int]]:
    """Build a controlled-composition pool while retaining random ordering."""

    families = _balanced_integers(count, tuple(range(family_count)), rng)
    low, high = size_interval
    sizes = _balanced_integers(count, tuple(range(low, high + 1)), rng)
    return [(int(family), int(size)) for family, size in zip(
        families, sizes, strict=True)]


def generate_job_records(
        *,
        target_load: float,
        initial_jobs: int,
        total_jobs: int,
        due_date_factor: float,
        processing_times: tuple[int, ...] | list[int],
        size_interval: tuple[int, int],
        machine_counts: tuple[int, ...] | list[int],
        machine_capacities: tuple[int, ...] | list[int],
        seed: int,
) -> list[JobRecord]:
    """Generate exactly ``total_jobs`` under a controlled dynamic load.

    ``target_load`` denotes the nominal finite-horizon load, including both
    initial work-in-process and dynamically arriving jobs. Exponential
    spacings retain local burstiness, while normalization fixes their total
    arrival span so that ``finite_instance_load`` equals ``target_load``.
    """

    if target_load <= 0:
        raise ValueError("target_load must be positive.")
    if total_jobs <= 0 or not 0 <= initial_jobs < total_jobs:
        raise ValueError(
            "Require total_jobs > 0 and 0 <= initial_jobs < total_jobs.")
    if not math.isfinite(due_date_factor) or due_date_factor < 1:
        raise ValueError(
            "due_date_factor must be finite and >= 1 so that due time is "
            "not earlier than arrival + nominal processing time.")
    if not processing_times or any(value <= 0 for value in processing_times):
        raise ValueError("processing_times must be nonempty and positive.")
    low, high = size_interval
    if low <= 0 or high < low:
        raise ValueError("size_interval must be a positive inclusive interval.")

    if (
            len(machine_counts) != len(machine_capacities)
            or not machine_counts
            or any(count < 0 for count in machine_counts)
            or any(capacity <= 0 for capacity in machine_capacities)):
        raise ValueError("Machine counts and capacities are invalid.")
    plant_capacity = float(sum(
        count * capacity for count, capacity in zip(
            machine_counts, machine_capacities, strict=True)))
    if plant_capacity <= 0:
        raise ValueError("The instance requires positive plant capacity.")

    rng = np.random.default_rng(seed)
    records: list[JobRecord] = []

    def add(arrival: float, family: int, size: int) -> None:
        due_time = float(
            arrival + processing_times[family] * due_date_factor)
        records.append(JobRecord(
            job_id=len(records),
            arrival_time=arrival,
            family=family,
            size=size,
            due_time=due_time,
        ))

    initial_pool = _job_pool(
        initial_jobs, len(processing_times), size_interval, rng)
    dynamic_pool = _job_pool(
        total_jobs - initial_jobs, len(processing_times), size_interval, rng)
    for family, size in initial_pool:
        add(0.0, family, size)

    initial_workload = float(sum(
        size * processing_times[family] for family, size in initial_pool))
    dynamic_workload = float(sum(
        size * processing_times[family] for family, size in dynamic_pool))
    total_workload = initial_workload + dynamic_workload
    arrival_span = total_workload / (target_load * plant_capacity)
    spacings = rng.exponential(1.0, size=len(dynamic_pool))
    arrivals = np.cumsum(spacings)
    arrivals *= arrival_span / float(arrivals[-1])
    for arrival, (family, size) in zip(
            arrivals, dynamic_pool, strict=True):
        add(float(arrival), family, size)
    return records


def instance_load(
        records: list[JobRecord],
        *,
        target_load: float,
        initial_jobs: int,
        processing_times: tuple[int, ...] | list[int],
        machine_counts: tuple[int, ...] | list[int],
        machine_capacities: tuple[int, ...] | list[int]) -> InstanceLoad:
    """Recompute load quantities from generated records for contract checks."""

    plant_capacity = float(sum(
        count * capacity for count, capacity in zip(
            machine_counts, machine_capacities, strict=True)))
    initial = records[:initial_jobs]
    dynamic = records[initial_jobs:]
    initial_work = float(sum(
        record.size * processing_times[record.family] for record in initial))
    dynamic_work = float(sum(
        record.size * processing_times[record.family] for record in dynamic))
    span = max(record.arrival_time for record in records)
    denominator = span * plant_capacity
    return InstanceLoad(
        target_load=float(target_load),
        arrival_span=float(span),
        dynamic_nominal_workload=dynamic_work,
        initial_nominal_workload=initial_work,
        total_nominal_workload=initial_work + dynamic_work,
        plant_capacity=plant_capacity,
        realized_dynamic_load=dynamic_work / denominator,
        finite_instance_load=(initial_work + dynamic_work) / denominator,
    )
