"Build the industrial-order-data-driven dynamic scheduling instances."

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DATASET_VERSION = "1.0.0"
GENERATION_PROTOCOL = "constrained_exponential_arrivals_fixed_industrial_orders_v1"
DATASET_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = DATASET_ROOT / "source" / "industrial_orders.csv"

TARGET_LOADS = (0.90, 1.00, 1.10)
INSTANCES_PER_LOAD = 20
INITIAL_JOB_COUNT = 18
INSTANCE_SEED_BASE = 41_000_000
SIMULATION_SEEDS = tuple(range(52_000_000, 52_000_010))

MACHINE_COUNTS = (1, 1, 1)
MACHINE_CAPACITIES = (120, 160, 200)
PROCESSING_TIMES = (128, 190, 104, 293, 202, 279)
MINIMUM_NOMINAL_SLACK_RATIO = 0.25

# Exact snapshot of the frozen stochastic degradation and maintenance model.
DEGRADATION_STATES = 4
FAILURE_TYPES = 3
DEGRADATION_RATE_RANGES = (
    (0.001, 0.005),
    (0.001, 0.003),
    (0.001, 0.002),
)
UNEXPECTED_FAILURE_RATE_RANGES = (
    (0.0001, 0.0003),
    (0.0001, 0.0003),
    (0.0001, 0.0003),
)
TERMINAL_DEGRADATION_RATES = (0.005, 0.003, 0.002)
FAILURE_TYPE_PROPORTIONS = (0.6, 0.3, 0.1)
IPM_RECOVERY_RATES = (
    (0.06931471805599453, 0.06019864021629608, 0.02153914580462271),
    (0.046209812037329684, 0.04013242681086405, 0.014359430536415141),
    (0.03465735902799726, 0.03009932010814804, 0.010769572902311356),
)
IPM_TIMES = (20, 30, 40)
PM_TIMES = (60, 90, 120)
CORRECTIVE_TIME_RANGES = ((20, 40), (40, 60), (60, 80))
MINIMAL_REPAIR_TIMES = (80, 120, 160)
PRODUCTION_RATE_RANGE = (0.9, 1.0)

SOURCE_FIELDS = (
    "source_order_id",
    "weight",
    "due_time",
    "family_code",
    "processing_time",
    "field_origin",
    "source_table",
    "source_pdf_pages",
)
INSTANCE_FIELDS = ("job_id", "arrival_time", "family", "size", "due_time")
PROVENANCE_FIELDS = (
    "job_id",
    "source_order_id",
    "arrival_time",
    "family",
    "source_family_code",
    "size",
    "processing_time",
    "due_time",
    "is_initial",
    "arrival_seed",
    "job_attributes_origin",
    "arrival_origin",
)


@dataclass(frozen=True)
class IndustrialOrder:
    source_order_id: int
    weight: int
    due_time: float
    family_code: int
    processing_time: int

    @property
    def family(self) -> int:
        """Zero-based family index expected by the scheduling simulator."""

        return self.family_code - 1


@dataclass(frozen=True)
class DynamicJob:
    job_id: int
    order: IndustrialOrder
    arrival_time: float
    arrival_seed: int

    @property
    def is_initial(self) -> bool:
        return self.arrival_time == 0.0

    @property
    def nominal_slack(self) -> float:
        return (
            self.order.due_time
            - self.arrival_time
            - self.order.processing_time
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen aerospace heat-treatment dynamic case.")
    parser.add_argument(
        "--source-pdf",
        type=Path,
        help=("Optional path to the source PDF. It is read only to record its "
              "name and SHA-256; the PDF is not copied."),
    )
    return parser.parse_args()


def _assert_within_dataset(path: Path) -> Path:
    resolved = path.resolve()
    root = DATASET_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing to write outside dataset root: {resolved}")
    return resolved


def _atomic_text(path: Path, text: str) -> None:
    target = _assert_within_dataset(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, target)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_csv(
        path: Path,
        fieldnames: Sequence[str],
        rows: Iterable[dict[str, object]]) -> None:
    target = _assert_within_dataset(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_float(value: float) -> str:
    return format(float(value), ".17g")


def load_source_orders() -> list[IndustrialOrder]:
    with SOURCE_CSV.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SOURCE_FIELDS:
            raise ValueError("Industrial source CSV schema does not match contract.")
        raw_rows = list(reader)

    orders = [
        IndustrialOrder(
            source_order_id=int(row["source_order_id"]),
            weight=int(row["weight"]),
            due_time=float(row["due_time"]),
            family_code=int(row["family_code"]),
            processing_time=int(row["processing_time"]),
        )
        for row in raw_rows
    ]
    if len(orders) != 89:
        raise ValueError(f"Expected 89 industrial orders, found {len(orders)}.")
    if [item.source_order_id for item in orders] != list(range(1, 90)):
        raise ValueError("Source order IDs must be contiguous from 1 to 89.")
    if any(row["field_origin"] != "observed_industrial" for row in raw_rows):
        raise ValueError("Every source order must be marked observed_industrial.")
    if any(row["source_table"] != "Table 4.1" for row in raw_rows):
        raise ValueError("Every order must retain its Table 4.1 provenance.")

    expected_by_family = {
        family: processing_time
        for family, processing_time in enumerate(PROCESSING_TIMES, start=1)
    }
    for item in orders:
        if item.family_code not in expected_by_family:
            raise ValueError(f"Unknown family code: {item.family_code}")
        if item.processing_time != expected_by_family[item.family_code]:
            raise ValueError(
                f"Processing time mismatch for order {item.source_order_id}.")
        if item.weight <= 0 or item.due_time <= item.processing_time:
            raise ValueError(
                f"Invalid physical fields for order {item.source_order_id}.")

    expected_counts = {1: 17, 2: 9, 3: 10, 4: 17, 5: 21, 6: 15}
    counts = Counter(item.family_code for item in orders)
    if dict(counts) != expected_counts:
        raise ValueError(f"Family counts differ from Table 4.1: {dict(counts)}")
    if sum(item.weight for item in orders) != 3234:
        raise ValueError("Total industrial order weight must equal 3234.")
    if sum(item.weight * item.processing_time for item in orders) != 655_840:
        raise ValueError("Weighted nominal workload must equal 655840.")
    if (min(item.due_time for item in orders),
            max(item.due_time for item in orders)) != (409.0, 2208.0):
        raise ValueError("Industrial due-time range does not match Table 4.1.")
    return orders


def _arrival_seed(load_index: int, instance_index: int) -> int:
    return INSTANCE_SEED_BASE + load_index * 1_000 + instance_index


def build_dynamic_jobs(
        orders: Sequence[IndustrialOrder],
        *,
        target_load: float,
        arrival_seed: int) -> tuple[list[DynamicJob], float]:
    total_workload = float(sum(
        item.weight * item.processing_time for item in orders))
    aggregate_capacity = float(sum(
        count * capacity for count, capacity in zip(
            MACHINE_COUNTS, MACHINE_CAPACITIES, strict=True)))
    arrival_span = total_workload / (target_load * aggregate_capacity)

    rng = np.random.default_rng(arrival_seed)
    dynamic_count = len(orders) - INITIAL_JOB_COUNT
    gaps = rng.exponential(1.0, size=dynamic_count)
    dynamic_slots = np.cumsum(gaps)
    dynamic_slots *= arrival_span / float(dynamic_slots[-1])
    slots = np.concatenate((np.zeros(INITIAL_JOB_COUNT), dynamic_slots))

    remaining = list(range(len(orders)))
    assignment: list[int | None] = [None] * len(orders)
    for slot_index in np.argsort(slots)[::-1]:
        slot = float(slots[slot_index])
        eligible = [
            order_index
            for order_index in remaining
            if (
                orders[order_index].due_time
                - (1.0 + MINIMUM_NOMINAL_SLACK_RATIO)
                * orders[order_index].processing_time
            ) >= slot - 1e-9
        ]
        if not eligible:
            raise RuntimeError(
                "No feasible order-to-arrival assignment for "
                f"target_load={target_load}, seed={arrival_seed}, slot={slot}.")

        # A later release is preferentially paired with a due allowance that is
        # only moderately larger than the slot. This preserves stochasticity
        # while avoiding implausible inversions of release and due urgency.
        latest_arrivals = np.asarray([
            orders[index].due_time
            - (1.0 + MINIMUM_NOMINAL_SLACK_RATIO)
            * orders[index].processing_time
            for index in eligible
        ], dtype=np.float64)
        selection_weights = 1.0 / (
            1.0 + np.maximum(0.0, latest_arrivals - slot))
        selection_weights /= selection_weights.sum()
        selected = int(rng.choice(eligible, p=selection_weights))
        assignment[int(slot_index)] = selected
        remaining.remove(selected)

    if remaining or any(index is None for index in assignment):
        raise AssertionError("Order-to-arrival assignment is incomplete.")

    jobs = [
        DynamicJob(
            job_id=job_id,
            order=orders[int(order_index)],
            arrival_time=float(slots[job_id]),
            arrival_seed=arrival_seed,
        )
        for job_id, order_index in enumerate(assignment)
    ]
    validate_dynamic_jobs(jobs, target_load=target_load)
    return jobs, arrival_span


def validate_dynamic_jobs(
        jobs: Sequence[DynamicJob], *, target_load: float) -> None:
    if len(jobs) != 89 or [job.job_id for job in jobs] != list(range(89)):
        raise ValueError("Dynamic jobs must be contiguous and contain 89 rows.")
    if sum(job.is_initial for job in jobs) != INITIAL_JOB_COUNT:
        raise ValueError("Dynamic instance has the wrong initial-job count.")
    arrivals = [job.arrival_time for job in jobs]
    if any(not math.isfinite(value) or value < 0 for value in arrivals):
        raise ValueError("Arrival times must be finite and nonnegative.")
    if arrivals != sorted(arrivals):
        raise ValueError("Arrival times must be nondecreasing.")
    if any(value != 0.0 for value in arrivals[:INITIAL_JOB_COUNT]):
        raise ValueError("Every initial job must arrive at time zero.")
    if any(value <= 0.0 for value in arrivals[INITIAL_JOB_COUNT:]):
        raise ValueError("Every dynamic job must arrive after time zero.")
    if len({job.order.source_order_id for job in jobs}) != 89:
        raise ValueError("Every industrial source order must appear exactly once.")
    for job in jobs:
        required_slack = (
            MINIMUM_NOMINAL_SLACK_RATIO * job.order.processing_time)
        if job.nominal_slack + 1e-8 < required_slack:
            raise ValueError(
                f"Order {job.order.source_order_id} violates the release/due "
                "feasibility buffer.")

    workload = float(sum(
        job.order.weight * job.order.processing_time for job in jobs))
    aggregate_capacity = float(sum(
        count * capacity for count, capacity in zip(
            MACHINE_COUNTS, MACHINE_CAPACITIES, strict=True)))
    realized_load = workload / (max(arrivals) * aggregate_capacity)
    if not math.isclose(realized_load, target_load, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"Finite-horizon load mismatch: {realized_load} != {target_load}")


def _load_label(load: float) -> str:
    return f"{load:.2f}".replace(".", "p")


def _write_instance(
        jobs: Sequence[DynamicJob],
        *,
        target_load: float,
        instance_number: int) -> tuple[Path, Path]:
    load_label = _load_label(target_load)
    stem = (
        f"industrial_orders_load-{load_label}_instance-{instance_number:03d}")
    job_path = (
        DATASET_ROOT / "instances" / "jobs" / f"load_{load_label}" /
        f"{stem}.csv")
    provenance_path = (
        DATASET_ROOT / "instances" / "provenance" / f"load_{load_label}" /
        f"{stem}_provenance.csv")

    _atomic_csv(job_path, INSTANCE_FIELDS, ({
        "job_id": job.job_id,
        "arrival_time": _format_float(job.arrival_time),
        "family": job.order.family,
        "size": job.order.weight,
        "due_time": _format_float(job.order.due_time),
    } for job in jobs))
    _atomic_csv(provenance_path, PROVENANCE_FIELDS, ({
        "job_id": job.job_id,
        "source_order_id": job.order.source_order_id,
        "arrival_time": _format_float(job.arrival_time),
        "family": job.order.family,
        "source_family_code": job.order.family_code,
        "size": job.order.weight,
        "processing_time": job.order.processing_time,
        "due_time": _format_float(job.order.due_time),
        "is_initial": int(job.is_initial),
        "arrival_seed": job.arrival_seed,
        "job_attributes_origin": "observed_industrial",
        "arrival_origin": "synthetically_generated",
    } for job in jobs))
    return job_path, provenance_path


def _write_configurations(source_pdf: Path | None) -> None:
    machine_classes = []
    for index in range(3):
        machine_classes.append({
            "machine_class": index,
            "count": MACHINE_COUNTS[index],
            "maximum_capacity": MACHINE_CAPACITIES[index],
            "configuration_origin": "scenario_configured",
            "degradation_rate_range": list(DEGRADATION_RATE_RANGES[index]),
            "unexpected_failure_rate_range": list(
                UNEXPECTED_FAILURE_RATE_RANGES[index]),
            "terminal_degradation_rate": TERMINAL_DEGRADATION_RATES[index],
            "ipm_recovery_rates": list(IPM_RECOVERY_RATES[index]),
            "ipm_time": IPM_TIMES[index],
            "pm_time": PM_TIMES[index],
            "corrective_time_range": list(CORRECTIVE_TIME_RANGES[index]),
            "minimal_repair_time": MINIMAL_REPAIR_TIMES[index],
            "degradation_and_maintenance_origin": (
                "unchanged_study_model_parameter"),
        })
    _atomic_json(
        DATASET_ROOT / "configuration" / "machine_configuration.json",
        {
            "schema_version": 1,
            "machine_classes": machine_classes,
            "aggregate_nominal_capacity": sum(
                count * capacity for count, capacity in zip(
                    MACHINE_COUNTS, MACHINE_CAPACITIES, strict=True)),
            "capacity_design_basis": (
                "The three-class capacity ratio of the study is retained and "
                "scaled to a maximum capacity of 200, consistent with the "
                "largest reported batch weight in the industrial case."),
        },
    )
    _atomic_json(
        DATASET_ROOT / "configuration" / "simulation_parameters.json",
        {
            "schema_version": 1,
            "parameter_origin": "unchanged_study_model_parameter",
            "degradation_states": DEGRADATION_STATES,
            "failure_types": FAILURE_TYPES,
            "failure_type_proportions": list(FAILURE_TYPE_PROPORTIONS),
            "production_rate_range": list(PRODUCTION_RATE_RANGE),
            "simulation_seeds": list(SIMULATION_SEEDS),
        },
    )
    _atomic_json(
        DATASET_ROOT / "configuration" / "generation_protocol.json",
        {
            "schema_version": 1,
            "dataset_version": DATASET_VERSION,
            "protocol": GENERATION_PROTOCOL,
            "target_nominal_finite_loads": list(TARGET_LOADS),
            "instances_per_load": INSTANCES_PER_LOAD,
            "initial_job_count": INITIAL_JOB_COUNT,
            "initial_job_fraction": INITIAL_JOB_COUNT / 89,
            "minimum_nominal_slack_ratio": MINIMUM_NOMINAL_SLACK_RATIO,
            "processing_times_by_source_family_1_to_6": list(PROCESSING_TIMES),
            "instance_seed_base": INSTANCE_SEED_BASE,
            "simulation_replications_per_instance": len(SIMULATION_SEEDS),
            "arrival_process": (
                "normalized exponential interarrival spacings with seeded "
                "due-feasible order-to-slot assignment"),
            "arrival_span_formula": (
                "sum(weight_i * processing_time_i) / "
                "(target_load * sum(machine_count_c * capacity_c))"),
            "due_time_rule": "preserve the observed industrial due time",
            "source_pdf": ({
                "file_name": source_pdf.name,
                "sha256": _sha256(source_pdf),
                "source_location": "Table 4.1, thesis pages 45-47",
                "copied_into_dataset": False,
            } if source_pdf is not None else {
                "file_name": None,
                "sha256": None,
                "source_location": "Table 4.1, thesis pages 45-47",
                "copied_into_dataset": False,
                "warning": "Source PDF was not supplied when rebuilding.",
            }),
        },
    )
    _atomic_json(
        DATASET_ROOT / "metadata" / "field_provenance.json",
        {
            "observed_industrial": [
                "source_order_id", "weight", "due_time", "family_code",
                "processing_time",
            ],
            "synthetically_generated": ["arrival_time", "is_initial"],
            "scenario_configured": ["machine_count", "maximum_capacity"],
            "unchanged_study_model_parameter": [
                "degradation parameters", "failure parameters",
                "IPM parameters", "PM parameters", "CM parameters",
                "production rate range",
            ],
            "family_index_mapping": (
                "source family codes 1-6 are mapped to simulator indices 0-5"),
        },
    )


def _family_summary(orders: Sequence[IndustrialOrder]) -> list[dict[str, object]]:
    grouped: dict[int, list[IndustrialOrder]] = defaultdict(list)
    for order in orders:
        grouped[order.family_code].append(order)
    output = []
    for family in sorted(grouped):
        items = grouped[family]
        output.append({
            "source_family_code": family,
            "simulator_family": family - 1,
            "job_count": len(items),
            "processing_time": items[0].processing_time,
            "total_weight": sum(item.weight for item in items),
            "mean_weight": _format_float(np.mean([
                item.weight for item in items])),
            "minimum_due_time": int(min(item.due_time for item in items)),
            "maximum_due_time": int(max(item.due_time for item in items)),
        })
    return output


def _write_checksums() -> None:
    checksum_path = DATASET_ROOT / "checksums.json"
    files = []
    for path in sorted(DATASET_ROOT.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        relative_parts = path.relative_to(DATASET_ROOT).parts
        if (
                "screening" in relative_parts
                or "__pycache__" in relative_parts
                or path.suffix == ".pyc"):
            continue
        files.append({
            "path": path.relative_to(DATASET_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    _atomic_json(checksum_path, {
        "schema_version": 1,
        "algorithm": "SHA-256",
        "files": files,
    })


def _find_default_source_pdf() -> Path | None:
    project_root = DATASET_ROOT.parents[1]
    candidates = sorted(project_root.glob("*热处理*真实数据*.pdf"))
    return candidates[0].resolve() if candidates else None


def _remove_stale_generated_csvs() -> None:
    instance_root = DATASET_ROOT / "instances"
    if not instance_root.exists():
        return
    for path in instance_root.rglob("*.csv"):
        _assert_within_dataset(path).unlink()


def main() -> None:
    args = parse_args()
    source_pdf = args.source_pdf.resolve() if args.source_pdf else (
        _find_default_source_pdf())
    if source_pdf is not None and not source_pdf.is_file():
        raise FileNotFoundError(source_pdf)

    orders = load_source_orders()
    _remove_stale_generated_csvs()
    _write_configurations(source_pdf)

    manifest_rows: list[dict[str, object]] = []
    all_minimum_slacks: list[float] = []
    for load_index, target_load in enumerate(TARGET_LOADS):
        for instance_number in range(1, INSTANCES_PER_LOAD + 1):
            arrival_seed = _arrival_seed(load_index, instance_number)
            jobs, arrival_span = build_dynamic_jobs(
                orders,
                target_load=target_load,
                arrival_seed=arrival_seed,
            )
            job_path, provenance_path = _write_instance(
                jobs,
                target_load=target_load,
                instance_number=instance_number,
            )
            slacks = [job.nominal_slack for job in jobs]
            all_minimum_slacks.append(min(slacks))
            initial_family_counts = Counter(
                job.order.family_code for job in jobs if job.is_initial)
            manifest_rows.append({
                "target_load": _format_float(target_load),
                "instance_number": instance_number,
                "arrival_seed": arrival_seed,
                "job_file": job_path.relative_to(DATASET_ROOT).as_posix(),
                "provenance_file": provenance_path.relative_to(
                    DATASET_ROOT).as_posix(),
                "job_count": len(jobs),
                "initial_job_count": sum(job.is_initial for job in jobs),
                "dynamic_job_count": sum(not job.is_initial for job in jobs),
                "arrival_span": _format_float(arrival_span),
                "weighted_nominal_workload": 655_840,
                "aggregate_nominal_capacity": 480,
                "realized_finite_load": _format_float(target_load),
                "minimum_nominal_slack": _format_float(min(slacks)),
                "mean_nominal_slack": _format_float(np.mean(slacks)),
                "initial_family_counts_1_to_6": json.dumps([
                    initial_family_counts.get(family, 0)
                    for family in range(1, 7)
                ], separators=(",", ":")),
                "job_file_sha256": _sha256(job_path),
                "provenance_file_sha256": _sha256(provenance_path),
            })

    manifest_fields = (
        "target_load", "instance_number", "arrival_seed", "job_file",
        "provenance_file", "job_count", "initial_job_count",
        "dynamic_job_count", "arrival_span", "weighted_nominal_workload",
        "aggregate_nominal_capacity", "realized_finite_load",
        "minimum_nominal_slack", "mean_nominal_slack",
        "initial_family_counts_1_to_6", "job_file_sha256",
        "provenance_file_sha256",
    )
    _atomic_csv(
        DATASET_ROOT / "metadata" / "instance_manifest.csv",
        manifest_fields,
        manifest_rows,
    )
    family_rows = _family_summary(orders)
    _atomic_csv(
        DATASET_ROOT / "audit" / "source_family_summary.csv",
        tuple(family_rows[0].keys()),
        family_rows,
    )
    _atomic_json(
        DATASET_ROOT / "audit" / "construction_audit.json",
        {
            "schema_version": 1,
            "status": "passed",
            "dataset_version": DATASET_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_order_count": len(orders),
            "source_total_weight": sum(item.weight for item in orders),
            "source_weighted_nominal_workload": sum(
                item.weight * item.processing_time for item in orders),
            "source_due_time_range": [
                min(item.due_time for item in orders),
                max(item.due_time for item in orders),
            ],
            "dynamic_instance_count": len(manifest_rows),
            "target_loads": list(TARGET_LOADS),
            "instances_per_load": INSTANCES_PER_LOAD,
            "simulation_replications_per_instance": len(SIMULATION_SEEDS),
            "all_instances_have_89_unique_source_orders": True,
            "all_instances_have_18_initial_jobs": True,
            "all_instances_match_target_finite_load": True,
            "all_release_due_constraints_passed": True,
            "global_minimum_nominal_slack": min(all_minimum_slacks),
            "machine_counts": list(MACHINE_COUNTS),
            "machine_capacities": list(MACHINE_CAPACITIES),
            "maintenance_parameters_modified": False,
        },
    )
    _write_checksums()
    print(
        "Industrial dynamic case built successfully: "
        f"{len(orders)} source orders, {len(manifest_rows)} instances, "
        f"{len(manifest_rows) * len(SIMULATION_SEEDS)} planned simulations.")
    print(f"Dataset root: {DATASET_ROOT}")


if __name__ == "__main__":
    main()
