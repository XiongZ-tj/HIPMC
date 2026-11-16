"""Shared, auditable evaluation protocol for every scheduling method."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class EvaluationCase:
    """One method-independent test instance and simulation replication."""

    instance_index: int
    replication_index: int
    instance_seed: int
    simulation_seed: int


@dataclass(frozen=True)
class EvaluationRecord:
    """The result and full random-number identity of one evaluation run."""

    instance_index: int
    replication_index: int
    instance_seed: int
    simulation_seed: int
    total_tardiness: float
    mean_tardiness: float = 0.0
    tardy_job_rate: float = 0.0
    p95_job_tardiness: float = 0.0
    max_job_tardiness: float = 0.0
    mean_flow_time: float = 0.0
    p95_flow_time: float = 0.0
    makespan: float = 0.0
    throughput: float = 0.0
    production_utilization: float = 0.0
    batch_fill_ratio: float = 0.0
    mean_queue_waiting: float = 0.0
    breakdowns: int = 0
    unexpected_breakdowns: int = 0
    critical_failures: int = 0
    failure_interruptions: int = 0
    maintenance_time: float = 0.0
    planned_maintenance_time: float = 0.0
    corrective_maintenance_time: float = 0.0
    ipm_count: int = 0
    pm_count: int = 0
    corrective_maintenance_count: int = 0
    maintenance_downtime_ratio: float = 0.0
    failed_batch_restarts: int = 0
    reprocessed_work: float = 0.0
    decision_latency_mean: float = 0.0
    decision_latency_p95: float = 0.0
    decision_latency_p99: float = 0.0
    terminated: bool = True
    unfinished_jobs: int = 0
    invalid_action_fallbacks: int = 0
    decisions: int = 0
    job_count: int = 0
    initial_job_count: int = 0
    dynamic_job_count: int = 0
    target_load: float = 0.0
    arrival_span: float = 0.0
    dynamic_nominal_workload: float = 0.0
    initial_nominal_workload: float = 0.0
    total_nominal_workload: float = 0.0
    realized_dynamic_load: float = 0.0
    finite_instance_load: float = 0.0
    actual_processing_load: float = 0.0
    actual_maintenance_load: float = 0.0

    @classmethod
    def from_case(
            cls,
            case: EvaluationCase,
            metrics: float | dict) -> "EvaluationRecord":
        values = (
            {"total_tardiness": float(metrics)}
            if not isinstance(metrics, dict)
            else dict(metrics)
        )
        accepted = cls.__dataclass_fields__.keys()
        payload = {
            name: values[name]
            for name in accepted
            if name in values
        }
        return cls(
            instance_index=case.instance_index,
            replication_index=case.replication_index,
            instance_seed=case.instance_seed,
            simulation_seed=case.simulation_seed,
            **payload,
        )


@dataclass(frozen=True)
class EvaluationProtocol:
    """Deterministic two-level seed schedule shared by every method.

    ``instance_seed`` fixes job arrivals and job attributes. A distinct
    ``simulation_seed`` fixes the stochastic simulation stream for each
    replication of that instance.
    """

    instance_count: int = 10
    replications: int = 5
    instance_seed_start: int = 10000
    simulation_seed_start: int = 20000

    def __post_init__(self) -> None:
        if self.instance_count <= 0:
            raise ValueError("instance_count must be positive.")
        if self.replications <= 0:
            raise ValueError("replications must be positive.")
        for name in ("instance_seed_start", "simulation_seed_start"):
            value = getattr(self, name)
            if not 0 <= value <= 2**32 - 1:
                raise ValueError(f"{name} must be in [0, 2**32 - 1].")
        last_simulation_seed = (
            self.simulation_seed_start
            + self.instance_count * self.replications
            - 1
        )
        last_instance_seed = self.instance_seed_start + self.instance_count - 1
        if last_instance_seed > 2**32 - 1:
            raise ValueError("The generated instance seed exceeds 2**32 - 1.")
        if last_simulation_seed > 2**32 - 1:
            raise ValueError("The generated simulation seed exceeds 2**32 - 1.")

    def cases(self) -> Iterator[EvaluationCase]:
        for instance_index in range(self.instance_count):
            instance_seed = self.instance_seed_start + instance_index
            for replication_index in range(self.replications):
                yield EvaluationCase(
                    instance_index=instance_index,
                    replication_index=replication_index,
                    instance_seed=instance_seed,
                    simulation_seed=(
                        self.simulation_seed_start
                        + instance_index * self.replications
                        + replication_index
                    ),
                )

    def to_dict(self) -> dict:
        return asdict(self)


def seed_simulation(seed: int) -> None:
    """Reset all random generators used by the current implementations."""

    random.seed(seed)
    np.random.seed(seed)

    # Torch is optional for rule-based methods and is imported lazily.
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def merge_audit_tables(
        target: dict[str, list[dict]],
        source: dict[str, list[dict]]) -> None:
    for name, rows in source.items():
        target.setdefault(name, []).extend(rows)


def save_evaluation_outputs(
        records: Sequence[EvaluationRecord],
        protocol: EvaluationProtocol,
        results_dir: str | Path,
        legacy_filename: str,
        audit_tables: dict[str, list[dict]] | None = None) -> np.ndarray:
    "Save evaluation values and structured CSV/JSON protocol metadata."

    expected_count = protocol.instance_count * protocol.replications
    if len(records) != expected_count:
        raise ValueError(
            f"Expected {expected_count} evaluation records, got {len(records)}.")

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = output_dir / legacy_filename
    suffix = legacy_path.stem.removeprefix("Total_Tardiness_")
    records_path = output_dir / f"Evaluation_Records_{suffix}.csv"
    protocol_path = output_dir / f"Evaluation_Protocol_{suffix}.json"

    delays = np.asarray(
        [record.total_tardiness for record in records],
        dtype=float,
    )
    np.savetxt(legacy_path, delays, fmt="%d")

    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(EvaluationRecord.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    protocol_metadata = {
        "schema_version": 2,
        "protocol": protocol.to_dict(),
        "record_count": len(records),
        "aggregation": {
            "excluded_runs": 0,
            "description": "All replications are retained without trimming.",
        },
        "files": {
            "legacy_total_tardiness": legacy_path.name,
            "evaluation_records": records_path.name,
        },
    }
    if audit_tables:
        audit_files = {}
        for table, rows in audit_tables.items():
            path = output_dir / f"{table.title().replace('_', '')}_{suffix}.csv"
            if rows:
                fields = list(dict.fromkeys(
                    key for row in rows for key in row))
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({
                            key: (
                                json.dumps(value, ensure_ascii=False)
                                if isinstance(value, (list, tuple, dict))
                                else value
                            )
                            for key, value in row.items()
                        })
            audit_files[table] = path.name
        protocol_metadata["files"]["audit_tables"] = audit_files
    protocol_path.write_text(
        json.dumps(protocol_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Evaluation results saved to: {legacy_path}")
    print(f"Evaluation audit records saved to: {records_path}")
    return delays
