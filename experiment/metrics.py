"""Canonical episode metrics shared by every experiment method."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EpisodeMetrics:
    total_tardiness: float
    mean_tardiness: float
    makespan: float
    breakdowns: int
    maintenance_time: float
    decisions: int
    tardy_job_rate: float = 0.0
    p95_job_tardiness: float = 0.0
    max_job_tardiness: float = 0.0
    mean_flow_time: float = 0.0
    p95_flow_time: float = 0.0
    throughput: float = 0.0
    production_utilization: float = 0.0
    batch_fill_ratio: float = 0.0
    mean_queue_waiting: float = 0.0
    unexpected_breakdowns: int = 0
    critical_failures: int = 0
    failure_interruptions: int = 0
    ipm_count: int = 0
    pm_count: int = 0
    corrective_maintenance_count: int = 0
    planned_maintenance_time: float = 0.0
    corrective_maintenance_time: float = 0.0
    maintenance_downtime_ratio: float = 0.0
    failed_batch_restarts: int = 0
    reprocessed_work: float = 0.0
    decision_latency_mean: float = 0.0
    decision_latency_p95: float = 0.0
    decision_latency_p99: float = 0.0
    terminated: bool = True
    unfinished_jobs: int = 0
    invalid_action_fallbacks: int = 0
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
    def from_mapping(cls, values: dict) -> "EpisodeMetrics":
        field_types = cls.__dataclass_fields__
        required = {
            "total_tardiness", "mean_tardiness", "makespan", "breakdowns",
            "maintenance_time", "decisions",
        }
        missing = required - values.keys()
        if missing:
            raise KeyError(f"Missing metric fields: {sorted(missing)}")
        kwargs = {
            name: values[name]
            for name in field_types
            if name in values
            and name not in {
                "decision_latency_mean",
                "decision_latency_p95",
                "decision_latency_p99",
            }
        }
        result = cls(**kwargs)
        if result.total_tardiness < 0 or result.mean_tardiness < 0:
            raise ValueError("Tardiness metrics cannot be negative.")
        if result.makespan < 0 or result.maintenance_time < 0:
            raise ValueError("Time metrics cannot be negative.")
        if result.breakdowns < 0 or result.decisions <= 0:
            raise ValueError("Count metrics are invalid.")
        return result

    def to_dict(self) -> dict:
        return asdict(self)
