"""Method-neutral detailed CSV rows for formal comparison evaluation."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment.evaluation import EvaluationRecord


CASE_FIELDS = (
    "instance_index", "replication_index", "instance_seed", "simulation_seed")
EPISODE_FIELDS = tuple(
    name for name in EvaluationRecord.__dataclass_fields__
    if name not in CASE_FIELDS)
ACTION_NAMES = {0: "Process", 1: "IPM", 2: "PM", 3: "Wait"}


def episode_row(identity, metrics, simulator, *, wall_time, error=None):
    row = {
        **identity,
        "status": "failed" if error is not None else "completed",
        "error_type": type(error).__name__ if error is not None else "",
        "error_message": (
            str(error).replace("\r", " ").replace("\n", " ")
            if error is not None else ""),
        "episode_wall_time_seconds": wall_time,
    }
    for field in EPISODE_FIELDS:
        row[field] = "" if metrics is None else metrics.get(field, "")
    if metrics is None:
        row.update({
            "batch_count": "",
            "mean_jobs_per_batch": "",
            "terminal_degradation_failures": "",
        })
        return row
    batch_count = sum(machine.batch_count for machine in simulator.machines)
    terminal_failures = sum(
        event.get("event") == "failure" and event.get("failure_type") == -1
        for machine in simulator.machines for event in machine.history)
    row.update({
        "batch_count": batch_count,
        "mean_jobs_per_batch": (
            metrics["job_count"] / batch_count if batch_count else 0.0),
        "terminal_degradation_failures": terminal_failures,
    })
    return row


def machine_rows(identity, simulator, makespan):
    rows = []
    horizon = max(1e-12, float(makespan))
    for machine in simulator.machines:
        failure_types = defaultdict(int)
        for event in machine.history:
            if event.get("event") == "failure":
                failure_types[int(event["failure_type"])] += 1
        rows.append({
            **identity,
            "machine_id": machine.machine_id,
            "machine_class": machine.class_id,
            "local_machine_id": machine.local_id,
            "capacity": machine.capacity,
            "final_health_state": machine.state,
            "batch_count": machine.batch_count,
            "loaded_size": machine.loaded_size,
            "capacity_committed": machine.capacity_committed,
            "batch_fill_ratio": (
                machine.loaded_size / machine.capacity_committed
                if machine.capacity_committed else 0.0),
            "processing_time": machine.processing_time,
            "processing_utilization": machine.processing_time / horizon,
            "maintenance_time": machine.maintenance_time,
            "maintenance_utilization": machine.maintenance_time / horizon,
            "planned_maintenance_time": machine.planned_maintenance_time,
            "corrective_maintenance_time": machine.corrective_maintenance_time,
            "ipm_count": machine.ipm_count,
            "pm_count": machine.pm_count,
            "corrective_maintenance_count": machine.corrective_count,
            "breakdown_count": machine.breakdown_count,
            "unexpected_breakdown_count": machine.unexpected_breakdown_count,
            "critical_failure_count": machine.critical_failure_count,
            "failure_interruptions": machine.failure_interruptions,
            "terminal_degradation_failures": failure_types[-1],
            "failure_type_0_count": failure_types[0],
            "failure_type_1_count": failure_types[1],
            "failure_type_2_count": failure_types[2],
            "history_event_count": len(machine.history),
        })
    return rows


def state_action_rows(identity, simulator):
    grouped = defaultdict(list)
    totals = defaultdict(int)
    for decision in simulator.decision_history:
        action = int(decision["maintenance_action"])
        key = (
            int(decision["machine_id"]),
            int(decision["machine_class"]),
            int(decision["machine_state"]),
            action,
        )
        grouped[key].append(float(decision["latency_seconds"]))
        totals[key[:3]] += 1
    rows = []
    for (machine_id, machine_class, health_state, action), latencies in sorted(
            grouped.items()):
        values = np.asarray(latencies, dtype=float)
        rows.append({
            **identity,
            "machine_id": machine_id,
            "machine_class": machine_class,
            "health_state": health_state,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "decision_count": len(latencies),
            "fraction_within_machine_health_state": (
                len(latencies) / totals[
                    (machine_id, machine_class, health_state)]),
            "decision_latency_mean": float(values.mean()),
            "decision_latency_median": float(np.median(values)),
            "decision_latency_p95": float(np.percentile(values, 95)),
            "decision_latency_p99": float(np.percentile(values, 99)),
        })
    return rows


def job_rows(identity, simulator):
    return [{
        **identity,
        "job_id": job.job_id,
        "arrival_time": job.arrival_time,
        "family": job.family,
        "size": job.size,
        "due_time": job.due_time,
        "first_start_time": job.first_start_time,
        "finish_time": job.finish_time,
        "queue_waiting_time": job.first_start_time - job.arrival_time,
        "flow_time": job.finish_time - job.arrival_time,
        "tardiness": max(0.0, job.finish_time - job.due_time),
        "tardy": job.finish_time > job.due_time,
    } for job in simulator.jobs]


def consolidate_csv(source_dir: Path, target: Path) -> int:
    files = sorted(source_dir.glob("*.csv"))
    temporary = target.with_name(f".{target.name}.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fields = None
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = None
        for path in files:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if fields is None:
                    fields = reader.fieldnames
                    writer = csv.DictWriter(output, fieldnames=fields)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
                    count += 1
    temporary.replace(target)
    return count
