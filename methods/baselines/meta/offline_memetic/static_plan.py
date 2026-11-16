"""Read-only execution contract for C++-generated offline memetic plans."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from methods.baselines.sdr.rules import proposal_from_batch
from methods.hierarchical.types import Job, MachineAction


STATIC_PLAN_SCHEMA_VERSION = 1
STATIC_PLAN_ACTIONS = ("PROCESS", "IPM", "PM")


@dataclass(frozen=True)
class StaticPlanOperation:
    operation_index: int
    sequence_index: int
    machine_id: int
    machine_class: int
    machine_capacity: int
    action: str
    planned_start: float
    planned_end: float
    family: int | None
    batch_size: int
    job_ids: tuple[int, ...]


@dataclass(frozen=True)
class StaticPlan:
    path: Path
    dataset_contract_sha256: str
    optimizer_contract_sha256: str
    condition_index: int
    condition_id: str
    instance_index: int
    instance_seed: int
    optimizer_seed: int
    objective_total_tardiness: float
    operations: tuple[StaticPlanOperation, ...]

    @property
    def operations_by_machine(
            self) -> dict[int, tuple[StaticPlanOperation, ...]]:
        grouped: dict[int, list[StaticPlanOperation]] = {}
        for operation in self.operations:
            grouped.setdefault(operation.machine_id, []).append(operation)
        return {
            machine_id: tuple(sorted(
                values,
                key=lambda item: (
                    item.planned_start,
                    item.operation_index,
                ),
            ))
            for machine_id, values in grouped.items()
        }


def _constant(rows: Sequence[dict[str, str]], field: str) -> str:
    values = {row[field] for row in rows}
    if len(values) != 1:
        raise ValueError(f"Static-plan field is not constant: {field}")
    return values.pop()


def load_static_plan(
        path: str | Path,
        *,
        expected_dataset_contract_sha256: str | None = None,
        expected_optimizer_contract_sha256: str | None = None,
        ) -> StaticPlan:
    """Load and strictly validate one immutable static plan CSV."""

    target = Path(path).resolve()
    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Static plan contains no operations: {target}")
    dataset_contract = _constant(rows, "dataset_contract_sha256")
    optimizer_contract = _constant(rows, "optimizer_contract_sha256")
    if (
            expected_dataset_contract_sha256 is not None
            and dataset_contract != expected_dataset_contract_sha256):
        raise ValueError(f"Static plan belongs to another dataset: {target}")
    if (
            expected_optimizer_contract_sha256 is not None
            and optimizer_contract != expected_optimizer_contract_sha256):
        raise ValueError(f"Static plan belongs to another optimizer: {target}")
    operations = []
    seen_operation_indices = set()
    for row in rows:
        action = row["action"]
        if action not in STATIC_PLAN_ACTIONS:
            raise ValueError(f"Unknown static-plan action {action!r}: {target}")
        operation_index = int(row["operation_index"])
        if operation_index in seen_operation_indices:
            raise ValueError(f"Duplicate static-plan operation: {target}")
        seen_operation_indices.add(operation_index)
        start = float(row["planned_start"])
        end = float(row["planned_end"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError(f"Invalid planned interval: {target}")
        job_ids = tuple(
            int(value) for value in row["job_ids"].split(";") if value)
        family = int(row["family"]) if row["family"] else None
        if action == "PROCESS" and (not job_ids or family is None):
            raise ValueError(f"Production operation has no batch: {target}")
        if action != "PROCESS" and (job_ids or family is not None):
            raise ValueError(f"Maintenance operation contains jobs: {target}")
        operations.append(StaticPlanOperation(
            operation_index=operation_index,
            sequence_index=int(row["sequence_index"]),
            machine_id=int(row["machine_id"]),
            machine_class=int(row["machine_class"]),
            machine_capacity=int(row["machine_capacity"]),
            action=action,
            planned_start=start,
            planned_end=end,
            family=family,
            batch_size=int(row["batch_size"]),
            job_ids=job_ids,
        ))
    plan = StaticPlan(
        path=target,
        dataset_contract_sha256=dataset_contract,
        optimizer_contract_sha256=optimizer_contract,
        condition_index=int(_constant(rows, "condition_index")),
        condition_id=_constant(rows, "condition_id"),
        instance_index=int(_constant(rows, "instance_index")),
        instance_seed=int(_constant(rows, "instance_seed")),
        optimizer_seed=int(_constant(rows, "optimizer_seed")),
        objective_total_tardiness=float(
            _constant(rows, "objective_total_tardiness")),
        operations=tuple(operations),
    )
    expected_indices = set(range(len(plan.operations)))
    if seen_operation_indices != expected_indices:
        raise ValueError(f"Static-plan operation indices are not contiguous: {target}")
    for machine_id, machine_operations in plan.operations_by_machine.items():
        previous_end = 0.0
        for operation in machine_operations:
            if operation.machine_id != machine_id \
                    or operation.planned_start + 1e-8 < previous_end:
                raise ValueError(f"Static-plan machine operations overlap: {target}")
            previous_end = operation.planned_end
    return plan


class StaticPlanController:
    """Execute a frozen plan without changing batches, machines, or actions."""

    control_trajectory_scope = "physical"
    allow_maintenance_at_healthy = True

    def __init__(self, plan: StaticPlan | None = None) -> None:
        self.plan = plan
        self._operations: dict[int, tuple[StaticPlanOperation, ...]] = {}
        self._cursor: dict[int, int] = {}
        self._jobs: dict[int, Job] = {}
        self._machine_capacities: dict[int, int] = {}
        self._active_process: dict[int, StaticPlanOperation] = {}
        self.execution_trace: list[dict] = []

    def set_plan(self, plan: StaticPlan) -> None:
        self.plan = plan

    def begin_episode(self, _: int) -> None:
        self._operations = (
            {} if self.plan is None else self.plan.operations_by_machine)
        self._cursor = {
            machine_id: 0 for machine_id in self._operations
        }
        self._jobs = {}
        self._machine_capacities = {}
        self._active_process = {}
        self.execution_trace = []

    def bind_episode(self, *, jobs: Sequence[Job], machines: Sequence) -> None:
        """Bind the plan to fresh simulator objects and verify exact coverage."""

        if self.plan is None:
            return
        if self.plan.instance_seed < 0:
            raise ValueError("Static plan has an invalid instance seed.")
        self._jobs = {job.job_id: job for job in jobs}
        self._machine_capacities = {
            machine.machine_id: machine.capacity for machine in machines}
        planned_job_ids = [
            job_id
            for operation in self.plan.operations
            if operation.action == "PROCESS"
            for job_id in operation.job_ids
        ]
        if sorted(planned_job_ids) != sorted(self._jobs):
            raise ValueError(
                "Static plan must cover every bound job exactly once.")
        for operation in self.plan.operations:
            if operation.machine_id not in self._machine_capacities:
                raise ValueError("Static plan references an unknown machine.")
            if operation.machine_capacity \
                    != self._machine_capacities[operation.machine_id]:
                raise ValueError("Static plan machine capacity has changed.")
            if operation.action != "PROCESS":
                continue
            selected = [self._jobs[job_id] for job_id in operation.job_ids]
            if {job.family for job in selected} != {operation.family}:
                raise ValueError("Static plan mixes incompatible job families.")
            if sum(job.size for job in selected) != operation.batch_size:
                raise ValueError("Static plan batch size does not match jobs.")
            if operation.batch_size > operation.machine_capacity:
                raise ValueError("Static plan exceeds machine capacity.")

    def has_pending_action(self, machine_id: int) -> bool:
        operations = self._operations.get(machine_id, ())
        return self._cursor.get(machine_id, 0) < len(operations)

    def _current(self, machine_id: int) -> StaticPlanOperation:
        if not self.has_pending_action(machine_id):
            raise RuntimeError(
                f"Static plan has no pending operation for machine {machine_id}.")
        return self._operations[machine_id][self._cursor[machine_id]]

    def _proposal_operation(self, machine_id: int) -> StaticPlanOperation:
        operations = self._operations[machine_id]
        for operation in operations[self._cursor[machine_id]:]:
            if operation.action == "PROCESS":
                return operation
        raise RuntimeError(
            "A planned maintenance action must precede a production batch.")

    def build_proposals(
            self,
            jobs,
            capacity,
            clock,
            processing_times,
            *,
            requester_id,
            machine_state,
            ):
        del jobs, machine_state
        if self.plan is None or not self.has_pending_action(requester_id):
            return []
        operation = self._current(requester_id)
        proposal_operation = (
            operation if operation.action == "PROCESS"
            else self._proposal_operation(requester_id)
        )
        selected = [self._jobs[job_id] for job_id in proposal_operation.job_ids]
        if operation.action == "PROCESS":
            if any(job.finished or job.reserved_by is not None for job in selected):
                raise RuntimeError(
                    "A planned production batch is already finished or reserved.")
            if any(job.arrival_time > clock + 1e-10 for job in selected):
                return []
        proposal = proposal_from_batch(
            (int(proposal_operation.family), proposal_operation.job_ids),
            list(self._jobs.values()),
            capacity,
            clock,
            tuple(processing_times),
            "OPT",
        )
        return [replace(
            proposal, proposal_id=proposal_operation.operation_index)]

    def select_action(
            self,
            observation,
            *,
            requester_id,
            machine_state,
            clock,
            retained_batch=False,
            **_,
            ) -> int:
        if retained_batch:
            operation = self._active_process.get(requester_id)
            if operation is None:
                raise RuntimeError(
                    "Corrective-maintenance resume has no active planned batch.")
            self._trace(
                operation, clock, machine_state,
                execution_kind="resume_after_corrective_maintenance")
            return 0
        operation = self._current(requester_id)
        proposal_count = observation.action_mask.size - 2
        if operation.action == "PROCESS":
            action = 0
            self._active_process[requester_id] = operation
        elif operation.action == "IPM":
            action = proposal_count
        else:
            action = proposal_count + 1
        self._cursor[requester_id] += 1
        self._trace(operation, clock, machine_state, execution_kind="planned")
        return action

    def _trace(
            self,
            operation: StaticPlanOperation,
            clock: float,
            machine_state: int,
            *,
            execution_kind: str) -> None:
        self.execution_trace.append({
            "trace_index": len(self.execution_trace),
            "plan_operation_index": operation.operation_index,
            "sequence_index": operation.sequence_index,
            "machine_id": operation.machine_id,
            "action": operation.action,
            "planned_start": operation.planned_start,
            "planned_end": operation.planned_end,
            "actual_decision_time": float(clock),
            "actual_health_state": int(machine_state),
            "family": "" if operation.family is None else operation.family,
            "batch_size": operation.batch_size,
            "job_ids": ";".join(map(str, operation.job_ids)),
            "execution_kind": execution_kind,
        })

    def observe(self, transition) -> None:
        del transition

    def assert_consumed(self) -> None:
        remaining = {
            machine_id: len(operations) - self._cursor[machine_id]
            for machine_id, operations in self._operations.items()
            if self._cursor[machine_id] != len(operations)
        }
        if remaining:
            raise RuntimeError(
                f"Episode ended before consuming the static plan: {remaining}")
