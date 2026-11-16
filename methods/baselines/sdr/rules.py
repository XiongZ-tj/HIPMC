"""Pure, testable batching and dispatching rules used by SDR18."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from methods.hierarchical.types import BatchProposal, Job


BATCH_FORMATION_RULES = ("CH", "EDD-J", "FFD")
BATCH_SELECTION_RULES = ("MMU", "EDD-B", "MAST")
MAINTENANCE_RULES = ("Threshold", "Period")


def _pack(order: Iterable[Job], capacity: int) -> tuple[int, ...]:
    remaining = capacity
    result: list[int] = []
    for job in order:
        if job.size <= remaining:
            result.append(job.job_id)
            remaining -= job.size
    return tuple(result)


def form_batches(
        jobs: list[Job],
        capacity: int,
        clock: float,
        processing_times: tuple[int, ...],
        rule: str,
) -> list[tuple[int, tuple[int, ...]]]:
    """Form one feasible candidate batch per currently available family."""

    if rule not in BATCH_FORMATION_RULES:
        raise ValueError(f"Unknown batch-formation rule: {rule}")
    result = []
    for family in sorted({job.family for job in jobs}):
        family_jobs = [job for job in jobs if job.family == family]
        if rule == "EDD-J":
            order = sorted(
                family_jobs, key=lambda job: (job.due_time, job.job_id))
        elif rule == "FFD":
            order = sorted(
                family_jobs,
                key=lambda job: (
                    -job.size,
                    job.due_time,
                    job.job_id,
                ),
            )
        else:
            urgent = sorted(
                (
                    job for job in family_jobs
                    if job.due_time - clock - processing_times[family] <= 0
                ),
                key=lambda job: (job.size, job.due_time, job.job_id),
            )
            nonurgent = sorted(
                (
                    job for job in family_jobs
                    if job.due_time - clock - processing_times[family] > 0
                ),
                key=lambda job: (
                    job.due_time - clock - processing_times[family],
                    job.job_id,
                ),
            )
            order = [*urgent, *nonurgent]
        packed = _pack(order, capacity)
        if packed:
            result.append((family, packed))
    return result


def select_batch(
        batches: list[tuple[int, tuple[int, ...]]],
        jobs: list[Job],
        clock: float,
        processing_times: tuple[int, ...],
        rule: str,
) -> int:
    """Return the selected batch index; all directions match their definitions."""

    if not batches:
        raise ValueError("Cannot select from an empty batch list.")
    if rule not in BATCH_SELECTION_RULES:
        raise ValueError(f"Unknown batch-selection rule: {rule}")
    by_id = {job.job_id: job for job in jobs}
    if rule == "MMU":
        return max(
            range(len(batches)),
            key=lambda index: (
                sum(by_id[job_id].size for job_id in batches[index][1]),
                -batches[index][0],
            ),
        )
    if rule == "EDD-B":
        return min(
            range(len(batches)),
            key=lambda index: (
                min(by_id[job_id].due_time for job_id in batches[index][1]),
                batches[index][0],
            ),
        )
    if rule == "MAST":
        return min(
            range(len(batches)),
            key=lambda index: (
                np.mean([
                    by_id[job_id].due_time
                    - clock
                    - processing_times[batches[index][0]]
                    for job_id in batches[index][1]
                ]),
                batches[index][0],
            ),
        )

    raise AssertionError(f"Unhandled batch-selection rule: {rule}")


def proposal_from_batch(
        batch: tuple[int, tuple[int, ...]],
        jobs: list[Job],
        capacity: int,
        clock: float,
        processing_times: tuple[int, ...],
        selection_rule: str,
) -> BatchProposal:
    family, job_ids = batch
    by_id = {job.job_id: job for job in jobs}
    selected = [by_id[job_id] for job_id in job_ids]
    sizes = np.asarray([job.size for job in selected], dtype=np.float32)
    due = np.asarray([job.due_time for job in selected], dtype=np.float32)
    slack = due - clock - processing_times[family]
    provenance = np.zeros(len(BATCH_SELECTION_RULES), dtype=np.float32)
    if selection_rule in BATCH_SELECTION_RULES:
        provenance[BATCH_SELECTION_RULES.index(selection_rule)] = 1.0
    maximum = max(processing_times)
    features = np.asarray([
        len(selected) / max(1, len(jobs)),
        sizes.sum() / capacity,
        sizes.mean() / capacity,
        np.clip((due.min() - clock) / maximum, -5, 5),
        np.clip((due.mean() - clock) / maximum, -5, 5),
        np.clip(slack.min() / maximum, -5, 5),
        np.clip(slack.mean() / maximum, -5, 5),
        processing_times[family] / maximum,
        family / max(1, len(processing_times) - 1),
        *provenance,
    ], dtype=np.float32)
    return BatchProposal(
        proposal_id=0,
        job_ids=job_ids,
        family=family,
        base_criteria=np.zeros(5, dtype=np.float32),
        features=features,
    )


def build_rule_proposal(
        jobs: list[Job],
        capacity: int,
        clock: float,
        processing_times: tuple[int, ...],
        formation_rule: str,
        selection_rule: str,
) -> list[BatchProposal]:
    batches = form_batches(
        jobs, capacity, clock, processing_times, formation_rule)
    if not batches:
        return []
    selected = select_batch(
        batches, jobs, clock, processing_times, selection_rule)
    return [proposal_from_batch(
        batches[selected], jobs, capacity, clock, processing_times,
        selection_rule,
    )]
