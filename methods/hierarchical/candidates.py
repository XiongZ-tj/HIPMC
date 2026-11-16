"Construct feasible family batches and their normalized descriptors."

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .types import BatchProposal, Job


BAA_CRITERIA = (
    "capacity_utilization",
    "mean_urgency",
    "worst_job_urgency",
    "waiting_starvation",
    "processing_efficiency",
    "uninterrupted_completion_probability",
)
RULE_NAMES = ("CH2", "MMU", "EDD-B", "MAST")


def _pack(order: Iterable[Job], capacity: int) -> tuple[int, ...]:
    remaining = capacity
    selected: list[int] = []
    for job in order:
        if job.size <= remaining:
            selected.append(job.job_id)
            remaining -= job.size
    return tuple(selected)


def _family_batches(
        jobs: list[Job],
        capacity: int,
        clock: float,
        processing_times: tuple[int, ...],
) -> list[tuple[int, tuple[int, ...]]]:
    "Pack urgent jobs first, then nonurgent jobs, for each feasible family."

    result: list[tuple[int, tuple[int, ...]]] = []
    for family in sorted({job.family for job in jobs}):
        family_jobs = [job for job in jobs if job.family == family]
        urgent = sorted(
            (job for job in family_jobs
             if job.due_time - clock - processing_times[family] <= 0),
            key=lambda job: (job.size, job.due_time, job.job_id),
        )
        nonurgent = sorted(
            (job for job in family_jobs
             if job.due_time - clock - processing_times[family] > 0),
            key=lambda job: (job.due_time, job.job_id),
        )
        packed = _pack((*urgent, *nonurgent), capacity)
        if packed:
            result.append((family, packed))
    return result


def _urgency(slack: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(slack / max(scale, 1e-12), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(normalized))


def _legacy_features(
        selected: list[Job],
        family: int,
        *,
        total_jobs: int,
        capacity: int,
        clock: float,
        processing_times: tuple[int, ...],
        provenance: np.ndarray | None = None) -> np.ndarray:
    sizes = np.asarray([job.size for job in selected], dtype=np.float32)
    due = np.asarray([job.due_time for job in selected], dtype=np.float32)
    max_pt = max(processing_times)
    pt = processing_times[family]
    sources = (
        np.zeros(len(RULE_NAMES), dtype=np.float32)
        if provenance is None else provenance.astype(np.float32, copy=False))
    return np.asarray([
        len(selected) / max(1, total_jobs),
        sizes.sum() / capacity,
        sizes.mean() / capacity,
        np.clip((due.min() - clock) / max_pt, -5, 5),
        np.clip((due.mean() - clock) / max_pt, -5, 5),
        np.clip((due.min() - clock - pt) / max_pt, -5, 5),
        np.clip((due.mean() - clock - pt) / max_pt, -5, 5),
        pt / max_pt,
        family / max(1, len(processing_times) - 1),
        *sources.tolist(),
    ], dtype=np.float32)


def build_proposals(
        jobs: list[Job],
        capacity: int,
        clock: float,
        processing_times: tuple[int, ...],
) -> list[BatchProposal]:
    """Return the complete feasible-family pool without rule preselection.

    The first five criteria are machine independent and lie in ``[0, 1]``.
    The environment appends the requesting-machine completion reliability as
    the sixth criterion before the BAA policy scores the candidates.
    """

    batches = _family_batches(jobs, capacity, clock, processing_times)
    if not batches:
        return []
    by_id = {job.job_id: job for job in jobs}
    max_pt = float(max(processing_times))
    raw_efficiencies = np.asarray([
        sum(by_id[job_id].size for job_id in job_ids)
        / float(processing_times[family])
        for family, job_ids in batches
    ], dtype=np.float64)
    efficiency_scale = max(float(raw_efficiencies.max()), 1e-12)

    proposals: list[BatchProposal] = []
    for proposal_id, ((family, job_ids), efficiency) in enumerate(zip(
            batches, raw_efficiencies, strict=True)):
        selected = [by_id[job_id] for job_id in job_ids]
        sizes = np.asarray([job.size for job in selected], dtype=np.float64)
        due = np.asarray([job.due_time for job in selected], dtype=np.float64)
        arrivals = np.asarray(
            [job.arrival_time for job in selected], dtype=np.float64)
        slack = due - clock - processing_times[family]
        urgency = _urgency(slack, max_pt)
        waiting = np.maximum(0.0, clock - arrivals)
        waiting_score = 1.0 - np.exp(-waiting / max_pt)
        base = np.asarray([
            np.clip(sizes.sum() / capacity, 0.0, 1.0),
            np.clip(urgency.mean(), 0.0, 1.0),
            np.clip(urgency.max(), 0.0, 1.0),
            np.clip(waiting_score.mean(), 0.0, 1.0),
            np.clip(efficiency / efficiency_scale, 0.0, 1.0),
        ], dtype=np.float32)
        # Keep the established 13-dimensional descriptor for shared-simulator
        # comparison controllers.  The proposed BAA/MMA policies consume the
        # separate six-dimensional criteria instead.
        features = _legacy_features(
            selected,
            family,
            total_jobs=len(jobs),
            capacity=capacity,
            clock=clock,
            processing_times=processing_times,
        )
        proposals.append(BatchProposal(
            proposal_id=proposal_id,
            job_ids=job_ids,
            family=family,
            base_criteria=base,
            features=features,
        ))
    return proposals


def _two_batch_tardiness(
        first: BatchProposal,
        second: BatchProposal,
        by_id: dict[int, Job],
        clock: float,
        processing_times: tuple[int, ...]) -> float:
    first_end = clock + processing_times[first.family]
    second_end = first_end + processing_times[second.family]
    return float(sum(
        max(0.0, first_end - by_id[job_id].due_time)
        for job_id in first.job_ids
    ) + sum(
        max(0.0, second_end - by_id[job_id].due_time)
        for job_id in second.job_ids
    ))


def _ch2_choice(
        proposals: list[BatchProposal],
        by_id: dict[int, Job],
        clock: float,
        processing_times: tuple[int, ...]) -> int:
    order = list(range(len(proposals)))
    changed = True
    while changed:
        changed = False
        for position in range(len(order) - 1):
            prefix = clock + sum(
                processing_times[proposals[order[index]].family]
                for index in range(position))
            left, right = order[position], order[position + 1]
            lr = _two_batch_tardiness(
                proposals[left], proposals[right], by_id, prefix,
                processing_times)
            rl = _two_batch_tardiness(
                proposals[right], proposals[left], by_id, prefix,
                processing_times)
            if rl < lr or (
                    np.isclose(rl, lr)
                    and processing_times[proposals[right].family]
                    < processing_times[proposals[left].family]):
                order[position], order[position + 1] = right, left
                changed = True
    return order[0]


def build_rule_proposals(
        jobs: list[Job],
        capacity: int,
        clock: float,
        processing_times: tuple[int, ...]) -> list[BatchProposal]:
    "Build the rule-based candidate pool for shared simulator controllers."

    complete = build_proposals(jobs, capacity, clock, processing_times)
    if not complete:
        return []
    by_id = {job.job_id: job for job in jobs}
    choices = [
        _ch2_choice(complete, by_id, clock, processing_times),
        max(range(len(complete)), key=lambda index: (
            sum(by_id[job_id].size for job_id in complete[index].job_ids),
            -complete[index].family)),
        min(range(len(complete)), key=lambda index: (
            min(by_id[job_id].due_time for job_id in complete[index].job_ids),
            complete[index].family)),
        min(range(len(complete)), key=lambda index: (
            np.mean([
                by_id[job_id].due_time - clock
                - processing_times[complete[index].family]
                for job_id in complete[index].job_ids]),
            complete[index].family)),
    ]
    sources: dict[tuple[int, ...], np.ndarray] = {}
    selected_by_ids = {item.job_ids: item for item in complete}
    for rule_index, proposal_index in enumerate(choices):
        job_ids = complete[proposal_index].job_ids
        sources.setdefault(
            job_ids, np.zeros(len(RULE_NAMES), dtype=np.float32))
        sources[job_ids][rule_index] = 1.0
    result = []
    for proposal_id, (job_ids, provenance) in enumerate(sources.items()):
        original = selected_by_ids[job_ids]
        selected = [by_id[job_id] for job_id in job_ids]
        result.append(BatchProposal(
            proposal_id=proposal_id,
            job_ids=job_ids,
            family=original.family,
            base_criteria=original.base_criteria.copy(),
            features=_legacy_features(
                selected,
                original.family,
                total_jobs=len(jobs),
                capacity=capacity,
                clock=clock,
                processing_times=processing_times,
                provenance=provenance,
            ),
        ))
    return result
