"Execute a static MILP schedule inside the dynamic simulation environment."

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from methods.hierarchical.config import HierarchyConfig
from methods.hierarchical.environment import HierarchicalSchedulingEnv
from methods.hierarchical.types import (
    BAAObservation,
    BatchProposal,
    DecisionContext,
    HandshakeMode,
    MachineAction,
    MMAObservation,
)


def plan_from_milp(result: dict) -> dict[int, list[dict]]:
    """Convert a MILP result dict into per-machine ordered batch queues.

    The ``schedule`` field lists batches with ``machine_id``, ``slot``,
    ``family`` and ``job_ids``.  The queues are ordered by slot, which is the
    processing order imposed by the static plan.
    """
    entries: dict[int, list[dict]] = {}
    for item in result["schedule"]:
        machine_id = int(item["machine_id"])
        entries.setdefault(machine_id, []).append({
            "slot": int(item["slot"]),
            "family": int(item["family"]),
            "job_ids": tuple(int(job_id) for job_id in item["job_ids"]),
        })
    queues = {
        machine_id: [
            {"family": batch["family"], "job_ids": batch["job_ids"]}
            for batch in sorted(batches, key=lambda b: b["slot"])
        ]
        for machine_id, batches in entries.items()
    }
    return queues


class MILPPlanExecutorEnv(HierarchicalSchedulingEnv):
    """Replay a static MILP plan in the dynamic production environment.

    Parameters
    ----------
    job_parameters:
        Same contract as ``HierarchicalSchedulingEnv``.
    machine_parameters:
        Same contract as ``HierarchicalSchedulingEnv``.
    plan_batches:
        machine id -> batches in processing order; each batch is a dict with
        ``job_ids`` (tuple of contiguous renumbered job ids) and ``family``.
    """

    def __init__(
            self,
            job_parameters: list[float | int],
            machine_parameters: list[int],
            *,
            plan_batches: dict[int, Sequence[dict]],
            processing_times: tuple[int, ...],
            handshake_mode: HandshakeMode = HandshakeMode.INTERACTIVE,
            degradation_multiplier: float = 1.0,
            device: str | object = "auto",
            calibrated_reward_scale: float | None = None,
            reward_scale_artifact_sha256: str | None = None,
    ) -> None:
        # ``super().__init__`` calls ``self.reset`` internally, so the source
        # queues must exist before the super call.  They are filled below.
        self._plan_queues_source: dict[int, list[dict]] = {}
        super().__init__(
            job_parameters,
            machine_parameters,
            processing_times=tuple(processing_times),
            handshake_mode=HandshakeMode(handshake_mode),
            config=HierarchyConfig(
                degradation_multiplier=degradation_multiplier,
                calibrated_reward_scale=calibrated_reward_scale,
                reward_scale_artifact_sha256=reward_scale_artifact_sha256,
            ),
            device=device,
        )
        self._plan_queues_source = {
            int(machine_id): [
                {
                    "family": int(batch["family"]),
                    "job_ids": tuple(int(job_id) for job_id in batch["job_ids"]),
                }
                for batch in batches
            ]
            for machine_id, batches in plan_batches.items()
        }
        self._validate_plan()
        self._plan_queues = {
            machine_id: list(batches)
            for machine_id, batches in self._plan_queues_source.items()
        }

    def _validate_plan(self) -> None:
        seen: set[int] = set()
        for machine_id, batches in self._plan_queues_source.items():
            for batch in batches:
                if not batch["job_ids"]:
                    raise ValueError("MILP plan contains an empty batch.")
                for job_id in batch["job_ids"]:
                    if job_id in seen:
                        raise ValueError(
                            f"Job {job_id} appears in more than one MILP batch.")
                    seen.add(job_id)

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self._plan_queues = {
            machine_id: [dict(batch) for batch in batches]
            for machine_id, batches in self._plan_queues_source.items()
        }
        return result

    def _next_plan_batch(self, machine) -> dict | None:
        queue = self._plan_queues.get(machine.machine_id)
        return queue[0] if queue else None

    def _pop_plan_batch(self, machine) -> dict | None:
        queue = self._plan_queues.get(machine.machine_id)
        return queue.pop(0) if queue else None

    def _request(
            self,
            machine,
            *,
            epsilon: float,
            training: bool) -> bool:
        if not machine.batch_ids:
            plan = self._next_plan_batch(machine)
            if plan is None:
                # No further batch is assigned to this machine by the plan.
                return False
            for job_id in plan["job_ids"]:
                job = self.jobs[job_id]
                if job.arrival_time > self.clock or job.finished:
                    # The plan batch is not fully released yet.  The machine
                    # stays idle and the next arrival event retriggers it.
                    return False
            self._pop_plan_batch(machine)
            machine.batch_ids = plan["job_ids"]
        return super()._request(machine, epsilon=epsilon, training=training)

    def _decide_mma(
            self,
            machine,
            proposal: BatchProposal,
            *,
            epsilon: float,
            training: bool,
            baa_observation: BAAObservation | None,
            baa_action: int,
            request_id: int,
            baa_eligible_family_count: int = 0) -> None:
        del epsilon, training
        observation = self._mma_observation(machine, proposal)
        self._close_previous_mma(machine, observation, terminated=False)
        action = int(MachineAction.PROCESS)
        probabilities = np.zeros(3, dtype=np.float32)
        probabilities[action] = 1.0
        context = DecisionContext(
            request_id=request_id,
            proposal_id=proposal.proposal_id,
            machine_id=machine.machine_id,
            started_at=self.clock,
            baa_observation=baa_observation,
            baa_action=baa_action,
            mma_observation=observation,
            mma_action=action,
            baa_policy_action=None,
            baa_log_probability=None,
            baa_value=None,
            baa_entropy=None,
            baa_weights=None,
            baa_scores=None,
            mma_log_probability=0.0,
            mma_value=0.0,
            mma_entropy=0.0,
            mma_probabilities=probabilities,
            baa_decision=baa_observation is not None,
            baa_eligible_family_count=baa_eligible_family_count,
        )
        self._start_action(machine, proposal, context)
