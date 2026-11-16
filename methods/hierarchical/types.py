"""Typed domain and reinforcement-learning contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np


class HandshakeMode(str, Enum):
    INTERACTIVE = "interactive"


class MachineAction(IntEnum):
    PROCESS = 0
    IPM = 1
    PM = 2


class MachinePhase(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    MAINTAINING = "maintaining"
    REPAIRING = "repairing"


@dataclass
class Job:
    job_id: int
    arrival_time: float
    family: int
    size: int
    due_time: float
    reserved_by: int | None = None
    finished: bool = False
    finish_time: float | None = None
    first_start_time: float | None = None


@dataclass
class BatchProposal:
    proposal_id: int
    job_ids: tuple[int, ...]
    family: int
    # Five machine-independent priority criteria.  Reliability is appended
    # by the environment for the requesting machine.
    base_criteria: np.ndarray
    features: np.ndarray


@dataclass
class BAAObservation:
    proposal_features: np.ndarray
    proposal_mask: np.ndarray
    machine_features: np.ndarray
    machine_mask: np.ndarray
    requester_index: int
    context_features: np.ndarray


@dataclass
class MMAObservation:
    batch_features: np.ndarray
    machine_features: np.ndarray
    global_machine_features: np.ndarray
    requester_index: int
    action_mask: np.ndarray
    context_features: np.ndarray
    maintenance_features: np.ndarray
    health_state: int = 0


@dataclass
class JointObservation:
    """One physical requester's CH candidates and maintenance actions.

    Actions ``0..K-1`` process the corresponding candidate batch; actions
    ``K`` and ``K+1`` are IPM and PM.  QMIX adds a private no-op only for
    inactive agents outside this environment-facing mask.
    """

    proposal_features: np.ndarray
    machine_features: np.ndarray
    requester_index: int
    action_mask: np.ndarray


@dataclass
class CentralizedObservation:
    """Global event state used by the centralized autoregressive PPO.

    ``proposal_machine_indices`` assigns each CH candidate to the physical
    machine that can execute it.  The policy first selects a requestable
    machine, then Process/IPM/PM, and finally a candidate if Process was
    selected.
    """

    proposal_features: np.ndarray
    proposal_machine_indices: np.ndarray
    machine_features: np.ndarray
    machine_mask: np.ndarray
    operation_mask: np.ndarray


@dataclass(frozen=True)
class CentralizedAction:
    machine_id: int
    operation: int
    proposal_index: int = -1


@dataclass
class ControlTransition:
    """One global decision transition emitted by the shared simulator."""

    state: JointObservation | CentralizedObservation
    action: int | CentralizedAction
    reward: float
    next_state: JointObservation | CentralizedObservation | None
    elapsed_time: float
    terminated: bool
    requester_class: int
    requester_id: int = -1

    def __post_init__(self) -> None:
        if self.elapsed_time < 0:
            raise ValueError("elapsed_time cannot be negative.")
        if self.terminated and self.next_state is not None:
            raise ValueError("A terminated transition cannot have next_state.")


@dataclass
class DecisionContext:
    request_id: int
    proposal_id: int
    machine_id: int
    started_at: float
    baa_observation: BAAObservation
    baa_action: int
    mma_observation: MMAObservation
    mma_action: int
    baa_policy_action: np.ndarray | None = None
    baa_log_probability: float | None = None
    baa_value: float | None = None
    baa_entropy: float | None = None
    baa_weights: np.ndarray | None = None
    baa_scores: np.ndarray | None = None
    mma_log_probability: float | None = None
    mma_value: float | None = None
    mma_entropy: float | None = None
    mma_probabilities: np.ndarray | None = None
    baa_decision: bool = False
    baa_eligible_family_count: int = 0


@dataclass
class Transition:
    agent: str
    agent_id: int
    request_id: int
    proposal_id: int
    state: object
    action: int | np.ndarray
    action_mask: np.ndarray
    reward: float
    next_state: object | None
    next_action_mask: np.ndarray
    elapsed_time: float
    terminated: bool
    truncated: bool = False
    old_log_probability: float = 0.0
    old_value: float = 0.0
    trajectory_id: int = 0
    reward_components: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.elapsed_time < 0:
            raise ValueError("elapsed_time cannot be negative.")
        if self.terminated and self.next_state is not None:
            raise ValueError("A terminated transition cannot have next_state.")
