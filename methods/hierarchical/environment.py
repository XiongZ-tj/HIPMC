"Event-driven production and maintenance simulation with hierarchical policies."

from __future__ import annotations

import heapq
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from experiment.artifacts import (
    atomic_csv_save,
    atomic_json_save,
    atomic_torch_save,
)
from experiment.device import (
    cuda_rng_state,
    resolve_torch_device,
    restore_cuda_rng_state,
)
from experiment.evaluation import (
    EvaluationProtocol,
    EvaluationRecord,
    merge_audit_tables,
    save_evaluation_outputs,
)
from experiment.instances import (
    INSTANCE_GENERATION_PROTOCOL,
    JobRecord,
    generate_job_records,
    instance_load,
    validate_job_records,
)
from experiment.metrics import EpisodeMetrics
from experiment.seeds import TrainingSeedProtocol
from experiment.training_logs import (
    TrainingConsoleLogger,
    baa_policy_summary,
    decision_action_summary,
    mma_probability_summary,
    save_training_log_bundle,
    validation_summary,
)

from .agents import HierarchicalMAPPO
from .candidates import BAA_CRITERIA, build_proposals, build_rule_proposals
from .config import HierarchyConfig, Scenario
from .reliability import (
    COMPLETION_PROBABILITY_PROTOCOL,
    uninterrupted_completion_probabilities,
)
from .features import (
    FEATURE_SCHEMA_VERSION,
    GLOBAL_CONTEXT_FEATURES,
    MMA_MAINTENANCE_FEATURES,
    POLICY_MACHINE_FEATURES,
    REFERENCE_JOB_SIZE,
    bounded_fraction,
    checked_unit_features,
    positive_saturation,
)
from .types import (
    BAAObservation,
    BatchProposal,
    CentralizedAction,
    CentralizedObservation,
    ControlTransition,
    DecisionContext,
    HandshakeMode,
    Job,
    JointObservation,
    MMAObservation,
    MachineAction,
    MachinePhase,
    Transition,
)

# Reward-normalization settings are runtime configuration (the calibrated
# scale and its artifact hash depend on the training dataset contract), not
# part of the frozen method architecture.  Loading weights across datasets
# (zero-shot evaluation, finetune warm-start) must therefore ignore them when
# comparing the recorded method configuration with the current environment.
_REWARD_SCALE_CONFIG_KEYS = frozenset({
    "calibrated_reward_scale",
    "reward_scale_artifact_sha256",
})


def _without_reward_scale(config) -> dict:
    """Strip runtime reward-normalization fields from a method contract."""
    if config is None:
        return {}
    return {
        key: value for key, value in config.items()
        if key not in _REWARD_SCALE_CONFIG_KEYS
    }


@dataclass
class Machine:
    machine_id: int
    class_id: int
    local_id: int
    capacity: int
    state: int = 0
    phase: MachinePhase = MachinePhase.IDLE
    batch_ids: tuple[int, ...] = ()
    remaining_work: float = 0.0
    activity_started_at: float = 0.0
    last_event_at: float = 0.0
    scheduled_event: str | None = None
    scheduled_at: float = math.inf
    failure_type: int | None = None
    context: DecisionContext | None = None
    maintenance_action: int | None = None
    breakdown_count: int = 0
    maintenance_time: float = 0.0
    planned_maintenance_time: float = 0.0
    corrective_maintenance_time: float = 0.0
    processing_time: float = 0.0
    ipm_count: int = 0
    pm_count: int = 0
    corrective_count: int = 0
    unexpected_breakdown_count: int = 0
    critical_failure_count: int = 0
    batch_count: int = 0
    loaded_size: int = 0
    capacity_committed: int = 0
    failure_interruptions: int = 0
    idle_since: float = 0.0
    generation: int = 0
    history: list[dict] = field(default_factory=list)


@dataclass
class PendingDecision:
    context: DecisionContext
    potential_at_start: float


@dataclass
class PendingControl:
    state: JointObservation | CentralizedObservation
    action: int | CentralizedAction
    requester_class: int
    requester_id: int
    started_at: float
    potential_at_start: float


class HierarchicalSchedulingEnv:
    """Shared environment for interactive and top-down hierarchical control."""

    proposal_dim = 6
    # ``machine_dim`` remains the frozen shared-simulator baseline contract.
    machine_dim = 14
    policy_machine_dim = len(POLICY_MACHINE_FEATURES)
    context_dim = len(GLOBAL_CONTEXT_FEATURES)
    maintenance_dim = len(MMA_MAINTENANCE_FEATURES)

    def __init__(
            self,
            job_parameters: list[float | int],
            machine_parameters: list[int],
            *,
            processing_times: tuple[int, ...],
            handshake_mode: HandshakeMode = HandshakeMode.INTERACTIVE,
            config: HierarchyConfig | None = None,
            controller: object | None = None,
            device: str | torch.device = "auto") -> None:
        self.target_load, self.initial_jobs, self.total_jobs, \
            self.due_date_factor, self.instance_seed = job_parameters
        self.job_parameters = list(job_parameters)
        self.scenario = Scenario(
            job_processing_times=tuple(processing_times),
            machine_counts=tuple(machine_parameters),
        )
        self.config = config or HierarchyConfig()
        self.controller = controller
        self.handshake_mode = HandshakeMode(handshake_mode)
        self.device = resolve_torch_device(device)
        self.max_episode = 500
        self.training_horizon: int | None = None
        self.validation_interval = 10
        self.validation_instances = 30
        self.checkpoint_interval = 25
        self.training_master_seed = self.instance_seed
        self.seed_protocol = TrainingSeedProtocol(self.training_master_seed)
        self.training_seed_schedule = None
        self.validation_seed_schedule = None
        # Optional runtime adapters let an external dataset configure the
        # freshly constructed validation environment and reset each frozen
        # case explicitly.  They are intentionally not checkpointed: runners
        # reinstall dataset contracts before either a fresh run or resume.
        # The default ``None`` path preserves the generated-instance protocol.
        self.validation_evaluator_setup = None
        self.validation_case_reset = None
        self.validation_summary_auditor = None
        self.best_validation = math.inf
        self.validation_metrics: list[dict] = []
        self.pretraining_baseline: dict | None = None
        self.pretraining_baseline_cases: list[dict] = []
        self.total_episode = 0
        self.total_step = 0
        self.clock = 0.0
        self.jobs: list[Job] = []
        self.machines: list[Machine] = []
        self._event_queue: list[tuple[float, int, int, str, int]] = []
        self._event_sequence = 0
        self._request_sequence = 0
        self._pending_baa: PendingDecision | None = None
        self._pending_mma: dict[int, PendingDecision] = {}
        self._pending_control: dict[int, PendingControl] = {}
        self._episode_training = False
        self._rng = np.random.default_rng(self.instance_seed)
        self._simulation_rng = np.random.default_rng(self.instance_seed)
        self._policy_rng = np.random.default_rng(self.instance_seed)
        self.episode_metrics: list[dict] = []
        self._decision_latencies: list[float] = []
        self.decision_history: list[dict] = []
        self.losses: list[float] = []
        self.update_diagnostics: list[dict[str, float]] = []
        self.rollout_diagnostics: list[dict] = []
        self._active_reward_normalizer = float(
            self.config.calibrated_reward_scale
            if self.config.calibrated_reward_scale is not None
            else self.config.reward_normalizer_floor)
        self._active_job_count = 1
        self._instance_load = None

        self.mappo = HierarchicalMAPPO(
            machine_class_count=len(self.scenario.machine_counts),
            health_state_count=self.config.degradation_states,
            proposal_dim=self.proposal_dim,
            machine_dim=self.policy_machine_dim,
            context_dim=self.context_dim,
            maintenance_dim=self.maintenance_dim,
            hidden_dim=self.config.hidden_dim,
            learning_rate=self.config.learning_rate,
            gamma_baa=self.config.gamma_baa,
            gamma_mma=self.config.gamma_mma,
            gae_lambda=self.config.gae_lambda,
            clip_ratio=self.config.ppo_clip_ratio,
            value_loss_weight=self.config.value_loss_weight,
            baa_entropy_weight=self.config.baa_entropy_weight,
            mma_entropy_weight=self.config.mma_entropy_weight,
            update_epochs=self.config.ppo_update_epochs,
            minibatch_size=self.config.minibatch_size,
            time_scale=self.config.discount_time_scale,
            gradient_clip=self.config.gradient_clip,
            dirichlet_min_concentration=(
                self.config.dirichlet_min_concentration),
            device=self.device,
        )
        self.reset(instance_seed=self.instance_seed, simulation_seed=self.instance_seed)

    @property
    def job_cluster_pt(self) -> tuple[int, ...]:
        return self.scenario.job_processing_times

    def _jobs_from_records(self, records: Sequence[JobRecord]) -> list[Job]:
        validate_job_records(
            records,
            initial_jobs=self.initial_jobs,
            total_jobs=self.total_jobs,
            due_date_factor=self.due_date_factor,
            processing_times=self.job_cluster_pt,
            size_interval=self.scenario.job_size_interval,
        )
        return [
            Job(
                job_id=record.job_id,
                arrival_time=float(record.arrival_time),
                family=record.family,
                size=record.size,
                due_time=float(record.due_time),
            )
            for record in records
        ]

    def _new_jobs(self, seed: int) -> list[Job]:
        return self._jobs_from_records(generate_job_records(
            target_load=self.target_load,
            initial_jobs=self.initial_jobs,
            total_jobs=self.total_jobs,
            due_date_factor=self.due_date_factor,
            processing_times=self.job_cluster_pt,
            size_interval=self.scenario.job_size_interval,
            machine_counts=self.scenario.machine_counts,
            machine_capacities=self.scenario.machine_capacities,
            seed=seed,
        ))

    def _new_machines(self) -> list[Machine]:
        machines: list[Machine] = []
        for class_id, count in enumerate(self.scenario.machine_counts):
            for local_id in range(count):
                machines.append(Machine(
                    machine_id=len(machines),
                    class_id=class_id,
                    local_id=local_id,
                    capacity=self.scenario.machine_capacities[class_id],
                ))
        return machines

    def reset(
            self,
            *,
            instance_seed: int | None = None,
            simulation_seed: int | None = None,
            policy_seed: int | None = None,
            job_records: Sequence[JobRecord] | None = None) -> None:
        if instance_seed is not None:
            self.instance_seed = int(instance_seed)
        self.clock = 0.0
        self.jobs = (
            self._new_jobs(self.instance_seed)
            if job_records is None
            else self._jobs_from_records(tuple(job_records))
        )
        self._instance_load = instance_load(
            self.jobs,
            target_load=self.target_load,
            initial_jobs=self.initial_jobs,
            processing_times=self.job_cluster_pt,
            machine_counts=self.scenario.machine_counts,
            machine_capacities=self.scenario.machine_capacities,
        )
        self._active_job_count = max(1, len(self.jobs))
        self.machines = self._new_machines()
        self._event_queue = []
        self._event_sequence = 0
        self._request_sequence = 0
        self._pending_baa = None
        self._pending_mma = {}
        self._pending_control = {}
        self._decision_latencies = []
        self.decision_history = []
        self._simulation_rng = np.random.default_rng(
            self.instance_seed if simulation_seed is None else simulation_seed)
        resolved_policy_seed = (
            self.instance_seed if policy_seed is None else int(policy_seed))
        self._policy_rng = np.random.default_rng(resolved_policy_seed)
        if (
                self.controller is not None
                and hasattr(self.controller, "begin_episode")):
            self.controller.begin_episode(resolved_policy_seed)
        if (
                self.controller is not None
                and hasattr(self.controller, "bind_episode")):
            self.controller.bind_episode(
                jobs=self.jobs,
                machines=self.machines,
            )

    def _available_jobs(self) -> list[Job]:
        return [
            job for job in self.jobs
            if job.arrival_time <= self.clock
            and not job.finished
            and job.reserved_by is None
        ]

    def _unfinished_jobs(self) -> list[Job]:
        return [job for job in self.jobs if not job.finished]

    def _potential(self) -> float:
        "Return arrived-job estimated tardiness with the instance-size normalization."
        if not self.jobs:
            return 0.0
        total = 0.0
        for job in self.jobs:
            if job.arrival_time > self.clock:
                continue
            if job.finished:
                completion = float(job.finish_time)
            else:
                completion = self.clock + self.job_cluster_pt[job.family]
            total += max(0.0, completion - job.due_time)
        return -total / len(self.jobs)

    def total_tardiness(self) -> float:
        if any(not job.finished for job in self.jobs):
            raise RuntimeError("Cannot score an unfinished episode.")
        return float(sum(
            max(0.0, float(job.finish_time) - job.due_time)
            for job in self.jobs
        ))

    def _production_rates(self, machine: Machine) -> np.ndarray:
        low, high = self.config.production_rate_range
        return np.linspace(
            high, low, self.config.degradation_states, dtype=float)

    def _degradation_rates(self, machine: Machine) -> np.ndarray:
        start, end = self.config.degradation_rate_ranges[machine.class_id]
        nonterminal = np.linspace(
            start, end, self.config.degradation_states - 1, dtype=float)
        return self.config.degradation_multiplier * np.append(
            nonterminal,
            self.config.terminal_degradation_rates[machine.class_id],
        )

    def _failure_rates(self, machine: Machine) -> np.ndarray:
        start, end = self.config.failure_rate_ranges[machine.class_id]
        total = np.linspace(
            start, end, self.config.degradation_states, dtype=float)
        return self.config.degradation_multiplier * total[:, None] * np.asarray(
            self.config.failure_type_proportions, dtype=float)

    def _failure_risk(
            self, machine: Machine, family: int | None = None) -> float:
        "Return the competing-event probability over a nominal processing interval."
        duration = (
            self.job_cluster_pt[family]
            if family is not None else max(self.job_cluster_pt)
        )
        rate = (
            self._degradation_rates(machine)[machine.state]
            + self._failure_rates(machine)[machine.state].sum()
        )
        return float(1.0 - math.exp(-rate * duration))

    def _policy_observed_remaining_work(self, machine: Machine) -> float:
        """Project booked work to now without consulting the next event."""

        work = float(machine.remaining_work)
        if machine.phase is MachinePhase.PROCESSING:
            elapsed = max(0.0, self.clock - machine.last_event_at)
            work -= elapsed * self._production_rates(machine)[machine.state]
        return max(0.0, float(work))

    def _policy_proposal_work(
            self, machine: Machine, proposal: BatchProposal) -> float:
        """Use residual work only for this machine's exact retained batch."""

        if machine.batch_ids and machine.batch_ids == proposal.job_ids:
            return self._policy_observed_remaining_work(machine)
        return float(self.job_cluster_pt[proposal.family])

    def _policy_completion_probability(
            self, machine: Machine, family: int | None = None,
            *, nominal_work: float | None = None) -> float:
        """Probability of immediate processing without failure interruption.

        Progressive degradation and the resulting speed/hazard changes are
        included. Terminal deterioration and every unexpected breakdown are
        absorbing interruptions. No preventive intervention or repair path
        contributes to this processing-only conditional probability.
        """

        work = nominal_work
        if work is None:
            work = (
                float(self.job_cluster_pt[family]) if family is not None
                else self._policy_observed_remaining_work(machine)
                if machine.batch_ids else float(max(self.job_cluster_pt)))
        survival = uninterrupted_completion_probabilities(
            work,
            self._production_rates(machine),
            self._degradation_rates(machine),
            self._failure_rates(machine).sum(axis=1),
        )
        return survival[machine.state]

    def _machine_feature(
            self,
            machine: Machine,
            *,
            requester_id: int,
            family: int | None = None) -> np.ndarray:
        state = np.zeros(self.config.degradation_states, dtype=np.float32)
        state[machine.state] = 1.0
        phases = np.asarray([
            machine.phase is MachinePhase.IDLE,
            machine.phase is MachinePhase.PROCESSING,
            machine.phase is MachinePhase.MAINTAINING,
            machine.phase is MachinePhase.REPAIRING,
        ], dtype=np.float32)
        remaining = (
            machine.remaining_work / max(self.job_cluster_pt)
            if machine.phase is MachinePhase.PROCESSING
            else max(0.0, machine.scheduled_at - self.clock)
            / max(self.config.pm_times)
            if machine.phase in (MachinePhase.MAINTAINING, MachinePhase.REPAIRING)
            else 0.0
        )
        return np.asarray([
            *state,
            *phases,
            positive_saturation(remaining, 1.0),
            machine.capacity / max(self.scenario.machine_capacities),
            machine.class_id / max(1, len(self.scenario.machine_counts) - 1),
            float(bool(machine.batch_ids)),
            self._failure_risk(machine, family),
            float(machine.machine_id == requester_id),
        ], dtype=np.float32)

    def _policy_machine_feature(
            self,
            machine: Machine,
            *,
            requester_id: int,
            family: int | None = None,
            nominal_work: float | None = None) -> np.ndarray:
        """Return the proposed method's bounded physical-machine descriptor."""

        state = np.zeros(self.config.degradation_states, dtype=np.float32)
        state[machine.state] = 1.0
        phases = np.asarray([
            machine.phase is MachinePhase.IDLE,
            machine.phase is MachinePhase.PROCESSING,
            machine.phase is MachinePhase.MAINTAINING,
            machine.phase is MachinePhase.REPAIRING,
        ], dtype=np.float32)
        if machine.phase is MachinePhase.PROCESSING:
            production_rate = max(
                float(self._production_rates(machine)[machine.state]), 1e-12)
            # Work is booked at this machine's own events. Observe its current
            # residual, not the stale residual at the preceding own event,
            # and never use the pre-sampled next processing event time.
            remaining_work = self._policy_observed_remaining_work(machine)
            remaining_duration = remaining_work / production_rate
        elif machine.phase in (MachinePhase.MAINTAINING, MachinePhase.REPAIRING):
            # These service durations are deterministic given an intervention
            # or an already-observed failure type. Do not expose the event
            # queue itself as a learned feature.
            if machine.phase is MachinePhase.MAINTAINING:
                duration = (
                    self.config.ipm_times[machine.class_id]
                    if machine.maintenance_action == int(MachineAction.IPM)
                    else self.config.pm_times[machine.class_id])
            elif machine.failure_type == -1:
                duration = self.config.minimal_repair_times[machine.class_id]
            elif machine.failure_type is not None:
                start, end = self.config.corrective_time_ranges[machine.class_id]
                duration = float(np.linspace(
                    start, end, self.config.failure_types)[machine.failure_type])
            else:
                raise ValueError("Repair observation requires an observed failure type.")
            service_started_at = (
                machine.activity_started_at
                if machine.phase is MachinePhase.MAINTAINING
                else machine.last_event_at)
            remaining_duration = max(
                0.0, duration - max(0.0, self.clock - service_started_at))
        else:
            remaining_duration = 0.0
        capacity_low = min(self.scenario.machine_capacities)
        capacity_high = max(self.scenario.machine_capacities)
        capacity_feature = (
            (machine.capacity - capacity_low) / (capacity_high - capacity_low)
            if capacity_high > capacity_low else 1.0)
        classes = np.zeros(len(self.scenario.machine_counts), dtype=np.float32)
        classes[machine.class_id] = 1.0
        assigned = [self.jobs[job_id] for job_id in machine.batch_ids]
        reserved_load = sum(job.size for job in assigned)
        elapsed = max(0.0, self.clock)
        realized_processing = float(machine.processing_time)
        if machine.phase is MachinePhase.PROCESSING:
            realized_processing += max(0.0, self.clock - machine.last_event_at)
        realized_maintenance = float(machine.maintenance_time)
        if machine.phase in (MachinePhase.MAINTAINING, MachinePhase.REPAIRING):
            realized_maintenance -= remaining_duration
        realized_processing = max(0.0, realized_processing)
        realized_maintenance = max(0.0, realized_maintenance)
        processing_utilization = (
            bounded_fraction(realized_processing, elapsed)
            if elapsed > 0 else 0.0)
        maintenance_utilization = (
            bounded_fraction(realized_maintenance, elapsed)
            if elapsed > 0 else 0.0)
        values = [
            *state,
            *phases,
            positive_saturation(remaining_duration, max(self.job_cluster_pt)),
            capacity_feature,
            *classes,
            bounded_fraction(reserved_load, machine.capacity),
            processing_utilization,
            maintenance_utilization,
            positive_saturation(machine.breakdown_count, 1.0),
            1.0 - self._policy_completion_probability(
                machine, family, nominal_work=nominal_work),
            float(machine.machine_id == requester_id),
        ]
        return checked_unit_features(
            "policy_machine_features",
            values,
            expected_dim=self.policy_machine_dim,
        )

    def _global_context_features(self) -> np.ndarray:
        """Sixteen causal queue/shop features shared by actors and critic.

        Only released orders, current machine observations and declared shop
        parameters are permitted. In particular, no finite-instance count,
        realized release horizon or generator condition label is consulted.
        """

        available = self._available_jobs()
        reserved = [
            job for job in self.jobs
            if job.arrival_time <= self.clock
            and not job.finished and job.reserved_by is not None]
        time_reference = float(max(self.job_cluster_pt))
        overdue_count = 0
        urgent_count = 0
        if available:
            slack = np.asarray([
                job.due_time - self.clock - self.job_cluster_pt[job.family]
                for job in available
            ], dtype=np.float64)
            normalized_slack = np.clip(
                slack / time_reference, -20.0, 20.0)
            urgency = 1.0 / (1.0 + np.exp(normalized_slack))
            overdue_count = sum(job.due_time < self.clock for job in available)
            urgent_count = sum(
                self.clock <= job.due_time
                < self.clock + self.job_cluster_pt[job.family]
                for job in available)
            waiting = np.asarray([
                max(0.0, self.clock - job.arrival_time)
                for job in available
            ], dtype=np.float64)
            waiting_features = waiting / (waiting + time_reference)
            mean_urgency = float(urgency.mean())
            maximum_urgency = float(urgency.max())
            mean_waiting = float(waiting_features.mean())
            maximum_waiting = float(waiting_features.max())
        else:
            mean_urgency = maximum_urgency = 0.0
            mean_waiting = maximum_waiting = 0.0
        machine_count = max(1, len(self.machines))
        phase_counts = {
            phase: sum(machine.phase is phase for machine in self.machines)
            for phase in MachinePhase
        }
        plant_capacity = float(sum(machine.capacity for machine in self.machines))
        queue_count = len(available)
        queue_work = sum(
            job.size * self.job_cluster_pt[job.family] for job in available)
        allocated_capacity = sum(job.size for job in reserved)
        # A family is eligible only if at least one of its available orders
        # fits a declared machine capacity. No future-family catalogue is read.
        largest_capacity = max(machine.capacity for machine in self.machines)
        eligible_families = len({
            job.family for job in available if job.size <= largest_capacity})
        health_capacity = sum(
            machine.capacity * machine.state / (self.config.degradation_states - 1)
            for machine in self.machines)
        severe_capacity = sum(
            machine.capacity for machine in self.machines
            if machine.state == self.config.degradation_states - 1)
        values = [
            positive_saturation(queue_count, plant_capacity / REFERENCE_JOB_SIZE),
            positive_saturation(queue_work, plant_capacity * time_reference),
            bounded_fraction(allocated_capacity, plant_capacity),
            bounded_fraction(overdue_count, queue_count) if queue_count else 0.0,
            bounded_fraction(urgent_count, queue_count) if queue_count else 0.0,
            mean_urgency,
            maximum_urgency,
            mean_waiting,
            maximum_waiting,
            bounded_fraction(phase_counts[MachinePhase.IDLE], machine_count),
            bounded_fraction(phase_counts[MachinePhase.PROCESSING], machine_count),
            bounded_fraction(phase_counts[MachinePhase.MAINTAINING], machine_count),
            bounded_fraction(phase_counts[MachinePhase.REPAIRING], machine_count),
            bounded_fraction(eligible_families, len(self.job_cluster_pt)),
            bounded_fraction(health_capacity, plant_capacity),
            bounded_fraction(severe_capacity, plant_capacity),
        ]
        return checked_unit_features(
            "global_context_features",
            values,
            expected_dim=self.context_dim,
        )

    def _maintenance_features(
            self, machine: Machine, proposal: BatchProposal) -> np.ndarray:
        class_id = machine.class_id
        state = machine.state
        if state > 0:
            recovery_rate = float(
                self.config.ipm_recovery_rates[class_id][state - 1])
            one_step_recovery = 1.0 - math.exp(
                -recovery_rate * self.config.ipm_times[class_id])
        else:
            one_step_recovery = 0.0
        values = [
            state / max(1, self.config.degradation_states - 1),
            1.0 - self._policy_completion_probability(
                machine, nominal_work=self._policy_proposal_work(machine, proposal)),
            positive_saturation(
                self.config.ipm_times[class_id], max(self.job_cluster_pt)),
            positive_saturation(
                self.config.pm_times[class_id], max(self.job_cluster_pt)),
            one_step_recovery,
        ]
        return checked_unit_features(
            "mma_maintenance_features",
            values,
            expected_dim=self.maintenance_dim,
        )

    def _baa_observation(
            self,
            requester: Machine,
            proposals: list[BatchProposal]) -> BAAObservation:
        return BAAObservation(
            proposal_features=np.stack(
                [self._proposal_features(requester, proposal)
                 for proposal in proposals]),
            proposal_mask=np.ones(len(proposals), dtype=np.bool_),
            machine_features=np.stack([
                self._policy_machine_feature(
                    machine,
                    requester_id=requester.machine_id,
                )
                for machine in self.machines
            ]),
            machine_mask=np.ones(len(self.machines), dtype=np.bool_),
            requester_index=requester.machine_id,
            context_features=self._global_context_features(),
        )

    def _proposal_features(
            self, requester: Machine, proposal: BatchProposal) -> np.ndarray:
        reliability = self._policy_completion_probability(
            requester, nominal_work=self._policy_proposal_work(requester, proposal))
        return checked_unit_features(
            "proposal_features",
            np.concatenate((
            proposal.base_criteria.astype(np.float32, copy=False),
            np.asarray([np.clip(reliability, 0.0, 1.0)], dtype=np.float32),
            )),
            expected_dim=self.proposal_dim,
        )

    @staticmethod
    def _baa_policy_diagnostics(context: DecisionContext | None) -> dict:
        if context is None or not context.baa_decision:
            return {"baa_decision": False}
        observation = context.baa_observation
        features = np.asarray(observation.proposal_features, dtype=float)
        scores = np.asarray(context.baa_scores, dtype=float)
        selected = int(context.baa_action)
        sorted_scores = np.sort(scores[np.isfinite(scores)])[::-1]
        margin = (
            float(sorted_scores[0] - sorted_scores[1])
            if sorted_scores.size >= 2 else 0.0)
        result: dict[str, int | float | bool] = {
            "baa_decision": True,
            "baa_candidate_count": int(len(features)),
            "baa_eligible_family_count": int(
                context.baa_eligible_family_count),
            "baa_policy_entropy": float(context.baa_entropy),
            "baa_selected_score": float(scores[selected]),
            "baa_score_margin": margin,
        }
        for index, key in enumerate(BAA_CRITERIA):
            descending = np.argsort(-features[:, index], kind="stable")
            rank = int(np.flatnonzero(descending == selected)[0]) + 1
            result[f"baa_selected_{key}"] = float(features[selected, index])
            result[f"baa_selected_rank_{key}"] = rank
        return result

    def _mma_observation(
            self,
            machine: Machine,
            proposal: BatchProposal) -> MMAObservation:
        action_mask = np.asarray([
            bool(proposal.job_ids),
            machine.state > 0,
            machine.state > 0,
        ], dtype=np.bool_)
        work = self._policy_proposal_work(machine, proposal)
        return MMAObservation(
            batch_features=self._proposal_features(machine, proposal),
            machine_features=self._policy_machine_feature(
                machine,
                requester_id=machine.machine_id,
                family=proposal.family,
                nominal_work=work,
            ),
            global_machine_features=np.stack([
                self._policy_machine_feature(
                    item,
                    requester_id=machine.machine_id,
                    family=proposal.family,
                    nominal_work=work,
                )
                for item in self.machines
            ]),
            requester_index=machine.machine_id,
            action_mask=action_mask,
            context_features=self._global_context_features(),
            maintenance_features=self._maintenance_features(machine, proposal),
            health_state=machine.state,
        )

    def _joint_observation(
            self,
            machine: Machine,
            proposals: list[BatchProposal]) -> JointObservation:
        maintenance_allowed = (
            machine.state > 0
            or bool(getattr(
                self.controller, "allow_maintenance_at_healthy", False))
        )
        action_mask = np.asarray([
            *([True] * len(proposals)),
            maintenance_allowed,
            maintenance_allowed,
        ], dtype=np.bool_)
        if getattr(self.controller, "shared_learning_features", False):
            from methods.baselines.learning_features import observation
            return observation(self, machine, proposals, action_mask)
        return JointObservation(
            proposal_features=np.stack([
                self._control_proposal_features(machine, proposal)
                for proposal in proposals
            ]),
            machine_features=np.stack([
                self._machine_feature(
                    item, requester_id=machine.machine_id)
                for item in self.machines
            ]),
            requester_index=machine.machine_id,
            action_mask=action_mask,
        )

    def _control_proposal_features(
            self, machine: Machine, proposal: BatchProposal) -> np.ndarray:
        """Strictly normalized descriptor shared by all DRL baselines."""

        selected = [self.jobs[job_id] for job_id in proposal.job_ids]
        sizes = np.asarray([job.size for job in selected], dtype=float)
        due = np.asarray([job.due_time for job in selected], dtype=float)
        arrivals = np.asarray(
            [job.arrival_time for job in selected], dtype=float)
        maximum = float(max(self.job_cluster_pt))
        slack = due - self.clock - self.job_cluster_pt[proposal.family]
        urgency = 1.0 / (1.0 + np.exp(np.clip(
            slack / maximum, -20.0, 20.0)))
        waiting = np.maximum(0.0, self.clock - arrivals)
        values = [
            *np.asarray(proposal.base_criteria, dtype=float).tolist(),
            proposal.family / max(1, len(self.job_cluster_pt) - 1),
            self.job_cluster_pt[proposal.family] / maximum,
            len(selected) / max(1, len(self.jobs)),
            float(sizes.mean()) / machine.capacity,
            float((1.0 - np.exp(-waiting / maximum)).max(initial=0.0)),
            float(urgency.max(initial=0.0)),
            len(self._unfinished_jobs()) / max(1, len(self.jobs)),
            1.0 - self._failure_risk(machine, proposal.family),
        ]
        result = np.asarray(values, dtype=np.float32)
        return np.clip(result, 0.0, 1.0)

    def _close_previous_control(
            self,
            next_state: JointObservation | CentralizedObservation | None,
            *,
            terminated: bool,
            trajectory_id: int) -> None:
        pending = self._pending_control.pop(trajectory_id, None)
        if pending is None:
            return
        transition = ControlTransition(
            state=pending.state,
            action=pending.action,
            reward=(
                self._potential() - pending.potential_at_start
            ) / self._active_reward_normalizer,
            next_state=None if terminated else next_state,
            elapsed_time=self.clock - pending.started_at,
            terminated=terminated,
            requester_class=pending.requester_class,
            requester_id=pending.requester_id,
        )
        if self._episode_training:
            self.controller.observe(transition)

    def _schedule(
            self, machine: Machine, at: float, event: str) -> None:
        machine.generation += 1
        machine.scheduled_at = at
        machine.scheduled_event = event
        self._event_sequence += 1
        heapq.heappush(
            self._event_queue,
            (at, self._event_sequence, machine.machine_id, event,
             machine.generation),
        )

    def _schedule_processing_event(self, machine: Machine) -> None:
        rate = self._production_rates(machine)[machine.state]
        completion_duration = machine.remaining_work / rate
        degradation_rate = self._degradation_rates(machine)[machine.state]
        failure_rates = self._failure_rates(machine)[machine.state]
        hazards = np.append(degradation_rate, failure_rates)
        hazard_sum = float(hazards.sum())
        if hazard_sum <= 0:
            self._schedule(machine, self.clock + completion_duration, "complete")
            return
        hazard_duration = float(self._simulation_rng.exponential(1 / hazard_sum))
        if completion_duration <= hazard_duration:
            self._schedule(machine, self.clock + completion_duration, "complete")
            return
        hazard_kind = int(self._simulation_rng.choice(
            len(hazards), p=hazards / hazard_sum))
        event = "degrade" if hazard_kind == 0 else f"failure:{hazard_kind - 1}"
        self._schedule(machine, self.clock + hazard_duration, event)

    def _reserve(
            self, machine: Machine, proposal: BatchProposal) -> None:
        machine.batch_ids = proposal.job_ids
        for job_id in proposal.job_ids:
            job = self.jobs[job_id]
            if job.reserved_by not in (None, machine.machine_id):
                raise RuntimeError("A job cannot be reserved by two machines.")
            job.reserved_by = machine.machine_id

    def _release(self, machine: Machine) -> None:
        for job_id in machine.batch_ids:
            if not self.jobs[job_id].finished:
                self.jobs[job_id].reserved_by = None
        machine.batch_ids = ()
        # Remaining work belongs to the released batch.  Keeping it after the
        # reservation is discarded can make an unrelated future batch look
        # like interrupted work that should be resumed.
        machine.remaining_work = 0.0

    def _proposal_for_reserved(self, machine: Machine) -> BatchProposal:
        family = self.jobs[machine.batch_ids[0]].family
        available = [
            job for job in self.jobs if job.job_id in machine.batch_ids]
        generated = build_proposals(
            available, machine.capacity, self.clock, self.job_cluster_pt)
        if generated and generated[0].job_ids == machine.batch_ids:
            return generated[0]
        if not generated:
            raise RuntimeError("A retained batch no longer forms a valid proposal.")
        template = generated[0]
        return BatchProposal(
            proposal_id=-1,
            job_ids=machine.batch_ids,
            family=family,
            base_criteria=template.base_criteria.copy(),
            features=template.features.copy(),
        )

    def _close_previous_baa(
            self,
            next_state: BAAObservation | None,
            *,
            terminated: bool) -> None:
        pending = self._pending_baa
        if pending is None:
            return
        raw_reward = self._potential() - pending.potential_at_start
        reward = raw_reward / self._active_reward_normalizer
        context = pending.context
        transition = Transition(
            agent="BAA",
            agent_id=0,
            request_id=context.request_id,
            proposal_id=context.proposal_id,
            state=context.baa_observation,
            action=np.asarray(context.baa_action, dtype=np.int64),
            action_mask=np.asarray(
                context.baa_observation.proposal_mask, dtype=np.bool_),
            reward=reward,
            next_state=None if terminated else next_state,
            next_action_mask=(
                np.zeros(0, dtype=np.bool_) if terminated
                else np.asarray(next_state.proposal_mask, dtype=np.bool_)
            ),
            elapsed_time=self.clock - context.started_at,
            terminated=terminated,
            old_log_probability=float(context.baa_log_probability),
            old_value=float(context.baa_value),
            trajectory_id=0,
            reward_components={
                "global_potential_delta": reward,
                "global_potential_delta_raw": raw_reward,
                "sdr_reward_normalizer": self._active_reward_normalizer,
            },
        )
        if self._episode_training:
            self.mappo.remember(0, transition)
        self._pending_baa = None

    def _close_previous_mma(
            self,
            machine: Machine,
            next_state: MMAObservation | None,
            *,
            terminated: bool) -> None:
        pending = self._pending_mma.pop(machine.machine_id, None)
        if pending is None:
            return
        context = pending.context
        elapsed = self.clock - context.started_at
        # T3 deliberately gives every MMA trajectory the same normalized
        # global potential signal as BAA.  A machine's transition remains open
        # until that physical machine makes its next decision; only completion
        # of the complete scheduling episode is terminal.
        raw_reward = self._potential() - pending.potential_at_start
        reward = raw_reward / self._active_reward_normalizer
        components = {
            "global_potential_delta": reward,
            "global_potential_delta_raw": raw_reward,
            "sdr_reward_normalizer": self._active_reward_normalizer,
        }
        transition = Transition(
            agent="MMA",
            agent_id=machine.class_id,
            request_id=context.request_id,
            proposal_id=context.proposal_id,
            state=context.mma_observation,
            action=context.mma_action,
            action_mask=context.mma_observation.action_mask.copy(),
            reward=reward,
            next_state=None if terminated else next_state,
            next_action_mask=(
                np.zeros(0, dtype=np.bool_) if terminated
                else next_state.action_mask.copy()
            ),
            elapsed_time=elapsed,
            terminated=terminated,
            old_log_probability=float(context.mma_log_probability),
            old_value=float(context.mma_value),
            trajectory_id=machine.machine_id,
            reward_components=components,
        )
        if self._episode_training:
            self.mappo.remember(machine.class_id + 1, transition)

    def _start_action(
            self,
            machine: Machine,
            proposal: BatchProposal,
            context: DecisionContext) -> None:
        action = MachineAction(context.mma_action)
        machine.context = context
        if self.controller is None:
            self._pending_mma[machine.machine_id] = PendingDecision(
                context=context,
                potential_at_start=self._potential(),
            )
        if action is MachineAction.PROCESS:
            retained_work = (
                machine.remaining_work
                if machine.batch_ids == proposal.job_ids
                and machine.remaining_work > 0
                else None
            )
            self._reserve(machine, proposal)
            machine.phase = MachinePhase.PROCESSING
            machine.remaining_work = (
                float(retained_work)
                if retained_work is not None
                else float(self.job_cluster_pt[proposal.family])
            )
            machine.activity_started_at = self.clock
            machine.last_event_at = self.clock
            if retained_work is None:
                machine.batch_count += 1
                loaded = sum(self.jobs[job_id].size for job_id in proposal.job_ids)
                machine.loaded_size += loaded
                machine.capacity_committed += machine.capacity
                for job_id in proposal.job_ids:
                    if self.jobs[job_id].first_start_time is None:
                        self.jobs[job_id].first_start_time = self.clock
            machine.history.append({
                "time": self.clock, "event": "process_start",
                "batch": list(proposal.job_ids), "state": machine.state,
            })
            self._schedule_processing_event(machine)
            return

        self._release(machine)
        machine.phase = MachinePhase.MAINTAINING
        machine.maintenance_action = int(action)
        duration = (
            self.config.ipm_times[machine.class_id]
            if action is MachineAction.IPM
            else self.config.pm_times[machine.class_id]
        )
        machine.maintenance_time += duration
        machine.planned_maintenance_time += duration
        if action is MachineAction.IPM:
            machine.ipm_count += 1
        else:
            machine.pm_count += 1
        machine.activity_started_at = self.clock
        machine.history.append({
            "time": self.clock,
            "event": "ipm_start" if action is MachineAction.IPM else "pm_start",
            "state": machine.state,
        })
        self._schedule(machine, self.clock + duration, "maintenance_complete")

    def _decide_mma(
            self,
            machine: Machine,
            proposal: BatchProposal,
            *,
            epsilon: float,
            training: bool,
            baa_observation: BAAObservation | None,
            baa_action: int,
            request_id: int,
            baa_eligible_family_count: int = 0) -> None:
        observation = self._mma_observation(machine, proposal)
        self._close_previous_mma(
            machine,
            observation,
            terminated=False,
        )
        selection = self.mappo.select_mma(
            machine.class_id,
            observation,
            training=training,
            rng=self._policy_rng,
        )
        action = selection.executed_action
        baa_context = (
            self._pending_baa.context
            if baa_observation is not None and self._pending_baa is not None
            else None
        )
        context = DecisionContext(
            request_id=request_id,
            proposal_id=proposal.proposal_id,
            machine_id=machine.machine_id,
            started_at=self.clock,
            baa_observation=baa_observation
            if baa_observation is not None
            else BAAObservation(
                proposal_features=self._proposal_features(
                    machine, proposal)[None, :],
                proposal_mask=np.ones(1, dtype=np.bool_),
                machine_features=np.stack([
                    self._policy_machine_feature(
                        item, requester_id=machine.machine_id)
                    for item in self.machines
                ]),
                machine_mask=np.ones(len(self.machines), dtype=np.bool_),
                requester_index=machine.machine_id,
                context_features=self._global_context_features(),
            ),
            baa_action=baa_action,
            mma_observation=observation,
            mma_action=action,
            baa_policy_action=(
                baa_context.baa_policy_action if baa_context else None),
            baa_log_probability=(
                baa_context.baa_log_probability if baa_context else None),
            baa_value=baa_context.baa_value if baa_context else None,
            baa_entropy=baa_context.baa_entropy if baa_context else None,
            baa_weights=baa_context.baa_weights if baa_context else None,
            baa_scores=baa_context.baa_scores if baa_context else None,
            mma_log_probability=selection.log_probability,
            mma_value=selection.value,
            mma_entropy=selection.entropy,
            mma_probabilities=selection.probabilities,
            baa_decision=baa_observation is not None,
            baa_eligible_family_count=baa_eligible_family_count,
        )
        self._start_action(machine, proposal, context)

    def _centralized_observation(
            self,
            requestable: list[Machine],
            proposals_by_machine: dict[int, list[BatchProposal]],
            ) -> tuple[CentralizedObservation, list[tuple[int, BatchProposal]]]:
        proposal_entries = [
            (machine.machine_id, proposal)
            for machine in requestable
            for proposal in proposals_by_machine.get(machine.machine_id, [])
        ]
        if not proposal_entries:
            raise ValueError("A centralized decision needs a feasible CH candidate.")
        machine_mask = np.zeros(len(self.machines), dtype=np.bool_)
        operation_mask = np.zeros((len(self.machines), 3), dtype=np.bool_)
        for machine in requestable:
            proposals = proposals_by_machine.get(machine.machine_id, [])
            if not proposals:
                continue
            machine_mask[machine.machine_id] = True
            operation_mask[machine.machine_id, 0] = True
            operation_mask[machine.machine_id, 1:] = machine.state > 0
        observation = CentralizedObservation(
            proposal_features=np.stack([
                self._control_proposal_features(
                    self.machines[machine_id], proposal)
                for machine_id, proposal in proposal_entries
            ]),
            proposal_machine_indices=np.asarray([
                machine_id for machine_id, _ in proposal_entries
            ], dtype=np.int64),
            machine_features=np.stack([
                self._machine_feature(machine, requester_id=-1)
                for machine in self.machines
            ]),
            machine_mask=machine_mask,
            operation_mask=operation_mask,
        )
        return observation, proposal_entries

    def _request_centralized(
            self,
            requestable: list[Machine],
            *,
            epsilon: float,
            training: bool) -> Machine | None:
        proposals_by_machine = {}
        available_jobs = self._available_jobs()
        for machine in requestable:
            proposals = (
                [self._proposal_for_reserved(machine)]
                if machine.batch_ids
                else self.controller.build_proposals(
                    available_jobs,
                    machine.capacity,
                    self.clock,
                    self.job_cluster_pt,
                    requester_id=machine.machine_id,
                    machine_state=machine.state,
                )
            )
            if proposals:
                proposals_by_machine[machine.machine_id] = proposals
        if not proposals_by_machine:
            return None
        observation, proposal_entries = self._centralized_observation(
            requestable, proposals_by_machine)
        self._close_previous_control(
            observation, terminated=False, trajectory_id=-1)
        action = self.controller.select_action(
            observation,
            training=training,
            epsilon=epsilon if training else 0.0,
            rng=self._policy_rng,
        )
        if not isinstance(action, CentralizedAction):
            raise TypeError(
                "Centralized controller must return CentralizedAction.")
        if (
                not 0 <= action.machine_id < len(self.machines)
                or not observation.machine_mask[action.machine_id]
                or not 0 <= action.operation < 3
                or not observation.operation_mask[
                    action.machine_id, action.operation]):
            raise RuntimeError("Centralized controller selected an invalid action.")
        machine = self.machines[action.machine_id]
        machine_proposals = proposals_by_machine[machine.machine_id]
        if action.operation == int(MachineAction.PROCESS):
            if (
                    not 0 <= action.proposal_index < len(proposal_entries)
                    or proposal_entries[action.proposal_index][0]
                    != machine.machine_id):
                raise RuntimeError(
                    "Centralized PPO selected a candidate owned by another machine.")
            proposal = proposal_entries[action.proposal_index][1]
            proposal_index = next(
                index for index, item in enumerate(machine_proposals)
                if item is proposal)
        else:
            proposal = machine_proposals[0]
            proposal_index = 0
        self._request_sequence += 1
        context = DecisionContext(
            request_id=self._request_sequence,
            proposal_id=proposal.proposal_id,
            machine_id=machine.machine_id,
            started_at=self.clock,
            baa_observation=self._baa_observation(machine, machine_proposals),
            baa_action=proposal_index,
            mma_observation=self._mma_observation(machine, proposal),
            mma_action=action.operation,
        )
        self._pending_control[-1] = PendingControl(
            state=observation,
            action=action,
            requester_class=machine.class_id,
            requester_id=machine.machine_id,
            started_at=self.clock,
            potential_at_start=self._potential(),
        )
        self._start_action(machine, proposal, context)
        return machine

    def _request(
            self,
            machine: Machine,
            *,
            epsilon: float,
            training: bool) -> bool:
        if self.controller is not None:
            proposals = (
                [self._proposal_for_reserved(machine)]
                if machine.batch_ids
                else (
                    self.controller.build_proposals(
                        self._available_jobs(),
                        machine.capacity,
                        self.clock,
                        self.job_cluster_pt,
                        requester_id=machine.machine_id,
                        machine_state=machine.state,
                    )
                    if hasattr(self.controller, "build_proposals")
                    else build_rule_proposals(
                        self._available_jobs(),
                        machine.capacity,
                        self.clock,
                        self.job_cluster_pt,
                    )
                )
            )
            if not proposals:
                return False
            observation = self._joint_observation(machine, proposals)
            trajectory_id = (
                machine.machine_id
                if getattr(
                    self.controller,
                    "control_trajectory_scope",
                    "global",
                ) == "physical"
                else -1
            )
            self._close_previous_control(
                observation,
                terminated=False,
                trajectory_id=trajectory_id,
            )
            action = int(self.controller.select_action(
                observation,
                requester_class=machine.class_id,
                requester_id=machine.machine_id,
                machine_state=machine.state,
                clock=self.clock,
                retained_batch=bool(machine.batch_ids),
                training=training,
                epsilon=epsilon if training else 0.0,
                rng=self._policy_rng,
            ))
            if (
                    action < 0
                    or action >= observation.action_mask.size
                    or not observation.action_mask[action]):
                raise RuntimeError(
                    "Controller selected an invalid shared-simulator action.")
            proposal_count = len(proposals)
            if action < proposal_count:
                proposal_index = action
                maintenance_action = int(MachineAction.PROCESS)
            else:
                proposal_index = 0
                maintenance_action = action - proposal_count + 1
            proposal = proposals[proposal_index]
            self._request_sequence += 1
            context = DecisionContext(
                request_id=self._request_sequence,
                proposal_id=proposal.proposal_id,
                machine_id=machine.machine_id,
                started_at=self.clock,
                baa_observation=self._baa_observation(machine, proposals),
                baa_action=proposal_index,
                mma_observation=self._mma_observation(machine, proposal),
                mma_action=maintenance_action,
            )
            self._pending_control[trajectory_id] = PendingControl(
                state=observation,
                action=action,
                requester_class=machine.class_id,
                requester_id=machine.machine_id,
                started_at=self.clock,
                potential_at_start=self._potential(),
            )
            self._start_action(machine, proposal, context)
            return True
        if machine.batch_ids:
            proposal = self._proposal_for_reserved(machine)
            self._request_sequence += 1
            self._decide_mma(
                machine, proposal, epsilon=epsilon, training=training,
                baa_observation=None, baa_action=0,
                request_id=self._request_sequence,
            )
            return True
        available_jobs = self._available_jobs()
        eligible_family_count = len({
            job.family for job in available_jobs
            if job.size <= machine.capacity
        })
        proposals = self._build_candidate_proposals(available_jobs,machine.capacity)
        if not proposals:
            return False
        observation = self._baa_observation(machine, proposals)
        self._close_previous_baa(observation, terminated=False)
        selection = self.mappo.select_baa(
            observation,
            training=training,
        )
        action = selection.executed_action
        proposal = proposals[action]
        self._request_sequence += 1
        request_id = self._request_sequence
        placeholder = DecisionContext(
            request_id=request_id,
            proposal_id=proposal.proposal_id,
            machine_id=machine.machine_id,
            started_at=self.clock,
            baa_observation=observation,
            baa_action=action,
            mma_observation=self._mma_observation(machine, proposal),
            mma_action=-1,
            baa_policy_action=np.asarray(
                selection.policy_action, dtype=np.int64),
            baa_log_probability=selection.log_probability,
            baa_value=selection.value,
            baa_entropy=selection.entropy,
            baa_weights=None,
            baa_scores=selection.scores,
            baa_decision=True,
            baa_eligible_family_count=eligible_family_count,
        )
        self._pending_baa = PendingDecision(
            context=placeholder,
            potential_at_start=self._potential(),
        )
        self._decide_mma(
            machine, proposal, epsilon=epsilon, training=training,
            baa_observation=observation, baa_action=action,
            request_id=request_id,
            baa_eligible_family_count=eligible_family_count,
        )
        return True

    def _build_candidate_proposals(self,jobs,capacity):
        """Default CH pool; inference-time ablations may replace packing only."""
        return build_proposals(jobs,capacity,self.clock,self.job_cluster_pt)

    def _next_arrival_time(self) -> float:
        future = [
            job.arrival_time for job in self.jobs
            if not job.finished
            and job.reserved_by is None
            and job.arrival_time > self.clock
        ]
        return min(future, default=math.inf)

    def _controller_has_pending_action(self, machine: Machine) -> bool:
        """Whether an offline controller needs an idle-machine request.

        Ordinary online controllers remain driven exclusively by available
        work.  A frozen static plan may need to execute maintenance before its
        next planned batch arrives, or deliberately wait for that exact batch.
        """

        return bool(
            self.controller is not None
            and hasattr(self.controller, "has_pending_action")
            and self.controller.has_pending_action(machine.machine_id)
        )

    def _handle_machine_event(
            self, machine: Machine, event: str, at: float) -> None:
        if machine.phase is MachinePhase.PROCESSING:
            elapsed = at - machine.last_event_at
            machine.processing_time += elapsed
            machine.remaining_work = max(
                0.0,
                machine.remaining_work
                - elapsed * self._production_rates(machine)[machine.state],
            )
            machine.last_event_at = at
        if event == "complete":
            # Completion clears numerical residuals before a new batch is assigned.
            machine.remaining_work = 0.0
            completed_ids = machine.batch_ids
            for job_id in completed_ids:
                job = self.jobs[job_id]
                job.finished = True
                job.finish_time = self.clock
                job.reserved_by = None
            machine.history.append({
                "time": self.clock, "event": "process_complete",
                "batch": list(completed_ids), "state": machine.state,
            })
            machine.batch_ids = ()
            machine.phase = MachinePhase.IDLE
            machine.idle_since = self.clock
            return
        if event == "degrade" and machine.state < self.config.degradation_states - 1:
            machine.state += 1
            machine.history.append({
                "time": self.clock, "event": "degrade", "state": machine.state,
            })
            self._schedule_processing_event(machine)
            return
        if event == "degrade":
            machine.failure_type = -1
            repair_time = self.config.minimal_repair_times[machine.class_id]
        elif event.startswith("failure:"):
            machine.failure_type = int(event.split(":")[1])
            start, end = self.config.corrective_time_ranges[machine.class_id]
            repair_times = np.linspace(
                start, end, self.config.failure_types, dtype=float)
            repair_time = float(repair_times[machine.failure_type])
        else:
            repair_time = 0.0
        if event == "maintenance_complete":
            if machine.maintenance_action == int(MachineAction.PM):
                machine.state = 0
            else:
                recovery_rates = np.insert(
                    np.asarray(
                        self.config.ipm_recovery_rates[machine.class_id],
                        dtype=float,
                    ),
                    0,
                    0.0,
                )
                remaining = float(self.config.ipm_times[machine.class_id])
                while machine.state > 0:
                    recovery_time = float(self._simulation_rng.exponential(
                        1 / recovery_rates[machine.state]))
                    if recovery_time > remaining:
                        break
                    remaining -= recovery_time
                    machine.state -= 1
            machine.phase = MachinePhase.IDLE
            machine.idle_since = self.clock
            machine.history.append({
                "time": self.clock, "event": "maintenance_complete",
                "state": machine.state,
            })
            return
        machine.phase = MachinePhase.REPAIRING
        machine.breakdown_count += 1
        machine.failure_interruptions += 1
        machine.corrective_count += 1
        machine.corrective_maintenance_time += repair_time
        machine.maintenance_time += repair_time
        if machine.failure_type is not None and machine.failure_type >= 0:
            machine.unexpected_breakdown_count += 1
            if machine.failure_type == self.config.failure_types - 1:
                machine.critical_failure_count += 1
        machine.history.append({
            "time": self.clock, "event": "failure",
            "failure_type": machine.failure_type, "state": machine.state,
        })
        self._schedule(machine, self.clock + repair_time, "repair_complete")

    def _handle_repair_completion(self, machine: Machine) -> None:
        if machine.failure_type == -1:
            machine.state = self.config.degradation_states - 1
        machine.phase = MachinePhase.IDLE
        machine.idle_since = self.clock
        machine.history.append({
            "time": self.clock, "event": "repair_complete",
            "state": machine.state,
        })
        # The failed batch is retained. Its transition therefore bootstraps
        # from this machine's next decision instead of being marked terminal.

    def _advance_with_repairs(self) -> None:
        """Advance one event, intercepting repair completion explicitly."""
        arrival_time = self._next_arrival_time()
        while self._event_queue:
            at, _, machine_id, event, generation = self._event_queue[0]
            if generation != self.machines[machine_id].generation:
                heapq.heappop(self._event_queue)
                continue
            break
        machine_time = self._event_queue[0][0] if self._event_queue else math.inf
        next_time = min(arrival_time, machine_time)
        if not math.isfinite(next_time):
            raise RuntimeError("Simulation deadlock: unfinished jobs but no event.")
        self.clock = next_time
        if arrival_time <= machine_time:
            return
        at, _, machine_id, event, generation = heapq.heappop(self._event_queue)
        machine = self.machines[machine_id]
        if generation != machine.generation:
            return
        if event == "repair_complete":
            self._handle_repair_completion(machine)
        else:
            self._handle_machine_event(machine, event, at)

    def _finish_episode_transitions(self) -> None:
        if self.controller is not None:
            for trajectory_id in tuple(self._pending_control):
                self._close_previous_control(
                    None,
                    terminated=True,
                    trajectory_id=trajectory_id,
                )
            return
        self._close_previous_baa(None, terminated=True)
        for machine in self.machines:
            self._close_previous_mma(machine, None, terminated=True)

    def _record_controller_decision(
            self, machine: Machine, decisions: int, latency: float) -> None:
        context = machine.context
        self.decision_history.append({
            "decision_index": decisions,
            "clock": self.clock,
            "request_id": context.request_id if context else None,
            "proposal_id": context.proposal_id if context else None,
            "machine_id": machine.machine_id,
            "machine_class": machine.class_id,
            "machine_state": machine.state,
            "job_ids": list(machine.batch_ids),
            "batch_action": context.baa_action if context else None,
            "maintenance_action": context.mma_action if context else None,
            "mma_probability_process": (
                float(context.mma_probabilities[0])
                if context is not None
                and context.mma_probabilities is not None else None),
            "mma_probability_ipm": (
                float(context.mma_probabilities[1])
                if context is not None
                and context.mma_probabilities is not None else None),
            "mma_probability_pm": (
                float(context.mma_probabilities[2])
                if context is not None
                and context.mma_probabilities is not None else None),
            "mma_policy_entropy": context.mma_entropy if context else None,
            "mma_failure_risk": (
                float(context.mma_observation.machine_features[-2])
                if context is not None else None),
            "latency_seconds": latency,
            **self._baa_policy_diagnostics(context),
        })

    def run_episode(
            self,
            *,
            training: bool,
            epsilon: float = 0.0) -> dict:
        self._episode_training = bool(training)
        decisions = 0
        while self._unfinished_jobs():
            requestable = sorted(
                (
                    machine for machine in self.machines
                    if machine.phase is MachinePhase.IDLE
                    and (
                        bool(machine.batch_ids)
                        or bool(self._available_jobs())
                        or self._controller_has_pending_action(machine)
                    )
                ),
                key=lambda item: (
                    item.idle_since, item.machine_id),
            )
            made_decision = False
            if (
                    self.controller is not None
                    and getattr(
                        self.controller,
                        "centralized_machine_selection",
                        False,
                    )
                    and requestable):
                while requestable:
                    started = time.perf_counter()
                    machine = self._request_centralized(
                        requestable, epsilon=epsilon, training=training)
                    if machine is None:
                        break
                    latency = time.perf_counter() - started
                    self._decision_latencies.append(latency)
                    decisions += 1
                    self._record_controller_decision(
                        machine, decisions, latency)
                    made_decision = True
                    requestable = sorted(
                        (
                            item for item in self.machines
                            if item.phase is MachinePhase.IDLE
                            and (
                                bool(item.batch_ids)
                                or bool(self._available_jobs())
                                or self._controller_has_pending_action(item)
                            )
                        ),
                        key=lambda item: (item.idle_since, item.machine_id),
                    )
            else:
                for machine in requestable:
                    started = time.perf_counter()
                    if self._request(
                            machine, epsilon=epsilon, training=training):
                        latency = time.perf_counter() - started
                        self._decision_latencies.append(latency)
                        decisions += 1
                        self._record_controller_decision(
                            machine, decisions, latency)
                        made_decision = True
            if self._unfinished_jobs():
                self._advance_with_repairs()
        self._finish_episode_transitions()
        missing_start_times = [
            job.job_id for job in self.jobs
            if job.finished and job.first_start_time is None
        ]
        tardiness = np.asarray([
            max(0.0, float(job.finish_time) - job.due_time)
            for job in self.jobs
        ])
        flow_times = np.asarray([
            float(job.finish_time) - job.arrival_time for job in self.jobs
        ])
        queue_waits = np.asarray([
            float(job.first_start_time) - job.arrival_time
            for job in self.jobs if job.first_start_time is not None
        ])
        makespan = max(float(job.finish_time) for job in self.jobs)
        planned = sum(machine.planned_maintenance_time for machine in self.machines)
        corrective = sum(
            machine.corrective_maintenance_time for machine in self.machines)
        processing = sum(machine.processing_time for machine in self.machines)
        capacity_committed = sum(
            machine.capacity_committed for machine in self.machines)
        latencies = np.asarray(self._decision_latencies, dtype=float)
        metrics = {
            "total_tardiness": float(tardiness.sum()),
            "mean_tardiness": float(tardiness.mean()),
            "tardy_job_rate": float(np.mean(tardiness > 0)),
            "p95_job_tardiness": float(np.percentile(tardiness, 95)),
            "max_job_tardiness": float(tardiness.max(initial=0.0)),
            "mean_flow_time": float(flow_times.mean()),
            "p95_flow_time": float(np.percentile(flow_times, 95)),
            "makespan": makespan,
            "throughput": float(len(self.jobs) / makespan) if makespan else 0.0,
            "job_count": len(self.jobs),
            "initial_job_count": self.initial_jobs,
            "dynamic_job_count": len(self.jobs) - self.initial_jobs,
            "target_load": float(self.target_load),
            "arrival_span": self._instance_load.arrival_span,
            "dynamic_nominal_workload": (
                self._instance_load.dynamic_nominal_workload),
            "initial_nominal_workload": (
                self._instance_load.initial_nominal_workload),
            "total_nominal_workload": (
                self._instance_load.total_nominal_workload),
            "realized_dynamic_load": (
                self._instance_load.realized_dynamic_load),
            "finite_instance_load": (
                self._instance_load.finite_instance_load),
            "actual_processing_load": float(
                sum(machine.processing_time * machine.capacity
                    for machine in self.machines)
                / max(1e-12, self._instance_load.arrival_span
                      * self._instance_load.plant_capacity)),
            "actual_maintenance_load": float(
                sum(machine.maintenance_time * machine.capacity
                    for machine in self.machines)
                / max(1e-12, self._instance_load.arrival_span
                      * self._instance_load.plant_capacity)),
            "breakdowns": sum(machine.breakdown_count for machine in self.machines),
            "unexpected_breakdowns": sum(
                machine.unexpected_breakdown_count for machine in self.machines),
            "critical_failures": sum(
                machine.critical_failure_count for machine in self.machines),
            "failure_interruptions": sum(
                machine.failure_interruptions for machine in self.machines),
            "maintenance_time": planned + corrective,
            "planned_maintenance_time": planned,
            "corrective_maintenance_time": corrective,
            "ipm_count": sum(machine.ipm_count for machine in self.machines),
            "pm_count": sum(machine.pm_count for machine in self.machines),
            "corrective_maintenance_count": sum(
                machine.corrective_count for machine in self.machines),
            "maintenance_downtime_ratio": float(
                (planned + corrective)
                / max(1e-12, processing + planned + corrective)),
            "production_utilization": float(
                processing / max(1e-12, makespan * len(self.machines))),
            "batch_fill_ratio": float(
                sum(machine.loaded_size for machine in self.machines)
                / capacity_committed) if capacity_committed else 0.0,
            "mean_queue_waiting": float(
                queue_waits.mean()) if queue_waits.size else 0.0,
            "failed_batch_restarts": 0,
            "reprocessed_work": 0.0,
            "decision_latency_mean": float(
                latencies.mean()) if latencies.size else 0.0,
            "decision_latency_p95": float(
                np.percentile(latencies, 95)) if latencies.size else 0.0,
            "decision_latency_p99": float(
                np.percentile(latencies, 99)) if latencies.size else 0.0,
            "terminated": True,
            "unfinished_jobs": 0,
            "invalid_action_fallbacks": 0,
            "decisions": decisions,
        }
        return metrics

    def audit_tables(self, case) -> dict[str, list[dict]]:
        """Return raw tables sufficient to independently recompute metrics."""

        identity = {
            "instance_index": case.instance_index,
            "replication_index": case.replication_index,
            "instance_seed": case.instance_seed,
            "simulation_seed": case.simulation_seed,
        }
        jobs = [{
            **identity,
            "job_id": job.job_id,
            "arrival_time": job.arrival_time,
            "family": job.family,
            "size": job.size,
            "due_time": job.due_time,
            "first_start_time": job.first_start_time,
            "finish_time": job.finish_time,
            "tardiness": max(0.0, float(job.finish_time) - job.due_time),
            "flow_time": float(job.finish_time) - job.arrival_time,
        } for job in self.jobs]
        machine_events = []
        for machine in self.machines:
            for event_index, event in enumerate(machine.history):
                machine_events.append({
                    **identity,
                    "machine_id": machine.machine_id,
                    "machine_class": machine.class_id,
                    "event_index": event_index,
                    **event,
                })
        decisions = [{**identity, **item} for item in self.decision_history]
        return {
            "jobs": jobs,
            "machine_events": machine_events,
            "decisions": decisions,
        }

    def _machine_responsibility_metrics(self) -> dict[str, float | int]:
        """Expose production, maintenance, and failure burden by actor scope."""

        result: dict[str, float | int] = {}
        total_processing = sum(machine.processing_time for machine in self.machines)
        total_maintenance = sum(machine.maintenance_time for machine in self.machines)
        total_loaded = sum(machine.loaded_size for machine in self.machines)
        total_batches = sum(machine.batch_count for machine in self.machines)
        total_capacity = sum(
            machine.capacity_committed for machine in self.machines)
        horizon = max(1.0, self.clock)
        production_utilizations = np.asarray([
            machine.processing_time / horizon for machine in self.machines
        ], dtype=float)
        maintenance_utilizations = np.asarray([
            machine.maintenance_time / horizon for machine in self.machines
        ], dtype=float)

        def coefficient_of_variation(values: np.ndarray) -> float:
            mean = float(values.mean()) if values.size else 0.0
            return float(values.std() / mean) if mean > 1e-12 else 0.0

        result.update({
            "responsibility_processing_time_total": total_processing,
            "responsibility_maintenance_time_total": total_maintenance,
            "responsibility_loaded_size_total": total_loaded,
            "responsibility_batch_count_total": total_batches,
            "responsibility_capacity_committed_total": total_capacity,
            "responsibility_machine_processing_utilization_cv": (
                coefficient_of_variation(production_utilizations)),
            "responsibility_machine_maintenance_utilization_cv": (
                coefficient_of_variation(maintenance_utilizations)),
        })
        for class_id in range(len(self.scenario.machine_counts)):
            machines = [
                machine for machine in self.machines
                if machine.class_id == class_id
            ]
            count = len(machines)
            processing = sum(machine.processing_time for machine in machines)
            maintenance = sum(machine.maintenance_time for machine in machines)
            planned = sum(
                machine.planned_maintenance_time for machine in machines)
            corrective = sum(
                machine.corrective_maintenance_time for machine in machines)
            loaded = sum(machine.loaded_size for machine in machines)
            batches = sum(machine.batch_count for machine in machines)
            committed = sum(
                machine.capacity_committed for machine in machines)
            prefix = f"responsibility_class_{class_id}"
            result.update({
                f"{prefix}_machine_count": count,
                f"{prefix}_batch_count": batches,
                f"{prefix}_loaded_size": loaded,
                f"{prefix}_capacity_committed": committed,
                f"{prefix}_processing_time": processing,
                f"{prefix}_maintenance_time": maintenance,
                f"{prefix}_planned_maintenance_time": planned,
                f"{prefix}_corrective_maintenance_time": corrective,
                f"{prefix}_breakdown_count": sum(
                    machine.breakdown_count for machine in machines),
                f"{prefix}_ipm_count": sum(
                    machine.ipm_count for machine in machines),
                f"{prefix}_pm_count": sum(
                    machine.pm_count for machine in machines),
                f"{prefix}_corrective_count": sum(
                    machine.corrective_count for machine in machines),
                f"{prefix}_processing_time_share": (
                    processing / total_processing
                    if total_processing > 0 else 0.0),
                f"{prefix}_maintenance_time_share": (
                    maintenance / total_maintenance
                    if total_maintenance > 0 else 0.0),
                f"{prefix}_loaded_size_share": (
                    loaded / total_loaded if total_loaded > 0 else 0.0),
                f"{prefix}_batch_count_share": (
                    batches / total_batches if total_batches > 0 else 0.0),
                f"{prefix}_processing_utilization": (
                    processing / (count * horizon) if count else 0.0),
                f"{prefix}_batch_fill_ratio": (
                    loaded / committed if committed > 0 else 0.0),
                f"{prefix}_batches_per_machine": (
                    batches / count if count else 0.0),
                f"{prefix}_loaded_size_per_machine": (
                    loaded / count if count else 0.0),
            })
        for machine in self.machines:
            prefix = f"responsibility_machine_{machine.machine_id}"
            result.update({
                f"{prefix}_class_id": machine.class_id,
                f"{prefix}_batch_count": machine.batch_count,
                f"{prefix}_loaded_size": machine.loaded_size,
                f"{prefix}_capacity_committed": machine.capacity_committed,
                f"{prefix}_processing_time": machine.processing_time,
                f"{prefix}_maintenance_time": machine.maintenance_time,
                f"{prefix}_planned_maintenance_time": (
                    machine.planned_maintenance_time),
                f"{prefix}_corrective_maintenance_time": (
                    machine.corrective_maintenance_time),
                f"{prefix}_breakdown_count": machine.breakdown_count,
                f"{prefix}_ipm_count": machine.ipm_count,
                f"{prefix}_pm_count": machine.pm_count,
                f"{prefix}_corrective_count": machine.corrective_count,
                f"{prefix}_processing_utilization": (
                    machine.processing_time / horizon),
                f"{prefix}_maintenance_utilization": (
                    machine.maintenance_time / horizon),
                f"{prefix}_batch_fill_ratio": (
                    machine.loaded_size / machine.capacity_committed
                    if machine.capacity_committed else 0.0),
            })
        return result

    def train_episode(
            self,
            *,
            instance_seed: int,
            simulation_seed: int,
            policy_seed: int | None = None) -> dict:
        if policy_seed is not None:
            random.seed(policy_seed)
        scheduled_episode = self.total_episode + 1
        mma_entropy_weight = self.config.mma_entropy_weight_at(
            scheduled_episode,
            self.training_horizon or self.max_episode,
        )
        self.mappo.set_mma_entropy_weight(mma_entropy_weight)
        if self.config.calibrated_reward_scale is None:
            raise RuntimeError(
                "Training requires an independently calibrated SDR reward "
                "scale artifact.")
        # The scale is estimated once on an independent calibration set and
        # remains independent of every sampled on-policy training episode.
        self._active_reward_normalizer = float(
            self.config.calibrated_reward_scale)
        self.reset(
            instance_seed=instance_seed,
            simulation_seed=simulation_seed,
            policy_seed=policy_seed,
        )
        self._active_job_count = max(1, len(self.jobs))
        self.mappo.begin_episode(
            instance_seed if policy_seed is None else policy_seed)
        metrics = self.run_episode(training=True, epsilon=0.0)
        completed_episode = self.total_episode + 1
        cumulative_decisions = self.total_step + metrics["decisions"]
        # Wall-clock latency is an evaluation metric, not deterministic
        # training state. Excluding it keeps checkpoint/resume comparisons
        # bitwise auditable while evaluation still records real timings.
        for name in (
                "decision_latency_mean",
                "decision_latency_p95",
                "decision_latency_p99"):
            metrics[name] = 0.0
        self.total_step += metrics["decisions"]
        self.total_episode += 1
        if self.total_episode % self.config.rollout_episodes == 0:
            loss = self.mappo.update()
            if loss is not None:
                self.losses.append(loss)
                for record in self.mappo.last_update_records:
                    diagnostics = record.copy()
                    diagnostics.update({
                        "method": self.handshake_mode.value,
                        "episode": completed_episode,
                        "cumulative_decisions": cumulative_decisions,
                    })
                    self.update_diagnostics.append(diagnostics)
                for record in self.mappo.last_rollout_records:
                    diagnostics = record.copy()
                    diagnostics.update({
                        "method": self.handshake_mode.value,
                        "episode": completed_episode,
                        "cumulative_decisions": cumulative_decisions,
                    })
                    self.rollout_diagnostics.append(diagnostics)
        metrics["instance_seed"] = int(instance_seed)
        metrics["simulation_seed"] = int(simulation_seed)
        metrics["policy_seed"] = int(
            instance_seed if policy_seed is None else policy_seed)
        metrics["episode"] = self.total_episode
        metrics["cumulative_decisions"] = self.total_step
        metrics["reward_normalizer"] = self._active_reward_normalizer
        metrics["reward_normalizer_job_count"] = self._active_job_count
        metrics["reward_normalization_mode"] = (
            self.config.reward_normalization_mode)
        metrics["reward_scale_artifact_sha256"] = (
            self.config.reward_scale_artifact_sha256 or "")
        metrics["reward_reference_rule"] = (
            f"{self.config.reward_reference_batch_rule}+"
            f"{self.config.reward_reference_allocation_rule}+"
            f"{self.config.reward_reference_maintenance_policy}")
        metrics["policy_updates_total"] = self.mappo.update_steps
        metrics["learning_rate"] = float(
            self.mappo.optimizer.param_groups[0]["lr"])
        metrics["baa_entropy_weight"] = self.config.baa_entropy_weight
        metrics["mma_entropy_weight"] = mma_entropy_weight
        metrics.update(self._machine_responsibility_metrics())
        baa_summary = self.mappo.episode_transition_summary(0, "baa")
        for component in (
                "global_potential_delta",
                "global_potential_delta_raw",
                "sdr_reward_normalizer"):
            baa_summary.setdefault(f"baa_{component}_sum", 0.0)
        metrics.update(baa_summary)
        mma_reward_sum = 0.0
        mma_trajectory_count = 0
        for agent_id in range(len(self.scenario.machine_counts)):
            prefix = f"mma_{agent_id}"
            summary = self.mappo.episode_transition_summary(
                agent_id + 1, prefix)
            for action in range(3):
                summary.setdefault(f"{prefix}_action_{action}_count", 0)
            for component in (
                    "global_potential_delta",
                    "global_potential_delta_raw",
                    "sdr_reward_normalizer"):
                summary.setdefault(f"{prefix}_{component}_sum", 0.0)
            mma_reward_sum += float(summary[f"{prefix}_reward_sum"])
            mma_trajectory_count += int(
                summary[f"{prefix}_terminal_transitions"])
            metrics.update(summary)
        metrics["mma_shared_reward_sum"] = mma_reward_sum
        metrics["mma_physical_trajectory_count"] = mma_trajectory_count
        metrics["mma_shared_reward_per_trajectory_mean"] = (
            mma_reward_sum / mma_trajectory_count
            if mma_trajectory_count else 0.0)
        metrics.update(baa_policy_summary(self.decision_history))
        metrics.update(decision_action_summary(
            self.decision_history,
            machine_class_count=len(self.scenario.machine_counts),
            health_state_count=self.config.degradation_states,
        ))
        metrics.update(mma_probability_summary(
            self.decision_history,
            machine_class_count=len(self.scenario.machine_counts),
            health_state_count=self.config.degradation_states,
        ))
        self.episode_metrics.append(metrics)
        return metrics

    def _validation_score(self) -> float:
        evaluator = HierarchicalSchedulingEnv(
            self.job_parameters,
            list(self.scenario.machine_counts),
            processing_times=self.scenario.job_processing_times,
            handshake_mode=self.handshake_mode,
            config=self.config,
            device=self.device,
        )
        if self.validation_evaluator_setup is not None:
            self.validation_evaluator_setup(evaluator)
        evaluator.mappo.load_state_dict(
            self.mappo.inference_state(), training=False)
        metrics = []
        schedule = self.validation_seed_schedule or [{
            "instance_seed": 3_000_000 + index,
            "simulation_seed": 4_000_000 + index,
            "policy_seed": 4_000_000 + index,
        } for index in range(self.validation_instances)]
        for item in schedule:
            if self.validation_case_reset is None:
                evaluator.reset(
                    instance_seed=int(item["instance_seed"]),
                    simulation_seed=int(item["simulation_seed"]),
                    policy_seed=int(item["policy_seed"]),
                )
            else:
                self.validation_case_reset(evaluator, item)
            metrics.append(EpisodeMetrics.from_mapping(
                evaluator.run_episode(training=False)))
        record = validation_summary(self.total_episode, metrics)
        if self.validation_summary_auditor is not None:
            self.validation_summary_auditor(record)
        score = float(record["mean_tardiness"])
        if self.pretraining_baseline is not None:
            baseline = float(self.pretraining_baseline["mean_tardiness"])
            improvement = baseline - score
            paired_improvements = np.asarray([
                float(reference["mean_tardiness"])
                - float(candidate.mean_tardiness)
                for reference, candidate in zip(
                    self.pretraining_baseline_cases, metrics, strict=True)
            ], dtype=float)
            record.update({
                "reference_policy": self.pretraining_baseline["policy"],
                "reference_mean_tardiness": baseline,
                "absolute_improvement_vs_reference": improvement,
                "relative_improvement_vs_reference_percent": (
                    100.0 * improvement / max(
                        self.config.reward_normalizer_floor, baseline)),
                "beats_reference": bool(score < baseline),
                "paired_win_rate": float(np.mean(paired_improvements > 0.0)),
                "paired_tie_rate": float(np.mean(paired_improvements == 0.0)),
                "paired_loss_rate": float(np.mean(paired_improvements < 0.0)),
                "paired_improvement_median": float(
                    np.median(paired_improvements)),
            })
        self.validation_metrics.append(record)
        return score

    def _run_pretraining_baseline(self, output: Path) -> dict:
        """Evaluate the frozen SDR reference on the exact validation cases."""

        from methods.baselines.sdr.shared import RuleController

        policy = (
            f"{self.config.reward_reference_batch_rule}+"
            f"{self.config.reward_reference_allocation_rule}+"
            f"{self.config.reward_reference_maintenance_policy}"
        )
        controller = RuleController(
            self.config.reward_reference_batch_rule,
            self.config.reward_reference_allocation_rule,
            self.config.reward_reference_maintenance_policy,
            max(self.scenario.job_processing_times),
        )
        # Constructing the rule-controlled environment also constructs unused
        # neural modules. Preserve the training RNG so this diagnostic cannot
        # alter the initial policy or the subsequent optimization trajectory.
        with torch.random.fork_rng(devices=[]):
            evaluator = HierarchicalSchedulingEnv(
                self.job_parameters,
                list(self.scenario.machine_counts),
                processing_times=self.scenario.job_processing_times,
                handshake_mode=HandshakeMode.INTERACTIVE,
                config=self.config,
                controller=controller,
                device="cpu",
            )
        cases: list[dict] = []
        episode_metrics: list[EpisodeMetrics] = []
        schedule = self.validation_seed_schedule or [{
            "instance_seed": 3_000_000 + index,
            "simulation_seed": 4_000_000 + index,
            "policy_seed": 4_000_000 + index,
        } for index in range(self.validation_instances)]
        for index, item in enumerate(schedule):
            instance_seed = int(item["instance_seed"])
            simulation_seed = int(item["simulation_seed"])
            policy_seed = int(item["policy_seed"])
            evaluator.reset(
                instance_seed=instance_seed,
                simulation_seed=simulation_seed,
                policy_seed=policy_seed,
            )
            metrics = EpisodeMetrics.from_mapping(
                evaluator.run_episode(training=False))
            episode_metrics.append(metrics)
            cases.append({
                "validation_case": index,
                "instance_seed": instance_seed,
                "simulation_seed": simulation_seed,
                "policy_seed": policy_seed,
                "policy": policy,
                **metrics.to_dict(),
            })

        summary = validation_summary(0, episode_metrics)
        summary.update({
            "schema_version": 1,
            "policy": policy,
            "purpose": "pretraining_validation_reference",
            "validation_cases": self.validation_instances,
            "instance_seed_start": int(schedule[0]["instance_seed"]),
            "simulation_seed_start": int(schedule[0]["simulation_seed"]),
            "policy_seed_start": int(schedule[0]["policy_seed"]),
            "pairing": "same_frozen_instances_and_simulation_seeds_as_model_validation",
        })
        self.pretraining_baseline = summary
        self.pretraining_baseline_cases = cases
        artifact_stem = policy.replace("+", "_")
        atomic_csv_save(
            cases, output / f"Pretraining_Baseline_{artifact_stem}.csv")
        atomic_json_save(
            summary, output / f"Pretraining_Baseline_{artifact_stem}.json")
        return summary

    def _save_training_logs(self, output: Path) -> None:
        updates = list(self.update_diagnostics)
        updates.sort(key=lambda item: (
            int(item.get("episode", 0)),
            int(item.get("cumulative_decisions", 0)),
            str(item.get("component", "")),
            int(item.get("agent_id", 0)),
            float(item.get("update_step", 0)),
        ))
        save_training_log_bundle(
            output,
            method=(
                "ours" if self.handshake_mode is HandshakeMode.INTERACTIVE
                else "hrl"
            ),
            episodes=self.episode_metrics,
            updates=updates,
            rollouts=self.rollout_diagnostics,
            validation=self.validation_metrics,
            metadata={
                "training_seed": self.training_master_seed,
                "training_condition": {
                    "nominal_finite_load": self.target_load,
                    "initial_jobs_configured": self.initial_jobs,
                    "total_jobs_configured": self.total_jobs,
                    "due_date_factor": self.due_date_factor,
                    "degradation_multiplier": (
                        self.config.degradation_multiplier),
                },
                "handshake_mode": self.handshake_mode.value,
                "algorithm": "heterogeneous_smdp_mappo",
                "actor_topology": (
                    "one_candidate_scoring_baa_actor_plus_one_categorical_actor_"
                    "per_machine_class"),
                "baa_parameters": sum(
                    parameter.numel()
                    for parameter in self.mappo.baa_actor.parameters()),
                "mma_parameters_per_agent": [
                    sum(parameter.numel()
                        for parameter in actor.parameters())
                    for actor in self.mappo.mma_actors
                ],
                "central_critic_parameters": sum(
                    parameter.numel()
                    for parameter in self.mappo.critic.parameters()),
                "feature_contract": {
                    "schema_version": FEATURE_SCHEMA_VERSION,
                    "range": "closed_unit_interval_[0,1]",
                    "runtime_validation": "finite_shape_and_bounds",
                    "online_running_statistics": False,
                    "information_contract": "released_orders_current_shop_and_known_physics",
                    "reference_job_size": REFERENCE_JOB_SIZE,
                    "reference_time": "maximum_of_declared_process_family_times",
                    "future_order_attributes_used": False,
                    "finite_instance_counts_used": False,
                    "generator_condition_labels_used": False,
                    "physics_assumption": (
                        "health_observed_and_rate_recovery_service_parameters_known"),
                    "completion_probability": {
                        "protocol": COMPLETION_PROBABILITY_PROTOCOL,
                        "solver": "float64_cpu_matrix_exponential",
                        "work": "nominal_family_work_or_observed_retained_residual",
                        "allows_progressive_degradation": True,
                        "absorbing_events": "unexpected_breakdowns_and_terminal_deterioration",
                        "includes_future_repairs": False,
                        "exact_work_and_physics_cache_limit": 4096,
                    },
                    "proposal_features": list(BAA_CRITERIA),
                    "policy_machine_features": list(POLICY_MACHINE_FEATURES),
                    "global_context_features": list(GLOBAL_CONTEXT_FEATURES),
                    "mma_maintenance_features": list(MMA_MAINTENANCE_FEATURES),
                    "mma_actor_global_machine_pooling": "encoded_mean_and_max",
                    "critic_pooling": "proposal_mean_max_machine_mean_max_requester",
                },
                "instance_generation": {
                    "protocol": INSTANCE_GENERATION_PROTOCOL,
                    "load_scope": (
                        "initial_and_dynamic_finite_horizon_workload"),
                    "load_definition": (
                        "total_nominal_workload/"
                        "(arrival_span*plant_capacity)"),
                    "nominal_finite_load": self.target_load,
                    "initial_jobs": self.initial_jobs,
                    "total_jobs": self.total_jobs,
                    "due_date_factor": self.due_date_factor,
                    "arrival_process": (
                        "normalized_exponential_spacings_with_exact_"
                        "nominal_finite_load"),
                    "family_composition": (
                        "separately_balanced_initial_and_dynamic_pools"),
                    "size_composition": (
                        "separately_stratified_initial_and_dynamic_pools"),
                },
                "gamma_baa": self.config.gamma_baa,
                "gamma_mma": self.config.gamma_mma,
                "gae_lambda": self.config.gae_lambda,
                "clip_ratio": self.config.ppo_clip_ratio,
                "update_epochs": self.config.ppo_update_epochs,
                "rollout_episodes": self.config.rollout_episodes,
                "minibatch_size": self.config.minibatch_size,
                "optimization_objective": {
                    "actor_reduction": "mean_over_all_actor_eligible_samples",
                    "critic_reduction": "mean_over_all_transitions",
                    "head_balancing": False,
                    "rollback_reason": (
                        "three_seed_validation_regression_and_rare_head_"
                        "gradient_amplification"),
                    "mma_actor_samples": "states_with_at_least_two_valid_actions",
                    "mma_forced_action_samples_train_critic": True,
                    "baa_actor_samples": "all_baa_transitions",
                    "baa_single_candidate_filtering": False,
                },
                "discount_protocol": "fixed_per_decision_transition",
                "baa_entropy_weight": self.config.baa_entropy_weight,
                "mma_entropy_schedule": {
                    "training_horizon": (
                        self.training_horizon or self.max_episode),
                    "initial_weight": self.config.mma_entropy_weight,
                    "final_weight": self.config.mma_entropy_final_weight,
                    "hold_fraction": (
                        self.config.mma_entropy_hold_fraction),
                    "linear_decay_fraction": (
                        self.config.mma_entropy_decay_fraction),
                    "final_fraction": 1.0 - (
                        self.config.mma_entropy_hold_fraction
                        + self.config.mma_entropy_decay_fraction),
                    "episode_indexing": "one_based",
                },
                "reward_normalization": {
                    "mode": self.config.reward_normalization_mode,
                    "reference_policy": (
                        f"{self.config.reward_reference_batch_rule}+"
                        f"{self.config.reward_reference_allocation_rule}+"
                        f"{self.config.reward_reference_maintenance_policy}"),
                    "calibration_independence": (
                        "disjoint_from_training_validation_and_test"),
                    "reference_metric": "mean_tardiness",
                    "floor": self.config.reward_normalizer_floor,
                    "calibrated_scale": (
                        self.config.calibrated_reward_scale),
                    "scale_artifact_sha256": (
                        self.config.reward_scale_artifact_sha256),
                    "calibration_estimator": (
                        "mean_of_per_instance_replication_means_after_"
                        "cross_instance_5_95_percent_winsorization"),
                    "baa_formula": "global_potential_delta / D_ref",
                    "mma_formula": "global_potential_delta / D_ref",
                    "mma_global_potential_reward": True,
                    "mma_reward_protocol": self.config.mma_reward_protocol,
                    "mma_terminal": "global_scheduling_episode_only",
                },
                "maintenance_action_map": {
                    "0": "process", "1": "IPM", "2": "PM"},
                "baa_categorical_action": {
                    "distribution": "masked_categorical",
                    "candidate_features": list(BAA_CRITERIA),
                    "action": "selected_candidate_index",
                    "validation_action": "argmax_candidate_logit",
                    "batch_selection": "learned_candidate_scoring",
                    "decision_scope": "true_baa_choices_only",
                    "mma_only_redecisions_excluded": True,
                    "candidate_pool": "one_constructible_batch_per_available_family",
                    "knowledge_rule_prefilter": False,
                },
                "mma_action_distribution": "masked_categorical",
                "maintenance_parameters_unchanged": True,
                "maintenance_parameter_snapshot": {
                    "degradation_rate_ranges": self.config.degradation_rate_ranges,
                    "failure_rate_ranges": self.config.failure_rate_ranges,
                    "terminal_degradation_rates": self.config.terminal_degradation_rates,
                    "ipm_recovery_rates": self.config.ipm_recovery_rates,
                    "ipm_times": self.config.ipm_times,
                    "pm_times": self.config.pm_times,
                    "corrective_time_ranges": self.config.corrective_time_ranges,
                    "minimal_repair_times": self.config.minimal_repair_times,
                },
                "machine_class_indexing": "zero_based",
                "health_state_indexing": "zero_based_0_is_healthy",
                "episode_indexing": "one_based_completed_episode",
                "validation_episode_indexing": (
                    "episode_0_initial_then_completed_episode"),
                "primary_training_curve": "total_tardiness",
                "primary_validation_curve": "mean_tardiness",
                "pretraining_validation_baseline": self.pretraining_baseline,
            },
        )
        atomic_json_save({
            "episodes": self.episode_metrics,
            "updates": self.update_diagnostics,
            "rollouts": self.rollout_diagnostics,
            "validation": self.validation_metrics,
            "pretraining_baseline": self.pretraining_baseline,
            "pretraining_baseline_cases": self.pretraining_baseline_cases,
            "best_validation": self.best_validation,
        }, output / "Training_Diagnostics.json")

    def checkpoint(self) -> dict:
        return {
            "schema_version": 11,
            "handshake_mode": self.handshake_mode.value,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "scenario": asdict(self.scenario),
            "config": self.config.to_dict(),
            "training": {
                "job_parameters": self.job_parameters,
                "total_episode": self.total_episode,
                "total_step": self.total_step,
                "episode_metrics": self.episode_metrics,
                "losses": self.losses,
                "update_diagnostics": self.update_diagnostics,
                "rollout_diagnostics": self.rollout_diagnostics,
                "validation_metrics": self.validation_metrics,
                "pretraining_baseline": self.pretraining_baseline,
                "pretraining_baseline_cases": self.pretraining_baseline_cases,
                "best_validation": self.best_validation,
                "training_master_seed": self.training_master_seed,
                "training_horizon": self.training_horizon or self.max_episode,
            },
            "mappo": self.mappo.checkpoint(),
            "numpy_random_state": self._simulation_rng.bit_generator.state,
            "policy_random_state": self._policy_rng.bit_generator.state,
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": cuda_rng_state(),
            "device": str(self.device),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        atomic_torch_save(self.checkpoint(), path)

    def save_inference(self, path: str | Path) -> None:
        atomic_torch_save({
            "schema_version": 9,
            "handshake_mode": self.handshake_mode.value,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "mappo": self.mappo.inference_state(),
        }, path)

    def load_inference(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.mappo.load_state_dict(state["mappo"], training=False)

    def load_checkpoint(
            self, path: str | Path, *, load_training: bool = False) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if load_training:
            pass
        self.mappo.load_state_dict(
            state["mappo"], training=load_training)
        if load_training:
            training = state["training"]
            self.total_episode = int(training["total_episode"])
            self.total_step = int(training["total_step"])
            self.episode_metrics = training["episode_metrics"]
            self.losses = training["losses"]
            self.update_diagnostics = training.get(
                "update_diagnostics", [])
            self.rollout_diagnostics = training.get(
                "rollout_diagnostics", [])
            self.validation_metrics = training.get(
                "validation_metrics", [])
            self.pretraining_baseline = training.get("pretraining_baseline")
            self.pretraining_baseline_cases = training.get(
                "pretraining_baseline_cases", [])
            self.best_validation = float(training.get(
                "best_validation", math.inf))
            self.training_master_seed = int(training.get(
                "training_master_seed", self.instance_seed))
            saved_horizon = int(training.get(
                "training_horizon", self.max_episode))
            self.training_horizon = saved_horizon
            self.seed_protocol = TrainingSeedProtocol(
                self.training_master_seed)
            if "numpy_random_state" in state:
                self._simulation_rng.bit_generator.state = (
                    state["numpy_random_state"])
            if "policy_random_state" in state:
                self._policy_rng.bit_generator.state = (
                    state["policy_random_state"])
            if "python_random_state" in state:
                random.setstate(state["python_random_state"])
            if "torch_random_state" in state:
                torch.set_rng_state(state["torch_random_state"].cpu())
            restore_cuda_rng_state(
                state.get("torch_cuda_random_state"),
                self.device,
            )

    def main(
            self,
            results_dir: str,
            resume_from: str | Path | None = None) -> None:
        output = Path(results_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.training_horizon is None:
            self.training_horizon = self.max_episode
        if resume_from is not None:
            self.load_checkpoint(resume_from, load_training=True)
        console = TrainingConsoleLogger(
            method=(
                "ours" if self.handshake_mode is HandshakeMode.INTERACTIVE
                else "hrl"
            ),
            device=str(self.device),
            total_episodes=self.max_episode,
            completed_episodes=self.total_episode,
        )
        if resume_from is None:
            baseline = self._run_pretraining_baseline(output)
            console.print_baseline(baseline)
            score = self._validation_score()
            self.best_validation = score
            initial = self.validation_metrics[-1]
            initial["is_best"] = True
            initial["best_mean_tardiness"] = score
            self.save_inference(output / "hierarchical_inference.pt")
            self.save_checkpoint(output / "hierarchical_checkpoint.pt")
            self._save_training_logs(output)
            console.print_validation(initial, label="initial")
        while self.total_episode < self.max_episode:
            update_start = len(self.update_diagnostics)
            seeds = (
                self.training_seed_schedule[self.total_episode]
                if self.training_seed_schedule is not None
                else self.seed_protocol.for_episode(self.total_episode)
            )
            episode_metrics = self.train_episode(
                instance_seed=seeds.instance_seed,
                simulation_seed=seeds.simulation_seed,
                policy_seed=seeds.policy_seed,
            )
            validation_record = None
            if self.total_episode % self.validation_interval == 0:
                score = self._validation_score()
                improved = score < self.best_validation
                if improved:
                    self.best_validation = score
                    self.save_inference(
                        output / "hierarchical_inference.pt")
                self.validation_metrics[-1]["is_best"] = improved
                self.validation_metrics[-1]["best_mean_tardiness"] = (
                    self.best_validation)
                validation_record = self.validation_metrics[-1]
            episode_updates = self.update_diagnostics[update_start:]
            console.print_episode(
                episode_metrics,
                updates=episode_updates,
                validation=validation_record,
            )
            if self.total_episode % self.checkpoint_interval == 0:
                self.save_checkpoint(
                    output / "hierarchical_checkpoint.pt")
                self._save_training_logs(output)
        if any(self.mappo.rollouts.values()):
            loss = self.mappo.update()
            if loss is not None:
                self.losses.append(loss)
                for record in self.mappo.last_update_records:
                    diagnostics = record.copy()
                    diagnostics.update({
                        "method": self.handshake_mode.value,
                        "episode": self.total_episode,
                        "cumulative_decisions": self.total_step,
                    })
                    self.update_diagnostics.append(diagnostics)
                for record in self.mappo.last_rollout_records:
                    diagnostics = record.copy()
                    diagnostics.update({
                        "method": self.handshake_mode.value,
                        "episode": self.total_episode,
                        "cumulative_decisions": self.total_step,
                    })
                    self.rollout_diagnostics.append(diagnostics)
                if self.episode_metrics:
                    self.episode_metrics[-1]["policy_updates_total"] = (
                        self.mappo.update_steps)
        self.save_checkpoint(output / "hierarchical_checkpoint.pt")
        if not (output / "hierarchical_inference.pt").exists():
            self.save_inference(output / "hierarchical_inference.pt")
        self.save_inference(output / "hierarchical_final_inference.pt")
        np.savetxt(
            output / "Training_Total_Tardiness.txt",
            np.asarray([
                item["total_tardiness"] for item in self.episode_metrics
            ]),
            fmt="%.6f",
        )
        self._save_training_logs(output)

    def evaluate_model(
            self,
            eval_episodes: int,
            results_dir: str,
            model_dir: str,
            evaluation_protocol: EvaluationProtocol | None = None) -> np.ndarray:
        protocol = evaluation_protocol or EvaluationProtocol(
            instance_count=eval_episodes)
        model_path = Path(model_dir)
        inference = model_path / "hierarchical_inference.pt"
        checkpoint = model_path / "hierarchical_checkpoint.pt"
        if inference.is_file():
            self.load_inference(inference)
        elif checkpoint.is_file():
            self.load_checkpoint(checkpoint, load_training=False)
        else:
            raise FileNotFoundError(
                f"No hierarchical MAPPO artifact found in {model_path}.")
        records: list[EvaluationRecord] = []
        audit_tables: dict[str, list[dict]] = {}
        for case in protocol.cases():
            self.reset(
                instance_seed=case.instance_seed,
                simulation_seed=case.simulation_seed,
            )
            metrics = self.run_episode(training=False)
            records.append(EvaluationRecord.from_case(
                case, metrics))
            merge_audit_tables(audit_tables, self.audit_tables(case))
        suffix = (
            f"Load_{self.target_load:g}_N0_{self.initial_jobs}_"
            f"N_{self.total_jobs}_DDT_{self.due_date_factor}_"
            f"Eta_{self.config.degradation_multiplier:g}"
        )
        return save_evaluation_outputs(
            records,
            protocol,
            results_dir,
            f"Total_Tardiness_{suffix}.txt",
            audit_tables=audit_tables,
        )
