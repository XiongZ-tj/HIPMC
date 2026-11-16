"""Immutable configuration for the hierarchical scheduling simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiment.problem import (
    CORRECTIVE_TIME_RANGES,
    DEGRADATION_RATE_RANGES,
    DEGRADATION_STATES,
    FAILURE_TYPE_PROPORTIONS,
    FAILURE_TYPES,
    IPM_RECOVERY_RATES,
    IPM_TIMES,
    JOB_SIZE_INTERVAL,
    MACHINE_CAPACITIES,
    MINIMAL_REPAIR_TIMES,
    PM_TIMES,
    PRODUCTION_RATE_RANGE,
    TERMINAL_DEGRADATION_RATES,
    UNEXPECTED_FAILURE_RATE_RANGES,
)
from experiment.reward_reference import (
    REWARD_REFERENCE_ALLOCATION_RULE,
    REWARD_REFERENCE_BATCH_RULE,
    REWARD_REFERENCE_MAINTENANCE_RULE,
)


OFFLINE_CALIBRATED_SDR_NORMALIZATION = "offline-calibrated-sdr"


@dataclass(frozen=True)
class Scenario:
    job_processing_times: tuple[int, ...] = (190, 158, 202, 128, 104)
    machine_counts: tuple[int, ...] = (2, 2, 1)
    machine_capacities: tuple[int, ...] = MACHINE_CAPACITIES
    job_size_interval: tuple[int, int] = JOB_SIZE_INTERVAL

    def __post_init__(self) -> None:
        if len(self.machine_counts) != len(self.machine_capacities):
            raise ValueError("Every machine class needs a capacity.")
        if any(value <= 0 for value in self.job_processing_times):
            raise ValueError("Processing times must be positive.")
        if any(value < 0 for value in self.machine_counts):
            raise ValueError("Machine counts cannot be negative.")


@dataclass(frozen=True)
class HierarchyConfig:
    mma_reward_protocol: str = "t3_global_shared"
    degradation_multiplier: float = 1.0
    degradation_states: int = DEGRADATION_STATES
    failure_types: int = FAILURE_TYPES
    failure_type_proportions: tuple[float, ...] = FAILURE_TYPE_PROPORTIONS
    degradation_rate_ranges: tuple[tuple[float, float], ...] = DEGRADATION_RATE_RANGES
    failure_rate_ranges: tuple[tuple[float, float], ...] = UNEXPECTED_FAILURE_RATE_RANGES
    terminal_degradation_rates: tuple[float, ...] = TERMINAL_DEGRADATION_RATES
    ipm_recovery_rates: tuple[tuple[float, ...], ...] = IPM_RECOVERY_RATES
    ipm_times: tuple[int, ...] = IPM_TIMES
    pm_times: tuple[int, ...] = PM_TIMES
    corrective_time_ranges: tuple[tuple[int, int], ...] = CORRECTIVE_TIME_RANGES
    minimal_repair_times: tuple[int, ...] = MINIMAL_REPAIR_TIMES
    production_rate_range: tuple[float, float] = PRODUCTION_RATE_RANGE
    gamma_baa: float = 0.99
    gamma_mma: float = 0.99
    discount_time_scale: float = 100.0  
    gae_lambda: float = 0.95
    ppo_clip_ratio: float = 0.2
    ppo_update_epochs: int = 5
    rollout_episodes: int = 10
    minibatch_size: int = 64
    hidden_dim: int = 64
    learning_rate: float = 2e-4
    value_loss_weight: float = 0.5
    baa_entropy_weight: float = 0.005
    mma_entropy_weight: float = 0.01
    mma_entropy_final_weight: float = 0.001
    mma_entropy_hold_fraction: float = 0.20
    mma_entropy_decay_fraction: float = 0.50
    gradient_clip: float = 1.0
    dirichlet_min_concentration: float = 0.5
    mma_actor_choice_only: bool = True
    baa_actor_choice_only: bool = False
    head_balanced_actor_critic: bool = False
    sample_weighted_actor_critic: bool = True
    reward_reference_batch_rule: str = REWARD_REFERENCE_BATCH_RULE
    reward_reference_allocation_rule: str = REWARD_REFERENCE_ALLOCATION_RULE
    reward_reference_maintenance_policy: str = (
        REWARD_REFERENCE_MAINTENANCE_RULE)
    reward_normalization_mode: str = OFFLINE_CALIBRATED_SDR_NORMALIZATION
    reward_normalizer_floor: float = 1.0
    calibrated_reward_scale: float | None = None
    reward_scale_artifact_sha256: str | None = None
    max_decisions_per_episode: int = 200_000

    def __post_init__(self) -> None:
        if self.degradation_multiplier <= 0:
            raise ValueError("degradation_multiplier must be positive.")
        if self.degradation_states < 2 or self.failure_types != 3:
            raise ValueError("The method requires >=2 degradation states and 3 failure types.")
        if (
                len(self.failure_type_proportions) != self.failure_types
                or any(value < 0 for value in self.failure_type_proportions)
                or abs(sum(self.failure_type_proportions) - 1.0) > 1e-9):
            raise ValueError(
                "Failure-type proportions must be nonnegative and sum to one.")
        if self.hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by four attention heads.")
        if not 0 < self.gamma_baa <= 1 or not 0 < self.gamma_mma <= 1:
            raise ValueError("Discount factors must be in (0, 1].")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be in [0, 1].")
        if not 0 < self.ppo_clip_ratio < 1:
            raise ValueError("ppo_clip_ratio must be in (0, 1).")
        if min(
                self.ppo_update_epochs,
                self.rollout_episodes,
                self.minibatch_size) <= 0:
            raise ValueError("MAPPO update counts must be positive.")
        if min(
                self.learning_rate,
                self.value_loss_weight,
                self.gradient_clip,
                self.dirichlet_min_concentration) <= 0:
            raise ValueError("MAPPO scales must be positive.")
        if min(
                self.baa_entropy_weight,
                self.mma_entropy_weight,
                self.mma_entropy_final_weight) < 0:
            raise ValueError("Entropy weights cannot be negative.")
        if self.mma_entropy_final_weight > self.mma_entropy_weight:
            raise ValueError(
                "The final MMA entropy weight cannot exceed its initial value.")
        if (
                not 0 <= self.mma_entropy_hold_fraction <= 1
                or not 0 <= self.mma_entropy_decay_fraction <= 1
                or self.mma_entropy_hold_fraction
                + self.mma_entropy_decay_fraction > 1):
            raise ValueError("MMA entropy schedule fractions are invalid.")
        class_count = len(self.degradation_rate_ranges)
        class_parameters = (
            self.failure_rate_ranges,
            self.terminal_degradation_rates,
            self.ipm_recovery_rates,
            self.ipm_times,
            self.pm_times,
            self.corrective_time_ranges,
            self.minimal_repair_times,
        )
        if any(len(values) != class_count for values in class_parameters):
            raise ValueError("Machine-class parameter tables must align.")
        if any(
                len(row) != self.degradation_states - 1
                or any(rate <= 0 for rate in row)
                for row in self.ipm_recovery_rates):
            raise ValueError(
                "Every machine class requires one positive IPM restoration "
                "rate for each degraded health state.")
        if any(
                ipm_time <= 0 or pm_time <= ipm_time
                for ipm_time, pm_time in zip(
                    self.ipm_times, self.pm_times, strict=True)):
            raise ValueError(
                "Maintenance times must be positive and PM must take longer "
                "than IPM for every machine class.")
        if self.reward_normalizer_floor <= 0:
            raise ValueError("reward_normalizer_floor must be positive.")
        if (
                self.reward_normalization_mode
                != OFFLINE_CALIBRATED_SDR_NORMALIZATION):
            raise ValueError(
                "The retired per-episode normalization mode is unsupported; "
                "use offline-calibrated-sdr.")
        if (
                (self.calibrated_reward_scale is None)
                != (self.reward_scale_artifact_sha256 is None)):
            raise ValueError(
                "An offline-calibrated SDR scale and its artifact SHA-256 "
                "must be configured together.")
        if (
                self.calibrated_reward_scale is not None
                and self.calibrated_reward_scale <= 0):
            raise ValueError(
                "The offline-calibrated SDR scale must be positive.")

    def to_dict(self) -> dict:
        return asdict(self)

    def mma_entropy_weight_at(
            self, episode: int, total_episodes: int) -> float:
        """Return the deterministic one-based piecewise entropy schedule."""

        if total_episodes <= 0:
            raise ValueError("total_episodes must be positive.")
        if episode < 1:
            raise ValueError("episode must be positive.")
        # Continuing a completed budget holds the final coefficient; it must
        # not restart or stretch the original annealing schedule.
        episode = min(episode, total_episodes)
        hold_end = int(total_episodes * self.mma_entropy_hold_fraction)
        decay_end = int(total_episodes * (
            self.mma_entropy_hold_fraction
            + self.mma_entropy_decay_fraction))
        if episode <= hold_end:
            return self.mma_entropy_weight
        if episode >= decay_end or decay_end <= hold_end:
            return self.mma_entropy_final_weight
        fraction = (episode - hold_end) / (decay_end - hold_end)
        return (
            self.mma_entropy_weight
            + fraction * (
                self.mma_entropy_final_weight - self.mma_entropy_weight)
        )
