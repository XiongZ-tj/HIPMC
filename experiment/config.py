"""Central configuration for all scheduling experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator
from experiment.revision_protocol import (
    TRAINING_SEEDS, FROZEN_TEST_INSTANCES_PER_CONDITION, SIMULATION_REPLICATIONS,
)


BASE_JOB_PROCESSING_TIMES = (190, 158, 202, 128, 104)


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    job_processing_times: tuple[int, ...]
    machine_counts: tuple[int, ...]
    description: str


@dataclass(frozen=True)
class MethodConfig:
    name: str
    module_name: str
    default_model_dir: Path | None
    model_artifacts: tuple[str, ...]
    supports_training: bool
    uses_accelerator: bool
    description: str


@dataclass(frozen=True)
class ConditionConfig:
    # Nominal finite-horizon load including initial work-in-process.
    target_load: float = 0.95
    initial_jobs: int = 30
    total_jobs: int = 150
    due_date_factor: float = 3.0
    degradation_multiplier: float = 1.0
    seed: int = 42

    def as_job_parameters(self) -> list[float | int]:
        return [
            self.target_load,
            self.initial_jobs,
            self.total_jobs,
            self.due_date_factor,
            self.seed,
        ]


SCENARIOS = {
    "base": ScenarioConfig(
        name="base",
        job_processing_times=BASE_JOB_PROCESSING_TIMES,
        machine_counts=(2, 2, 1),
        description="Reference five-family, five-machine operating scale.",
    ),
}


METHODS = {
    "ours": MethodConfig("ours", "methods.hierarchical", None,
        ("hierarchical_inference.pt",), True, True, "HIPMC batch scheduling and maintenance."),
    "ddqnb": MethodConfig("ddqnb", "methods.baselines.ddqnb.shared", None,
        ("ddqnb_model.pth",), True, True, "Double-DQN rule selection and maintenance."),
    "psa_mappo": MethodConfig("psa_mappo", "methods.baselines.psa_mappo.shared", None,
        ("psa_mappo_model.pth",), True, True, "Machine-class actors with a centralized critic."),
    "hdrl_option": MethodConfig("hdrl_option", "methods.baselines.hdrl_option.shared", None,
        ("hdrl_option_model.pth",), True, True, "Option-based hierarchical control."),
    "sdr": MethodConfig("sdr", "methods.baselines.sdr.shared", None,
        (), False, False, "Dispatching rules with preventive maintenance."),
}


NOMINAL_FINITE_LOADS = (0.90, 1.00, 1.10)
TOTAL_JOB_COUNTS = (100, 150, 200)
DUE_DATE_FACTORS = (2.0, 2.5, 3.0)
DEGRADATION_MULTIPLIERS = (0.75, 1.0, 1.25)
E1_INITIAL_JOB_FRACTION = 0.20
E1_DEGRADATION_MULTIPLIER = 1.0
E1_INSTANCE_COUNT = FROZEN_TEST_INSTANCES_PER_CONDITION
E1_SIMULATION_REPLICATIONS = SIMULATION_REPLICATIONS
E1_INSTANCE_SEED_START = 20_000_000
E1_SIMULATION_SEED_START = 30_000_000
# Separate from instance/simulation common random numbers.  Every stochastic
# formal comparison method must use this same replication list for its
# training or optimizer randomness; deterministic SDR methods do not consume it.
E1_STOCHASTIC_METHOD_SEEDS = TRAINING_SEEDS


def iter_full_factorial(
        *,
        seed: int = 42,
) -> Iterator[ConditionConfig]:
    """Yield the frozen E1 production-task grid.

    E1 varies load, due-date tightness, and total job count.  Machine scale and
    degradation physics stay fixed; the initial queue remains 20% of the
    total job count.
    """

    for target_load in NOMINAL_FINITE_LOADS:
        for due_date_factor in DUE_DATE_FACTORS:
            for total_jobs in TOTAL_JOB_COUNTS:
                yield ConditionConfig(
                    target_load=target_load,
                    initial_jobs=int(round(
                        E1_INITIAL_JOB_FRACTION * total_jobs)),
                    total_jobs=total_jobs,
                    due_date_factor=due_date_factor,
                    degradation_multiplier=E1_DEGRADATION_MULTIPLIER,
                    seed=seed,
                )


def serialize_config(value: object) -> dict:
    data = asdict(value)
    for key, item in tuple(data.items()):
        if isinstance(item, Path):
            data[key] = str(item)
    return data
