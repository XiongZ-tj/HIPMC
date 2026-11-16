"""Industrial arrival schedules, validation and reference-policy evaluation."""

from __future__ import annotations
from pathlib import Path
from types import MethodType
import numpy as np
from comparison_runners.drl_common import simulation_core
from comparison_runners.industrial_case.case_common import INITIAL_JOBS, PROCESSING_TIMES, TOTAL_JOBS, adapt_simulator, audit_industrial_episode_input, atomic_csv, atomic_json, generate_records, reset_industrial_simulator
from experiment.seeds import TrainingSeeds
from methods.baselines.sdr.shared import RuleController

VALIDATION_INSTANCE_SEED_BASE = 81_000_000
VALIDATION_SIMULATION_SEED_BASE = 82_000_000


def schedules(training_seed: int, episodes: int, validation_instances: int):
    training = []
    load_by_seed = {}
    for episode in range(episodes):
        load_index = episode % 3
        instance_seed = 71_000_000 + training_seed * 10_000 + episode
        load_by_seed[instance_seed] = (0.9, 1.0, 1.1)[load_index]
        training.append(TrainingSeeds(
            episode=episode,
            instance_seed=instance_seed,
            simulation_seed=72_000_000 + training_seed * 10_000 + episode,
            policy_seed=73_000_000 + training_seed * 10_000 + episode,
        ))
    validation = []
    for index in range(validation_instances):
        instance_seed = VALIDATION_INSTANCE_SEED_BASE + index
        load_by_seed[instance_seed] = (0.9, 1.0, 1.1)[index % 3]
        validation.append({
            "instance_seed": instance_seed,
            "simulation_seed": VALIDATION_SIMULATION_SEED_BASE + index,
            "policy_seed": VALIDATION_SIMULATION_SEED_BASE + index,
        })
    return training, validation, load_by_seed


def install_record_schedule(environment, load_by_seed: dict[int, float]):
    core = simulation_core(environment)
    records = {
        seed: generate_records(target_load=load, arrival_seed=seed)
        for seed, load in load_by_seed.items()
    }
    original_reset = core.reset

    def scheduled_reset(self, *, instance_seed=None, simulation_seed=None,
                        policy_seed=None, job_records=None):
        if instance_seed in records:
            load = load_by_seed[int(instance_seed)]
            self.target_load = load
            self.job_parameters = [load, INITIAL_JOBS, TOTAL_JOBS, 1.0, int(instance_seed)]
            if job_records is None:
                job_records = records[int(instance_seed)]
        result = original_reset(
            instance_seed=instance_seed,
            simulation_seed=simulation_seed,
            policy_seed=policy_seed,
            job_records=job_records,
        )
        if instance_seed in records:
            audit_industrial_episode_input(
                self,
                records[int(instance_seed)],
                target_load=load_by_seed[int(instance_seed)],
            )
        return result

    core.reset = MethodType(scheduled_reset, core)
    return records


def install_industrial_validation(
        environment,
        validation_schedule: list[dict],
        records,
        load_by_seed: dict[int, float]) -> None:
    """Bind the proposed method's detached evaluator to industrial cases."""

    validation_seeds = [
        int(item["instance_seed"]) for item in validation_schedule]
    if len(validation_seeds) != len(set(validation_seeds)):
        raise ValueError("Industrial validation instance seeds must be unique.")
    missing = [
        seed for seed in validation_seeds
        if seed not in records or seed not in load_by_seed
    ]
    if missing:
        raise ValueError(
            f"Industrial validation cases were not materialized: {missing}")
    expected_loads = np.asarray(
        [load_by_seed[seed] for seed in validation_seeds], dtype=float)

    def setup_evaluator(evaluator) -> None:
        adapt_simulator(evaluator)

    def reset_case(evaluator, item) -> None:
        instance_seed = int(item["instance_seed"])
        reset_industrial_simulator(
            evaluator,
            target_load=load_by_seed[instance_seed],
            instance_seed=instance_seed,
            simulation_seed=int(item["simulation_seed"]),
            policy_seed=int(item["policy_seed"]),
            records=records[instance_seed],
        )

    def audit_summary(record) -> None:
        expected = {
            "validation_cases": len(validation_seeds),
            "target_load_mean": float(expected_loads.mean()),
            "target_load_std": float(expected_loads.std()),
            "finite_instance_load_mean": float(expected_loads.mean()),
            "finite_instance_load_std": float(expected_loads.std()),
            "job_count_mean": float(TOTAL_JOBS),
            "job_count_std": 0.0,
            "initial_job_count_mean": float(INITIAL_JOBS),
            "initial_job_count_std": 0.0,
            "dynamic_job_count_mean": float(TOTAL_JOBS - INITIAL_JOBS),
            "dynamic_job_count_std": 0.0,
        }
        for key, value in expected.items():
            observed = float(record[key])

    environment.validation_evaluator_setup = setup_evaluator
    environment.validation_case_reset = reset_case
    environment.validation_summary_auditor = audit_summary


def install_industrial_pretraining_reference(environment, validation_schedule, records, output: Path):
    def run_reference(self, _output):
        controller = RuleController("CH", "MMU", "Threshold", max(PROCESSING_TIMES))
        from comparison_runners.industrial_case.case_common import build_static_simulator
        simulator = build_static_simulator(controller, target_load=1.0,
                                            arrival_seed=validation_schedule[0]["instance_seed"])
        cases = []
        values = []
        for index, item in enumerate(validation_schedule):
            seed = int(item["instance_seed"])
            reset_industrial_simulator(
                simulator,
                target_load=(0.9, 1.0, 1.1)[index % 3],
                instance_seed=seed,
                simulation_seed=int(item["simulation_seed"]),
                policy_seed=int(item["policy_seed"]),
                records=records[seed],
            )
            metrics = simulator.run_episode(training=False)
            values.append(float(metrics["mean_tardiness"]))
            cases.append({
                "validation_case": index,
                "instance_seed": seed,
                "simulation_seed": int(item["simulation_seed"]),
                "policy_seed": int(item["policy_seed"]),
                "policy": "CH+MMU+Threshold",
                **metrics,
            })
        summary = {
            "schema_version": 1,
            "policy": "CH+MMU+Threshold",
            "validation_cases": len(cases),
            "mean_tardiness": float(np.mean(values)),
            "mean_tardiness_std": float(np.std(values)),
            "pairing": "same frozen validation arrivals and simulation seeds as every learned method",
        }
        self.pretraining_baseline = summary
        self.pretraining_baseline_cases = cases
        atomic_csv(output / "pretraining_sdr_reference_cases.csv", cases)
        atomic_json(output / "pretraining_sdr_reference_summary.json", summary)
        return summary

    environment._run_pretraining_baseline = MethodType(run_reference, environment)
