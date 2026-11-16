"""Dispatching-rule baseline on the shared event-driven simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment.evaluation import (
    EvaluationProtocol,
    EvaluationRecord,
    merge_audit_tables,
    save_evaluation_outputs,
)
from experiment.shared_simulator import build_shared_simulator
from .rules import (
    BATCH_FORMATION_RULES,
    BATCH_SELECTION_RULES,
    MAINTENANCE_RULES,
    build_rule_proposal,
)


job_cluster_pt = [190, 158, 202, 128, 104]


def configure_scenario(processing_times) -> None:
    global job_cluster_pt
    job_cluster_pt = list(processing_times)


class RuleController:
    def __init__(
            self,
            batch_rule: str,
            allocation_rule: str,
            maintenance_policy: str,
            maintenance_period: float) -> None:
        if batch_rule not in BATCH_FORMATION_RULES:
            raise ValueError(f"Unknown batch-formation rule: {batch_rule}")
        if allocation_rule not in BATCH_SELECTION_RULES:
            raise ValueError(f"Unknown batch-selection rule: {allocation_rule}")
        if maintenance_policy not in MAINTENANCE_RULES:
            raise ValueError(f"Unknown maintenance rule: {maintenance_policy}")
        self.batch_rule = batch_rule
        self.allocation_rule = allocation_rule
        self.maintenance_policy = maintenance_policy
        self.maintenance_period = maintenance_period
        self.last_maintenance = {}

    def begin_episode(self, policy_seed: int) -> None:
        """Reset all episode-local rule state.

        ``policy_seed`` is accepted to share the controller lifecycle contract
        used by learned policies; deterministic rules do not consume it.
        """

        del policy_seed
        self.last_maintenance.clear()

    def build_proposals(
            self, jobs, capacity, clock, processing_times, **_):
        return build_rule_proposal(
            jobs,
            capacity,
            clock,
            processing_times,
            self.batch_rule,
            self.allocation_rule,
        )

    def select_action(
            self,
            observation,
            *,
            requester_class,
            machine_state,
            clock,
            **_) -> int:
        del requester_class
        proposal = 0
        maintenance = 0
        machine_id = observation.requester_index
        if machine_state > 0:
            if self.maintenance_policy == "Threshold":
                if machine_state >= 3:
                    maintenance = 2
                elif machine_state >= 2:
                    maintenance = 1
            elif (
                    self.maintenance_policy == "Period"
                    and clock - self.last_maintenance.get(machine_id, 0.0)
                    >= self.maintenance_period):
                maintenance = 1
            if maintenance:
                self.last_maintenance[machine_id] = clock
        action = proposal * 3 + maintenance
        if not observation.action_mask[action]:
            process_action = proposal * 3
            if not observation.action_mask[process_action]:
                raise RuntimeError("SDR selected a proposal with no feasible action.")
            action = process_action
        return action

    def observe(self, transition) -> None:
        del transition


class env:
    def __init__(
            self,
            job_parameters,
            machine_parameters,
            sdrs=("CH", "MMU"),
            maint_policy="Threshold",
            degradation_multiplier=1.0):
        self.job_parameters = list(job_parameters)
        self.machine_parameters = list(machine_parameters)
        self.sdrs = tuple(sdrs)
        self.maint_policy = maint_policy
        self.controller = RuleController(
            self.sdrs[0],
            self.sdrs[1],
            maint_policy,
            max(job_cluster_pt),
        )
        self.core = build_shared_simulator(
            job_parameters=self.job_parameters,
            machine_parameters=self.machine_parameters,
            processing_times=tuple(job_cluster_pt),
            controller=self.controller,
            degradation_multiplier=degradation_multiplier,
        )

    def evaluate_model(
            self,
            eval_episodes=10,
            results_dir="sdr_results",
            evaluation_protocol=None):
        protocol = evaluation_protocol or EvaluationProtocol(
            instance_count=eval_episodes)
        records = []
        audit_tables = {}
        for case in protocol.cases():
            self.core.reset(
                instance_seed=case.instance_seed,
                simulation_seed=case.simulation_seed,
                policy_seed=case.simulation_seed,
            )
            metrics = self.core.run_episode(training=False)
            records.append(EvaluationRecord.from_case(
                case, metrics))
            merge_audit_tables(
                audit_tables, self.core.audit_tables(case))
        target_load, initial_jobs, total_jobs, due, _ = self.job_parameters
        filename = (
            f"Total_Tardiness_Load_{target_load:g}_N0_{initial_jobs}_"
            f"N_{total_jobs}_DDT_{due}_"
            f"Eta_{self.core.config.degradation_multiplier:g}.txt"
        )
        values = save_evaluation_outputs(
            records, protocol, results_dir, filename,
            audit_tables=audit_tables)
        return float(np.mean(values))
