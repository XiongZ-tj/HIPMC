"Preventive maintenance and event-driven waiting during empty-queue periods."

from __future__ import annotations

from collections import Counter
import heapq
import json
from pathlib import Path

import numpy as np
import torch

from experiment.artifacts import atomic_csv_save, atomic_json_save, atomic_torch_save
from experiment.metrics import EpisodeMetrics
from experiment.training_logs import decision_action_summary, validation_summary
from methods.hierarchical.environment import HierarchicalSchedulingEnv, PendingDecision
from methods.hierarchical.features import MMA_MAINTENANCE_FEATURES, positive_saturation
from methods.hierarchical.types import (
    BAAObservation, BatchProposal, DecisionContext, HandshakeMode, MachineAction,
    MachinePhase, MMAObservation,
)
from .agents import IdleMaintenanceMAPPO, WAIT


PROTOCOLS = ("idle_opportunity",)
ACTION_MAP = {"0": "process", "1": "IPM", "2": "PM", "3": "wait"}


class IdleMaintenanceEnv(HierarchicalSchedulingEnv):
    maintenance_dim = 6

    def __init__(self, *args, maintenance_protocol="idle_opportunity", **kwargs):
        if kwargs.get("controller") is not None:
            raise ValueError("This isolated variant accepts only its own HIPMC policies.")
        self.maintenance_protocol = maintenance_protocol
        super().__init__(*args, **kwargs)
        if self.handshake_mode is not HandshakeMode.INTERACTIVE:
            raise ValueError("The idle-maintenance comparison retains interactive coordination.")
        self.mappo = IdleMaintenanceMAPPO(
            machine_class_count=len(self.scenario.machine_counts),
            health_state_count=self.config.degradation_states,
            proposal_dim=self.proposal_dim, machine_dim=self.policy_machine_dim,
            context_dim=self.context_dim, maintenance_dim=self.maintenance_dim,
            hidden_dim=self.config.hidden_dim, learning_rate=self.config.learning_rate,
            gamma_baa=self.config.gamma_baa, gamma_mma=self.config.gamma_mma,
            gae_lambda=self.config.gae_lambda, clip_ratio=self.config.ppo_clip_ratio,
            value_loss_weight=self.config.value_loss_weight,
            baa_entropy_weight=self.config.baa_entropy_weight,
            mma_entropy_weight=self.config.mma_entropy_weight,
            update_epochs=self.config.ppo_update_epochs,
            minibatch_size=self.config.minibatch_size,
            time_scale=self.config.discount_time_scale, gradient_clip=self.config.gradient_clip,
            dirichlet_min_concentration=self.config.dirichlet_min_concentration,
            device=self.device,
        )

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._last_empty_decision = {}
        self.idle_interventions = []
        self.wait_intervals = []

    def variant_contract(self):
        return {"version": 1, "maintenance_protocol": self.maintenance_protocol,
                "action_map": ACTION_MAP, "maintenance_features": [
                    *MMA_MAINTENANCE_FEATURES, "has_tentative_batch"],
                "waiting": "until_next_observed_event",
                "decision_boundary": "all_simultaneous_physical_events_processed",
                "empty_queue_scope": "global_released_unallocated_queue",
                "context_dim": self.context_dim}

    def _maintenance_features(self, machine, proposal):
        state = machine.state
        recovery = (1-np.exp(-self.config.ipm_recovery_rates[machine.class_id][state-1]
                            *self.config.ipm_times[machine.class_id])) if state else 0.
        return np.asarray([
            state/(self.config.degradation_states-1),
            1-self._policy_completion_probability(machine,
                nominal_work=self._policy_proposal_work(machine, proposal)),
            positive_saturation(self.config.ipm_times[machine.class_id], max(self.job_cluster_pt)),
            positive_saturation(self.config.pm_times[machine.class_id], max(self.job_cluster_pt)),
            recovery, 1.,
        ], dtype=np.float32)

    def _mma_observation(self, machine, proposal):
        observation = super()._mma_observation(machine, proposal)
        observation.action_mask = np.append(observation.action_mask, False)
        return observation

    def _empty_observation(self, machine):
        if machine.batch_ids or self._available_jobs() or machine.phase is not MachinePhase.IDLE:
            raise ValueError("An empty-queue decision needs an idle, unassigned machine and empty queue.")
        state = machine.state
        recovery = (1 - np.exp(-self.config.ipm_recovery_rates[machine.class_id][state-1]
                              * self.config.ipm_times[machine.class_id])) if state else 0.
        risk = 1 - self._policy_completion_probability(machine)
        maintenance = np.asarray([
            state / (self.config.degradation_states-1), risk,
            positive_saturation(self.config.ipm_times[machine.class_id], max(self.job_cluster_pt)),
            positive_saturation(self.config.pm_times[machine.class_id], max(self.job_cluster_pt)),
            recovery, 0.,
        ], dtype=np.float32)
        return MMAObservation(
            batch_features=np.zeros(self.proposal_dim, dtype=np.float32),
            machine_features=self._policy_machine_feature(machine, requester_id=machine.machine_id),
            global_machine_features=np.stack([
                self._policy_machine_feature(m, requester_id=machine.machine_id) for m in self.machines]),
            requester_index=machine.machine_id,
            action_mask=np.asarray([False, state > 0, state > 0, True]),
            context_features=self._global_context_features(), maintenance_features=maintenance,
            health_state=state,
        )

    def _controller_has_pending_action(self, machine):
        # No test of whether more orders WILL arrive: the actor must not know.
        return (not machine.batch_ids and not self._available_jobs()
                and self._last_empty_decision.get(machine.machine_id) != self.clock)

    def _request(self, machine, *, epsilon, training):
        if machine.batch_ids or self._available_jobs():
            return super()._request(machine, epsilon=epsilon, training=training)
        if not self._controller_has_pending_action(machine):
            return False
        observation = self._empty_observation(machine)
        self._close_previous_mma(machine, observation, terminated=False)
        selected = self.mappo.select_mma(
            machine.class_id, observation, training=training, rng=self._policy_rng)
        action = int(selected.executed_action)
        if action < 0 or action >= 4 or not observation.action_mask[action]:
            raise RuntimeError("Invalid empty-queue maintenance decision.")
        self._last_empty_decision[machine.machine_id] = self.clock
        self._request_sequence += 1
        # A nonexecuted placeholder, never submitted to the BAA actor/critic.
        no_baa = BAAObservation(
            proposal_features=np.zeros((1, self.proposal_dim), dtype=np.float32),
            proposal_mask=np.zeros(1, dtype=bool),
            machine_features=observation.global_machine_features,
            machine_mask=np.ones(len(self.machines), dtype=bool),
            requester_index=machine.machine_id, context_features=observation.context_features,
        )
        context = DecisionContext(
            request_id=self._request_sequence, proposal_id=-1,
            machine_id=machine.machine_id, started_at=self.clock,
            baa_observation=no_baa, baa_action=-1, mma_observation=observation,
            mma_action=action, mma_log_probability=selected.log_probability,
            mma_value=selected.value, mma_entropy=selected.entropy,
            mma_probabilities=selected.probabilities, baa_decision=False,
        )
        placeholder = BatchProposal(-1, (), -1, np.zeros(5), np.zeros(13))
        if action == WAIT:
            machine.context = context
            self._pending_mma[machine.machine_id] = PendingDecision(context, self._potential())
            machine.history.append({"time": self.clock, "event": "idle_wait"})
        else:
            super()._start_action(machine, placeholder, context)
            duration = self.config.ipm_times[machine.class_id] if action == 1 else self.config.pm_times[machine.class_id]
            self.idle_interventions.append({"machine_id": machine.machine_id,
                "machine_class": machine.class_id, "action": action,
                "started_at": self.clock, "planned_end": self.clock + duration,
                "planned_duration": duration})
        return True

    def _close_previous_mma(self, machine, next_state, *, terminated):
        pending = self._pending_mma.get(machine.machine_id)
        if pending is not None and pending.context.mma_action == WAIT:
            self.wait_intervals.append({"machine_id": machine.machine_id,
                "started_at": pending.context.started_at, "ended_at": self.clock,
                "elapsed_time": self.clock-pending.context.started_at, "terminated": terminated})
        return super()._close_previous_mma(machine, next_state, terminated=terminated)

    def _advance_with_repairs(self):
        super()._advance_with_repairs()
        # Settle ties before any next policy call. Neither protocol can act
        # between two simultaneous completion/failure/maintenance events.
        while self._event_queue:
            at, _, machine_id, event, generation = self._event_queue[0]
            machine = self.machines[machine_id]
            if generation != machine.generation:
                heapq.heappop(self._event_queue)
                continue
            if at > self.clock:
                break
            heapq.heappop(self._event_queue)
            if event == "repair_complete":
                self._handle_repair_completion(machine)
            else:
                self._handle_machine_event(machine, event, at)

    def _record_controller_decision(self, machine, decisions, latency):
        super()._record_controller_decision(machine, decisions, latency)
        row = self.decision_history[-1]
        row["empty_queue_decision"] = bool(machine.context.mma_observation.maintenance_features[-1] == 0)
        row["mma_probability_wait"] = float(machine.context.mma_probabilities[WAIT])

    def _realized_maintenance(self, machine):
        planned = float(machine.planned_maintenance_time)
        corrective = float(machine.corrective_maintenance_time)
        if machine.phase is MachinePhase.MAINTAINING:
            duration = (self.config.ipm_times[machine.class_id] if machine.maintenance_action == 1
                        else self.config.pm_times[machine.class_id])
            planned -= max(0., duration - (self.clock-machine.activity_started_at))
        elif machine.phase is MachinePhase.REPAIRING:
            duration = (self.config.minimal_repair_times[machine.class_id]
                        if machine.failure_type == -1 else np.linspace(
                            *self.config.corrective_time_ranges[machine.class_id],
                            self.config.failure_types)[machine.failure_type])
            corrective -= max(0., duration - (self.clock-machine.last_event_at))
        return max(0., planned), max(0., corrective)

    def run_episode(self, *, training, epsilon=0.):
        metrics = super().run_episode(training=training, epsilon=epsilon)
        totals = [self._realized_maintenance(m) for m in self.machines]
        planned, corrective = map(float, np.sum(totals, axis=0))
        metrics.update(maintenance_time_booked=metrics["maintenance_time"],
            planned_maintenance_time=planned, corrective_maintenance_time=corrective,
            maintenance_time=planned+corrective,
            maintenance_downtime_ratio=(planned+corrective)/max(
                1e-12, sum(m.processing_time for m in self.machines)+planned+corrective),
            actual_maintenance_load=sum((p+c)*m.capacity for m,(p,c) in zip(self.machines, totals))
                /max(1e-12, self._instance_load.arrival_span*self._instance_load.plant_capacity))
        for row in self.idle_interventions:
            row["ended_at"] = min(row["planned_end"], self.clock)
            row["actual_duration"] = row["ended_at"]-row["started_at"]
            row["new_fitting_arrival_during_maintenance"] = any(
                row["started_at"] < j.arrival_time < row["ended_at"]
                and j.size <= self.machines[row["machine_id"]].capacity for j in self.jobs)
        empty = [r for r in self.decision_history if r["empty_queue_decision"]]
        counts = Counter(r["maintenance_action"] for r in empty)
        metrics.update(idle_decisions=len(empty), idle_wait_count=counts[WAIT],
            idle_ipm_count=counts[1], idle_pm_count=counts[2],
            idle_maintenance_actual_time=sum(r["actual_duration"] for r in self.idle_interventions),
            idle_maintenance_arrival_overlap_count=sum(r["new_fitting_arrival_during_maintenance"] for r in self.idle_interventions),
            idle_wait_elapsed_time=sum(r["elapsed_time"] for r in self.wait_intervals),
            mma_wait_probability_mean=float(np.mean([r["mma_probability_wait"] for r in empty])) if empty else 0.)
        return metrics

    def _machine_responsibility_metrics(self):
        # Keep event-booked counters intact; only reported realized metrics
        # exclude the unexecuted tail after the last job completes.
        original = [(m.maintenance_time, m.planned_maintenance_time, m.corrective_maintenance_time)
                    for m in self.machines]
        try:
            for m in self.machines:
                p, c = self._realized_maintenance(m)
                m.maintenance_time, m.planned_maintenance_time, m.corrective_maintenance_time = p+c, p, c
            return super()._machine_responsibility_metrics()
        finally:
            for m, values in zip(self.machines, original):
                m.maintenance_time, m.planned_maintenance_time, m.corrective_maintenance_time = values

    def train_episode(self, **kwargs):
        metrics = super().train_episode(**kwargs)
        metrics.update(decision_action_summary(self.decision_history,
            machine_class_count=len(self.scenario.machine_counts),
            health_state_count=self.config.degradation_states, action_count=4))
        return metrics

    def _validation_score(self):
        evaluator = type(self)(self.job_parameters, list(self.scenario.machine_counts),
            processing_times=self.job_cluster_pt, config=self.config, device=self.device,
            maintenance_protocol=self.maintenance_protocol)
        if self.validation_evaluator_setup is not None:
            self.validation_evaluator_setup(evaluator)
        evaluator.mappo.load_state_dict(self.mappo.inference_state(), training=False)
        schedule = self.validation_seed_schedule or [dict(instance_seed=3000000+i,
            simulation_seed=4000000+i, policy_seed=4000000+i) for i in range(self.validation_instances)]
        if len(schedule) != self.validation_instances:
            raise ValueError("Validation schedule size differs from configured budget.")
        metrics, idle_metrics = [], []
        for item in schedule:
            if self.validation_case_reset is None:
                evaluator.reset(**item)
            else:
                self.validation_case_reset(evaluator, item)
            result = evaluator.run_episode(training=False)
            metrics.append(EpisodeMetrics.from_mapping(result))
            idle_metrics.append(result)
        record = validation_summary(self.total_episode, metrics)
        for name in ("idle_decisions", "idle_wait_count", "idle_ipm_count", "idle_pm_count",
                     "idle_maintenance_actual_time", "idle_maintenance_arrival_overlap_count"):
            record[f"{name}_mean"] = float(np.mean([m[name] for m in idle_metrics]))
        if self.validation_summary_auditor is not None:
            self.validation_summary_auditor(record)
        if self.pretraining_baseline is not None:
            reference = float(self.pretraining_baseline["mean_tardiness"])
            differences = [float(r["mean_tardiness"])-m.mean_tardiness
                           for r,m in zip(self.pretraining_baseline_cases, metrics, strict=True)]
            record.update(reference_policy=self.pretraining_baseline["policy"],
                reference_mean_tardiness=reference,
                absolute_improvement_vs_reference=reference-float(record["mean_tardiness"]),
                relative_improvement_vs_reference_percent=100*(reference-float(record["mean_tardiness"]))
                    /max(self.config.reward_normalizer_floor, reference),
                beats_reference=bool(record["mean_tardiness"] < reference),
                paired_win_rate=float(np.mean(np.asarray(differences)>0)),
                paired_tie_rate=float(np.mean(np.asarray(differences)==0)),
                paired_loss_rate=float(np.mean(np.asarray(differences)<0)),
                paired_improvement_median=float(np.median(differences)))
        self.validation_metrics.append(record)
        return float(record["mean_tardiness"])

    def _save_training_logs(self, output):
        super()._save_training_logs(output)
        path = Path(output)/"Training_Log_Manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        metadata = manifest["metadata"]
        metadata["idle_maintenance_contract"] = self.variant_contract()
        metadata["maintenance_action_map"] = ACTION_MAP
        metadata["feature_contract"]["mma_maintenance_features"] = [*MMA_MAINTENANCE_FEATURES, "has_tentative_batch"]
        atomic_json_save(manifest, path)

    def audit_tables(self, case):
        tables = super().audit_tables(case)
        identity = dict(instance_index=case.instance_index, replication_index=case.replication_index,
                        instance_seed=case.instance_seed, simulation_seed=case.simulation_seed)
        tables["idle_maintenance"] = [{**identity, **r} for r in self.idle_interventions]
        tables["wait_intervals"] = [{**identity, **r} for r in self.wait_intervals]
        return tables

    def checkpoint(self):
        state = super().checkpoint()
        state["idle_maintenance_contract"] = self.variant_contract()
        return state

    def save_inference(self, path):
        state = self.checkpoint()
        state["schema_version"] = 9
        state["mappo"] = self.mappo.inference_state()
        for key in ("training", "numpy_random_state", "policy_random_state", "python_random_state",
                    "torch_random_state", "torch_cuda_random_state"):
            state.pop(key, None)
        atomic_torch_save(state, path)


