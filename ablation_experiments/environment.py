"Inference-time replacement of batch scoring, maintenance, or batch formation."

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from methods.baselines.sdr.rules import select_batch
from methods.hierarchical.agents import PolicySelection
from methods.hierarchical.config import HierarchyConfig
from methods.hierarchical.environment import (
    HierarchicalSchedulingEnv,
    Machine,
)
from methods.hierarchical.types import (
    BAAObservation,
    BatchProposal,
    DecisionContext,
    HandshakeMode,
    MachineAction,
    MachinePhase,
    MMAObservation,
)

from .variants import AblationVariant, VariantSpec, variant_spec
from .candidates import formation_proposals
from comparison_runners.composite_priority.rule import select_candidate


class AblationSchedulingEnv(HierarchicalSchedulingEnv):
    """Proposed environment with exactly one frozen component replacement."""

    def __init__(
            self,
            job_parameters: list[float | int],
            machine_parameters: list[int],
            *,
            processing_times: tuple[int, ...],
            variant: str | AblationVariant,
            config: HierarchyConfig,
            device: str | torch.device = "auto",
            composite_weights=None) -> None:
        self.variant_spec: VariantSpec = variant_spec(variant)
        self.ablation_variant = self.variant_spec.variant
        self._periodic_last_maintenance: dict[int, float] = {}
        self.maintenance_period = float(max(processing_times))
        self._active_proposals: list[BatchProposal] | None = None
        self.composite_weights=composite_weights
        super().__init__(
            job_parameters,
            machine_parameters,
            processing_times=processing_times,
            handshake_mode=self.variant_spec.handshake_mode,
            config=config,
            device=device,
        )

    @classmethod
    def create(
            cls,
            *,
            variant: str | AblationVariant,
            job_parameters: list[float | int],
            machine_parameters: list[int],
            processing_times: tuple[int, ...],
            degradation_multiplier: float,
            device: str | torch.device,
            calibrated_reward_scale: float | None,
            reward_scale_artifact_sha256: str | None,
            hyperparameters: dict | None = None,
            composite_weights=None):
        config_kwargs = dict(
            degradation_multiplier=degradation_multiplier,
            calibrated_reward_scale=calibrated_reward_scale,
            reward_scale_artifact_sha256=reward_scale_artifact_sha256,
        )
        if hyperparameters:
            config_kwargs.update({
                "learning_rate": float(hyperparameters.get(
                    "learning_rate", 2e-4)),
                "ppo_clip_ratio": float(hyperparameters.get(
                    "clip_ratio",
                    hyperparameters.get("ppo_clip_ratio", 0.2))),
                "gamma_baa": float(hyperparameters.get("gamma_baa", 0.99)),
                "gamma_mma": float(hyperparameters.get("gamma_mma", 0.99)),
            })
        return cls(
            job_parameters,
            machine_parameters,
            processing_times=tuple(processing_times),
            variant=variant,
            config=HierarchyConfig(**config_kwargs),
            device=device,
            composite_weights=composite_weights,
        )

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)
        self._periodic_last_maintenance.clear()
        self._active_proposals = None


    def _baa_observation(
            self,
            machine: Machine,
            proposals: list[BatchProposal]) -> BAAObservation:
        """Expose the current candidate set to the fixed rule callback."""
        self._active_proposals = list(proposals)
        return super()._baa_observation(machine, proposals)

    def _build_candidate_proposals(self,jobs,capacity):
        if self.variant_spec.formation_rule=='CH':
            return super()._build_candidate_proposals(jobs,capacity)
        return formation_proposals(jobs,capacity,self.clock,self.job_cluster_pt,
                                   self.variant_spec.formation_rule)

    def _rule_scores(self) -> np.ndarray:
        """Rule value of every active candidate (sign kept, ascending best)."""
        if self._active_proposals is None or not self._active_proposals:
            raise RuntimeError("Rule selection needs a nonempty candidate set.")
        rule = self.variant_spec.baa_rule
        by_id = {job.job_id: job for job in self.jobs}
        clock = self.clock
        values = []
        for proposal in self._active_proposals:
            jobs = [by_id[job_id] for job_id in proposal.job_ids]
            if rule == "MMU":
                value = float(-sum(job.size for job in jobs))
            elif rule == "EDD-B":
                value = float(min(job.due_time for job in jobs))
            elif rule == "MAST":
                value = float(np.mean([
                    job.due_time - clock
                    - self.job_cluster_pt[proposal.family]
                    for job in jobs
                ]))
            else:
                raise RuntimeError(f"Unknown BAA rule: {rule}")
            values.append(value)
        return np.asarray(values, dtype=np.float32)

    def _fixed_baa_selection(
            self, observation: BAAObservation) -> PolicySelection:
        """Select a candidate with the frozen batch-selection rule."""
        rule = self.variant_spec.baa_rule
        if rule is None:
            raise RuntimeError("This variant does not replace the BAA actor.")
        batches = [
            (proposal.family, proposal.job_ids)
            for proposal in self._active_proposals or []
        ]
        action = int(select_candidate(observation.proposal_features,self.composite_weights,
            [p.job_ids for p in self._active_proposals])) if rule=='Composite' else int(select_batch(
            batches,
            self.jobs,
            self.clock,
            self.job_cluster_pt,
            rule,
        ))
        scores = (observation.proposal_features@np.asarray(self.composite_weights)
                  if rule=='Composite' else self._rule_scores())
        return PolicySelection(
            executed_action=action,
            policy_action=np.asarray([action], dtype=np.int64),
            log_probability=0.0,
            value=0.0,
            entropy=0.0,
            weights=None,
            scores=scores,
        )

    def _request(
            self,
            machine: Machine,
            *,
            epsilon: float,
            training: bool) -> bool:
        if self.variant_spec.baa_rule is None:
            return super()._request(
                machine, epsilon=epsilon, training=training)
        original = self.mappo.select_baa

        def select_fixed(
                observation: BAAObservation,
                *,
                training: bool) -> PolicySelection:
            del training
            return self._fixed_baa_selection(observation)

        # Reuse the canonical request/reservation implementation while
        # replacing only its BAA selection callback for this synchronous call.
        self.mappo.select_baa = select_fixed
        try:
            return super()._request(
                machine, epsilon=epsilon, training=training)
        finally:
            self.mappo.select_baa = original

    def _close_previous_baa(
            self,
            next_state: BAAObservation | None,
            *,
            terminated: bool) -> None:
        if self.variant_spec.baa_rule is None:
            super()._close_previous_baa(next_state, terminated=terminated)
        else:
            # The fixed rule creates neither an actor nor an RL trajectory.
            self._pending_baa = None

    def _close_previous_mma(
            self,
            machine: Machine,
            next_state: MMAObservation | None,
            *,
            terminated: bool) -> None:
        if self.variant_spec.mma_rule is None:
            super()._close_previous_mma(
                machine, next_state, terminated=terminated)
        else:
            # Fixed maintenance rules do not create an MMA policy trajectory.
            self._pending_mma.pop(machine.machine_id, None)

    def _fixed_mma_action(
            self, machine: Machine, observation: MMAObservation) -> int:
        rule = self.variant_spec.mma_rule
        if rule is None:
            raise RuntimeError("This variant does not replace the MMA actor.")
        action = int(MachineAction.PROCESS)
        if rule == "Period":
            if (
                    machine.state > 0
                    and self.clock - self._periodic_last_maintenance.get(
                        machine.machine_id, 0.0) >= self.maintenance_period):
                action = int(MachineAction.IPM)
        elif rule == "Threshold":
            if machine.state >= 3:
                action = int(MachineAction.PM)
            elif machine.state >= 2:
                action = int(MachineAction.IPM)
        elif rule != "CM":
            raise RuntimeError(f"Unknown MMA rule: {rule}")
        if not observation.action_mask[action]:
            action = int(MachineAction.PROCESS)
        if action != int(MachineAction.PROCESS):
            self._periodic_last_maintenance[machine.machine_id] = self.clock
        return action

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
        if self.variant_spec.mma_rule is None:
            super()._decide_mma(
                machine,
                proposal,
                epsilon=epsilon,
                training=training,
                baa_observation=baa_observation,
                baa_action=baa_action,
                request_id=request_id,
                baa_eligible_family_count=baa_eligible_family_count,
            )
            return
        del epsilon, training
        observation = self._mma_observation(machine, proposal)
        self._close_previous_mma(machine, observation, terminated=False)
        action = self._fixed_mma_action(machine, observation)
        probabilities = np.zeros(3, dtype=np.float32)
        probabilities[action] = 1.0
        baa_context = (
            self._pending_baa.context
            if baa_observation is not None and self._pending_baa is not None
            else None
        )
        effective_baa_observation = (
            baa_observation
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
            )
        )
        context = DecisionContext(
            request_id=request_id,
            proposal_id=proposal.proposal_id,
            machine_id=machine.machine_id,
            started_at=self.clock,
            baa_observation=effective_baa_observation,
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
            mma_log_probability=0.0,
            mma_value=0.0,
            mma_entropy=0.0,
            mma_probabilities=probabilities,
            baa_decision=baa_observation is not None,
            baa_eligible_family_count=baa_eligible_family_count,
        )
        self._start_action(machine, proposal, context)

