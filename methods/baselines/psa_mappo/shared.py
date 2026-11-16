"PSA-MAPPO: one actor per machine class with a centralized critic."

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiment.artifacts import atomic_json_save, atomic_torch_save
from experiment.device import (
    cuda_rng_state,
    move_optimizer_state,
    resolve_torch_device,
    restore_cuda_rng_state,
)
from experiment.evaluation import (
    EvaluationProtocol,
    EvaluationRecord,
    merge_audit_tables,
    save_evaluation_outputs,
)
from experiment.seeds import TrainingSeedProtocol
from experiment.shared_simulator import build_shared_simulator
from experiment.training_logs import TrainingConsoleLogger, TransitionAccumulator
from methods.baselines.common import (
    JointActionNetwork,
    compute_smdp_gae,
    observation_summary,
    padded_joint_actions,
)
from methods.baselines.sdr.rules import build_rule_proposal
from methods.hierarchical.config import HierarchyConfig
from methods.baselines.policy_gradient import PolicyGradientTrainingEnv


job_cluster_pt = [190, 158, 202, 128, 104]


def configure_scenario(processing_times) -> None:
    global job_cluster_pt
    job_cluster_pt = list(processing_times)


FORMATION_RULE = "CH"
SELECTION_RULES = ("MMU", "EDD-B", "MAST")


@dataclass
class RolloutItem:
    transition: object
    log_probability: float
    value: float


from methods.baselines.learning_features import MACHINE_DIM, PROPOSAL_DIM

class PSAMAPPOController:
    shared_learning_features = True
    control_trajectory_scope = "physical"

    def __init__(
            self, class_count=3, learning_rate=2e-4, gamma=0.99,
            gae_lambda=0.95, clip_ratio=0.2, update_epochs=5,
            batch_size=64, time_scale=100.0, device="auto"):
        self.device = resolve_torch_device(device)
        self.class_count = int(class_count)
        self.actors = nn.ModuleList([
            JointActionNetwork(proposal_dim=PROPOSAL_DIM, machine_dim=MACHINE_DIM, hidden_dim=64) for _ in range(class_count)
        ]).to(self.device)
        self.central_critic = nn.Sequential(
            nn.Linear(PROPOSAL_DIM + 2 * MACHINE_DIM + 1, 64), nn.ReLU(), nn.Linear(64, 64),
            nn.ReLU(), nn.Linear(64, 1),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            [*self.actors.parameters(), *self.central_critic.parameters()],
            lr=learning_rate,
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.time_scale = time_scale
        self.rollout: list[RolloutItem] = []
        self._selected = {}
        self._update_rng = np.random.default_rng(0)
        self.update_steps = 0
        self.last_diagnostics = {}
        self.last_update_records = []
        self.episode_transitions = TransitionAccumulator()

    def begin_episode(self, policy_seed):
        self._update_rng = np.random.default_rng(policy_seed)
        self._selected = {}
        self.episode_transitions.reset()

    def build_proposals(self, jobs, capacity, clock, processing_times, **_):
        """One proposal per selection rule under the shared CH formation."""
        proposals = []
        for selection_rule in SELECTION_RULES:
            proposals.extend(build_rule_proposal(
                jobs, capacity, clock, processing_times,
                FORMATION_RULE, selection_rule,
            ))
        return proposals

    def _critic_batch(self, observations):
        summaries = torch.stack([
            observation_summary(item, self.device) for item in observations
        ])
        return self.central_critic(summaries).squeeze(-1)

    def select_action(
            self, observation, *, requester_class, requester_id, training, rng, **_):
        actor_id = int(requester_class)
        with torch.no_grad():
            logits = self.actors[actor_id].forward_observation(
                observation, self.device)
            value = self._critic_batch([observation])[0]
            probabilities = torch.softmax(logits, dim=-1)
        valid = np.flatnonzero(observation.action_mask)
        if training:
            local = probabilities[valid].cpu().numpy()
            local /= local.sum()
            action = int(rng.choice(valid, p=local))
            self._selected[requester_id] = (
                float(torch.log(probabilities[action].clamp_min(1e-12)).cpu()),
                float(value.cpu()),
            )
        else:
            action = int(logits.argmax().item())
        return action

    def observe(self, transition):
        selected = self._selected.pop(transition.requester_id, None)
        if selected is not None:
            self.rollout.append(RolloutItem(
                transition, selected[0], selected[1]))
            self.episode_transitions.add(transition)

    def update(self):
        if not self.rollout:
            return None
        transitions = [item.transition for item in self.rollout]
        rewards = torch.as_tensor(
            [item.reward for item in transitions],
            dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(
            [item.value for item in self.rollout],
            dtype=torch.float32, device=self.device)
        old_log = torch.as_tensor(
            [item.log_probability for item in self.rollout],
            dtype=torch.float32, device=self.device)
        actions = [item.action for item in transitions]
        next_values = {}
        with torch.no_grad():
            available = [
                (index, item.next_state)
                for index, item in enumerate(transitions)
                if item.next_state is not None
            ]
            if available:
                values = self._critic_batch([item[1] for item in available])
                next_values = {
                    item[0]: values[position]
                    for position, item in enumerate(available)
                }
        aligned_next_values = torch.zeros_like(old_values)
        for index, next_value in next_values.items():
            aligned_next_values[index] = next_value
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        # SMDP-GAE is computed independently for each physical machine.  This
        # prevents one machine's transition from leaking credit into another
        # machine's trajectory merely because their events interleave.
        requester_ids = sorted({
            item.requester_id for item in transitions})
        for requester_id in requester_ids:
            indices = [
                index for index, item in enumerate(transitions)
                if item.requester_id == requester_id
            ]
            if not indices:
                continue
            index_tensor = torch.as_tensor(
                indices, dtype=torch.long, device=self.device)
            local_advantages, local_returns = compute_smdp_gae(
                rewards[index_tensor],
                old_values[index_tensor],
                aligned_next_values[index_tensor],
                [transitions[index].elapsed_time for index in indices],
                [transitions[index].terminated for index in indices],
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                time_scale=self.time_scale,
            )
            advantages[index_tensor] = local_advantages
            returns[index_tensor] = local_returns
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

        losses = []
        actor_losses = []
        critic_losses = []
        entropies = []
        approximate_kls = []
        clip_fractions = []
        gradient_norms = []
        update_records = []
        count = len(transitions)
        for _ in range(self.update_epochs):
            permutation = self._update_rng.permutation(count)
            for start in range(0, count, self.batch_size):
                indices = permutation[start:start + self.batch_size]
                batch = torch.as_tensor(
                    indices, dtype=torch.long, device=self.device)
                new_log = torch.empty(len(indices), device=self.device)
                entropy = torch.empty(len(indices), device=self.device)
                for class_id in range(self.class_count):
                    positions = [
                        position for position, index in enumerate(indices)
                        if transitions[index].requester_class == class_id
                    ]
                    if not positions:
                        continue
                    states = [
                        transitions[indices[position]].state
                        for position in positions
                    ]
                    logits = self.actors[class_id].forward_batch(
                        states, self.device)
                    distribution = torch.distributions.Categorical(logits=logits)
                    local_actions = padded_joint_actions(
                        states,
                        [actions[indices[position]] for position in positions],
                        device=self.device,
                    )
                    target = torch.as_tensor(
                        positions, dtype=torch.long, device=self.device)
                    new_log[target] = distribution.log_prob(local_actions)
                    entropy[target] = distribution.entropy()
                values = self._critic_batch([
                    transitions[index].state for index in indices])
                ratio = torch.exp(new_log - old_log[batch])
                surrogate = torch.minimum(
                    ratio * advantages[batch],
                    torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages[batch],
                )
                actor_loss = -surrogate.mean()
                critic_loss = nn.functional.smooth_l1_loss(
                    values, returns[batch])
                entropy_mean = entropy.mean()
                loss = (
                    actor_loss
                    + 0.5 * critic_loss
                    - 0.01 * entropy_mean
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("MAPPO loss became non-finite.")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    [*self.actors.parameters(),
                     *self.central_critic.parameters()], 1.0)
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))
                entropies.append(float(entropy_mean.detach().cpu()))
                approximate_kls.append(float(
                    (old_log[batch] - new_log).mean().detach().cpu()))
                clip_fractions.append(float(
                    (torch.abs(ratio - 1.0) > self.clip_ratio)
                    .float().mean().detach().cpu()))
                gradient_norms.append(float(
                    gradient_norm.detach().cpu()))
                self.update_steps += 1
                update_records.append({
                    "update_step": float(self.update_steps),
                    "loss": float(loss.detach().cpu()),
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "critic_loss": float(critic_loss.detach().cpu()),
                    "entropy": float(entropy_mean.detach().cpu()),
                    "approximate_kl": float(
                        (old_log[batch] - new_log)
                        .mean().detach().cpu()),
                    "clip_fraction": float(
                        (torch.abs(ratio - 1.0) > self.clip_ratio)
                        .float().mean().detach().cpu()),
                    "gradient_norm": float(
                        gradient_norm.detach().cpu()),
                    "mini_batch_size": float(len(indices)),
                    "learning_rate": float(
                        self.optimizer.param_groups[0]["lr"]),
                })
        self.rollout.clear()
        self.last_update_records = update_records
        self.last_diagnostics = {
            "loss": float(np.mean(losses)),
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "approximate_kl": float(np.mean(approximate_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "gradient_norm": float(np.mean(gradient_norms)),
            "update_step": float(self.update_steps),
            "optimizer_steps_this_update": float(len(losses)),
            "rollout_transitions": float(count),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }
        return self.last_diagnostics["loss"]

    def inference_state(self):
        return {
            "actors": self.actors.state_dict(),
            "central_critic": self.central_critic.state_dict(),
        }

    def state_dict(self):
        return {
            **self.inference_state(),
            "optimizer": self.optimizer.state_dict(),
            "rollout": self.rollout,
            "update_steps": self.update_steps,
            "update_rng": self._update_rng.bit_generator.state,
        }

    def load_state_dict(self, state, *, training):
        self.actors.load_state_dict(state["actors"])
        self.central_critic.load_state_dict(state["central_critic"])
        if training:
            self.optimizer.load_state_dict(state["optimizer"])
            move_optimizer_state(self.optimizer, self.device)
            self.rollout = list(state.get("rollout", []))
            self.update_steps = int(state.get("update_steps", 0))
            if "update_rng" in state:
                self._update_rng.bit_generator.state = state["update_rng"]


class env(PolicyGradientTrainingEnv):
    update_interval = 10

    def __init__(
            self, job_parameters, machine_parameters, device="auto",
            degradation_multiplier=1.0, calibrated_reward_scale=None,
            reward_scale_artifact_sha256=None):
        self.job_parameters = list(job_parameters)
        self.machine_parameters = list(machine_parameters)
        self.method_name = "psa_mappo"
        self.training_log_metadata = {
            "policy": "machine_class_actors_centralized_critic",
            "class_agent_count": len(self.machine_parameters),
            "trajectory_scope": "one_smdp_trajectory_per_physical_machine",
        }
        self.master_seed = int(job_parameters[-1])
        self.seed_protocol = TrainingSeedProtocol(self.master_seed)
        self.training_seed_schedule = None
        self.validation_seed_schedule = None
        self.controller = PSAMAPPOController(
            class_count=len(self.machine_parameters), device=device)
        self.training_log_metadata.update({
            "actor_parameters": sum(
                parameter.numel()
                for parameter in self.controller.actors.parameters()),
            "central_critic_parameters": sum(
                parameter.numel()
                for parameter in self.controller.central_critic.parameters()),
            "gamma": self.controller.gamma,
            "gae_lambda": self.controller.gae_lambda,
            "clip_ratio": self.controller.clip_ratio,
            "batch_size": self.controller.batch_size,
            "update_epochs": self.controller.update_epochs,
        })
        self.core = build_shared_simulator(
            job_parameters=self.job_parameters,
            machine_parameters=self.machine_parameters,
            processing_times=tuple(job_cluster_pt),
            controller=self.controller,
            config=HierarchyConfig(
                calibrated_reward_scale=calibrated_reward_scale,
                reward_scale_artifact_sha256=reward_scale_artifact_sha256,
            ),
            degradation_multiplier=degradation_multiplier,
            device=str(self.controller.device),
        )
        self.training_log_metadata.update({
            "reward_normalization": "offline_calibrated_sdr",
            "reward_scale": self.core._active_reward_normalizer,
            "reward_scale_artifact_sha256": reward_scale_artifact_sha256,
        })
        self.max_episode = 500
        self.total_episode = 0
        self.total_step = 0
        self.best_validation = math.inf
        self.training_metrics = []
        self.validation_metrics = []
        self.losses = []
        self.update_diagnostics = []
        self.rollout_diagnostics = []

    def checkpoint(self):
        return {
            "schema_version": 1,
            "method": "psa_mappo",
            "master_seed": self.master_seed,
            "total_episode": self.total_episode,
            "total_step": self.total_step,
            "best_validation": self.best_validation,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "losses": self.losses,
            "update_diagnostics": self.update_diagnostics,
            "rollout_diagnostics": self.rollout_diagnostics,
            "controller": self.controller.state_dict(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": cuda_rng_state(),
            "device": str(self.controller.device),
        }

    def load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.total_episode = int(state["total_episode"])
        self.total_step = int(state.get(
            "total_step",
            sum(item.get("decisions", 0)
                for item in state["training_metrics"]),
        ))
        self.best_validation = float(state["best_validation"])
        self.training_metrics = list(state["training_metrics"])
        self.validation_metrics = list(state["validation_metrics"])
        self.losses = list(state["losses"])
        self.update_diagnostics = list(
            state.get("update_diagnostics", []))
        self.rollout_diagnostics = list(
            state.get("rollout_diagnostics", []))
        self.controller.load_state_dict(state["controller"], training=True)
        torch.set_rng_state(state["torch_random_state"].cpu())
        restore_cuda_rng_state(
            state.get("torch_cuda_random_state"), self.controller.device)

    def main(self, results_dir, resume_from=None):
        output = Path(results_dir)
        output.mkdir(parents=True, exist_ok=True)
        if resume_from:
            self.load_checkpoint(resume_from)
        else:
            atomic_torch_save(
                self.checkpoint(), output / "psa_mappo_checkpoint.pt")
        console = TrainingConsoleLogger(
            method=self.method_name,
            device=str(self.controller.device),
            total_episodes=self.max_episode,
            completed_episodes=self.total_episode,
        )
        if self.total_episode == 0 and not self.validation_metrics:
            score = self._validation_score()
            self.best_validation = score
            self.validation_metrics[-1].update({
                "is_best": True,
                "best_mean_tardiness": score,
                "checkpoint_stage": "initial_policy",
            })
            atomic_torch_save(
                self.controller.inference_state(),
                output / "psa_mappo_model.pth")
            atomic_torch_save(
                self.checkpoint(), output / "psa_mappo_checkpoint.pt")
            self._save_training_logs(output)
            print(
                f"[VALIDATE][MAPPO][{self.controller.device}] | "
                f"initial Ep 0/{self.max_episode} | "
                f"ValTard={score:.3f}* | Best={score:.3f}",
                flush=True,
            )
        while self.total_episode < self.max_episode:
            self._run_episode(self.total_episode, training=True)
            self.total_episode += 1
            episode_updates = []
            if self.total_episode % 4 == 0:
                loss = self.controller.update()
                if loss is not None:
                    self.losses.append(loss)
                    for record in self.controller.last_update_records:
                        diagnostics = record.copy()
                        diagnostics.update({
                            "method": "psa_mappo",
                            "component": "actors_and_central_critic",
                            "episode": self.total_episode,
                            "cumulative_decisions": self.total_step,
                        })
                        self.update_diagnostics.append(diagnostics)
                        episode_updates.append(diagnostics)
                    self.training_metrics[-1]["policy_updates_total"] = (
                        self.controller.update_steps)
            validation_record = None
            if self.total_episode % self.validation_interval == 0:
                score = self._validation_score()
                improved = score < self.best_validation
                if improved:
                    self.best_validation = score
                    atomic_torch_save(
                        self.controller.inference_state(),
                        output / "psa_mappo_model.pth")
                self.validation_metrics[-1]["is_best"] = improved
                self.validation_metrics[-1]["best_mean_tardiness"] = (
                    self.best_validation)
                validation_record = self.validation_metrics[-1]
            console.print_episode(
                self.training_metrics[-1],
                updates=episode_updates,
                validation=validation_record,
            )
            if self.total_episode % self.checkpoint_interval == 0:
                atomic_torch_save(
                    self.checkpoint(), output / "psa_mappo_checkpoint.pt")
                self._save_training_logs(output)
        if not (output / "psa_mappo_model.pth").exists():
            atomic_torch_save(
                self.controller.inference_state(),
                output / "psa_mappo_model.pth")
        atomic_torch_save(
            self.checkpoint(), output / "psa_mappo_checkpoint.pt")
        self._save_training_logs(output)

    def evaluate_model(
            self, eval_episodes, results_dir, model_dir,
            evaluation_protocol=None):
        self.load_inference_model(model_dir)
        protocol = evaluation_protocol or EvaluationProtocol(
            instance_count=eval_episodes)
        records = []
        audit_tables = {}
        for case in protocol.cases():
            self.core.reset(
                instance_seed=case.instance_seed,
                simulation_seed=case.simulation_seed,
                policy_seed=case.simulation_seed)
            records.append(EvaluationRecord.from_case(
                case, self.core.run_episode(training=False)))
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

    def load_inference_model(self, model_dir) -> None:
        state = torch.load(
            Path(model_dir) / "psa_mappo_model.pth",
            map_location=self.controller.device,
            weights_only=True,
        )
        self.controller.load_state_dict(state, training=False)
