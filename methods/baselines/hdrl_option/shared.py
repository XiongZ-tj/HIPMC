"HDRL: option-framework hierarchy with three independent Double DQNs."

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiment.artifacts import atomic_json_save, atomic_torch_save
from experiment.device import (
    resolve_torch_device,
)
from experiment.evaluation import (
    EvaluationProtocol,
    EvaluationRecord,
    merge_audit_tables,
    save_evaluation_outputs,
)
from experiment.metrics import EpisodeMetrics
from experiment.seeds import TrainingSeedProtocol
from experiment.shared_simulator import build_shared_simulator
from experiment.training_logs import (
    TrainingConsoleLogger,
    TransitionAccumulator,
    decision_action_summary,
    save_training_log_bundle,
    validation_summary,
)
from methods.baselines.sdr.rules import build_rule_proposal
from methods.hierarchical.config import HierarchyConfig


job_cluster_pt = [190, 158, 202, 128, 104]


def configure_scenario(processing_times) -> None:
    global job_cluster_pt
    job_cluster_pt = list(processing_times)


N_ACTIONS = 5
RULE_COUNT = 3
FORMATION_RULE = "CH"
SELECTION_RULES = ("MMU", "EDD-B", "MAST")
from methods.baselines.learning_features import MACHINE_DIM, PROPOSAL_DIM
N_OPTIONS = 2
N_MAINTENANCE = 2


class DoubleDQN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def _pooled(features: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.size == 0 or arr.ndim != 2:
        return np.zeros(2 * dim, dtype=np.float32)
    return np.concatenate([
        arr.mean(axis=0), arr.max(axis=0),
    ]).astype(np.float32)


class HDRLController:
    shared_learning_features = True
    """Option-framework controller: option / production / maintenance DQNs."""

    def __init__(
            self,
            *,
            learning_rate=2e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_episodes=200,
            buffer_size=20000,
            batch_size=32,
            target_sync_steps=50,
            replay_update_interval=8,
            device="auto") -> None:
        self.device = resolve_torch_device(device)
        self.input_dim = (
            2 * MACHINE_DIM + 2 * PROPOSAL_DIM + MACHINE_DIM + N_ACTIONS)
        self.nets: dict[str, dict] = {}
        for key, n_actions in (
                ("option", N_OPTIONS),
                ("production", RULE_COUNT),
                ("maintenance", N_MAINTENANCE)):
            q1 = DoubleDQN(self.input_dim, n_actions=n_actions).to(self.device)
            q2 = DoubleDQN(self.input_dim, n_actions=n_actions).to(self.device)
            q2.load_state_dict(q1.state_dict())
            self.nets[key] = {
                "q1": q1,
                "q2": q2,
                "optimizer": torch.optim.Adam(
                    q1.parameters(), lr=learning_rate),
            }
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon = epsilon_start
        self.buffers: dict[str, deque] = {
            key: deque(maxlen=buffer_size) for key in self.nets}
        self.batch_size = batch_size
        self.target_sync_steps = target_sync_steps
        self.replay_update_interval = replay_update_interval
        self.replay_update_credit = {key: 0 for key in self.nets}
        self.train_steps = {key: 0 for key in self.nets}
        self.update_steps = 0
        self.last_losses: dict[str, float] = {}
        self.episode_transitions = TransitionAccumulator()
        self._rng = np.random.default_rng(0)
        self._pending: list[tuple[str, np.ndarray, int]] = []

    def begin_episode(self, policy_seed: int) -> None:
        self._rng = np.random.default_rng(policy_seed)
        self.episode_transitions.reset()
        self._pending = []

    def decay_epsilon(self, episode: int, total: int) -> None:
        del total
        frac = min(1.0, episode / max(1, self.epsilon_decay_episodes))
        self.epsilon = (
            self.epsilon_start
            + (self.epsilon_end - self.epsilon_start) * frac)

    # ---- environment-facing interface -----------------------------------

    def build_proposals(self, jobs, capacity, clock, processing_times, **_):
        proposals = []
        for selection_rule in SELECTION_RULES:
            proposals.extend(build_rule_proposal(
                jobs,
                capacity,
                clock,
                processing_times,
                FORMATION_RULE,
                selection_rule,
            ))
        return proposals

    def select_action(self, observation, *, training, epsilon, rng, **_):
        from methods.baselines.learning_features import fixed_mask
        state = self._encode(observation)
        valid = fixed_mask(observation)
        self._active_masks = {
            'option': np.asarray([valid[:3].any(), valid[3:].any()]),
            'production': valid[:3], 'maintenance': valid[3:]}
        self._active_epsilon = epsilon if training else 0.0
        option = self._egreedy('option', state, 2, rng)
        key = 'production' if option == 0 else 'maintenance'
        choice = self._egreedy(key, state, 3 if option == 0 else 2, rng)
        self._pending = [('option', state, option), (key, state, choice)]
        return choice if option == 0 else len(observation.proposal_features) + choice

    def _next_masks(self, states, key):
        valid = states[:, -5:] > 0
        if key == 'option':
            return torch.stack((valid[:, :3].any(dim=1), valid[:, 3:].any(dim=1)), dim=1)
        return valid[:, :3] if key == 'production' else valid[:, 3:]

    def observe(self, transition) -> None:
        self.episode_transitions.add(transition)
        if not self._pending:
            return
        next_state = (
            self._encode(transition.next_state)
            if transition.next_state is not None else None)
        done = transition.terminated or transition.next_state is None
        for key, state, action in self._pending:
            self.buffers[key].append((
                state, action, float(transition.reward), next_state, done, transition.elapsed_time))
            self.replay_update_credit[key] += 1
        self._pending = []

    def _egreedy(self, key, state, n_actions, rng):
        valid = self._active_masks[key]
        if rng.random() < self._active_epsilon:
            return int(rng.choice(np.flatnonzero(valid)))
        with torch.no_grad():
            values = self.nets[key]['q1'](torch.as_tensor(
                state, dtype=torch.float32, device=self.device)).cpu().numpy()
        return int(np.where(valid, values, -np.inf).argmax())

    def learn_from_replay(self) -> list[dict]:
        """Each head earns updates only from its own newly stored samples."""
        records = []
        for key in self.nets:
            if len(self.buffers[key]) < self.batch_size:
                continue
            steps, self.replay_update_credit[key] = divmod(
                self.replay_update_credit[key], self.replay_update_interval)
            for _ in range(steps):
                losses = self.update(components=(key,))
                records.append(dict(component=key, loss=losses[key],
                    train_steps=self.train_steps[key],
                    target_syncs=self.train_steps[key] // self.target_sync_steps,
                    target_synced=self.train_steps[key] % self.target_sync_steps == 0,
                    **{f'{key}_loss': losses[key], f'{key}_replay_size': len(self.buffers[key])}))
        return records

    def update(self, components=None) -> dict[str, float] | None:
        losses: dict[str, float] = {}
        for key, nets in self.nets.items():
            if components is not None and key not in components:
                continue
            buffer = self.buffers[key]
            if len(buffer) < self.batch_size:
                continue
            indices = self._rng.choice(
                len(buffer), size=self.batch_size, replace=False)
            states, actions, rewards, dones, discounts = [], [], [], [], []
            next_states = []
            for index in indices:
                state, action, reward, next_state, done, elapsed = buffer[index]
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(1.0 if done else 0.0)
                discounts.append(self.gamma)
                next_states.append(
                    np.zeros(self.input_dim, dtype=np.float32)
                    if next_state is None else next_state)
            state_tensor = torch.as_tensor(
                np.stack(states), dtype=torch.float32, device=self.device)
            next_tensor = torch.as_tensor(
                np.stack(next_states), dtype=torch.float32, device=self.device)
            action_tensor = torch.as_tensor(
                actions, dtype=torch.long, device=self.device)
            reward_tensor = torch.as_tensor(
                rewards, dtype=torch.float32, device=self.device)
            done_tensor = torch.as_tensor(
                dones, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                next_q = nets["q1"](next_tensor)
                next_q = next_q.masked_fill(~self._next_masks(next_tensor, key), -1e9)
                best_actions = next_q.argmax(dim=1)
                target = nets["q2"](next_tensor).gather(
                    1, best_actions.unsqueeze(1)).squeeze(1)
                target = target.masked_fill(~self._next_masks(next_tensor, key).any(dim=1), 0.0)
                target = (
                    reward_tensor
                    + torch.as_tensor(discounts, dtype=reward_tensor.dtype, device=self.device) * target * (1.0 - done_tensor))
            predicted = nets["q1"](state_tensor).gather(
                1, action_tensor.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(predicted, target)
            nets["optimizer"].zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(nets["q1"].parameters(), 1.0)
            nets["optimizer"].step()
            self.train_steps[key] += 1
            if self.train_steps[key] % self.target_sync_steps == 0:
                nets["q2"].load_state_dict(nets["q1"].state_dict())
            losses[key] = float(loss.detach().cpu())
        self.update_steps += 1
        self.last_losses = losses
        return losses or None

    # ---- internal ---------------------------------------------------------

    def _encode(self, observation) -> np.ndarray:
        machines = np.asarray(
            observation.machine_features, dtype=np.float32)
        proposals = np.asarray(
            observation.proposal_features, dtype=np.float32)
        machine_stats = _pooled(machines, MACHINE_DIM)
        proposal_stats = _pooled(proposals, PROPOSAL_DIM)
        requester = (
            machines[observation.requester_index]
            if 0 <= observation.requester_index < len(machines)
            else np.zeros(MACHINE_DIM, dtype=np.float32))
        from methods.baselines.learning_features import fixed_mask
        mask = fixed_mask(observation).astype(np.float32)
        state = np.concatenate([
            machine_stats, proposal_stats, requester, mask,
        ]).astype(np.float32)
        if state.shape != (self.input_dim,):
            raise ValueError(
                f"HDRL state dimension {state.shape} != {self.input_dim}.")
        return state

    def state_dict(self) -> dict:
        return {
            "nets": {
                key: {
                    "q1": nets["q1"].state_dict(),
                    "q2": nets["q2"].state_dict(),
                    "optimizer": nets["optimizer"].state_dict(),
                }
                for key, nets in self.nets.items()
            },
            "epsilon": self.epsilon,
            "train_steps": dict(self.train_steps),
            "replay_update_credit": dict(self.replay_update_credit),
            "replay_update_interval": self.replay_update_interval,
            "target_sync_steps": self.target_sync_steps,
            "update_steps": self.update_steps,
            "buffers": {
                key: list(buffer) for key, buffer in self.buffers.items()},
            "rng": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict, *, training: bool) -> None:
        for key, nets in self.nets.items():
            nets["q1"].load_state_dict(state["nets"][key]["q1"])
            nets["q2"].load_state_dict(state["nets"][key]["q2"])
            if training:
                nets["optimizer"].load_state_dict(
                    state["nets"][key]["optimizer"])
        if training:
            self.replay_update_credit = {key: int(state.get("replay_update_credit", {}).get(key, 0))
                                         for key in self.nets}
            self.replay_update_interval = int(state.get("replay_update_interval", self.replay_update_interval))
            self.target_sync_steps = int(state.get("target_sync_steps", self.target_sync_steps))
            self.epsilon = float(state.get("epsilon", self.epsilon))
            for key in self.nets:
                self.train_steps[key] = int(
                    state.get("train_steps", {}).get(key, 0))
            self.update_steps = int(state.get("update_steps", 0))
            for key, buffer in self.buffers.items():
                self.buffers[key] = deque(
                    [tuple(item) for item in state.get("buffers", {}).get(
                        key, [])],
                    maxlen=buffer.maxlen)
            if "rng" in state:
                self._rng.bit_generator.state = state["rng"]


class env:
    validation_interval = 10
    validation_instances = 30
    checkpoint_interval = 25
    update_interval = 1  # Episode-end scheduling; each head uses its own samples.

    def __init__(
            self,
            job_parameters,
            machine_parameters,
            device="auto",
            degradation_multiplier=1.0,
            calibrated_reward_scale=None,
            reward_scale_artifact_sha256=None):
        self.job_parameters = list(job_parameters)
        self.machine_parameters = list(machine_parameters)
        self.method_name = "hdrl_option"
        self.training_log_metadata = {
            "policy": "option_framework_three_independent_double_dqn",
        }
        self.master_seed = int(job_parameters[-1])
        self.seed_protocol = TrainingSeedProtocol(self.master_seed)
        self.training_seed_schedule = None
        self.validation_seed_schedule = None
        self.controller = HDRLController(device=device)
        self.training_log_metadata.update({
            "trainable_parameters": sum(
                sum(parameter.numel()
                    for parameter in nets["q1"].parameters())
                for nets in self.controller.nets.values()),
            "gamma": self.controller.gamma,
            "epsilon_start": self.controller.epsilon_start,
            "epsilon_end": self.controller.epsilon_end,
            "batch_size": self.controller.batch_size,
            "target_sync_steps": self.controller.target_sync_steps,
            "target_sync_unit": "per_network_optimizer_steps",
            "replay_update_interval": self.controller.replay_update_interval,
            "replay_update_unit": "per_network_new_stored_transitions",
            "optimizer_execution": "episode_end",
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
        self.training_metrics: list[dict] = []
        self.validation_metrics: list[dict] = []
        self.losses: list[float] = []
        self.update_diagnostics: list[dict] = []
        self.rollout_diagnostics: list[dict] = []

    def _run_episode(self, episode: int, *, training: bool) -> EpisodeMetrics:
        seeds = (
            self.training_seed_schedule[episode]
            if self.training_seed_schedule is not None
            else self.seed_protocol.for_episode(episode)
        )
        self.core.reset(
            instance_seed=seeds.instance_seed,
            simulation_seed=seeds.simulation_seed,
            policy_seed=seeds.policy_seed,
        )
        values = EpisodeMetrics.from_mapping(
            self.core.run_episode(training=training, epsilon=self.controller.epsilon if training else 0.0))
        result = values.to_dict()
        result.update(seeds.to_dict())
        if training:
            self.total_step += values.decisions
            result.update({
                "episode": episode + 1,
                "cumulative_decisions": self.total_step,
                "policy_updates_total": self.controller.update_steps,
                "epsilon": self.controller.epsilon,
            })
            transition_summary = (
                self.controller.episode_transitions.summary("policy"))
            result.update(transition_summary)
            result.update(decision_action_summary(
                self.core.decision_history,
                machine_class_count=len(self.core.scenario.machine_counts),
                health_state_count=self.core.config.degradation_states,
            ))
            self.training_metrics.append(result)
            self.rollout_diagnostics.append({
                "method": self.method_name,
                "episode": episode + 1,
                "training_seed": self.master_seed,
                **transition_summary,
            })
        return values

    def _validation_score(self) -> float:
        metrics = []
        schedule = self.validation_seed_schedule or [
            {
                "instance_seed": 3_000_000 + index,
                "simulation_seed": 4_000_000 + index,
                "policy_seed": 4_000_000 + index,
            }
            for index in range(self.validation_instances)
        ]
        for item in schedule:
            self.core.reset(
                instance_seed=int(item["instance_seed"]),
                simulation_seed=int(item["simulation_seed"]),
                policy_seed=int(item["policy_seed"]),
            )
            metrics.append(EpisodeMetrics.from_mapping(
                self.core.run_episode(training=False)))
        record = validation_summary(self.total_episode, metrics)
        score = float(record["mean_tardiness"])
        self.validation_metrics.append(record)
        return score

    def _save_training_logs(self, output: Path) -> None:
        save_training_log_bundle(
            output,
            method=self.method_name,
            episodes=self.training_metrics,
            updates=self.update_diagnostics,
            rollouts=self.rollout_diagnostics,
            validation=self.validation_metrics,
            metadata={
                **self.training_log_metadata,
                "training_seed": self.master_seed,
                "training_condition": {
                    "nominal_finite_load": self.job_parameters[0],
                    "initial_jobs_configured": self.job_parameters[1],
                    "total_jobs_configured": self.job_parameters[2],
                    "due_date_factor": self.job_parameters[3],
                    "degradation_multiplier": (
                        self.core.config.degradation_multiplier),
                },
                "episode_indexing": "one_based_completed_episode",
                "primary_training_curve": "total_tardiness",
                "primary_validation_curve": "mean_tardiness",
            },
        )
        atomic_json_save({
            "training": self.training_metrics,
            "updates": self.update_diagnostics,
            "rollouts": self.rollout_diagnostics,
            "validation": self.validation_metrics,
            "losses": self.losses,
            "best_validation": self.best_validation,
        }, output / "Training_Diagnostics.json")

    def checkpoint(self) -> dict:
        return {
            "schema_version": 1,
            "method": "hdrl_option",
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
            "device": str(self.controller.device),
        }

    def load_checkpoint(self, path) -> None:
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

    def main(self, results_dir, resume_from=None) -> None:
        output = Path(results_dir)
        output.mkdir(parents=True, exist_ok=True)
        if resume_from:
            self.load_checkpoint(resume_from)
        else:
            atomic_torch_save(
                self.checkpoint(), output / "hdrl_option_checkpoint.pt")
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
            self.save_inference(output)
            atomic_torch_save(
                self.checkpoint(), output / "hdrl_option_checkpoint.pt")
            self._save_training_logs(output)
            print(
                f"[VALIDATE][HDRL][{self.controller.device}] | "
                f"initial Ep 0/{self.max_episode} | "
                f"ValTard={score:.3f}* | Best={score:.3f}",
                flush=True,
            )
        while self.total_episode < self.max_episode:
            self._run_episode(self.total_episode, training=True)
            self.total_episode += 1
            self.controller.decay_epsilon(
                self.total_episode, self.max_episode)
            episode_updates = []
            if self.total_episode % self.update_interval == 0:
                for update in self.controller.learn_from_replay():
                    mean_loss = update['loss']
                    self.losses.append(mean_loss)
                    record = {
                        "method": self.method_name,
                        "component": "policy",
                        **update,
                        "episode": self.total_episode,
                        "loss": mean_loss,
                        "epsilon": self.controller.epsilon,
                        "cumulative_decisions": self.total_step,
                    }
                    self.update_diagnostics.append(record)
                    episode_updates.append(record)
            validation_record = None
            if self.total_episode % self.validation_interval == 0:
                score = self._validation_score()
                improved = score < self.best_validation
                if improved:
                    self.best_validation = score
                    self.save_inference(output)
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
                    self.checkpoint(), output / "hdrl_option_checkpoint.pt")
                self._save_training_logs(output)
        if not (output / "hdrl_option_model.pth").exists():
            self.save_inference(output)
        atomic_torch_save(
            self.checkpoint(), output / "hdrl_option_checkpoint.pt")
        self._save_training_logs(output)

    def save_inference(self, output: Path) -> None:
        inference = {
            key: nets["q1"].state_dict()
            for key, nets in self.controller.nets.items()
        }
        atomic_torch_save(inference, output / "hdrl_option_model.pth")

    def evaluate_model(
            self,
            eval_episodes,
            results_dir,
            model_dir,
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

    def load_inference_model(self, model_dir) -> None:
        weights = torch.load(
            Path(model_dir) / "hdrl_option_model.pth",
            map_location=self.controller.device,
            weights_only=True,
        )
        for key, nets in self.controller.nets.items():
            nets["q1"].load_state_dict(weights[key])
