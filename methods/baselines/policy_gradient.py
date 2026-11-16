"""Shared policy-gradient training and evaluation lifecycle."""

from __future__ import annotations
from pathlib import Path
from experiment.artifacts import atomic_json_save
from experiment.metrics import EpisodeMetrics
from experiment.training_logs import decision_action_summary, save_training_log_bundle, validation_summary


class PolicyGradientTrainingEnv:
    """Shared episode execution, validation and logging for policy-gradient controllers."""
    validation_interval = 10
    validation_instances = 10
    checkpoint_interval = 25
    update_interval = 4


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
            self.core.run_episode(training=training))
        result = values.to_dict()
        result.update(seeds.to_dict())
        if training:
            self.total_step += values.decisions
            result.update({
                "episode": episode + 1,
                "cumulative_decisions": self.total_step,
                "policy_updates_total": self.controller.update_steps,
                "learning_rate": float(
                    self.controller.optimizer.param_groups[0]["lr"]),
                "reward_normalizer": self.core._active_reward_normalizer,
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





