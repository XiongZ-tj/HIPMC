"""Unified, curve-ready training records for every DRL method."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from experiment.artifacts import atomic_csv_save, atomic_json_save
from experiment.metrics import EpisodeMetrics


TRAINING_LOG_SCHEMA_VERSION = 12
CANONICAL_METHOD_IDS = {
    "ours": "heterogeneous_smdp_mappo",
    "ppo": "centralized_ppo",
    "qmix": "qmix",
    "mappo": "mappo",
    "hrl": "hierarchical_reinforcement_learning",
    "ddqnb": "ddqnb",
    "psa_mappo": "psa_mappo",
    "hdrl_option": "hdrl_option",
}
BAA_RULE_KEYS = ("ch2", "mmu", "edd_b", "mast")
BAA_CRITERION_KEYS = (
    "capacity_utilization",
    "mean_urgency",
    "worst_job_urgency",
    "waiting_starvation",
    "processing_efficiency",
    "uninterrupted_completion_probability",
)


def _duration_text(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _mean_field(records: Iterable[dict], name: str) -> float | None:
    values = [
        float(record[name])
        for record in records
        if record.get(name) is not None
    ]
    return float(np.mean(values)) if values else None


class TrainingConsoleLogger:
    """Print one flushed, comparable progress line per completed episode."""

    def __init__(
            self,
            *,
            method: str,
            device: str,
            total_episodes: int,
            completed_episodes: int = 0) -> None:
        self.method = method.upper()
        self.device = str(device)
        self.total_episodes = int(total_episodes)
        self.completed_at_start = int(completed_episodes)
        self.started_at = perf_counter()

    def print_validation(self, validation: dict, *, label: str) -> str:
        """Print a standalone validation record, including episode zero."""

        marker = "*" if bool(validation.get("is_best")) else ""
        fields = [
            f"[VALIDATE][{self.method}][{self.device}]",
            f"{label} Ep {int(validation['episode'])}/{self.total_episodes}",
            f"ValTard={float(validation['mean_tardiness']):.3f}{marker}",
        ]
        best = validation.get("best_mean_tardiness")
        if best is not None:
            fields.append(f"Best={float(best):.3f}")
        self._append_reference_comparison(fields, validation)
        line = " | ".join(fields)
        print(line, flush=True)
        return line

    def print_baseline(self, baseline: dict) -> str:
        """Print the frozen pretraining reference before model validation."""

        line = " | ".join((
            f"[BASELINE][{self.method}][cpu]",
            str(baseline["policy"]),
            f"Cases={int(baseline['validation_cases'])}",
            f"MeanTard={float(baseline['mean_tardiness']):.3f}",
            f"Std={float(baseline['mean_tardiness_std']):.3f}",
        ))
        print(line, flush=True)
        return line

    @staticmethod
    def _append_reference_comparison(fields: list[str], validation: dict) -> None:
        baseline = validation.get("reference_mean_tardiness")
        if baseline is None:
            return
        fields.append(f"Base={float(baseline):.3f}")
        improvement = float(validation["absolute_improvement_vs_reference"])
        relative = float(
            validation["relative_improvement_vs_reference_percent"])
        fields.append(f"Improve={improvement:+.3f} ({relative:+.2f}%)")
        if validation.get("paired_win_rate") is not None:
            fields.append(
                f"Wins={100.0 * float(validation['paired_win_rate']):.1f}%")

    def print_episode(
            self,
            metrics: dict,
            *,
            updates: Iterable[dict] = (),
            validation: dict | None = None) -> str:
        """Print and return the exact progress line for testability."""

        episode = int(metrics["episode"])
        completed_here = max(1, episode - self.completed_at_start)
        elapsed = perf_counter() - self.started_at
        rate = elapsed / completed_here
        remaining = max(0, self.total_episodes - episode) * rate
        percent = 100.0 * episode / max(1, self.total_episodes)
        records = list(updates)

        fields = [
            f"[TRAIN][{self.method}][{self.device}]",
            f"Ep {episode}/{self.total_episodes} ({percent:5.1f}%)",
            f"Tard={float(metrics['total_tardiness']):.3f}",
            f"MeanTard={float(metrics['mean_tardiness']):.3f}",
            f"Makespan={float(metrics['makespan']):.3f}",
            f"Dec={int(metrics['decisions'])}",
            f"Fail={int(metrics.get('breakdowns', 0))}",
            f"IPM={int(metrics.get('ipm_count', 0))}",
            f"PM={int(metrics.get('pm_count', 0))}",
        ]
        if metrics.get("epsilon") is not None:
            fields.append(f"Eps={float(metrics['epsilon']):.4f}")
        if metrics.get("reward_normalizer") is not None:
            fields.append(
                f"Norm={float(metrics['reward_normalizer']):.4f}")
        if metrics.get("mma_entropy_weight") is not None:
            fields.append(
                f"MMAEntW={float(metrics['mma_entropy_weight']):.5f}")

        if metrics.get("baa_reward_sum") is not None:
            fields.append(f"BAA_R={float(metrics['baa_reward_sum']):.4f}")
            if metrics.get("mma_shared_reward_per_trajectory_mean") is not None:
                fields.append(
                    "MMA_Ravg="
                    f"{float(metrics['mma_shared_reward_per_trajectory_mean']):.4f}")
            else:
                mma_reward = sum(
                    float(metrics.get(f"mma_{index}_reward_sum", 0.0))
                    for index in range(3)
                )
                fields.append(f"MMA_R={mma_reward:.4f}")
        elif metrics.get("policy_reward_sum") is not None:
            fields.append(
                f"Reward={float(metrics['policy_reward_sum']):.4f}")

        baa_loss = _mean_field(
            (record for record in records
             if record.get("component") == "baa"),
            "loss",
        )
        mma_loss = _mean_field(
            (record for record in records
             if record.get("component") == "mma"),
            "loss",
        )
        if baa_loss is not None:
            fields.append(f"BAA_L={baa_loss:.5f}")
        if mma_loss is not None:
            fields.append(f"MMA_L={mma_loss:.5f}")
        if baa_loss is None and mma_loss is None:
            loss = _mean_field(records, "loss")
            fields.append("Loss=-" if loss is None else f"Loss={loss:.5f}")

        if validation is not None:
            marker = "*" if bool(validation.get("is_best")) else ""
            fields.append(
                f"ValTard={float(validation['mean_tardiness']):.3f}{marker}")
            best = validation.get("best_mean_tardiness")
            if best is not None:
                fields.append(f"Best={float(best):.3f}")
            self._append_reference_comparison(fields, validation)

        fields.extend((
            f"Elapsed={_duration_text(elapsed)}",
            f"ETA={_duration_text(remaining)}",
        ))
        line = " | ".join(fields)
        print(line, flush=True)
        return line


class TransitionAccumulator:
    """Collect one episode of transition diagnostics without storing states."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.rewards: list[float] = []
        self.elapsed_times: list[float] = []
        self.actions: list[int] = []
        self.terminal_count = 0
        self.reward_components: Counter[str] = Counter()

    def add(self, transition) -> None:
        self.rewards.append(float(transition.reward))
        self.elapsed_times.append(float(transition.elapsed_time))
        action = np.asarray(transition.action)
        if action.ndim == 0:
            try:
                self.actions.append(int(action))
            except TypeError:
                # Autoregressive centralized actions are structured; the
                # operation head (Process/IPM/PM) is the comparable action
                # category used by episode diagnostics.
                operation = getattr(transition.action, "operation", None)
                if operation is not None:
                    self.actions.append(int(operation))
        self.terminal_count += int(bool(transition.terminated))
        for name, value in getattr(
                transition, "reward_components", {}).items():
            if isinstance(value, (int, float, np.number)):
                self.reward_components[name] += float(value)

    def summary(self, prefix: str) -> dict[str, float | int]:
        rewards = np.asarray(self.rewards, dtype=float)
        elapsed = np.asarray(self.elapsed_times, dtype=float)
        result: dict[str, float | int] = {
            f"{prefix}_transitions": len(self.rewards),
            f"{prefix}_terminal_transitions": self.terminal_count,
            f"{prefix}_reward_sum": float(rewards.sum()) if rewards.size else 0.0,
            f"{prefix}_reward_mean": float(rewards.mean()) if rewards.size else 0.0,
            f"{prefix}_reward_std": float(rewards.std()) if rewards.size else 0.0,
            f"{prefix}_reward_min": float(rewards.min()) if rewards.size else 0.0,
            f"{prefix}_reward_max": float(rewards.max()) if rewards.size else 0.0,
            f"{prefix}_elapsed_mean": (
                float(elapsed.mean()) if elapsed.size else 0.0),
            f"{prefix}_elapsed_max": (
                float(elapsed.max()) if elapsed.size else 0.0),
        }
        for action, count in sorted(Counter(self.actions).items()):
            result[f"{prefix}_action_{action}_count"] = count
        for name, value in sorted(self.reward_components.items()):
            result[f"{prefix}_{name}_sum"] = float(value)
        return result


def baa_candidate_summary(decisions: Iterable[dict]) -> dict[str, float | int]:
    """Aggregate true BAA choices without counting MMA-only re-decisions."""

    records = [item for item in decisions if bool(item.get("baa_decision"))]
    candidate_counts = np.asarray([
        int(item["baa_candidate_count"]) for item in records], dtype=int)
    eligible_counts = np.asarray([
        int(item["baa_eligible_family_count"]) for item in records], dtype=int)
    family_counts = np.asarray([
        int(item["baa_candidate_family_count"]) for item in records], dtype=int)
    decision_count = len(records)
    candidate_total = int(candidate_counts.sum()) if decision_count else 0
    result: dict[str, float | int] = {
        "baa_candidate_decisions": decision_count,
        "baa_candidate_total": candidate_total,
        "baa_candidate_count_mean": (
            float(candidate_counts.mean()) if decision_count else 0.0),
        "baa_candidate_count_std": (
            float(candidate_counts.std()) if decision_count else 0.0),
        "baa_candidate_count_min": (
            int(candidate_counts.min()) if decision_count else 0),
        "baa_candidate_count_max": (
            int(candidate_counts.max()) if decision_count else 0),
        "baa_eligible_family_count_mean": (
            float(eligible_counts.mean()) if decision_count else 0.0),
        "baa_candidate_family_count_mean": (
            float(family_counts.mean()) if decision_count else 0.0),
        "baa_candidate_exclusion_gap_mean": (
            float((eligible_counts - candidate_counts).mean())
            if decision_count else 0.0),
        "baa_candidate_coverage_mean": (
            float(np.mean(candidate_counts / np.maximum(1, eligible_counts)))
            if decision_count else 0.0),
        "baa_full_family_coverage_rate": (
            float(np.mean(candidate_counts == eligible_counts))
            if decision_count else 0.0),
        "baa_candidate_family_identity_rate": (
            float(np.mean(candidate_counts == family_counts))
            if decision_count else 0.0),
        "baa_effective_choice_rate": (
            float(np.mean(candidate_counts >= 2))
            if decision_count else 0.0),
        "baa_no_choice_rate": (
            float(np.mean(candidate_counts == 1))
            if decision_count else 0.0),
        "baa_four_candidate_rate": (
            float(np.mean(candidate_counts == 4))
            if decision_count else 0.0),
    }
    for count in range(1, 5):
        occurrences = int(np.sum(candidate_counts == count))
        result[f"baa_candidate_count_{count}_count"] = occurrences
        result[f"baa_candidate_count_{count}_rate"] = (
            occurrences / decision_count if decision_count else 0.0)
    ge_five = int(np.sum(candidate_counts >= 5))
    result["baa_candidate_count_ge_5_count"] = ge_five
    result["baa_candidate_count_ge_5_rate"] = (
        ge_five / decision_count if decision_count else 0.0)

    for source_count in range(1, 5):
        candidate_occurrences = sum(
            int(item.get(
                f"baa_candidates_with_{source_count}_sources", 0))
            for item in records
        )
        selected_occurrences = sum(
            int(item.get("baa_selected_source_count", 0)) == source_count
            for item in records
        )
        result[
            f"baa_candidates_with_{source_count}_sources_count"
        ] = candidate_occurrences
        result[
            f"baa_candidates_with_{source_count}_sources_rate"
        ] = candidate_occurrences / candidate_total if candidate_total else 0.0
        result[
            f"baa_selected_source_count_{source_count}_count"
        ] = selected_occurrences
        result[
            f"baa_selected_source_count_{source_count}_rate"
        ] = selected_occurrences / decision_count if decision_count else 0.0

    for key in BAA_RULE_KEYS:
        selected = sum(bool(item.get(
            f"baa_selected_source_{key}")) for item in records)
        result[f"baa_selected_source_{key}_count"] = selected
        result[f"baa_selected_source_{key}_rate"] = (
            selected / decision_count if decision_count else 0.0)
    for left in range(len(BAA_RULE_KEYS)):
        for right in range(left + 1, len(BAA_RULE_KEYS)):
            name = (
                f"baa_rule_collision_{BAA_RULE_KEYS[left]}_"
                f"{BAA_RULE_KEYS[right]}")
            collisions = sum(bool(item.get(name)) for item in records)
            result[f"{name}_count"] = collisions
            result[f"{name}_rate"] = (
                collisions / decision_count if decision_count else 0.0)
    all_same = sum(bool(item.get(
        "baa_all_four_rules_same")) for item in records)
    result["baa_all_four_rules_same_count"] = all_same
    result["baa_all_four_rules_same_rate"] = (
        all_same / decision_count if decision_count else 0.0)
    return result


def baa_policy_summary(decisions: Iterable[dict]) -> dict[str, float | int]:
    """Aggregate candidate scores, feature values and selection diagnostics."""

    records = [item for item in decisions if bool(item.get("baa_decision"))]
    count = len(records)
    candidates = np.asarray([
        int(item["baa_candidate_count"]) for item in records], dtype=int)
    eligible = np.asarray([
        int(item["baa_eligible_family_count"]) for item in records], dtype=int)
    result: dict[str, float | int] = {
        "baa_decisions": count,
        "baa_candidate_count_mean": float(candidates.mean()) if count else 0.0,
        "baa_candidate_count_min": int(candidates.min()) if count else 0,
        "baa_candidate_count_max": int(candidates.max()) if count else 0,
        "baa_eligible_family_count_mean": float(eligible.mean()) if count else 0.0,
        "baa_candidate_coverage_mean": float(np.mean(
            candidates / np.maximum(1, eligible))) if count else 0.0,
        "baa_full_family_coverage_rate": float(np.mean(
            candidates == eligible)) if count else 0.0,
        "baa_no_choice_rate": float(np.mean(candidates == 1)) if count else 0.0,
        "baa_policy_entropy_mean": _mean_field(records, "baa_policy_entropy") or 0.0,
        "baa_selected_score_mean": _mean_field(records, "baa_selected_score") or 0.0,
        "baa_score_margin_mean": _mean_field(records, "baa_score_margin") or 0.0,
    }
    if not count:
        for key in BAA_CRITERION_KEYS:
            for suffix in (
                    "selected_value_mean", "selected_rank_mean"):
                result[f"baa_{key}_{suffix}"] = 0.0
        return result
    for key in BAA_CRITERION_KEYS:
        result[f"baa_{key}_selected_value_mean"] = float(np.mean([
            float(item[f"baa_selected_{key}"]) for item in records]))
        result[f"baa_{key}_selected_rank_mean"] = float(np.mean([
            float(item[f"baa_selected_rank_{key}"]) for item in records]))
    return result


def mma_probability_summary(
        decisions: Iterable[dict],
        *,
        machine_class_count: int,
        health_state_count: int) -> dict[str, float | int]:
    """Aggregate MMA probability calibration by class and health state."""

    records = [
        item for item in decisions
        if item.get("mma_probability_process") is not None
    ]
    result: dict[str, float | int] = {
        "mma_probability_decisions": len(records),
        "mma_policy_entropy_mean": _mean_field(
            records, "mma_policy_entropy") or 0.0,
        "mma_selected_process_failure_risk_mean": (
            _mean_field(
                (item for item in records
                 if int(item.get("maintenance_action", -1)) == 0),
                "mma_failure_risk",
            ) or 0.0
        ),
    }
    action_names = ("process", "ipm", "pm")
    for name in action_names:
        result[f"mma_probability_{name}_mean"] = (
            _mean_field(records, f"mma_probability_{name}") or 0.0)
    for class_id in range(machine_class_count):
        for health_state in range(health_state_count):
            group = [
                item for item in records
                if int(item["machine_class"]) == class_id
                and int(item["machine_state"]) == health_state
            ]
            prefix = f"mma_class_{class_id}_health_{health_state}"
            result[f"{prefix}_decisions"] = len(group)
            result[f"{prefix}_entropy_mean"] = (
                _mean_field(group, "mma_policy_entropy") or 0.0)
            result[f"{prefix}_failure_risk_mean"] = (
                _mean_field(group, "mma_failure_risk") or 0.0)
            for name in action_names:
                result[f"{prefix}_{name}_probability_mean"] = (
                    _mean_field(group, f"mma_probability_{name}") or 0.0)
    return result


def decision_action_summary(
        decisions: Iterable[dict],
        *,
        machine_class_count: int | None = None,
        health_state_count: int | None = None,
        action_count: int = 3) -> dict[str, float | int]:
    records = list(decisions)
    valid = [
        item for item in records
        if item.get("maintenance_action") is not None
        and int(item["maintenance_action"]) >= 0
        and item.get("machine_class") is not None
        and item.get("machine_state") is not None
    ]
    maintenance = Counter(
        int(item["maintenance_action"]) for item in valid)
    machine_classes = Counter(
        int(item["machine_class"])
        for item in valid
    )
    health_states = Counter(int(item["machine_state"]) for item in valid)
    class_state_actions = Counter(
        (int(item["machine_class"]), int(item["machine_state"]),
         int(item["maintenance_action"]))
        for item in valid
    )
    degraded = Counter(
        int(item["maintenance_action"])
        for item in valid
        if int(item["machine_state"]) > 0
    )
    total = sum(maintenance.values())
    result: dict[str, float | int] = {
        "process_action_count": maintenance.get(0, 0),
        "ipm_action_count": maintenance.get(1, 0),
        "pm_action_count": maintenance.get(2, 0),
        "maintenance_action_rate": (
            (maintenance.get(1, 0) + maintenance.get(2, 0)) / total
            if total else 0.0
        ),
        "degraded_process_action_count": degraded.get(0, 0),
        "degraded_ipm_action_count": degraded.get(1, 0),
        "degraded_pm_action_count": degraded.get(2, 0),
        "degraded_maintenance_action_rate": (
            (degraded.get(1, 0) + degraded.get(2, 0))
            / sum(degraded.values())
            if degraded else 0.0
        ),
    }
    classes = (
        range(machine_class_count) if machine_class_count is not None
        else sorted(machine_classes)
    )
    states = (
        range(health_state_count) if health_state_count is not None
        else sorted(health_states)
    )
    for class_id in classes:
        result[f"machine_class_{class_id}_decision_count"] = (
            machine_classes.get(class_id, 0))
    for state in states:
        result[f"health_state_{state}_decision_count"] = (
            health_states.get(state, 0))
    classes = tuple(classes)
    states = tuple(states)
    for class_id in classes:
        for state in states:
            for action in range(action_count):
                result[
                    f"machine_class_{class_id}_health_state_{state}_"
                    f"action_{action}_count"
                ] = class_state_actions.get((class_id, state, action), 0)
    return result


def validation_summary(
        episode: int,
        metrics: Iterable[EpisodeMetrics]) -> dict[str, float | int | bool]:
    records = list(metrics)
    if not records:
        raise ValueError("Validation requires at least one episode.")
    result: dict[str, float | int | bool] = {
        "episode": int(episode),
        "validation_cases": len(records),
        "is_best": False,
    }
    for name in EpisodeMetrics.__dataclass_fields__:
        values = np.asarray(
            [float(getattr(item, name)) for item in records], dtype=float)
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_std"] = float(values.std())
    # Retain the established model-selection field for compatibility.
    result["mean_tardiness"] = result["mean_tardiness_mean"]
    return result


def save_training_log_bundle(
        output_dir: str | Path,
        *,
        method: str,
        episodes: list[dict],
        updates: list[dict],
        rollouts: list[dict] | None = None,
        validation: list[dict],
        metadata: dict | None = None) -> None:
    """Write normalized episode, optimizer, rollout, and validation tables."""

    output = Path(output_dir)
    metadata = dict(metadata or {})
    if method not in CANONICAL_METHOD_IDS:
        raise ValueError(f"Unregistered training-log method key: {method}")
    if "training_seed" not in metadata:
        raise ValueError("Training-log metadata requires training_seed.")
    condition = metadata.get("training_condition")
    required_condition_fields = {
        "nominal_finite_load",
        "initial_jobs_configured",
        "total_jobs_configured",
        "due_date_factor",
        "degradation_multiplier",
    }
    if not isinstance(condition, dict) or not required_condition_fields.issubset(
            condition):
        raise ValueError(
            "Training-log metadata requires the normalized training_condition.")
    method_id = CANONICAL_METHOD_IDS[method]
    identity = {
        "log_schema_version": TRAINING_LOG_SCHEMA_VERSION,
        "method_id": method_id,
        "method_key": method,
        "training_seed": int(metadata["training_seed"]),
        **{field: condition[field] for field in sorted(
            required_condition_fields)},
    }

    def identified(records: list[dict]) -> list[dict]:
        return [{**identity, **record} for record in records]

    episodes = identified(episodes)
    updates = identified(updates)
    rollout_records = identified(list(rollouts or []))
    validation = identified(validation)
    episode_path = output / "Training_Episodes.csv"
    update_path = output / "Training_Updates.csv"
    rollout_path = output / "Training_Rollouts.csv"
    validation_path = output / "Training_Validation.csv"
    atomic_csv_save(episodes, episode_path)
    atomic_csv_save(updates, update_path)
    atomic_csv_save(rollout_records, rollout_path)
    atomic_csv_save(validation, validation_path)
    manifest = {
        "schema_version": TRAINING_LOG_SCHEMA_VERSION,
        "method_id": method_id,
        "method_key": method,
        "training_seed": int(metadata["training_seed"]),
        "training_condition": condition,
        "record_counts": {
            "episodes": len(episodes),
            "updates": len(updates),
            "rollouts": len(rollout_records),
            "validation": len(validation),
        },
        "files": {
            "episodes": episode_path.name,
            "updates": update_path.name,
            "rollouts": rollout_path.name,
            "validation": validation_path.name,
        },
        "columns": {
            "episodes": list(dict.fromkeys(
                key for row in episodes for key in row)),
            "updates": list(dict.fromkeys(
                key for row in updates for key in row)),
            "rollouts": list(dict.fromkeys(
                key for row in rollout_records for key in row)),
            "validation": list(dict.fromkeys(
                key for row in validation for key in row)),
        },
        "metadata": metadata,
    }
    atomic_json_save(manifest, output / "Training_Log_Manifest.json")
