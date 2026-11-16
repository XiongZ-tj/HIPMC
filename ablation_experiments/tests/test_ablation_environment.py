"""Shared small-instance fixture for component-replacement tests."""
from ablation_experiments.common import build_ablation_environment
from experiment.config import ConditionConfig


def small_environment(variant):
    condition = ConditionConfig(target_load=1., initial_jobs=4, total_jobs=20,
        due_date_factor=2.5, degradation_multiplier=1., seed=101)
    return build_ablation_environment(variant, condition, device="cpu",
        calibrated_reward_scale=1., reward_scale_artifact_sha256="test-artifact")[0]
