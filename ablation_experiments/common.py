"""Shared model-loading and environment helpers for component replacement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiment.config import SCENARIOS, ConditionConfig
from experiment.device import resolve_torch_device

from .environment import AblationSchedulingEnv
from .variants import AblationVariant, variant_spec

WORKSPACE = Path(__file__).resolve().parents[1]
ABLATION_ROOT = Path(__file__).resolve().parent
MODEL_ARTIFACT = "hierarchical_inference.pt"
CHECKPOINT_ARTIFACT = "hierarchical_checkpoint.pt"


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ablation_source_fingerprint() -> str:
    """Hash the component-replacement source files."""

    digest = hashlib.sha256()
    paths = sorted(
        (path for path in ABLATION_ROOT.rglob("*.py")
         if "__pycache__" not in path.parts),
        key=lambda path: path.relative_to(ABLATION_ROOT).as_posix(),
    )
    for path in paths:
        name = path.relative_to(ABLATION_ROOT).as_posix()
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


def build_ablation_environment(
        variant: str | AblationVariant,
        condition: ConditionConfig,
        *,
        device: str,
        calibrated_reward_scale: float | None,
        reward_scale_artifact_sha256: str | None,
        hyperparameters: dict | None = None,
        composite_weights=None):
    scenario = SCENARIOS["base"]
    resolved_device = str(resolve_torch_device(device))
    environment = AblationSchedulingEnv.create(
        variant=variant,
        job_parameters=condition.as_job_parameters(),
        machine_parameters=list(scenario.machine_counts),
        processing_times=scenario.job_processing_times,
        degradation_multiplier=condition.degradation_multiplier,
        device=resolved_device,
        calibrated_reward_scale=calibrated_reward_scale,
        reward_scale_artifact_sha256=reward_scale_artifact_sha256,
        hyperparameters=hyperparameters,
        composite_weights=composite_weights,
    )
    return environment, resolved_device


def load_ablation_model(
        environment: AblationSchedulingEnv,
        model_dir: str | Path) -> dict[str, str]:
    """Load the retained trained-model parameters for component evaluation."""
    path = Path(model_dir).resolve() / MODEL_ARTIFACT
    environment.load_inference(path)
    return {MODEL_ARTIFACT: file_sha256(path)}
