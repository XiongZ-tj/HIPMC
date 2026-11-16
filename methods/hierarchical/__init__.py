"HIPMC environment construction and machine configuration."

from __future__ import annotations

from .config import (
    OFFLINE_CALIBRATED_SDR_NORMALIZATION,
    HierarchyConfig,
    Scenario,
)
from .environment import HierarchicalSchedulingEnv
from .types import HandshakeMode


job_cluster_pt = [190, 158, 202, 128, 104]


def configure_scenario(processing_times: tuple[int, ...] | list[int]) -> None:
    global job_cluster_pt
    job_cluster_pt = list(processing_times)


def env(
        job_parameters: list[float | int],
        machine_parameters: list[int],
        *,
        handshake_mode: str = HandshakeMode.INTERACTIVE.value,
        degradation_multiplier: float = 1.0,
        device: str = "auto",
        reward_normalization_mode: str = OFFLINE_CALIBRATED_SDR_NORMALIZATION,
        calibrated_reward_scale: float | None = None,
        reward_scale_artifact_sha256: str | None = None,
        learning_rate: float = 2e-4,
        ppo_clip_ratio: float = 0.2,
        gamma_baa: float = 0.99,
        gamma_mma: float = 0.99,
) -> HierarchicalSchedulingEnv:
    return HierarchicalSchedulingEnv(
        job_parameters,
        machine_parameters,
        processing_times=tuple(job_cluster_pt),
        handshake_mode=HandshakeMode(handshake_mode),
        config=HierarchyConfig(
            degradation_multiplier=degradation_multiplier,
            reward_normalization_mode=reward_normalization_mode,
            calibrated_reward_scale=calibrated_reward_scale,
            reward_scale_artifact_sha256=reward_scale_artifact_sha256,
            learning_rate=learning_rate,
            ppo_clip_ratio=ppo_clip_ratio,
            gamma_baa=gamma_baa,
            gamma_mma=gamma_mma,
        ),
        device=device,
    )


__all__ = [
    "HandshakeMode",
    "HierarchyConfig",
    "HierarchicalSchedulingEnv",
    "OFFLINE_CALIBRATED_SDR_NORMALIZATION",
    "Scenario",
    "configure_scenario",
    "env",
]
