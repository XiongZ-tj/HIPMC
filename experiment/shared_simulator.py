"""The only simulator factory permitted for comparative experiments."""

from __future__ import annotations

from dataclasses import replace

from methods.hierarchical.config import HierarchyConfig
from methods.hierarchical.environment import HierarchicalSchedulingEnv
from methods.hierarchical.types import HandshakeMode


def build_revision_simulator(*, scene, job_records, job_parameters,
        processing_times, controller=None, config=None, device="auto",
        handshake_mode=HandshakeMode.INTERACTIVE.value,
        instance_seed=None, simulation_seed=None, policy_seed=None):
    "Opt-in factory on unchanged frozen orders for the scenes."
    from experiment.revision_scenarios import apply_revision_scene
    simulator = build_shared_simulator(job_parameters=job_parameters,
        machine_parameters=[2,2,1],processing_times=processing_times,
        controller=controller,config=config,device=device,
        degradation_multiplier=(config.degradation_multiplier if config else 1.0),
        handshake_mode=handshake_mode)
    return apply_revision_scene(simulator,scene,job_records=job_records,
        instance_seed=instance_seed,simulation_seed=simulation_seed,policy_seed=policy_seed)


def build_shared_simulator(
        *,
        job_parameters: list[float | int],
        machine_parameters: list[int],
        processing_times: tuple[int, ...],
        controller: object | None = None,
        handshake_mode: str = HandshakeMode.INTERACTIVE.value,
        config: HierarchyConfig | None = None,
        degradation_multiplier: float = 1.0,
        device: str = "auto",
) -> HierarchicalSchedulingEnv:
    """Construct the shared event-driven job/machine/degradation simulator."""

    effective_config = replace(
        config or HierarchyConfig(),
        degradation_multiplier=float(degradation_multiplier),
    )
    return HierarchicalSchedulingEnv(
        job_parameters,
        machine_parameters,
        processing_times=processing_times,
        handshake_mode=HandshakeMode(handshake_mode),
        config=effective_config,
        controller=controller,
        device=device,
    )
