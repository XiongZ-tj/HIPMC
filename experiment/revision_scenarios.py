"Physical parameters for base, degradation-stress and equipment scenarios."
from dataclasses import asdict, dataclass, replace
from .config import iter_full_factorial


@dataclass(frozen=True)
class RevisionScene:
    scene_id: str
    description: str
    degradation_factor: float = 1.0
    failure_factor: float = 1.0
    maintenance_factor: float = 1.0
    machine_capacities: tuple[int, ...] = (60, 80, 100)
    machine_counts: tuple[int, ...] = (2, 2, 1)

    @property
    def total_capacity(self):
        return sum(c*n for c,n in zip(self.machine_capacities,self.machine_counts))

    def effective_nominal_load(self, reference_load):
        return float(reference_load)*380.0/self.total_capacity

    def physics(self, base_config=None):
        from methods.hierarchical.config import HierarchyConfig
        base = base_config if base_config is not None else HierarchyConfig()
        def ranges(values, factor):
            return tuple(tuple(float(x)*factor for x in row) for row in values)
        def vector(values, factor):
            return tuple(float(x)*factor for x in values)
        return replace(base,
            degradation_rate_ranges=ranges(base.degradation_rate_ranges,self.degradation_factor),
            terminal_degradation_rates=vector(base.terminal_degradation_rates,self.degradation_factor),
            failure_rate_ranges=ranges(base.failure_rate_ranges,self.failure_factor),
            ipm_times=vector(base.ipm_times,self.maintenance_factor),
            pm_times=vector(base.pm_times,self.maintenance_factor),
            corrective_time_ranges=ranges(base.corrective_time_ranges,self.maintenance_factor),
            minimal_repair_times=vector(base.minimal_repair_times,self.maintenance_factor))

    def parameters(self, base_config=None):
        """Return JSON-serializable scene parameters and the effective physical model."""
        return {**asdict(self), 'total_capacity':self.total_capacity,
                'reference_total_capacity':380,
                'physics':self.physics(base_config).to_dict()}


REVISION_SCENES = {
    'S0':RevisionScene('S0','Reference conditions'),
    'S1':RevisionScene('S1','Accelerated degradation and breakdowns',2.,2.),
    'S2':RevisionScene('S2','Longer maintenance durations',maintenance_factor=2.),
    'S3':RevisionScene('S3','Accelerated degradation and longer maintenance',2.,2.,2.),
    'S4':RevisionScene('S4','Increased class-specific capacities',machine_capacities=(70,90,120)),
    'S5':RevisionScene('S5','Changed machine-class composition',machine_counts=(2,1,2)),
}


def get_revision_scene(scene):
    return REVISION_SCENES[scene] if isinstance(scene,str) else scene


def iter_revision_conditions():
    """27 S0 and 25 generalization combinations, preserving original seeds."""
    from .revision_selection import selected_conditions
    from comparison_runners.common_e1 import e1_condition_id
    for scene in REVISION_SCENES.values():
        for condition in iter_full_factorial():
            if selected_conditions([e1_condition_id(condition)],scene.scene_id):
                yield scene,condition


def machine_stream_seed(simulation_seed, class_id, local_id):
    "Stable class-local seed for native/per-machine stream adapters."
    import numpy as np
    return int(np.random.SeedSequence([int(simulation_seed),int(class_id),
        int(local_id),927]).generate_state(1,dtype=np.uint32)[0])


def apply_revision_scene(simulator, scene, *, job_records,
                         instance_seed=None, simulation_seed=None, policy_seed=None):
    """Configure a fresh episode on supplied frozen records, without rebatching.

    Works with the simulator or a baseline wrapper exposing ``core``. Capture
    the reference config once to avoid compounded multipliers on scene changes.
    Call only between episodes. No order-generation or on-disk writes occur.
    """
    scene=get_revision_scene(scene)
    core=getattr(simulator,'core',simulator)
    if not hasattr(core,'_revision_reference_config'):
        core._revision_reference_config=core.config
    core.config=scene.physics(core._revision_reference_config)
    core.scenario=replace(core.scenario,machine_counts=scene.machine_counts,
                          machine_capacities=scene.machine_capacities)
    core.machine_parameters=list(scene.machine_counts)
    if core is not simulator:
        simulator.machine_parameters=list(scene.machine_counts)
    core.revision_scene_id=scene.scene_id
    core.reset(instance_seed=instance_seed,simulation_seed=simulation_seed,
               policy_seed=policy_seed,job_records=job_records)
    return core
