"Compile scene-specific native planners from the shared physical parameters."
from pathlib import Path
import json
import re
import shutil
import subprocess
from experiment.revision_scenarios import get_revision_scene

ROOT=Path(__file__).resolve().parents[1]
MA_SOURCE=ROOT/'comparison_runners/offline_memetic_ma/cpp/offline_memetic_planner.cpp'
RH_SOURCE=ROOT/'comparison_runners/rolling_horizon_alns/planner.cpp'


def initializer(values):
    return '{'+','.join(initializer(x) if isinstance(x,(list,tuple)) else
                        format(x,'.17g') for x in values)+'}'


def scene_source(scene,*,processing_times=None,job_size_max=None):
    scene=get_revision_scene(scene)
    cfg=scene.physics()
    classes=[c for c,n in enumerate(scene.machine_counts) for _ in range(n)]
    capacities=[cap for cap,n in zip(scene.machine_capacities,scene.machine_counts) for _ in range(n)]
    arrays={
        'kMachineClass':classes,'kMachineCapacity':capacities,
        'kIpmTime':cfg.ipm_times,'kPmTime':cfg.pm_times,
        'kMinimalRepairTime':cfg.minimal_repair_times,
        'kCorrectiveRange':cfg.corrective_time_ranges,
        'kDegradationRange':cfg.degradation_rate_ranges,
        'kTerminalDegradationRate':cfg.terminal_degradation_rates,
        'kFailureRange':cfg.failure_rate_ranges,
        'kFailureProportion':cfg.failure_type_proportions,
        'kIpmRecoveryRate':cfg.ipm_recovery_rates,
    }
    source=MA_SOURCE.read_text(encoding='utf-8')
    if len(classes)!=5:
        source=source.replace('constexpr int kMachineCount = 5;',f'constexpr int kMachineCount = {len(classes)};')
        source=source.replace('current.machine = 4;',f'current.machine = {len(classes)-1};')
    if processing_times is not None:
        source=source.replace('constexpr int kFamilyCount = 5;',f'constexpr int kFamilyCount = {len(processing_times)};')
        source=re.sub(r'(kProcessingTime)\s*\{.*?\};',
            lambda m:m[1]+initializer(processing_times)+';',source,count=1,flags=re.S)
    if job_size_max is not None:source=source.replace('job.size > 30',f'job.size > {job_size_max}')
    for name,values in arrays.items():
        text=initializer(values)
        if isinstance(values[0],(list,tuple)):text='{'+text+'}'
        source=re.sub(r'('+re.escape(name)+r')\s*\{.*?\};',
                      lambda match:match[1]+text+';',source,count=1,flags=re.S)
    capacity=max(scene.machine_capacities)
    source=source.replace('for (int capacity : {60, 80, 100})',
                          'for (int capacity : '+initializer(scene.machine_capacities)+')')
    source=source.replace('batch_size(merged, instance_.jobs) <= 100',
                          f'batch_size(merged, instance_.jobs) <= {capacity}')
    source=source.replace('size + next <= 100',f'size + next <= {capacity}')
    source=source.replace('kMachineCapacity[machine] - size) / 100.0',
                          f'kMachineCapacity[machine] - size) / {float(capacity)}')
    return source


def compile_scene(scene, directory, *, ma=True, rh=True,processing_times=None,job_size_max=None):
    scene=get_revision_scene(scene)
    directory=Path(directory)/scene.scene_id
    directory.mkdir(parents=True,exist_ok=True)
    generated=directory/'offline_memetic_scene.cpp'
    generated.write_text(scene_source(scene,processing_times=processing_times,job_size_max=job_size_max),encoding='utf-8')
    (directory/'scene_parameters.json').write_text(json.dumps(scene.parameters(),indent=2),encoding='utf-8')
    compiler=shutil.which('g++') or shutil.which('clang++')
    if not compiler:raise RuntimeError('A C++20 compiler is required in PATH.')
    results={}
    for method,enabled in [('ma',ma),('rh_alns',rh)]:
        if not enabled:continue
        source=generated
        if method=='rh_alns':
            source=directory/'rh_alns_scene.cpp'
            source.write_text(RH_SOURCE.read_text(encoding='utf-8').replace(
                '#include "../offline_memetic_ma/cpp/offline_memetic_planner.cpp"',
                '#include "'+generated.resolve().as_posix()+'"'),encoding='utf-8')
        binary=directory/(method+'.exe')
        print(f'[NATIVE-SCENE] compiling {scene.scene_id} {method}',flush=True)
        subprocess.run([compiler,'-std=c++20','-O3','-DNDEBUG','-pthread',str(source),'-o',str(binary)],check=True)
        results[method]=binary
    return results
