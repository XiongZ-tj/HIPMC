"Shared model construction, transfer and training helpers."
from dataclasses import replace
import importlib
from pathlib import Path
import csv
import time
from comparison_runners.drl_common import load_model,simulation_core,checkpoint_name
from comparison_runners.training_budgets import configure_learning,run_training_budget
from experiment.artifacts import atomic_json_save,atomic_csv_save
from experiment.device import resolve_torch_device
from experiment.selected_hyperparameters import selected_parameters


def construct(method,parameters,counts,processing_times,*,device,scale=None,scale_sha=None):
    module=importlib.import_module('methods.hierarchical' if method in ('ours','ours_categorical_baa')
                                 else 'methods.baselines.'+method+'.shared')
    module.configure_scenario(processing_times)
    return module.env(parameters,list(counts),device=str(resolve_torch_device(device)),
        calibrated_reward_scale=scale,reward_scale_artifact_sha256=scale_sha)


def warm_start(environment,method,directory):
    load_model(environment,method,Path(directory))
    controller=getattr(environment,'controller',None)
    if hasattr(controller,'q1'):controller.q2.load_state_dict(controller.q1.state_dict())
    if hasattr(controller,'nets'):
        for nets in controller.nets.values():nets['q2'].load_state_dict(nets['q1'].state_dict())


def parameters_for(environment,selection,method):
    return configure_learning(environment,selected_parameters(selection,method))


def finish_training(environment,method,output,*,episodes,contract,reference_episodes=500):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    atomic_json_save(contract,output/'run_config.json')
    path=output/'training_runtime.csv'
    timings=[]
    if path.exists():
        with path.open(encoding='utf-8',newline='') as stream:timings=list(csv.DictReader(stream))
    started=time.perf_counter();status='interrupted_or_failed'
    try:
        run_training_budget(environment,method,output,episodes,reference_episodes=reference_episodes)
        status='complete'
    finally:
        timings.append(dict(invocation=len(timings)+1,requested_episodes=episodes,
            completed_episodes=environment.total_episode,wall_seconds=time.perf_counter()-started,status=status))
        atomic_csv_save(timings,path)
    atomic_json_save(dict(status='complete',episodes=environment.total_episode,
        best_validation_mean_tardiness=environment.best_validation,
        training_and_validation_wall_seconds=sum(float(r['wall_seconds']) for r in timings)),output/'run_complete.json')
