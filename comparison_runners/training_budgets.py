"""Budget snapshots and restartable training continuation."""
import json
from dataclasses import replace
from pathlib import Path
import shutil
import csv
import time
from experiment.artifacts import atomic_json_save,atomic_csv_save
from experiment.revision_protocol import STANDARD_EPISODES
from .drl_common import checkpoint_name


def configure_learning(environment, parameters):
    """Apply explicitly selected parameters, retaining algorithm-specific defaults."""
    parameters={k:v for k,v in parameters.items() if v is not None}
    core=getattr(environment,'core',environment)
    agent=getattr(environment,'mappo',getattr(environment,'controller',None))
    changes={}
    for key in ('learning_rate','clip_ratio','gamma'):
        if key not in parameters:continue
        value=float(parameters[key])
        if key=='gamma':
            changes.update(gamma_baa=value,gamma_mma=value)
            for name in ('gamma','gamma_baa','gamma_mma'):
                if hasattr(agent,name):setattr(agent,name,value)
        elif key=='clip_ratio':
            changes['ppo_clip_ratio']=value
            if hasattr(agent,'clip_ratio'):agent.clip_ratio=value
        else:
            changes['learning_rate']=value
            optimizers=[]
            if hasattr(agent,'optimizer'):optimizers.append(agent.optimizer)
            if hasattr(agent,'nets'):
                optimizers.extend(net['optimizer'] for net in agent.nets.values())
            for optimizer in optimizers:
                for group in optimizer.param_groups:group['lr']=value
    core.config=replace(core.config,**changes)
    effective=dict(learning_rate=core.config.learning_rate if hasattr(environment,'mappo') else
        next(iter(agent.optimizer.param_groups))['lr'] if hasattr(agent,'optimizer') else
        next(iter(agent.nets.values()))['optimizer'].param_groups[0]['lr'],
        clip_ratio=getattr(agent,'clip_ratio',None),
        gamma_baa=getattr(agent,'gamma_baa',getattr(agent,'gamma',None)),
        gamma_mma=getattr(agent,'gamma_mma',getattr(agent,'gamma',None)),
        discount_protocol='fixed_per_decision_transition')
    if hasattr(agent, 'replay_update_interval'):
        effective.update(replay_update_interval=agent.replay_update_interval,
            replay_update_unit='new_stored_transitions_per_network',
            target_sync_steps=agent.target_sync_steps,
            target_sync_unit='optimizer_steps_per_network',
            optimizer_execution='episode_end')
    if hasattr(environment,'training_log_metadata'):
        environment.training_log_metadata.update(learning_rate=effective['learning_rate'],
            gamma=effective['gamma_baa'],clip_ratio=effective['clip_ratio'],
            applied_hyperparameters=effective)
    return effective


def snapshot_budget(run_dir, episodes):
    """Copy completed budget evidence once; never replace an earlier snapshot."""
    run_dir=Path(run_dir)
    target=run_dir/'budgets'/f'episodes-{episodes}'
    if (target/'budget_complete.json').exists():return target
    target.mkdir(parents=True,exist_ok=True)
    for source in run_dir.iterdir():
        if source.is_file() and source.name!='run_complete.json':
            shutil.copy2(source,target/source.name)
    config=target/'run_config.json'
    if config.exists():
        data=json.loads(config.read_text(encoding='utf-8'))
        data['episodes']=episodes
        data['selected_budget']=episodes
        atomic_json_save(data,config)
    atomic_json_save({'episodes':episodes,'selection':'best_validation_up_to_budget'},
                     target/'budget_complete.json')
    return target


def run_training_budget(environment, method, run_dir, episodes,
                        reference_episodes=STANDARD_EPISODES):
    run_dir=Path(run_dir)
    checkpoint=run_dir/checkpoint_name(method)
    if checkpoint.exists():
        if hasattr(environment,'mappo'):
            environment.load_checkpoint(checkpoint,load_training=True)
        else:environment.load_checkpoint(checkpoint)
    # A fixed original horizon keeps the first 500 episodes identical and
    # holds the terminal entropy coefficient during later continuation.
    if hasattr(environment,'training_horizon'):
        environment.training_horizon=reference_episodes
    if environment.total_episode==reference_episodes:
        snapshot_budget(run_dir,reference_episodes)
        freeze_budget_runtime(run_dir,reference_episodes)
    milestones=sorted({min(reference_episodes,episodes),episodes})
    for stop in milestones:
        if environment.total_episode>=stop:continue
        environment.max_episode=stop
        timing_path=run_dir/'training_segments.csv'
        timing=[]
        if timing_path.exists():
            with timing_path.open(encoding='utf-8',newline='') as stream:timing=list(csv.DictReader(stream))
        started=time.perf_counter();completed=False;start_episode=environment.total_episode
        try:
            environment.main(run_dir,resume_from=checkpoint if checkpoint.exists() else None)
            completed=True
        finally:
            timing.append(dict(requested_stop=stop,start_episode=start_episode,completed_episode=environment.total_episode,
                wall_seconds=time.perf_counter()-started,completed=completed))
            atomic_csv_save(timing,timing_path)
        snapshot_budget(run_dir,stop)
        freeze_budget_runtime(run_dir,stop)
    return environment.total_episode


def freeze_budget_runtime(run_dir,episodes):
    """Preserve measured training+validation cost before later continuation."""
    target=Path(run_dir)/'budgets'/f'episodes-{episodes}'/'training_budget_runtime.json'
    if target.exists():return
    path=Path(run_dir)/'training_segments.csv';seconds=None
    if path.exists():
        with path.open(encoding='utf-8',newline='') as stream:
            selected=[r for r in csv.DictReader(stream) if int(r['requested_stop'])<=episodes]
        if selected:seconds=sum(float(r['wall_seconds']) for r in selected)
    if seconds is None:
        marker=Path(run_dir)/'run_complete.json'
        data=json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else {}
        if int(data.get('episodes',-1))==episodes:seconds=data.get('elapsed_seconds')
    atomic_json_save(dict(episodes=episodes,training_validation_wall_seconds=seconds,
        scope='measured training segments including validation and resumed attempts; excludes instance preparation',
        available=seconds is not None),target)


def budget_model_dir(root,method,seed,budget=None):
    directory=Path(root)/method/f'seed-{seed}'
    return directory if budget is None else directory/'budgets'/f'episodes-{budget}'
