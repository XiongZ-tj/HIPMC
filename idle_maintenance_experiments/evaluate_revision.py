"""Evaluate the empty-queue maintenance extension against the existing HIPMC results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment.revision_protocol import TEST_INSTANCES_PER_CONDITION
from experiment.revision_selection import selected_conditions
from idle_maintenance_experiments.environment import IdleMaintenanceEnv,PROTOCOLS
from comparison_runners.evaluate_revision_nonlearning import read,resolve,consolidate_selected
from comparison_runners.detailed_metrics import episode_row,machine_rows,state_action_rows,job_rows
from experiment.artifacts import atomic_csv_save,atomic_json_save
from experiment.instances import load_job_records_csv
from experiment.evaluation import EvaluationCase
from experiment.revision_protocol import TRAINING_SEEDS
from experiment.revision_scenarios import REVISION_SCENES,apply_revision_scene
from methods.hierarchical.config import HierarchyConfig


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocols',nargs='+',choices=('idle_opportunity',),default=['idle_opportunity'])
    p.add_argument('--training-seeds',nargs='+',type=int,default=list(TRAINING_SEEDS))
    p.add_argument('--scenes',nargs='+',choices=tuple(REVISION_SCENES),default=list(REVISION_SCENES))
    p.add_argument('--model-root',type=Path,default=ROOT/'runs/revision_idle_maintenance_training')
    p.add_argument('--model-budget',type=int,default=500)
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/revision_idle_maintenance_evaluation')
    p.add_argument('--dataset-dir',type=Path,default=ROOT/'datasets/e1_formal_instances')
    p.add_argument('--case-manifest',type=Path,default=ROOT/'experiment_manifests/e1_formal_comparison_cases.csv')
    p.add_argument('--device',default='auto')
    p.add_argument('--instances',type=int,default=TEST_INSTANCES_PER_CONDITION)
    p.add_argument('--replications',type=int,default=10)
    p.add_argument('--condition-limit',type=int)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--save-job-records',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args()


def execute(args):
    data=resolve(args.dataset_dir);out=resolve(args.output_dir)
    structural=read(data/'instance_manifest.csv')
    all_ids=list(dict.fromkeys(r['condition_id'] for r in structural))
    ids_by_scene={s:selected_conditions(all_ids,s,limit=args.condition_limit) for s in args.scenes}
    conditions=list(dict.fromkeys(c for ids in ids_by_scene.values() for c in ids))
    cases=[r for r in read(resolve(args.case_manifest)) if r['condition_id'] in conditions
           and int(r['instance_index'])<args.instances and int(r['replication_index'])<args.replications]
    plan=dict(protocols=args.protocols,seeds=args.training_seeds,scenes=args.scenes,
              conditions_by_scene=ids_by_scene,
              runs=len(args.protocols)*len(args.training_seeds)*sum(sum(r['condition_id'] in ids for r in cases) for ids in ids_by_scene.values()))
    if args.dry_run:print(json.dumps(plan,indent=2));return plan
    sources={(r['condition_id'],int(r['instance_index'])):r for r in structural}
    blocks=[]
    for protocol in args.protocols:
        for seed in args.training_seeds:
            model=resolve(args.model_root)/protocol/f'seed-{seed}'/'budgets'/f'episodes-{args.model_budget}'/'hierarchical_inference.pt'
            state=torch.load(model,map_location='cpu',weights_only=True)
            first=cases[0]
            env=IdleMaintenanceEnv([float(first['nominal_finite_load']),int(first['initial_jobs']),int(first['total_jobs']),
                float(first['due_date_factor']),int(first['instance_seed'])],[2,2,1],processing_times=(190,158,202,128,104),
                config=HierarchyConfig(**state['config']),device=args.device,maintenance_protocol=protocol)
            env.mappo.load_state_dict(state['mappo'],training=False)
            for scene_id in args.scenes:
                for condition in ids_by_scene[scene_id]:
                    selected=[r for r in cases if r['condition_id']==condition]
                    key=hashlib.sha256(json.dumps([(r['instance_seed'],r['simulation_seed']) for r in selected]).encode()).hexdigest()[:10]
                    stem=f'{scene_id}__{protocol}__seed-{seed}__budget-{args.model_budget}__{condition}__{key}'
                    completed=out/'blocks'/(stem+'.json')
                    if completed.exists():blocks.append(json.loads(completed.read_text(encoding='utf-8')));continue
                    tables={n:[] for n in ('run_metrics','machine_metrics','state_action_metrics','decisions','idle_maintenance','wait_intervals')}
                    if args.save_job_records:tables['job_records']=[]
                    for i,row in enumerate(selected,1):
                        identity={**row,'protocol':protocol,'training_seed':seed,'scene_id':scene_id,'model_file':str(model)}
                        metrics=error=None;started=time.perf_counter()
                        try:
                            apply_revision_scene(env,scene_id,job_records=load_job_records_csv(data/sources[(condition,int(row['instance_index']))]['job_file']),
                                instance_seed=int(row['instance_seed']),simulation_seed=int(row['simulation_seed']),policy_seed=int(row['simulation_seed']))
                            metrics=env.run_episode(training=False)
                        except Exception as exc:error=exc
                        result=episode_row(identity,metrics,env,wall_time=time.perf_counter()-started,error=error)
                        if metrics:
                            result.update({k:v for k,v in metrics.items() if k.startswith(('idle_','mma_wait_'))})
                            # Report actual executed maintenance, not the unexecuted
                            # tail of an idle action after the last job completes.
                            original=[(m.maintenance_time,m.planned_maintenance_time,m.corrective_maintenance_time) for m in env.machines]
                            try:
                                for m in env.machines:
                                    planned,corrective=env._realized_maintenance(m)
                                    m.maintenance_time=planned+corrective;m.planned_maintenance_time=planned;m.corrective_maintenance_time=corrective
                                tables['machine_metrics'].extend(machine_rows(identity,env,metrics['makespan']))
                            finally:
                                for m,values in zip(env.machines,original):m.maintenance_time,m.planned_maintenance_time,m.corrective_maintenance_time=values
                            tables['state_action_metrics'].extend(state_action_rows(identity,env))
                            tables['decisions'].extend({**identity,**r} for r in env.decision_history)
                            tables['idle_maintenance'].extend({**identity,**r} for r in env.idle_interventions)
                            tables['wait_intervals'].extend({**identity,**r} for r in env.wait_intervals)
                            if args.save_job_records:tables['job_records'].extend(job_rows(identity,env))
                        tables['run_metrics'].append(result)
                        if i==1 or i==len(selected) or error or args.log_every and i%args.log_every==0:
                            print(f'[IDLE-EVAL] {stem} {i}/{len(selected)} status={result["status"]} error={error or ""}',flush=True)
                    files={}
                    for name,rows in tables.items():
                        if rows:
                            path=out/'partitions'/name/(stem+'.csv');atomic_csv_save(rows,path);files[name]=str(path.relative_to(out))
                    block=dict(block=stem,runs=len(selected),failed_runs=sum(r['status']!='completed' for r in tables['run_metrics']),files=files)
                    atomic_json_save(block,completed);blocks.append(block)
    for name in ('run_metrics','machine_metrics','state_action_metrics','decisions','idle_maintenance','wait_intervals','job_records'):
        if name=='job_records' and not args.save_job_records:continue
        consolidate_selected(out,blocks,name)
    atomic_json_save(dict(plan=plan,blocks=blocks),out/'completion_summary.json')
    return blocks


if __name__=='__main__':execute(parse_args())
