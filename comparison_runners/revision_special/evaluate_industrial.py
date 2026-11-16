"Frozen industrial deployment modes and small-instance follow-up."
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from types import MethodType
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from comparison_runners.revision_special.common import construct,parameters_for
from comparison_runners.drl_common import load_model,simulation_core
from comparison_runners.evaluate_revision_nonlearning import read,resolve,consolidate_selected
from comparison_runners.detailed_metrics import episode_row,machine_rows,state_action_rows,job_rows
from comparison_runners.industrial_case.case_common import (
    DATASET_ROOT,PROCESSING_TIMES,MACHINE_COUNTS,adapt_simulator)
from experiment.artifacts import atomic_csv_save,atomic_json_save
from experiment.instances import load_job_records_csv,JobRecord
from experiment.revision_protocol import LEARNED_METHODS,TRAINING_SEEDS
from experiment.selected_hyperparameters import load_selection,source_directory
from methods.hierarchical.environment import HierarchicalSchedulingEnv
from methods.hierarchical.config import HierarchyConfig
from methods.hierarchical.types import Job


def deployment_environment(method,directory,device):
    directory=Path(directory)
    if method=='ours_categorical_baa':
        state=torch.load(directory/'hierarchical_inference.pt',map_location='cpu',weights_only=True)
        env=HierarchicalSchedulingEnv([1.,18,89,1.,101],list(MACHINE_COUNTS),processing_times=PROCESSING_TIMES,
                                     config=HierarchyConfig(**state['config']),device=device)
        env.mappo.load_state_dict(state['mappo'],training=False)
    else:
        env=construct(method,[1.,18,89,1.,101],MACHINE_COUNTS,PROCESSING_TIMES,device=device)
        load_model(env,method,directory)
    core=simulation_core(env);adapt_simulator(core)
    return env,core


def small_loader(self,records):
    return [Job(r.job_id,r.arrival_time,r.family,r.size,r.due_time) for r in records]


def small_records(path):
    raw=load_job_records_csv(path)
    return [JobRecord(i,r.arrival_time,r.family,r.size,r.due_time) for i,r in enumerate(raw)]


def industrial_cases(dataset_dir,small=False,instance_limit=None,replications=10):
    root=Path(dataset_dir)
    if small:
        rows=[r for r in read(root/'manifest.csv') if int(r['n_jobs']) in (10,20)]
        rows=rows[:instance_limit]
        return [dict(**r,condition_id=f'jobs-{r["n_jobs"]}',instance_index=int(r['instance_number']),
            instance_seed=84010000+int(r['instance_number']),simulation_seed=52000000+rep,
            replication_index=rep) for r in rows for rep in range(replications)]
    rows=read(root/'metadata/instance_manifest.csv')[:instance_limit]
    seeds=json.loads((root/'configuration/simulation_parameters.json').read_text(encoding='utf-8'))['simulation_seeds'][:replications]
    return [dict(**r,condition_id=f'industrial_load_{float(r["target_load"]):.2f}'.replace('.','p'),
        instance_index=int(r['instance_number']),instance_seed=int(r['arrival_seed']),simulation_seed=int(seed),
        replication_index=rep) for r in rows for rep,seed in enumerate(seeds)]


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',choices=LEARNED_METHODS,default=list(LEARNED_METHODS))
    p.add_argument('--training-seeds',nargs='+',type=int,default=list(TRAINING_SEEDS))
    p.add_argument('--modes',nargs='+',choices=('zero_shot','scratch','finetune'),default=['zero_shot','scratch','finetune'])
    p.add_argument('--small',action='store_true')
    p.add_argument('--milp-results',type=Path,help='Optional small-instance solver JSONs for reference/gap fields.')
    p.add_argument('--source-model-root',type=Path,default=ROOT/'runs/e1_learned_method_training')
    p.add_argument('--source-layout',choices=('standard','grid'),default='standard')
    p.add_argument('--source-budget',type=int,default=500)
    p.add_argument('--scratch-budget',type=int,default=500)
    p.add_argument('--finetuning-budget',type=int,default=200)
    p.add_argument('--model-root',type=Path,default=ROOT/'runs/revision_industrial_training')
    p.add_argument('--hyperparameter-file',type=Path)
    p.add_argument('--dataset-dir',type=Path)
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/revision_industrial_evaluation')
    p.add_argument('--replications',type=int,default=10)
    p.add_argument('--instance-limit',type=int)
    p.add_argument('--save-job-records',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--device',default='auto')
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args()


def execute(args):
    data=resolve(args.dataset_dir) if args.dataset_dir else ROOT/'datasets/industrial_case_small' if args.small else DATASET_ROOT
    # Small-scale comparison uses scratch unless modes were explicitly changed.
    modes=['scratch'] if args.small and args.modes==['zero_shot','scratch','finetune'] else args.modes
    cases=industrial_cases(data,args.small,args.instance_limit,args.replications)
    plan=dict(domain='small_industrial' if args.small else 'industrial',methods=args.methods,modes=modes,
        training_seeds=args.training_seeds,cases_per_model=len(cases),
        runs=len(cases)*len(args.methods)*len(modes)*len(args.training_seeds))
    if args.dry_run:print(json.dumps(plan,indent=2));return plan
    selection=load_selection(resolve(args.hyperparameter_file)) if args.hyperparameter_file else {'methods':{}}
    out=resolve(args.output_dir);blocks=[]
    for mode in modes:
        for method in args.methods:
            for seed in args.training_seeds:
                directory=(source_directory(resolve(args.source_model_root),method,seed,args.source_budget,
                    selection=selection,grid_layout=args.source_layout=='grid') if mode=='zero_shot' else
                    resolve(args.model_root)/mode/method/f'seed-{seed}'/'budgets'/f'episodes-{args.scratch_budget if mode=="scratch" else args.finetuning_budget}')
                # Read the selected validation model for the declared budget.
                environment,core=deployment_environment(method,directory,args.device)
                if args.small:core._jobs_from_records=MethodType(small_loader,core)
                for condition in dict.fromkeys(r['condition_id'] for r in cases):
                    selected=[r for r in cases if r['condition_id']==condition]
                    stem=f'{mode}__{method}__seed-{seed}__{condition}__n{len(selected)}'
                    completed=out/'blocks'/(stem+'.json')
                    if completed.exists():
                        blocks.append(json.loads(completed.read_text(encoding='utf-8')));continue
                    tables={n:[] for n in ('run_metrics','machine_metrics','state_action_metrics')}
                    if args.save_job_records:tables['job_records']=[]
                    for i,row in enumerate(selected,1):
                        records=small_records(data/row['job_file']) if args.small else load_job_records_csv(data/row['job_file'])
                        core.total_jobs=len(records);core.initial_jobs=sum(r.arrival_time==0 for r in records)
                        core.target_load=float(row['target_load']);core.job_parameters=[core.target_load,core.initial_jobs,len(records),1.,int(row['instance_seed'])]
                        identity={**row,'method':method,'deployment_mode':mode,'training_seed':seed,'model_directory':str(directory)}
                        reference=milp_reference(args.milp_results,row) if args.small and args.milp_results else {}
                        identity.update(reference)
                        error=metrics=None;started=time.perf_counter()
                        try:
                            core.reset(instance_seed=int(row['instance_seed']),simulation_seed=int(row['simulation_seed']),
                                policy_seed=int(row['simulation_seed']),job_records=records)
                            metrics=core.run_episode(training=False)
                        except Exception as exc:error=exc
                        result=episode_row(identity,metrics,core,wall_time=time.perf_counter()-started,error=error)
                        add_milp_gap(result,reference,metrics)
                        tables['run_metrics'].append(result)
                        if metrics is not None:
                            tables['machine_metrics'].extend(machine_rows(identity,core,metrics['makespan']))
                            tables['state_action_metrics'].extend(state_action_rows(identity,core))
                            if args.save_job_records:tables['job_records'].extend(job_rows(identity,core))
                        if i==1 or i==len(selected) or error or args.log_every and i%args.log_every==0:
                            print(f'[INDUSTRIAL-EVAL] {stem} {i}/{len(selected)} TTD={metrics["total_tardiness"] if metrics else "-"} error={error or ""}',flush=True)
                    files={}
                    for name,rows in tables.items():
                        if rows:
                            path=out/'partitions'/name/(stem+'.csv');atomic_csv_save(rows,path);files[name]=str(path.relative_to(out))
                    block=dict(block=stem,runs=len(selected),failed_runs=sum(r['status']!='completed' for r in tables['run_metrics']),files=files)
                    atomic_json_save(block,completed);blocks.append(block)
    for name in ('run_metrics','machine_metrics','state_action_metrics','job_records'):
        if name!='job_records' or args.save_job_records:consolidate_selected(out,blocks,name)
    atomic_json_save(dict(plan=plan,blocks=blocks),out/'completion_summary.json')
    return blocks


def milp_reference(directory,row):
    path=resolve(directory)/f'jobs-{int(row["n_jobs"])}_instance-{int(row["instance_number"]):03d}_milp.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    return dict(milp_reference_ttd=data.get('objective'),milp_lower_bound=data.get('obj_bound'),
        milp_solver_seconds=data.get('elapsed_seconds'),milp_status=data.get('status_name'),
        milp_relative_solver_gap=data.get('mip_gap'),milp_proven_optimal=bool(data.get('is_proven_global_production_optimum',data.get('is_optimal',False))),
        milp_reference_type='no_incumbent' if data.get('objective') is None else
        'proven_optimum' if data.get('is_proven_global_production_optimum',data.get('is_optimal',False)) else 'time_limited_incumbent')


def add_milp_gap(result,reference,metrics):
    value=reference.get('milp_reference_ttd')
    result['gap_to_milp_reference_percent']=100*(metrics['total_tardiness']/value-1) if metrics and value is not None and value>0 else ''


if __name__=='__main__':execute(parse_args())
