"""Replay existing small-instance MILP schedules with corrective maintenance only."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import MethodType
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from comparison_runners.revision_special.evaluate_industrial import (
    industrial_cases,small_records,small_loader,milp_reference,add_milp_gap)
from comparison_runners.industrial_case.milp_plan_executor import MILPPlanExecutorEnv,plan_from_milp
from comparison_runners.industrial_case.case_common import PROCESSING_TIMES,adapt_simulator
from comparison_runners.evaluate_revision_nonlearning import read,resolve,consolidate_selected
from comparison_runners.detailed_metrics import episode_row,machine_rows,state_action_rows,job_rows
from experiment.artifacts import atomic_csv_save,atomic_json_save


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--milp-results',type=Path,default=ROOT/'runs/milp_small_industrial')
    p.add_argument('--dataset-dir',type=Path,default=ROOT/'datasets/industrial_case_small')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/revision_small_milp_execution')
    p.add_argument('--instance-limit',type=int)
    p.add_argument('--replications',type=int,default=10)
    p.add_argument('--device',default='cpu')
    p.add_argument('--save-job-records',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args()


def execute(args):
    data=resolve(args.dataset_dir);out=resolve(args.output_dir)
    cases=industrial_cases(data,True,args.instance_limit,args.replications)
    plan=dict(method='MILP+CM',runs=len(cases),replications=args.replications)
    if args.dry_run:print(json.dumps(plan,indent=2));return plan
    blocks=[]
    for key in dict.fromkeys((int(r['n_jobs']),int(r['instance_number'])) for r in cases):
        selected=[r for r in cases if (int(r['n_jobs']),int(r['instance_number']))==key]
        stem=f'jobs-{key[0]}__instance-{key[1]}__r{len(selected)}'
        completed=out/'blocks'/(stem+'.json')
        if completed.exists():blocks.append(json.loads(completed.read_text(encoding='utf-8')));continue
        source=selected[0];path=resolve(args.milp_results)/f'jobs-{key[0]}_instance-{key[1]:03d}_milp.json'
        solution=json.loads(path.read_text(encoding='utf-8'))
        original=read(data/source['job_file']);mapping={int(r['job_id']):i for i,r in enumerate(original)}
        queues=plan_from_milp({**solution,'schedule':solution.get('schedule') or []})
        queues={m:[dict(family=b['family'],job_ids=tuple(mapping[j] for j in b['job_ids'])) for b in batches]
                for m,batches in queues.items()}
        core=MILPPlanExecutorEnv([1.,int(source['initial_job_count']),key[0],1.,int(source['instance_seed'])],[1,1,1],
            plan_batches=queues,processing_times=PROCESSING_TIMES,device=args.device)
        adapt_simulator(core);core._jobs_from_records=MethodType(small_loader,core)
        records=small_records(data/source['job_file'])
        reference=milp_reference(args.milp_results,source)
        tables={name:[] for name in ('run_metrics','machine_metrics','state_action_metrics')}
        if args.save_job_records:tables['job_records']=[]
        for row in selected:
            identity={**row,**reference,'method':'MILP+CM'}
            metrics=error=None;started=time.perf_counter()
            try:
                core.reset(instance_seed=int(row['instance_seed']),simulation_seed=int(row['simulation_seed']),
                    policy_seed=int(row['simulation_seed']),job_records=records)
                metrics=core.run_episode(training=False)
            except Exception as exc:error=exc
            result=episode_row(identity,metrics,core,wall_time=time.perf_counter()-started,error=error)
            add_milp_gap(result,reference,metrics);tables['run_metrics'].append(result)
            if metrics:
                tables['machine_metrics'].extend(machine_rows(identity,core,metrics['makespan']))
                tables['state_action_metrics'].extend(state_action_rows(identity,core))
                if args.save_job_records:tables['job_records'].extend(job_rows(identity,core))
            print(f'[SMALL-MILP] {stem} seed={row["simulation_seed"]} status={result["status"]} error={error or ""}',flush=True)
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


if __name__=='__main__':execute(parse_args())
