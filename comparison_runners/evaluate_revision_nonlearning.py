"Scene-aware SDR18, offline MA, rolling-horizon ALNS and tuned composite rule."
import argparse
import csv
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment.revision_protocol import TEST_INSTANCES_PER_CONDITION
from experiment.revision_selection import selected_conditions
from experiment.artifacts import atomic_csv_save,atomic_json_save
from experiment.instances import load_job_records_csv
from experiment.revision_scenarios import REVISION_SCENES,apply_revision_scene
from comparison_runners.detailed_metrics import episode_row,job_rows,machine_rows,state_action_rows
from comparison_runners.native_revision_scene import compile_scene
from comparison_runners.rolling_horizon_alns.controller import Controller
from comparison_runners.rolling_horizon_alns.environment import Environment as RollingEnvironment
from methods.hierarchical.environment import HierarchicalSchedulingEnv
from methods.baselines.sdr.shared import RuleController
from methods.baselines.meta.offline_memetic.static_plan import load_static_plan,StaticPlanController
from comparison_runners.composite_priority.rule import (
    Controller as CompositeController,Environment as CompositeEnvironment,load_weights,DEFAULT_WEIGHTS_FILE)


class NonLearningEnvironment(HierarchicalSchedulingEnv):
    def _proposal_features(self,requester,proposal):
        # A static plan can deliberately wait for a future batch. No learned
        # policy consumes its feature placeholders.
        return np.zeros(self.proposal_dim,dtype=np.float32)


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def resolve(path):
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def consolidate_selected(out,blocks,name):
    """Stream only this invocation's blocks, not other budgets/subsets."""
    paths=[out/b['files'][name] for b in blocks if name in b['files']]
    if not paths:return
    fields=[]
    for path in paths:
        with path.open(encoding='utf-8-sig',newline='') as stream:
            fields.extend(x for x in (csv.DictReader(stream).fieldnames or []) if x not in fields)
    target=out/(name+'.csv');temporary=out/('.'+name+'.tmp')
    with temporary.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for path in paths:
            with path.open(encoding='utf-8-sig',newline='') as source:
                writer.writerows(csv.DictReader(source))
    os.replace(temporary,target)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',choices=('sdr','ma','rh_alns','composite'),default=['sdr','ma','rh_alns','composite'])
    p.add_argument('--weights-file',type=Path,default=DEFAULT_WEIGHTS_FILE,help='Frozen tuned composite-priority weights JSON.')
    p.add_argument('--scenes',nargs='+',choices=tuple(REVISION_SCENES),default=list(REVISION_SCENES))
    p.add_argument('--phase',choices=('plan','evaluate','all'),default='all')
    p.add_argument('--dataset-dir',type=Path,default=ROOT/'datasets/e1_formal_instances')
    p.add_argument('--case-manifest',type=Path,default=ROOT/'experiment_manifests/e1_formal_comparison_cases.csv')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/revision_nonlearning_evaluation')
    p.add_argument('--plan-dir',type=Path,default=ROOT/'runs/revision_offline_ma_plans')
    p.add_argument('--condition-limit',type=int)
    p.add_argument('--instances',type=int,default=TEST_INSTANCES_PER_CONDITION)
    p.add_argument('--replications',type=int,default=10)
    p.add_argument('--rh-evaluations',type=int,default=8000)
    p.add_argument('--ma-population',type=int,default=100)
    p.add_argument('--ma-generations',type=int,default=2000)
    p.add_argument('--ma-refinement-generations',type=int,default=200)
    p.add_argument('--ma-local-search-evaluations',type=int,default=500)
    p.add_argument('--ma-threads',type=int,default=1)
    p.add_argument('--save-job-records',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args()


def execute(args,*,scene_map=None,processing_times=(190,158,202,128,104),machine_counts=(2,2,1),
            reset_adapter=None,native_options=None,reference_capacity=380):
    scenes=REVISION_SCENES if scene_map is None else scene_map
    data,out,plans=map(resolve,(args.dataset_dir,args.output_dir,args.plan_dir))
    structural=read(data/'instance_manifest.csv')
    all_ids=list(dict.fromkeys(row['condition_id'] for row in structural))
    ids_by_scene={s:selected_conditions(all_ids,s,limit=args.condition_limit) for s in args.scenes}
    condition_ids=list(dict.fromkeys(c for ids in ids_by_scene.values() for c in ids))
    structural=[r for r in structural if r['condition_id'] in condition_ids and int(r['instance_index'])<args.instances]
    cases=[r for r in read(resolve(args.case_manifest)) if r['condition_id'] in condition_ids
           and int(r['instance_index'])<args.instances and int(r['replication_index'])<args.replications]
    variants=[]
    if 'sdr' in args.methods:
        variants.extend((f'SDR-{f}-{s}-{m}',(f,s,m)) for f,s,m in
            product(('CH','EDD-J','FFD'),('MMU','EDD-B','MAST'),('Threshold','Period')))
    variants.extend((m,None) for m in ('ma','rh_alns','composite') if m in args.methods)
    task=dict(scenes=args.scenes,methods=[v[0] for v in variants],phase=args.phase,
              instances=len(structural),cases_per_scene_method=len(cases),
              conditions_by_scene=ids_by_scene,
              online_runs=len(variants)*sum(sum(r['condition_id'] in ids for r in cases) for ids in ids_by_scene.values()) if args.phase!='plan' else 0,
              offline_plans=sum(sum(r['condition_id'] in ids for r in structural) for ids in ids_by_scene.values()) if 'ma' in args.methods and args.phase!='evaluate' else 0)
    if args.dry_run:
        print(json.dumps(task,indent=2));return task
    weights=load_weights(resolve(args.weights_file)) if 'composite' in args.methods else None
    out.mkdir(parents=True,exist_ok=True)
    atomic_json_save({**task,**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, 'output_dir':str(out),'plan_dir':str(plans),
        'dataset_dir':str(data),'case_manifest':str(resolve(args.case_manifest))},out/'execution_plan.json')
    source_map={(r['condition_id'],int(r['instance_index'])):r for r in structural}
    ma_budget=dict(population=args.ma_population,generations=args.ma_generations,
                   refinement_generations=args.ma_refinement_generations,
                   local_search_evaluations=args.ma_local_search_evaluations)
    ma_key=hashlib.sha256(json.dumps(ma_budget,sort_keys=True).encode()).hexdigest()[:10]
    blocks=[]
    with tempfile.TemporaryDirectory(prefix='revision_native_build_') as temp:
        for scene_id in args.scenes:
            scene_structural=[r for r in structural if r['condition_id'] in ids_by_scene[scene_id]]
            scene=scenes[scene_id]
            scene_metadata={**scene.parameters(),'reference_total_capacity':reference_capacity}
            native=compile_scene(scene,Path(temp),ma='ma' in args.methods and args.phase!='evaluate',
                                 rh='rh_alns' in args.methods and args.phase!='plan',**(native_options or {})) if (
                                 'ma' in args.methods and args.phase!='evaluate' or
                                 'rh_alns' in args.methods and args.phase!='plan') else {}
            plan_root=plans/scene_id/ma_key
            if 'ma' in args.methods and args.phase!='evaluate':
                plan_root.mkdir(parents=True,exist_ok=True)
                manifest=plan_root/'input_manifest.csv'
                atomic_csv_save(scene_structural,manifest,fieldnames=tuple(scene_structural[0]))
                atomic_json_save(scene_metadata,plan_root/'scene_parameters.json')
                digest=hashlib.sha256(json.dumps(scene_metadata,sort_keys=True).encode()).hexdigest()
                subprocess.run([str(native['ma']),'--dataset-dir',str(data),'--manifest',str(manifest),
                    '--output-dir',str(plan_root),'--dataset-contract-sha256',
                    hashlib.sha256((data/'instance_manifest.csv').read_bytes()).hexdigest(),
                    '--optimizer-contract-sha256',digest,'--population',str(args.ma_population),
                    '--generations',str(args.ma_generations),'--refinement-generations',str(args.ma_refinement_generations),
                    '--local-search-evaluations',str(args.ma_local_search_evaluations),
                    '--threads',str(args.ma_threads)],check=True)
            if args.phase=='plan':continue
            plan_rows={(r['condition_id'],int(r['instance_index'])):r
                       for r in read(plan_root/'plan_manifest.csv')} if 'ma' in args.methods else {}
            worker=Controller(native['rh_alns'],args.rh_evaluations) if 'rh_alns' in args.methods else None
            try:
                for method,rules in variants:
                    for condition_id in ids_by_scene[scene_id]:
                        selected=[r for r in cases if r['condition_id']==condition_id]
                        key_data={'cases':[(r['instance_seed'],r['simulation_seed']) for r in selected],
                                  'budget':ma_budget if method=='ma' else
                                  args.rh_evaluations if method=='rh_alns' else
                                  weights.tolist() if method=='composite' else None,
                                  'save_job_records':args.save_job_records}
                        key=hashlib.sha256(json.dumps(key_data,sort_keys=True).encode()).hexdigest()[:10]
                        stem=f'{scene_id}__{method}__{condition_id}__{key}'
                        completed=out/'blocks'/(stem+'.json')
                        if completed.exists():
                            blocks.append(json.loads(completed.read_text(encoding='utf-8')))
                            print(f'[REVISION-BASELINE] SKIP {stem}',flush=True);continue
                        tables={name:[] for name in ('run_metrics','machine_metrics','state_action_metrics','reoptimization')}
                        if args.save_job_records:tables['job_records']=[]
                        for index,row in enumerate(selected,1):
                            source=source_map[(condition_id,int(row['instance_index']))]
                            records=load_job_records_csv(data/source['job_file'])
                            controller=RuleController(*rules,maintenance_period=float(max(processing_times))) if rules else (
                                StaticPlanController() if method=='ma' else
                                CompositeController(weights) if method=='composite' else worker)
                            cls=RollingEnvironment if method=='rh_alns' else (
                                CompositeEnvironment if method=='composite' else NonLearningEnvironment)
                            env=cls([float(row['nominal_finite_load']),int(row['initial_jobs']),int(row['total_jobs']),
                                float(row['due_date_factor']),int(row['instance_seed'])],list(machine_counts),
                                processing_times=processing_times,controller=controller,device='cpu')
                            if reset_adapter is not None:reset_adapter(env)
                            planned=plan_rows.get((condition_id,int(row['instance_index'])))
                            if method=='ma':controller.set_plan(load_static_plan(plan_root/planned['plan_file']))
                            identity={**row,'scene_id':scene_id,'method':method,
                                'scene_parameters_json':json.dumps(scene_metadata,sort_keys=True),
                                'effective_nominal_load':float(row['nominal_finite_load'])*reference_capacity/scene.total_capacity,
                                'ma_offline_ttd':float(planned['objective_total_tardiness']) if method=='ma' else '',
                                'ma_optimization_seconds':float(planned['runtime_seconds']) if method=='ma' else ''}
                            identity['composite_weights_json']=json.dumps(weights.tolist()) if method=='composite' else ''
                            identity['search_budget_json']=json.dumps(ma_budget if method=='ma' else
                                {'evaluations_per_replan':args.rh_evaluations} if method=='rh_alns' else {},sort_keys=True)
                            metrics=error=None;started=time.perf_counter()
                            try:
                                apply_revision_scene(env,scene,job_records=records,instance_seed=int(row['instance_seed']),
                                    simulation_seed=int(row['simulation_seed']),policy_seed=int(row['simulation_seed']))
                                metrics=env.run_episode(training=False)
                            except Exception as exc:error=exc
                            result=episode_row(identity,metrics,env,wall_time=time.perf_counter()-started,error=error)
                            if metrics is not None:
                                tables['machine_metrics'].extend(machine_rows(identity,env,metrics['makespan']))
                                tables['state_action_metrics'].extend(state_action_rows(identity,env))
                                if args.save_job_records:tables['job_records'].extend(job_rows(identity,env))
                            if method=='rh_alns':tables['reoptimization'].extend({**identity,**t} for t in worker.trace)
                            if method=='ma' and metrics is not None:
                                value=float(planned['objective_total_tardiness'])
                                result['ma_execution_delta']=metrics['total_tardiness']-value
                                result['ma_execution_degradation_percent']=100*(metrics['total_tardiness']/value-1) if value>0 else ''
                            else:result.update(ma_execution_delta='',ma_execution_degradation_percent='')
                            tables['run_metrics'].append(result)
                            if index==1 or index==len(selected) or error or args.log_every and index%args.log_every==0:
                                timing=''
                                if method=='rh_alns':
                                    count=len(worker.trace)
                                    mean_solve=sum(t['seconds'] for t in worker.trace)/count if count else 0.
                                    timing=(f' wall={result["episode_wall_time_seconds"]:.2f}s'
                                            f' replans={count} mean_search={1000*mean_solve:.2f}ms')
                                print(f'[REVISION-BASELINE] {stem} {index}/{len(selected)} seed={row["simulation_seed"]} '
                                      f'status={result["status"]} TTD={result["total_tardiness"]}{timing} error={error or ""}',flush=True)
                        files={}
                        for name,rows in tables.items():
                            if not rows:continue
                            path=out/'partitions'/name/(stem+'.csv')
                            atomic_csv_save(rows,path,fieldnames=tuple(rows[0]))
                            files[name]=str(path.relative_to(out))
                        summary=dict(block=stem,scene_id=scene_id,method=method,condition_id=condition_id,
                            runs=len(selected),failed_runs=sum(r['status']!='completed' for r in tables['run_metrics']),files=files)
                        atomic_json_save(summary,completed);blocks.append(summary)
            finally:
                if worker:worker.close()
    if args.phase!='plan':
        for name in ('run_metrics','machine_metrics','state_action_metrics','job_records','reoptimization'):
            if name=='job_records' and not args.save_job_records:continue
            consolidate_selected(out,blocks,name)
    atomic_json_save({'plan':task,'blocks':blocks},out/'completion_summary.json')
    return blocks


if __name__=='__main__':execute(parse_args())
