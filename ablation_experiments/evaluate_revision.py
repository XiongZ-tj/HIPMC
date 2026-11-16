"Frozen-component replacements on all six scenes and frozen E1 orders."
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
from experiment.revision_selection import COMPONENT_SCENES,selected_conditions
from ablation_experiments.environment import AblationSchedulingEnv
from ablation_experiments.variants import ABLATION_VARIANTS,variant_spec
from comparison_runners.composite_priority.rule import load_weights,DEFAULT_WEIGHTS_FILE
from comparison_runners.detailed_metrics import episode_row,machine_rows,state_action_rows,job_rows
from comparison_runners.evaluate_revision_nonlearning import read,resolve,consolidate_selected
from comparison_runners.training_budgets import budget_model_dir
from experiment.artifacts import atomic_csv_save,atomic_json_save
from experiment.instances import load_job_records_csv
from experiment.revision_protocol import TRAINING_SEEDS
from experiment.revision_scenarios import REVISION_SCENES,apply_revision_scene
from methods.hierarchical.config import HierarchyConfig
from methods.hierarchical.environment import HierarchicalSchedulingEnv

REFERENCE='full_hipmc'
def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants',nargs='*',choices=(*ABLATION_VARIANTS,REFERENCE),default=list(ABLATION_VARIANTS))
    p.add_argument('--training-seeds',nargs='+',type=int,default=list(TRAINING_SEEDS))
    p.add_argument('--scenes',nargs='+',choices=tuple(REVISION_SCENES),default=list(COMPONENT_SCENES))
    p.add_argument('--model-root',type=Path,default=ROOT/'runs/e1_learned_method_training')
    p.add_argument('--model-method',default='ours_categorical_baa')
    p.add_argument('--model-budget',type=int,default=500)
    p.add_argument('--model-layout',choices=('standard','grid'),default='standard')
    p.add_argument('--hyperparameter-file',type=Path)
    p.add_argument('--weights-file',type=Path,
        default=DEFAULT_WEIGHTS_FILE)
    p.add_argument('--dataset-dir',type=Path,default=ROOT/'datasets/e1_formal_instances')
    p.add_argument('--case-manifest',type=Path,default=ROOT/'experiment_manifests/e1_formal_comparison_cases.csv')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/revision_component_ablation')
    p.add_argument('--device',default='auto')
    p.add_argument('--instances',type=int,default=TEST_INSTANCES_PER_CONDITION)
    p.add_argument('--replications',type=int,default=10)
    p.add_argument('--condition-limit',type=int)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--save-job-records',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args()


def frozen_modules(environment):
    return (environment.mappo.baa_actor,environment.mappo.mma_actors,environment.mappo.critic)


def build_environment(variant,parameters,*,config,device,model_state=None,weights=None):
    kwargs=dict(processing_times=(190,158,202,128,104),config=config,device=device)
    if variant==REFERENCE:
        environment=HierarchicalSchedulingEnv(parameters,[2,2,1],**kwargs)
    else:
        environment=AblationSchedulingEnv(parameters,[2,2,1],variant=variant,
                                          composite_weights=weights,**kwargs)
    if model_state is not None:
        environment.mappo.load_state_dict(model_state['mappo'],training=False)
    for module in frozen_modules(environment):
        module.eval()
        for parameter in module.parameters():parameter.requires_grad_(False)
    return environment


def execute(args):
    out=resolve(args.output_dir);data=resolve(args.dataset_dir)
    structural=read(data/'instance_manifest.csv')
    conditions=selected_conditions(list(dict.fromkeys(r['condition_id'] for r in structural)),
                                   args.scenes[0],components=True,limit=args.condition_limit)
    cases=[r for r in read(resolve(args.case_manifest)) if r['condition_id'] in conditions
           and int(r['instance_index'])<args.instances and int(r['replication_index'])<args.replications]
    task=dict(variants=args.variants,training_seeds=args.training_seeds,
        scenes=args.scenes,conditions=len(conditions),cases_per_scene=len(cases),model_budget=args.model_budget,
        learned_runs=len(args.variants)*len(args.training_seeds)*len(args.scenes)*len(cases))
    if args.dry_run:print(json.dumps(task,indent=2));return task
    weights=load_weights(resolve(args.weights_file)) if 'tuned_composite_batch_selection' in args.variants else None
    out.mkdir(parents=True,exist_ok=True)
    atomic_json_save({**task,**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},out/'execution_plan.json')
    sources={(r['condition_id'],int(r['instance_index'])):r for r in structural}
    from experiment.selected_hyperparameters import load_selection,source_directory
    selection=load_selection(resolve(args.hyperparameter_file)) if args.hyperparameter_file else {'methods':{}}
    blocks=[]
    for seed in args.training_seeds:
        state=None;model_path=None;model_sha=''
        if seed is not None:
            if not args.variants:continue
            model_path=source_directory(resolve(args.model_root),args.model_method,seed,args.model_budget,
                selection=selection,grid_layout=args.model_layout=='grid')/'hierarchical_inference.pt'
            model_sha=hashlib.sha256(model_path.read_bytes()).hexdigest()
            state=torch.load(model_path,map_location='cpu',weights_only=True)
        for variant in args.variants:
            config=HierarchyConfig(**state['config']) if state is not None else HierarchyConfig()
            first=cases[0]
            parameters=[float(first['nominal_finite_load']),int(first['initial_jobs']),int(first['total_jobs']),
                        float(first['due_date_factor']),int(first['instance_seed'])]
            environment=build_environment(variant,parameters,config=config,device=args.device,
                                          model_state=state,weights=weights)
            spec=variant_spec(variant) if variant in ABLATION_VARIANTS else None
            for scene_id in args.scenes:
                scene=REVISION_SCENES[scene_id]
                for condition in conditions:
                    selected=[r for r in cases if r['condition_id']==condition]
                    key=hashlib.sha256(json.dumps(dict(
                        cases=[(r['instance_seed'],r['simulation_seed']) for r in selected],model=model_sha,
                        weights=weights.tolist() if variant=='tuned_composite_batch_selection' else None,
                        save_jobs=args.save_job_records),sort_keys=True).encode()).hexdigest()[:12]
                    stem=f'{scene_id}__{variant}__seed-{seed}__{condition}__{key}'
                    completed=out/'blocks'/(stem+'.json')
                    if completed.exists():
                        blocks.append(json.loads(completed.read_text(encoding='utf-8')))
                        print(f'[COMPONENT-EVAL] SKIP {stem}',flush=True);continue
                    tables={name:[] for name in ('run_metrics','machine_metrics','state_action_metrics')}
                    if args.save_job_records:tables['job_records']=[]
                    for index,row in enumerate(selected,1):
                        source=sources[(condition,int(row['instance_index']))]
                        identity={**row,'scene_id':scene_id,'ablation_variant':variant,
                            'training_seed':seed if seed is not None else '',
                            'publication_name':spec.publication_name if spec else 'HIPMC',
                            'target_component':spec.target_component if spec else 'reference',
                            'model_budget':args.model_budget if seed is not None else '',
                            'model_file':str(model_path) if model_path else '',
                            'model_sha256':model_sha,'formation_rule':spec.formation_rule if spec else 'CH',
                            'handshake_mode':environment.handshake_mode.value,
                            'composite_weights_json':json.dumps(weights.tolist()) if variant=='tuned_composite_batch_selection' else '',
                            'scene_parameters_json':json.dumps(scene.parameters(config),sort_keys=True),
                            'effective_nominal_load':scene.effective_nominal_load(float(row['nominal_finite_load'])),
                            'device':str(environment.device)}
                        metrics=error=None;started=time.perf_counter()
                        try:
                            apply_revision_scene(environment,scene,job_records=load_job_records_csv(data/source['job_file']),
                                instance_seed=int(row['instance_seed']),simulation_seed=int(row['simulation_seed']),
                                policy_seed=int(row['simulation_seed']))
                            metrics=environment.run_episode(training=False)
                        except Exception as exc:error=exc
                        result=episode_row(identity,metrics,environment,wall_time=time.perf_counter()-started,error=error)
                        tables['run_metrics'].append(result)
                        if metrics is not None:
                            tables['machine_metrics'].extend(machine_rows(identity,environment,metrics['makespan']))
                            tables['state_action_metrics'].extend(state_action_rows(identity,environment))
                            if args.save_job_records:tables['job_records'].extend(job_rows(identity,environment))
                        if index==1 or index==len(selected) or error or args.log_every and index%args.log_every==0:
                            print(f'[COMPONENT-EVAL] {stem} {index}/{len(selected)} '
                                f'status={result["status"]} TTD={result["total_tardiness"]} error={error or ""}',flush=True)
                    files={}
                    for name,rows in tables.items():
                        if not rows:continue
                        path=out/'partitions'/name/(stem+'.csv')
                        atomic_csv_save(rows,path,fieldnames=tuple(rows[0]));files[name]=str(path.relative_to(out))
                    block=dict(block=stem,scene_id=scene_id,variant=variant,training_seed=seed,
                        condition_id=condition,runs=len(selected),
                        failed_runs=sum(r['status']!='completed' for r in tables['run_metrics']),files=files)
                    atomic_json_save(block,completed);blocks.append(block)
    for name in ('run_metrics','machine_metrics','state_action_metrics','job_records'):
        if name=='job_records' and not args.save_job_records:continue
        consolidate_selected(out,blocks,name)
    atomic_json_save(dict(plan=task,blocks=blocks),out/'completion_summary.json')
    return blocks


if __name__=='__main__':execute(parse_args())
