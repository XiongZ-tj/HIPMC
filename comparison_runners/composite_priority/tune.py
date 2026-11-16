"Search weights only on the generated-data training production condition."
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiment.artifacts import atomic_csv_save,atomic_json_save
from experiment.instances import generate_job_records,save_job_records_csv,load_job_records_csv
from comparison_runners.detailed_metrics import episode_row
from comparison_runners.composite_priority.rule import Controller,Environment,weight_artifact

CONDITION=dict(target_load=1.,initial_jobs=30,total_jobs=150,due_date_factor=2.5,
    processing_times=(190,158,202,128,104),size_interval=(10,30),
    machine_counts=(2,2,1),machine_capacities=(60,80,100))
_ENV=None


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def weight_id(weights):
    return hashlib.sha256(json.dumps([float(x).hex() for x in weights]).encode()).hexdigest()[:20]


def initialize_worker():
    global _ENV
    import torch
    torch.set_num_threads(1)
    _ENV=Environment([1.,30,150,2.5,610000000],[2,2,1],
        processing_times=CONDITION['processing_times'],controller=Controller(np.ones(6)/6),device='cpu')


def simulate(task):
    global _ENV
    if _ENV is None:initialize_worker()
    row,weights=task
    _ENV.controller.weights=np.asarray(weights)
    _ENV.reset(instance_seed=int(row['instance_seed']),simulation_seed=int(row['simulation_seed']),
        policy_seed=int(row['simulation_seed']),job_records=load_job_records_csv(row['job_file']))
    started=time.perf_counter()
    metrics=_ENV.run_episode(training=False)
    return episode_row({**row,'weight_id':weight_id(weights)},metrics,_ENV,
                       wall_time=time.perf_counter()-started)


def materialize(root,args):
    """Persist two disjoint training-condition order sets, never test orders."""
    sets={}
    for split,count,reps,base,simbase in (
        ('search',args.search_instances,args.search_replications,610000000,630000000),
        ('selection',args.selection_instances,args.selection_replications,620000000,640000000)):
        path=root/'data'/f'{split}_cases_n{count}_r{reps}.csv'
        if path.exists():
            sets[split]=read(path);continue
        rows=[]
        for i in range(count):
            seed=base+i
            job_file=root/'data'/split/f'instance-{i:03d}.csv'
            save_job_records_csv(generate_job_records(**CONDITION,seed=seed),job_file)
            for r in range(reps):
                rows.append(dict(split=split,instance_index=i,replication_index=r,
                    instance_seed=seed,simulation_seed=simbase+i*100+r,
                    job_file=str(job_file.relative_to(root))))
        atomic_csv_save(rows,path);sets[split]=rows
    return sets


class Evaluator:
    def __init__(self, root, cases, pool=None, log_every=30):
        self.root=root;self.cases=cases;self.pool=pool;self.log_every=log_every

    def __call__(self, weights, split):
        key=weight_id(weights)
        cases_key=hashlib.sha256(json.dumps([
            (int(r['instance_seed']),int(r['simulation_seed'])) for r in self.cases[split]
        ]).encode()).hexdigest()[:12]
        target=self.root/'evaluations'/split/cases_key/(key+'.csv')
        summary=target.with_suffix('.json')
        if summary.exists():return json.loads(summary.read_text(encoding='utf-8'))
        previous=read(target) if target.exists() else []
        finished={(int(r['instance_index']),int(r['replication_index'])) for r in previous}
        tasks=[]
        for row in self.cases[split]:
            if (int(row['instance_index']),int(row['replication_index'])) not in finished:
                tasks.append(({**row,'job_file':str(self.root/row['job_file'])},weights))
        iterator=self.pool.map(simulate,tasks,chunksize=1) if self.pool else map(simulate,tasks)
        target.parent.mkdir(parents=True,exist_ok=True)
        for result in iterator:
            # Keep paths portable when the tuning directory is copied.
            result['job_file']=str(Path(result['job_file']).relative_to(self.root))
            with target.open('a',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(result))
                if stream.tell()==0:writer.writeheader()
                writer.writerow(result);stream.flush()
            previous.append(result)
            if self.log_every and len(previous)%self.log_every==0:
                print(f'[COMPOSITE-CASES] split={split} weight={key} '
                      f'{len(previous)}/{len(self.cases[split])}',flush=True)
        values=np.asarray([float(r['total_tardiness']) for r in previous])
        result=dict(weight_id=key,weights=list(map(float,weights)),split=split,
            mean_ttd=float(values.mean()),std_ttd=float(values.std(ddof=1)) if len(values)>1 else 0.,
            runs=len(values))
        atomic_json_save(result,summary)
        return result


def search(seed,evaluate,random_count=128,local_per_step=16):
    rng=np.random.default_rng(seed);history=[];best=None
    def visit(w,stage):
        nonlocal best
        result=evaluate(w,'search')
        if best is None or result['mean_ttd']<best['mean_ttd']:best=result
        history.append(dict(optimization_seed=seed,evaluation_index=len(history)+1,stage=stage,
            weight_id=result['weight_id'],mean_ttd=result['mean_ttd'],
            incumbent_ttd=best['mean_ttd'],**{f'w{i+1}':v for i,v in enumerate(w)}))
        print(f'[COMPOSITE-TUNE] seed={seed} candidate={len(history)} stage={stage} '
              f'TTD={result["mean_ttd"]:.3f} best={best["mean_ttd"]:.3f}',flush=True)
    for w in np.eye(6):visit(w,'single_feature')
    visit(np.ones(6)/6,'equal_weight')
    for _ in range(random_count):visit(rng.dirichlet(np.ones(6)),'random')
    for step in (.20,.10,.05,.02):
        for _ in range(local_per_step):
            w=np.asarray(best['weights']).copy()
            donors=np.flatnonzero(w>0)
            donor=int(rng.choice(donors));recipient=int(rng.choice([i for i in range(6) if i!=donor]))
            amount=min(step,w[donor]);w[donor]-=amount;w[recipient]+=amount
            visit(w,f'local_{step}')
    return history,best


def finalists(histories,count=10):
    unique={}
    for row in histories:unique.setdefault(row['weight_id'],row)
    ordered=sorted(unique.values(),key=lambda r:(r['mean_ttd'],r['weight_id']))
    anchors=[next(r for r in ordered if r['stage']=='equal_weight'),
             min((r for r in ordered if r['stage']=='single_feature'),key=lambda r:(r['mean_ttd'],r['weight_id']))]
    selected={r['weight_id']:r for r in ordered[:max(0,count-2)]}
    selected.update({r['weight_id']:r for r in anchors})
    for row in ordered:
        if len(selected)>=count:break
        selected[row['weight_id']]=row
    return list(selected.values())


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/composite_priority_tuning')
    p.add_argument('--optimization-seeds',type=int,nargs='+',default=[101])
    p.add_argument('--search-instances',type=int,default=30)
    p.add_argument('--selection-instances',type=int,default=30)
    p.add_argument('--search-replications',type=int,default=3)
    p.add_argument('--selection-replications',type=int,default=10)
    p.add_argument('--random-candidates',type=int,default=128)
    p.add_argument('--local-per-step',type=int,default=16)
    p.add_argument('--finalists',type=int,default=10)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--log-every',type=int,default=30)
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args()


def execute(args):
    root=args.output_dir
    root=(root if root.is_absolute() else ROOT/root).resolve()
    plan=dict(condition=CONDITION,scene='S0',optimization_seeds=args.optimization_seeds,
        search_instances=args.search_instances,selection_instances=args.selection_instances,
        search_replications=args.search_replications,selection_replications=args.selection_replications,
        random_candidates=args.random_candidates,local_per_step=args.local_per_step,finalists=args.finalists,
        nominal_simulations=len(args.optimization_seeds)*(7+args.random_candidates+4*args.local_per_step)
            *args.search_instances*args.search_replications
            +args.finalists*args.selection_instances*args.selection_replications)
    if args.dry_run:print(json.dumps(plan,indent=2));return plan
    root.mkdir(parents=True,exist_ok=True)
    atomic_json_save(plan,root/'tuning_plan.json')
    cases=materialize(root,args)
    pool=ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker) if args.workers>1 else None
    try:
        evaluate=Evaluator(root,cases,pool,args.log_every);histories=[];seed_bests=[];stage_rows=[]
        for seed in args.optimization_seeds:
            history,best=search(seed,evaluate,args.random_candidates,args.local_per_step)
            atomic_csv_save(history,root/f'search_seed-{seed}.csv')
            histories.extend(history);seed_bests.append(dict(optimization_seed=seed,**best))
            for stage in ('single_feature','equal_weight','random','local'):
                rows=[r for r in history if r['stage'].startswith(stage)]
                if rows:
                    row=min(rows,key=lambda r:(r['mean_ttd'],r['weight_id']))
                    stage_rows.append({**row,'comparison_stage':stage})
        atomic_csv_save(stage_rows,root/'search_stage_comparison.csv')
        candidates=finalists(histories,args.finalists);selection=[]
        for rank,row in enumerate(candidates,1):
            w=[row[f'w{i+1}'] for i in range(6)]
            result=evaluate(w,'selection');selection.append(result)
            print(f'[COMPOSITE-SELECT] {rank}/{len(candidates)} TTD={result["mean_ttd"]:.3f}',flush=True)
        selected=min(selection,key=lambda r:(r['mean_ttd'],r['weight_id']))
        atomic_csv_save([dict(weight_id=r['weight_id'],mean_ttd=r['mean_ttd'],
            std_ttd=r['std_ttd'],runs=r['runs'],selected=r['weight_id']==selected['weight_id'],
            **{f'w{i+1}':w for i,w in enumerate(r['weights'])}) for r in selection],
            root/'selection_comparison.csv')
        atomic_json_save(dict(search_seed_results=seed_bests,selection_results=selection),root/'tuning_results.json')
        artifact=weight_artifact(selected['weights'],training_condition=CONDITION,scene='S0',
            selection_mean_ttd=selected['mean_ttd'],selection_runs=selected['runs'],
            tuning_plan='tuning_plan.json',selection_results='tuning_results.json')
        atomic_json_save(artifact,root/'tuned_composite_priority_weights.json')
        print(f'[COMPOSITE-TUNE] DONE weights={selected["weights"]} selection_TTD={selected["mean_ttd"]:.3f}',flush=True)
        return artifact
    finally:
        if pool:pool.shutdown()


if __name__=='__main__':execute(parse_args())
