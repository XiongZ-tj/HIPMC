"""Industrial SDR18, static full-information MA, RH-ALNS, and frozen composite rule."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from comparison_runners import evaluate_revision_nonlearning as base
from comparison_runners.revision_special.evaluate_industrial import industrial_cases
from comparison_runners.industrial_case.case_common import (
    DATASET_ROOT,PROCESSING_TIMES,MACHINE_COUNTS,MACHINE_CAPACITIES,adapt_simulator)
from experiment.artifacts import atomic_csv_save
from experiment.revision_scenarios import RevisionScene
from comparison_runners.composite_priority.rule import DEFAULT_WEIGHTS_FILE


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',choices=('sdr','ma','rh_alns','composite'),default=['sdr','ma','rh_alns','composite'])
    p.add_argument('--phase',choices=('plan','evaluate','all'),default='all')
    p.add_argument('--dataset-dir',type=Path,default=DATASET_ROOT)
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/revision_industrial_nonlearning')
    p.add_argument('--plan-dir',type=Path,default=ROOT/'runs/revision_industrial_ma_plans')
    p.add_argument('--weights-file',type=Path,default=DEFAULT_WEIGHTS_FILE)
    p.add_argument('--instance-limit',type=int)
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


def execute(args):
    data=base.resolve(args.dataset_dir)
    cases=industrial_cases(data,False,args.instance_limit,args.replications)
    structural={}
    adapted=[]
    for row in cases:
        index=int(row['instance_number'])-1
        item={**row,'condition_index':(0.9,1.0,1.1).index(round(float(row['target_load']),1)),
            'instance_index':index,'nominal_finite_load':float(row['target_load']),
            'initial_jobs':18,'total_jobs':89,'due_date_factor':1.,'degradation_multiplier':1.,
            'job_file':(data/row['job_file']).as_posix()}
        adapted.append(item)
        structural[(row['condition_id'],index)]=item
    # Adapter metadata is an output artifact; original orders stay untouched.
    directory=base.resolve(args.output_dir)/'dataset_adapter'
    if not args.dry_run:
        atomic_csv_save(list(structural.values()),directory/'instance_manifest.csv')
        atomic_csv_save(adapted,directory/'cases.csv')
    else:
        n=len(structural);methods=18*('sdr' in args.methods)+sum(m in args.methods for m in ('ma','rh_alns','composite'))
        plan=dict(instances=n,online_runs=methods*len(cases) if args.phase!='plan' else 0,
                  offline_plans=n if 'ma' in args.methods and args.phase!='evaluate' else 0)
        import json
        print(json.dumps(plan,indent=2));return plan
    args.dataset_dir=directory;args.case_manifest=directory/'cases.csv'
    args.scenes=['industrial'];args.instances=20;args.condition_limit=None
    scene=RevisionScene('industrial','Industrial-order-data-driven case',
                        machine_counts=MACHINE_COUNTS,machine_capacities=MACHINE_CAPACITIES)
    return base.execute(args,scene_map={'industrial':scene},processing_times=PROCESSING_TIMES,
        machine_counts=MACHINE_COUNTS,reset_adapter=adapt_simulator,
        native_options=dict(processing_times=PROCESSING_TIMES,job_size_max=60),reference_capacity=480)


if __name__=='__main__':execute(parse_args())
