"""Retune and evaluate a composite priority rule in isolated output directories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiment.artifacts import atomic_json_save

ACTIONS=('tune','evaluate-generated','evaluate-industrial','evaluate-components')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=ACTIONS)
    p.add_argument('--output-root',type=Path,default=Path('runs/revision_reward_and_rule_followup/composite_retuning'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--device',default='auto')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    if args.workers<1:p.error('--workers must be positive')
    return args


def command(args):
    root=(args.output_root if args.output_root.is_absolute() else ROOT/args.output_root).resolve()
    if root==ROOT or ROOT.is_relative_to(root):raise ValueError('Use a dedicated output directory.')
    weights=root/'inputs/frozen_selected_weights.json'
    if args.action=='tune':
        script='comparison_runners/composite_priority/tune.py'
        flags=['--output-dir',str(root/'tuning'),'--optimization-seeds','101',
            '--search-instances','30','--selection-instances','30',
            '--search-replications','3','--selection-replications','10',
            '--random-candidates','128','--local-per-step','16','--finalists','10',
            '--workers',str(args.workers)]
    elif args.action=='evaluate-components':
        script='ablation_experiments/evaluate_revision.py'
        flags=['--variants','tuned_composite_batch_selection','--training-seeds','101','202','303',
            '--scenes','S0','S1','S2','S3','S4','S5','--instances','10','--replications','10',
            '--model-root','runs/revision_hyperparameter_training','--model-layout','grid',
            '--hyperparameter-file','experiment_manifests/selected_generated_hyperparameters.json',
            '--model-budget','500','--device',args.device,'--weights-file',str(weights),
            '--no-save-job-records','--output-dir',str(root/'component_evaluation')]
    else:
        industrial=args.action=='evaluate-industrial'
        script=('comparison_runners/revision_special/evaluate_industrial_nonlearning.py' if industrial
                else 'comparison_runners/evaluate_revision_nonlearning.py')
        flags=['--methods','composite','--phase','evaluate','--replications','10',
               '--weights-file',str(weights),'--no-save-job-records',
               '--output-dir',str(root/('industrial_evaluation' if industrial else 'generated_evaluation'))]
        if not industrial:flags+=['--scenes','S0','S1','S2','S3','S4','S5','--instances','10']
    return root,[sys.executable,str(ROOT/script),*flags],weights


def execute(args):
    root,cmd,weights=command(args)
    if args.dry_run:print(subprocess.list2cmdline(cmd));return cmd
    if args.action!='tune':
        artifact=root/'tuning/tuned_composite_priority_weights.json'
        if not artifact.is_file():raise FileNotFoundError('Run the tune action first.')
        data=artifact.read_bytes()
        if weights.exists() and weights.read_bytes()!=data:
            raise ValueError('Selected weights changed after evaluation; use a new output root.')
        weights.parent.mkdir(parents=True,exist_ok=True)
        if not weights.exists():weights.write_bytes(data)
    sources=['comparison_runners/composite_priority/rule.py','comparison_runners/composite_priority/tune.py',
             'methods/hierarchical/environment.py','methods/baselines/sdr/shared.py',
             'ablation_experiments/environment.py']
    record=dict(action=args.action,command=cmd,source_sha256={
        f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources},
        original_budget=True,selection_uses_test_results=False,
        repair_policy='health_threshold_before_retained_batch_resumption')
    if args.action!='tune':
        tuning_contract=root/'execution_contracts/tune.json'
        if not tuning_contract.is_file():raise FileNotFoundError('Missing tuning execution contract.')
        if json.loads(tuning_contract.read_text(encoding='utf-8'))['source_sha256']!=record['source_sha256']:
            raise ValueError('Implementation changed since tuning; use a new tuning output root.')
    if args.action!='tune':record['weights_sha256']=hashlib.sha256(weights.read_bytes()).hexdigest()
    path=root/'execution_contracts'/(args.action+'.json')
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=record:
        raise ValueError('Inputs differ from the saved execution contract; use a new output root.')
    atomic_json_save(record,path)
    log=root/'logs'/(args.action+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    print(subprocess.list2cmdline(cmd),flush=True)
    with log.open('a',encoding='utf-8') as stream:
        with subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',
                env={**os.environ,'PYTHONIOENCODING':'utf-8'}) as process:
            for line in process.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
            code=process.wait()
    atomic_json_save(dict(returncode=code),root/'execution_status'/(args.action+'.json'))
    if code:raise subprocess.CalledProcessError(code,cmd)
    return cmd


if __name__=='__main__':execute(parse_args())
