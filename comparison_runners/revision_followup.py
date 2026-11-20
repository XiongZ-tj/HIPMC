"""Isolated reward-shaping and composite-policy follow-up experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment.artifacts import atomic_json_save

SCALE='reward_scales/offline_calibrated_sdr_reward_scale_base_finite_load100_jobs150_due2p5_deg100.json'
WEIGHTS='experiment_manifests/tuned_composite_priority_weights.json'
ACTIONS=('train-shaping','summarize-shaping','evaluate-shaping',
         'evaluate-composite-generated','evaluate-composite-industrial')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=ACTIONS)
    p.add_argument('--output-root',type=Path,default=Path('runs/revision_reward_and_rule_followup'))
    p.add_argument('--device',default='auto')
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args(argv)


def commands(args):
    root=(args.output_root if args.output_root.is_absolute() else ROOT/args.output_root).resolve()
    if root==ROOT or ROOT.is_relative_to(root):
        raise ValueError('Choose a dedicated follow-up output directory.')
    training=root/'shaping_training'
    weights=root/'inputs/frozen_composite_weights.json'
    method=['--methods','ours_categorical_baa','--training-seeds','101','202','303']
    device=['--device',args.device]
    action=args.action
    if action=='train-shaping':
        script='comparison_runners/train_drl_baselines.py'
        flags=[*method,*device,'--episodes','500','--validation-instances','10',
               '--validation-interval','10','--checkpoint-interval','25','--learning-rate','0.0001',
               '--discount-factor','0.90','--ppo-clip-ratio','0.20',
               '--reward-formulation','discount_consistent_potential','--reward-scale-file',SCALE,
               '--output-dir',str(training)]
    elif action=='summarize-shaping':
        script='result_analysis/training.py'
        flags=['--training-roots',str(training),'--budget','500','--late-window','50',
               '--output-dir',str(root/'shaping_training_statistics')]
    elif action=='evaluate-shaping':
        script='comparison_runners/evaluate_learned_methods.py'
        flags=[*method,*device,'--scenes','S0','--instances','10','--model-budget','500',
               '--model-layout','standard','--model-root',str(training),'--no-save-job-records',
               '--output-dir',str(root/'shaping_evaluation')]
    else:
        industrial=action=='evaluate-composite-industrial'
        script=('comparison_runners/revision_special/evaluate_industrial_nonlearning.py' if industrial
                else 'comparison_runners/evaluate_revision_nonlearning.py')
        flags=['--methods','composite','--phase','evaluate','--replications','10',
               '--weights-file',str(weights),'--no-save-job-records',
               '--output-dir',str(root/'composite_frozen'/('industrial' if industrial else 'generated'))]
        if not industrial:flags+=['--scenes','S0','S1','S2','S3','S4','S5','--instances','10']
    return root,[sys.executable,str(ROOT/script),*flags],weights


def execute(args):
    root,command,weights=commands(args)
    if args.dry_run:
        print(subprocess.list2cmdline(command));return command
    if args.action.startswith('evaluate-composite'):
        source=(ROOT/WEIGHTS).read_bytes()
        if weights.exists() and weights.read_bytes()!=source:
            raise ValueError('Frozen input weights changed; use another output root.')
        weights.parent.mkdir(parents=True,exist_ok=True)
        if not weights.exists():weights.write_bytes(source)
    root.mkdir(parents=True,exist_ok=True)
    record=dict(action=args.action,command=command,working_directory=str(ROOT),
                output_root=str(root),reward_formulation='discount_consistent_potential',
                terminal_reward='(-total_tardiness - current_potential) / fixed_scale',
                weights_mode='frozen',
                composite_repair_policy='health_threshold_before_retained_batch_resumption')
    sources=['methods/hierarchical/config.py','methods/hierarchical/environment.py',
             'comparison_runners/composite_priority/rule.py',
             'comparison_runners/train_drl_baselines.py','comparison_runners/training_budgets.py',
             'experiment_manifests/e1_learned_method_training_cases.csv',
             'experiment_manifests/e1_learned_method_validation_cases.csv',SCALE]
    record['input_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}
    if weights.is_file():record['weights_sha256']=hashlib.sha256(weights.read_bytes()).hexdigest()
    key=args.action
    contract=root/'execution_contracts'/(key+'.json')
    if contract.exists() and json.loads(contract.read_text(encoding='utf-8'))!=record:
        raise ValueError('Follow-up inputs changed; use a separate output root.')
    atomic_json_save(record,contract)
    log=root/'logs'/(key+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    print(subprocess.list2cmdline(command),flush=True)
    with log.open('a',encoding='utf-8') as stream:
        with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace',
                              env={**__import__('os').environ,'PYTHONIOENCODING':'utf-8'}) as process:
            for line in process.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
            code=process.wait()
    atomic_json_save(dict(returncode=code),root/'execution_status'/(key+'.json'))
    if code:raise subprocess.CalledProcessError(code,command)
    return command


if __name__=='__main__':execute(parse_args())
