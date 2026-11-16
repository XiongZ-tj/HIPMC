"""Equal-budget twelve-configuration tuning of all four methods, generated only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment.revision_protocol import LEARNED_METHODS,TRAINING_SEEDS,VALIDATION_INSTANCES
from params_analysis.grid import select_configurations


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',choices=LEARNED_METHODS,default=list(LEARNED_METHODS))
    p.add_argument('--configuration-ids',nargs='+')
    p.add_argument('--training-seeds',nargs='+',type=int,default=list(TRAINING_SEEDS))
    p.add_argument('--episodes',type=int,default=500)
    p.add_argument('--validation-instances',type=int,default=VALIDATION_INSTANCES)
    p.add_argument('--validation-interval',type=int,default=10)
    p.add_argument('--checkpoint-interval',type=int,default=25)
    p.add_argument('--device',default='auto')
    p.add_argument('--reward-scale-file',type=Path,required=True)
    p.add_argument('--training-manifest',type=Path,default=ROOT/'experiment_manifests/e1_learned_method_training_cases.csv')
    p.add_argument('--validation-manifest',type=Path,default=ROOT/'experiment_manifests/e1_learned_method_validation_cases.csv')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/revision_hyperparameter_training')
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args()


def commands(args):
    result=[]
    for cfg in select_configurations(args.configuration_ids):
        command=[sys.executable,str(ROOT/'comparison_runners/train_drl_baselines.py'),
            '--methods',*args.methods,'--training-seeds',*map(str,args.training_seeds),
            '--learning-rate',str(cfg.learning_rate),'--discount-factor',str(cfg.discount_factor),
            '--ppo-clip-ratio',str(cfg.clip_ratio),'--episodes',str(args.episodes),
            '--validation-instances',str(args.validation_instances),'--validation-interval',str(args.validation_interval),
            '--checkpoint-interval',str(args.checkpoint_interval),'--device',args.device,
            '--reward-scale-file',str(args.reward_scale_file),
            '--training-manifest',str(args.training_manifest),'--validation-manifest',str(args.validation_manifest),
            '--output-dir',str(args.output_dir/cfg.configuration_id)]
        result.append(command)
    return result


def execute(args):
    jobs=commands(args)
    plan=dict(data_domain='generated',configurations=len(jobs),methods=args.methods,
        training_seeds=args.training_seeds,episodes=args.episodes,
        training_tasks=len(jobs)*len(args.methods)*len(args.training_seeds),commands=jobs)
    if args.dry_run:print(json.dumps(plan,indent=2));return plan
    from experiment.artifacts import atomic_json_save
    atomic_json_save(plan,args.output_dir/'grid_execution_plan.json')
    for i,command in enumerate(jobs,1):
        print(f'[PARAM-GRID] {i}/{len(jobs)} START',flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    return plan


if __name__=='__main__':execute(parse_args())
