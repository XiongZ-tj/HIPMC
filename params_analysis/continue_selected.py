"""Continue selected grid runs from their full 500-episode checkpoints to 3000."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment.selected_hyperparameters import load_selection
from experiment.revision_protocol import LEARNED_METHODS,TRAINING_SEEDS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',choices=LEARNED_METHODS,default=list(LEARNED_METHODS))
    p.add_argument('--training-seeds',nargs='+',type=int,default=list(TRAINING_SEEDS))
    p.add_argument('--hyperparameter-file',type=Path,required=True)
    p.add_argument('--reward-scale-file',type=Path,required=True)
    p.add_argument('--model-root',type=Path,default=ROOT/'runs/revision_hyperparameter_training')
    p.add_argument('--episodes',type=int,default=3000)
    p.add_argument('--device',default='auto')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();selection=load_selection(args.hyperparameter_file);commands=[]
    for method in args.methods:
        directory=args.model_root/selection['methods'][method]['configuration_id']
        commands.append([sys.executable,str(ROOT/'comparison_runners/train_drl_baselines.py'),
            '--methods',method,'--training-seeds',*map(str,args.training_seeds),
            '--hyperparameter-file',str(args.hyperparameter_file),'--reward-scale-file',str(args.reward_scale_file),
            '--episodes',str(args.episodes),'--device',args.device,'--output-dir',str(directory)])
    if args.dry_run:print(json.dumps(dict(commands=commands),indent=2));return
    for command in commands:subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()
