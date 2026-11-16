"""Train only the empty-queue maintenance extension; reuse standard HIPMC as reference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from comparison_runners.drl_common import ensure_training_manifest,ensure_validation_manifest
from comparison_runners.revision_special.common import finish_training
from comparison_runners.evaluate_revision_nonlearning import resolve
from experiment.evaluation import seed_simulation
from experiment.seeds import TrainingSeeds
from experiment.selected_hyperparameters import load_selection,selected_parameters
from experiment.revision_protocol import TRAINING_SEEDS,VALIDATION_INSTANCES
from idle_maintenance_experiments.common import build_environment,training_condition,DEFAULT_SCALE
from idle_maintenance_experiments.environment import PROTOCOLS


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocols',nargs='+',choices=('idle_opportunity',),default=['idle_opportunity'])
    p.add_argument('--training-seeds',nargs='+',type=int,default=list(TRAINING_SEEDS))
    p.add_argument('--episodes',type=int,default=500)
    p.add_argument('--validation-instances',type=int,default=VALIDATION_INSTANCES)
    p.add_argument('--validation-interval',type=int,default=10)
    p.add_argument('--checkpoint-interval',type=int,default=25)
    p.add_argument('--device',default='auto')
    p.add_argument('--hyperparameter-file',type=Path,required=True)
    p.add_argument('--reward-scale-file',type=Path,default=DEFAULT_SCALE)
    p.add_argument('--training-manifest',type=Path,default=ROOT/'experiment_manifests/e1_learned_method_training_cases.csv')
    p.add_argument('--validation-manifest',type=Path,default=ROOT/'experiment_manifests/e1_learned_method_validation_cases.csv')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/revision_idle_maintenance_training')
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args()


def execute(args):
    plan=dict(protocols=args.protocols,training_seeds=args.training_seeds,episodes=args.episodes,
        tasks=len(args.protocols)*len(args.training_seeds),reference='existing three-action HIPMC; no separate control training')
    if args.dry_run:print(json.dumps(plan,indent=2));return plan
    hp=selected_parameters(load_selection(resolve(args.hyperparameter_file)),'ours_categorical_baa')
    scale_path=resolve(args.reward_scale_file)
    artifact=json.loads(scale_path.read_text(encoding='utf-8'))
    scale=float(artifact.get('value',artifact.get('scale',{}).get('value')))
    scale_sha=hashlib.sha256(scale_path.read_bytes()).hexdigest()
    training=ensure_training_manifest(resolve(args.training_manifest),args.training_seeds,args.episodes)
    validation=ensure_validation_manifest(resolve(args.validation_manifest),args.validation_instances)
    for protocol in args.protocols:
        for seed in args.training_seeds:
            output=resolve(args.output_dir)/protocol/f'seed-{seed}'
            marker=output/'run_complete.json'
            if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))['episodes']>=args.episodes:
                print(f'[IDLE-TRAIN] SKIP {protocol} seed={seed}',flush=True);continue
            seed_simulation(seed)
            env=build_environment(protocol,training_condition(seed),device=args.device,scale=scale,
                                   scale_sha=scale_sha,hyperparameters=hp)
            digest=hashlib.sha256()
            for name,state in env.mappo.inference_state().items():
                for key,tensor in state.items():digest.update((name+key).encode());digest.update(tensor.cpu().numpy().tobytes())
            env.training_seed_schedule=[TrainingSeeds(int(r['episode'])-1,int(r['instance_seed']),
                int(r['simulation_seed']),int(r['policy_seed'])) for r in training.rows
                if int(r['training_seed'])==seed and int(r['episode'])<=args.episodes]
            env.validation_seed_schedule=[{k:int(r[k]) for k in ('instance_seed','simulation_seed','policy_seed')}
                                          for r in validation.rows]
            env.validation_instances=args.validation_instances;env.validation_interval=args.validation_interval
            env.checkpoint_interval=args.checkpoint_interval
            contract=dict(experiment='empty-queue preventive-maintenance extension',protocol=protocol,training_seed=seed,
                episodes=args.episodes,initial_parameter_sha256=digest.hexdigest(),hyperparameters=hp,
                feature_contract=env.variant_contract(),reward_normalization=dict(value=scale,artifact_sha256=scale_sha))
            print(f'[IDLE-TRAIN] START {protocol} seed={seed} episodes={args.episodes}',flush=True)
            finish_training(env,'ours_categorical_baa',output,episodes=args.episodes,contract=contract)
    return plan


if __name__=='__main__':execute(parse_args())
