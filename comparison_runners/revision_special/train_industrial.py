"""Industrial scratch/fine-tuning with each method's generated-domain selection."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from comparison_runners.revision_special.common import construct,warm_start,parameters_for,finish_training
from comparison_runners.industrial_case.training_schedule import (
    schedules,install_record_schedule,install_industrial_validation,install_industrial_pretraining_reference)
from comparison_runners.industrial_case.case_common import (
    PROCESSING_TIMES,MACHINE_COUNTS,INITIAL_JOBS,TOTAL_JOBS,adapt_simulator)
from comparison_runners.drl_common import simulation_core,checkpoint_name
from comparison_runners.evaluate_revision_nonlearning import resolve
from experiment.artifacts import atomic_csv_save
from experiment.evaluation import seed_simulation
from experiment.revision_protocol import LEARNED_METHODS,TRAINING_SEEDS,VALIDATION_INSTANCES
from experiment.selected_hyperparameters import load_selection,source_directory


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',choices=LEARNED_METHODS,default=list(LEARNED_METHODS))
    p.add_argument('--training-seeds',nargs='+',type=int,default=list(TRAINING_SEEDS))
    p.add_argument('--mode',choices=('scratch','finetune'),default='scratch')
    p.add_argument('--episodes',type=int,help='Default 500 scratch, 200 fine-tuning.')
    p.add_argument('--validation-instances',type=int,default=VALIDATION_INSTANCES)
    p.add_argument('--validation-interval',type=int,default=10)
    p.add_argument('--checkpoint-interval',type=int,default=25)
    p.add_argument('--device',default='auto')
    p.add_argument('--hyperparameter-file',type=Path,required=True)
    p.add_argument('--reward-scale-file',type=Path,default=ROOT/'reward_scales/offline_calibrated_sdr_reward_scale_aerospace_heat_treatment_industrial_case.json')
    p.add_argument('--source-model-root',type=Path,default=ROOT/'runs/e1_learned_method_training')
    p.add_argument('--source-layout',choices=('standard','grid'),default='standard')
    p.add_argument('--source-budget',type=int,default=500)
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/revision_industrial_training')
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args()


def execute(args):
    episodes=args.episodes if args.episodes is not None else 500 if args.mode=='scratch' else 200
    plan=dict(domain='industrial',mode=args.mode,methods=args.methods,seeds=args.training_seeds,
        episodes=episodes,tasks=len(args.methods)*len(args.training_seeds),
        hyperparameter_selection_domain='generated')
    if args.dry_run:print(json.dumps(plan,indent=2));return plan
    selection=load_selection(resolve(args.hyperparameter_file))
    reward_path=resolve(args.reward_scale_file)
    reward=json.loads(reward_path.read_text(encoding='utf-8'))
    from comparison_runners.drl_common import file_sha256
    scale=float(reward.get('value',reward.get('scale',{}).get('value')))
    for method in args.methods:
        for seed in args.training_seeds:
            output=resolve(args.output_dir)/args.mode/method/f'seed-{seed}'
            marker=output/'run_complete.json'
            if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))['episodes']>=episodes:
                print(f'[INDUSTRIAL-TRAIN] SKIP {method} seed={seed} mode={args.mode}',flush=True);continue
            seed_simulation(seed)
            environment=construct(method,[1.,INITIAL_JOBS,TOTAL_JOBS,1.,seed],MACHINE_COUNTS,PROCESSING_TIMES,
                device=args.device,scale=scale,scale_sha=file_sha256(reward_path))
            core=simulation_core(environment);adapt_simulator(core)
            hp=parameters_for(environment,selection,method)
            training,validation,loads=schedules(seed,episodes,args.validation_instances)
            records=install_record_schedule(environment,loads)
            environment.training_seed_schedule=training;environment.validation_seed_schedule=validation
            environment.validation_instances=args.validation_instances;environment.validation_interval=args.validation_interval
            environment.checkpoint_interval=args.checkpoint_interval
            if hasattr(environment,'mappo'):
                install_industrial_validation(environment,validation,records,loads)
                install_industrial_pretraining_reference(environment,validation,records,output)
            source=None
            if args.mode=='finetune':
                source=source_directory(resolve(args.source_model_root),method,seed,args.source_budget,
                    selection=selection,grid_layout=args.source_layout=='grid')
                if not (output/checkpoint_name(method)).exists():warm_start(environment,method,source)
            contract=dict(experiment='industrial-case revision training',method=method,training_seed=seed,
                mode=args.mode,episodes=episodes,hyperparameters=hp,hyperparameter_selection=selection,
                source_model=str(source) if source else None,source_budget=args.source_budget if source else None,
                machine_counts=list(MACHINE_COUNTS),machine_capacities=[120,160,200],
                processing_times=list(PROCESSING_TIMES),formal_test_dataset_used_for_training=False,
                reward_normalization=dict(value=scale,artifact_sha256=file_sha256(reward_path)),
                validation_instances=args.validation_instances,validation_interval=args.validation_interval,
                initialization='full_inference_networks_fresh_optimizer_replay_and_exploration' if source else 'random')
            output.mkdir(parents=True,exist_ok=True)
            atomic_csv_save([dict(training_seed=seed,**item.to_dict(),target_load=loads[item.instance_seed])
                             for item in training],output/'training_cases.csv')
            atomic_csv_save([dict(**item,target_load=loads[item['instance_seed']]) for item in validation],output/'validation_cases.csv')
            print(f'[INDUSTRIAL-TRAIN] START {method} seed={seed} mode={args.mode} episodes={episodes} hp={hp}',flush=True)
            finish_training(environment,method,output,episodes=episodes,contract=contract)
    return plan


if __name__=='__main__':execute(parse_args())
