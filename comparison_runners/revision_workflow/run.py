"Dispatch independently runnable training, evaluation and aggregation tasks."
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment.revision_protocol import LEARNED_METHODS, TRAINING_SEEDS, VALIDATION_INSTANCES, TEST_INSTANCES_PER_CONDITION
from experiment.revision_selection import COMPONENT_SCENES

SELECTION = 'experiment_manifests/selected_generated_hyperparameters.json'
GENERATED_SCALE = 'reward_scales/offline_calibrated_sdr_reward_scale_base_finite_load100_jobs150_due2p5_deg100.json'
INDUSTRIAL_SCALE = 'reward_scales/offline_calibrated_sdr_reward_scale_aerospace_heat_treatment_industrial_case.json'
GRID = 'runs/revision_hyperparameter_training'
INDUSTRIAL = 'runs/revision_industrial_training'
PLANS = 'runs/revision_offline_ma_plans'
REFERENCE = 'lr-2e-04_clip-0p20_gamma-0p95'
ALIASES = {'hipmc': 'ours_categorical_baa', 'hdrl': 'hdrl_option'}

ACTIONS = (
    'calibrate-generated', 'calibrate-industrial',
    'train-reference', 'train-grid', 'summarize-generated-training',
    'train-industrial', 'summarize-industrial-training',
    'evaluate-generated', 'evaluate-nonlearning', 'evaluate-ablations',
    'evaluate-industrial', 'evaluate-industrial-nonlearning',
    'evaluate-small-milp', 'evaluate-small-drl',
    'continue-extended', 'evaluate-extended', 'summarize-extended-training',
    'train-idle', 'evaluate-idle',
    'summarize-results', 'summarize-ma', 'summarize-rh',
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('action', choices=ACTIONS)
    p.add_argument('--methods', nargs='+', help='hipmc, ddqnb, psa_mappo, hdrl; or non-learning method IDs.')
    p.add_argument('--training-seeds', nargs='+', type=int, default=list(TRAINING_SEEDS))
    p.add_argument('--scenes', nargs='+', default=None, help='Default S0-S5 for component ablations; S0 for other actions.')
    p.add_argument('--device', default='auto')
    p.add_argument('--instances', type=int, default=TEST_INSTANCES_PER_CONDITION,
                   help='Generated evaluations only; keeps original per-instance seeds.')
    p.add_argument('--mode', choices=('scratch', 'finetune'), default='scratch')
    p.add_argument('--phase', choices=('plan', 'evaluate', 'all'), default='evaluate')
    p.add_argument('--hyperparameter-file', default=SELECTION)
    p.add_argument('--reference-configuration', default=REFERENCE,
                   help='One grid cell for initial review, not the selected final configuration.')
    p.add_argument('--output-dir', help='Override this task output; update analysis sources if performance paths change.')
    p.add_argument('--plan-root', default=PLANS, help='For MA search-summary task.')
    p.add_argument('--dry-run', action='store_true', help='Print CMD-compatible commands without running any child or writing files.')
    args,extra=p.parse_known_args(argv)
    if args.scenes is None:
        args.scenes=list(COMPONENT_SCENES) if args.action == 'evaluate-ablations' else ['S0']
    return args,extra


def build_commands(args, extra=()):
    """Pure command construction. Do not read selected models or create files."""
    action = args.action
    commands = []
    chosen = [ALIASES.get(m, m) for m in args.methods] if args.methods else list(LEARNED_METHODS)
    seeds = ['--training-seeds', *map(str, args.training_seeds)]
    methods = ['--methods', *chosen]
    selected = ['--hyperparameter-file', args.hyperparameter_file]
    device = ['--device', args.device]
    scenes = ['--scenes', *args.scenes]
    instances = ['--instances', str(args.instances)]

    def task(script, flags=(), output=None):
        command = [sys.executable, str(ROOT / script), *map(str, flags)]
        target = args.output_dir or output
        if target is not None:
            command += ['--output-dir', target]
        command += list(extra)
        commands.append(command)

    def generated_models(budget=500):
        return ['--model-root', GRID, '--model-layout', 'grid', *selected, '--model-budget', str(budget)]

    if action == 'calibrate-generated':
        task('tools/calibrate_reward_scale.py', ['--scenario', 'base', '--target-load', '1.00',
            '--initial-jobs', '30', '--total-jobs', '150', '--due-date-factor', '2.5',
            '--instances', '200', '--replications', '3', '--output', GENERATED_SCALE])
    elif action == 'calibrate-industrial':
        task('comparison_runners/industrial_case/calibrate_reward_scale.py',
             ['--instances-per-load', '60', '--replications', '3', '--output', INDUSTRIAL_SCALE])
    elif action in ('train-reference', 'train-grid'):
        flags = [*methods, *seeds, *device, '--episodes', '500', '--validation-instances', str(VALIDATION_INSTANCES),
                 '--validation-interval', '10', '--checkpoint-interval', '25', '--reward-scale-file', GENERATED_SCALE]
        if action == 'train-reference':
            flags += ['--configuration-ids', args.reference_configuration]
        task('params_analysis/train_revision.py', flags, GRID)
    elif action in ('summarize-generated-training', 'summarize-industrial-training', 'summarize-extended-training'):
        budget = 3000 if action == 'summarize-extended-training' else 200 if action == 'summarize-industrial-training' and args.mode == 'finetune' else 500
        source = INDUSTRIAL + '/' + args.mode if action == 'summarize-industrial-training' else GRID
        folder = {'summarize-generated-training': 'revision_hyperparameter_statistics',
                  'summarize-industrial-training': f'revision_industrial_{args.mode}_training_statistics',
                  'summarize-extended-training': 'revision_extended_training_statistics'}[action]
        task('result_analysis/training.py', ['--training-roots', source, '--budget', str(budget),
             '--late-window', '20' if budget == 200 else '50'], 'runs/' + folder)
    elif action == 'train-industrial':
        flags = [*methods, *seeds, *device, *selected, '--mode', args.mode,
                 '--episodes', '500' if args.mode == 'scratch' else '200', '--reward-scale-file', INDUSTRIAL_SCALE,
                 '--validation-instances', str(VALIDATION_INSTANCES), '--validation-interval', '10', '--checkpoint-interval', '25']
        if args.mode == 'finetune':
            flags += ['--source-model-root', GRID, '--source-layout', 'grid', '--source-budget', '500']
        task('comparison_runners/revision_special/train_industrial.py', flags, INDUSTRIAL)
    elif action in ('evaluate-generated', 'evaluate-extended'):
        budget = 3000 if action == 'evaluate-extended' else 500
        task('comparison_runners/evaluate_learned_methods.py',
             [*methods, *seeds, *device, *scenes, *instances, *generated_models(budget), '--no-save-job-records', '--log-every', '10'],
             'runs/revision_extended_budget_evaluation' if budget == 3000 else 'runs/revision_generated_drl_evaluation')
    elif action in ('evaluate-nonlearning', 'evaluate-industrial-nonlearning'):
        nonlearning = args.methods or ['sdr', 'ma', 'rh_alns', 'composite']
        industrial = action == 'evaluate-industrial-nonlearning'
        flags = ['--methods', *nonlearning, '--phase', args.phase, '--replications', '10',
                 '--rh-evaluations', '8000', '--ma-population', '100', '--ma-generations', '2000',
                 '--ma-refinement-generations', '200', '--ma-local-search-evaluations', '500',
                 '--weights-file', 'experiment_manifests/tuned_composite_priority_weights.json',
                 '--plan-dir', 'runs/revision_industrial_ma_plans' if industrial else PLANS,
                 '--no-save-job-records', '--log-every', '10']
        if not industrial:
            flags += [*scenes, *instances]
        # Match the declared statistics sources, even for separately run methods.
        if industrial:
            for method in nonlearning:
                one = flags.copy(); one[1:1 + len(nonlearning)] = [method]
                task('comparison_runners/revision_special/evaluate_industrial_nonlearning.py', one,
                     'runs/revision_industrial_' + method)
        else:
            task('comparison_runners/evaluate_revision_nonlearning.py', flags, 'runs/revision_nonlearning_evaluation')
    elif action == 'evaluate-ablations':
        flags = [*seeds, *device, *scenes, *instances, '--replications', '10',
                 '--no-save-job-records', '--log-every', '10']
        flags += generated_models()
        output = 'runs/revision_component_ablation'
        task('ablation_experiments/evaluate_revision.py', flags, output)
    elif action in ('evaluate-industrial', 'evaluate-small-drl'):
        small = action == 'evaluate-small-drl'
        flags = [*methods, *seeds, *device, *selected, '--model-root', INDUSTRIAL,
                 '--scratch-budget', '500', '--finetuning-budget', '200', '--replications', '10',
                 '--no-save-job-records', '--log-every', '10']
        if small:
            flags += ['--small', '--modes', 'scratch', '--milp-results', 'runs/milp_small_industrial']
        else:
            flags += ['--modes', 'zero_shot', 'scratch', 'finetune', '--source-model-root', GRID,
                      '--source-layout', 'grid', '--source-budget', '500']
        task('comparison_runners/revision_special/evaluate_industrial.py', flags,
             'runs/revision_small_drl_evaluation' if small else 'runs/revision_industrial_evaluation')
    elif action == 'evaluate-small-milp':
        task('comparison_runners/revision_special/evaluate_small_milp.py',
             ['--milp-results', 'runs/milp_small_industrial', '--replications', '10', '--no-save-job-records'], 'runs/revision_small_milp_execution')
    elif action == 'continue-extended':
        task('params_analysis/continue_selected.py', [*methods, *seeds, *device, *selected,
             '--reward-scale-file', GENERATED_SCALE, '--model-root', GRID, '--episodes', '3000'])
    elif action == 'train-idle':
        task('idle_maintenance_experiments/train_revision.py', [*seeds, *device, *selected, '--protocols', 'idle_opportunity', '--validation-instances', str(VALIDATION_INSTANCES),
             '--episodes', '500', '--reward-scale-file', GENERATED_SCALE], 'runs/revision_idle_maintenance_training')
    elif action == 'evaluate-idle':
        task('idle_maintenance_experiments/evaluate_revision.py', [*seeds, *device, *scenes, '--protocols', 'idle_opportunity',
             '--model-root', 'runs/revision_idle_maintenance_training', '--model-budget', '500',
             *instances, '--replications', '10', '--no-save-job-records', '--log-every', '10'],
             'runs/revision_idle_maintenance_evaluation')
    elif action == 'summarize-results':
        task('result_analysis/summarize.py', ['--input-spec', 'experiment_manifests/revision_analysis_sources.json', '--instances', str(args.instances),
             '--bootstrap-draws', '2000', '--expected-replications', '10'], 'runs/revision_result_statistics')
    elif action == 'summarize-ma':
        task('result_analysis/optimization.py', ['--plan-root', args.plan_root, *([] if 'industrial' in args.plan_root else instances)], 'runs/revision_ma_search_statistics')
    elif action == 'summarize-rh':
        task('result_analysis/timing.py', ['--rh-trace', 'runs/revision_nonlearning_evaluation/partitions/reoptimization', *instances],
             'runs/revision_rh_timing_statistics')
    return commands


def execute(args, extra=()):
    commands = build_commands(args, extra)
    for i, command in enumerate(commands, 1):
        # Display PATH-based python; actual child inherits the interpreter used
        # to launch this script. Passing a list safely handles spaces on Windows.
        print(f'[REVISION-TASK] {args.action} {i}/{len(commands)}', flush=True)
        print(subprocess.list2cmdline(['python', *command[1:]]), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    return commands


if __name__ == '__main__':
    options, forwarded = parse_args()
    execute(options, forwarded)
