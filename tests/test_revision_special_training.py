"""CPU-only temporary training, transfer, industrial deployment and idle tests."""
import contextlib
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from experiment.artifacts import atomic_json_save,atomic_torch_save
from experiment.revision_protocol import LEARNED_METHODS
from experiment.selected_hyperparameters import load_selection,selected_parameters,source_directory
from comparison_runners.drl_common import simulation_core
from comparison_runners.revision_special import train_industrial as training
from comparison_runners.revision_special import evaluate_industrial as evaluation
from comparison_runners.revision_special import evaluate_small_milp as small_milp
from comparison_runners.revision_special.common import construct,warm_start
from comparison_runners.evaluate_revision_nonlearning import read
from params_analysis import train_revision as grid
from params_analysis.grid import full_factorial_grid
from idle_maintenance_experiments import train_revision as idle_train
from idle_maintenance_experiments import evaluate_revision as idle_eval


def parsed(module):
    argv=['test']
    if module in (training,idle_train):argv+=['--hyperparameter-file','not-yet-selected.json']
    if module is grid:argv+=['--reward-scale-file','not-yet-calibrated.json']
    with patch('sys.argv',argv):return module.parse_args()


class SpecialTrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_independent_author_selections_and_selected_model_path(self):
        from params_analysis import select_revision
        with tempfile.TemporaryDirectory(prefix='revision_selection_cpu_') as temp:
            root=Path(temp);path=root/'selected.json'
            for i,method in enumerate(LEARNED_METHODS):
                configuration=full_factorial_grid()[i]
                with patch('sys.argv',['select','--method',method,'--configuration-id',configuration.configuration_id,
                    '--reason','Temporary test, not a scientific parameter choice','--output',str(path)]),contextlib.redirect_stdout(io.StringIO()):
                    select_revision.main()
            selection=load_selection(path)
            self.assertEqual(set(selection['methods']),set(LEARNED_METHODS))
            self.assertEqual(selected_parameters(selection,'ddqnb')['gamma'],.95)
            directory=source_directory(root,'ours_categorical_baa',101,selection=selection,grid_layout=True)
            self.assertIn(selection['methods']['ours_categorical_baa']['configuration_id'],str(directory))
            self.assertFalse(selection['test_results_used_for_selection'])

    def test_tuning_matrix_and_dry_run_budgets(self):
        self.assertEqual(len(full_factorial_grid()),12)
        self.assertEqual({c.clip_ratio for c in full_factorial_grid()},{.2})
        self.assertEqual({c.discount_factor for c in full_factorial_grid()},{.9,.95,.99,1.})
        args=parsed(grid);args.dry_run=True
        with contextlib.redirect_stdout(io.StringIO()):plan=grid.execute(args)
        self.assertEqual(plan['training_tasks'],144)
        for module,expected in ((training,12),(idle_train,3),(idle_eval,15600),(evaluation,21600)):
            args=parsed(module);args.dry_run=True
            with contextlib.redirect_stdout(io.StringIO()):plan=module.execute(args)
            self.assertEqual(plan.get('tasks',plan.get('runs')),expected)
        args=parsed(evaluation);args.small=True;args.dry_run=True
        with contextlib.redirect_stdout(io.StringIO()):plan=evaluation.execute(args)
        self.assertEqual(plan['modes'],['scratch'])
        self.assertEqual(plan['runs'],2400)

    def fixture(self,root):
        selection=root/'selection.json';scale=root/'scale.json'
        atomic_json_save({'selection_domain':'generated','methods':{method:{'configuration_id':'lr-1e-04_clip-0p20_gamma-0p95',
            'parameters':dict(learning_rate=1e-4,clip_ratio=.2,gamma=.95)} for method in LEARNED_METHODS}},selection)
        atomic_json_save(dict(value=1.,scale={'value':1.},contract_sha256='temporary-test'),scale)
        return selection,scale

    def source_models(self,root):
        for method in LEARNED_METHODS:
            env=construct(method,[1.,30,150,2.5,101],[2,2,1],(190,158,202,128,104),device='cpu')
            directory=root/method/'seed-101/budgets/episodes-500';directory.mkdir(parents=True)
            if method=='ours_categorical_baa':env.save_inference(directory/'hierarchical_inference.pt')
            elif method=='ddqnb':atomic_torch_save(env.controller.q1.state_dict(),directory/'ddqnb_model.pth')
            elif method=='psa_mappo':atomic_torch_save(env.controller.inference_state(),directory/'psa_mappo_model.pth')
            else:env.save_inference(directory)

    def test_warm_start_synchronizes_all_dqn_targets(self):
        with tempfile.TemporaryDirectory(prefix='revision_transfer_cpu_') as temp:
            root=Path(temp);self.source_models(root)
            for method in ('ddqnb','hdrl_option'):
                env=construct(method,[1.,18,89,1.,101],[1,1,1],(128,190,104,293,202,279),device='cpu')
                warm_start(env,method,root/method/'seed-101/budgets/episodes-500')
                pairs=[(env.controller.q1,env.controller.q2)] if method=='ddqnb' else [(n['q1'],n['q2']) for n in env.controller.nets.values()]
                self.assertTrue(all(torch.equal(a,b) for q1,q2 in pairs for a,b in zip(q1.parameters(),q2.parameters())))

    def test_real_industrial_training_validation_three_modes_and_small_cases(self):
        with tempfile.TemporaryDirectory(prefix='revision_industrial_cpu_') as temp:
            root=Path(temp);selection,scale=self.fixture(root)
            self.source_models(root/'source')
            args=parsed(training)
            args.hyperparameter_file=selection;args.reward_scale_file=scale
            args.output_dir=root/'trained';args.source_model_root=root/'source';args.training_seeds=[101]
            args.episodes=2;args.validation_instances=1;args.validation_interval=1;args.checkpoint_interval=1;args.device='cpu'
            actual_construct=training.construct
            def fast(*a,**kw):
                env=actual_construct(*a,**kw)
                if hasattr(env,'mappo'):env.config=replace(env.config,rollout_episodes=2)
                if hasattr(env,'update_interval'):env.update_interval=2
                return env
            with patch.object(training,'construct',side_effect=fast),contextlib.redirect_stdout(io.StringIO()):
                for mode in ('scratch','finetune'):args.mode=mode;training.execute(args)
            for mode in ('scratch','finetune'):
                for method in LEARNED_METHODS:
                    directory=root/'trained'/mode/method/'seed-101'
                    config=json.loads((directory/'run_config.json').read_text(encoding='utf-8'))
                    self.assertEqual(config['hyperparameters']['learning_rate'],1e-4)
                    self.assertEqual(config['hyperparameters']['gamma_baa'],.95)
                    # HIPMC's detached evaluator must actually see the 89 orders.
                    logs=read(directory/'Training_Validation.csv')
                    self.assertTrue(all(float(r['job_count_mean'])==89 for r in logs))
                    self.assertTrue(all(float(r['initial_job_count_mean'])==18 for r in logs))
                    self.assertTrue(all(abs(float(r['target_load_mean'])-.9)<1e-8 for r in logs))
            ev=parsed(evaluation);ev.source_model_root=root/'source';ev.model_root=root/'trained'
            ev.scratch_budget=ev.finetuning_budget=2;ev.training_seeds=[101]
            ev.instance_limit=1;ev.replications=1;ev.device='cpu';ev.output_dir=root/'evaluation'
            with contextlib.redirect_stdout(io.StringIO()):blocks=evaluation.execute(ev)
            self.assertEqual(sum(b['runs'] for b in blocks),12)
            self.assertEqual(sum(b['failed_runs'] for b in blocks),0)
            ev.small=True;ev.output_dir=root/'small'
            with contextlib.redirect_stdout(io.StringIO()):blocks=evaluation.execute(ev)
            self.assertEqual(sum(b['runs'] for b in blocks),4)
            self.assertEqual(sum(b['failed_runs'] for b in blocks),0)
            self.assertTrue(all(float(r['job_count'])==10 for r in read(ev.output_dir/'run_metrics.csv')))

    def test_extension_only_training_and_six_scene_replay(self):
        with tempfile.TemporaryDirectory(prefix='revision_idle_cpu_') as temp:
            root=Path(temp);selection,scale=self.fixture(root)
            args=parsed(idle_train);args.hyperparameter_file=selection;args.reward_scale_file=scale
            args.output_dir=root/'models';args.training_manifest=root/'train.csv';args.validation_manifest=root/'validation.csv'
            args.training_seeds=[101];args.episodes=2;args.validation_instances=1
            args.validation_interval=args.checkpoint_interval=1;args.device='cpu'
            original=idle_train.build_environment
            def fast(*a,**kw):
                env=original(*a,**kw);env.config=replace(env.config,rollout_episodes=2);return env
            with patch.object(idle_train,'build_environment',side_effect=fast),contextlib.redirect_stdout(io.StringIO()):idle_train.execute(args)
            configs=[json.loads((args.output_dir/p/'seed-101/run_config.json').read_text(encoding='utf-8')) for p in args.protocols]
            self.assertEqual(args.protocols,['idle_opportunity'])
            self.assertEqual(len(configs),1)
            ev=parsed(idle_eval);ev.model_root=args.output_dir;ev.output_dir=root/'evaluation';ev.model_budget=2
            ev.training_seeds=[101];ev.instances=ev.replications=1;ev.condition_limit=1;ev.device='cpu'
            with contextlib.redirect_stdout(io.StringIO()):blocks=idle_eval.execute(ev)
            self.assertEqual(len(blocks),6)
            self.assertEqual(sum(b['failed_runs'] for b in blocks),0)
            # Reported per-machine executed downtime matches episode totals.
            totals={}
            for row in read(ev.output_dir/'machine_metrics.csv'):
                key=(row['scene_id'],row['protocol'])
                totals[key]=totals.get(key,0.)+float(row['maintenance_time'])
            for row in read(ev.output_dir/'run_metrics.csv'):
                self.assertAlmostEqual(totals[(row['scene_id'],row['protocol'])],float(row['maintenance_time']))

    def test_small_milp_replay_and_reference_fields(self):
        with tempfile.TemporaryDirectory(prefix='revision_small_milp_cpu_') as temp:
            root=Path(temp);args=parsed(small_milp);args.milp_results=root/'solver';args.output_dir=root/'evaluation'
            args.instance_limit=1;args.replications=2
            cases=evaluation.industrial_cases(args.dataset_dir,True,1,2)
            raw=read(args.dataset_dir/cases[0]['job_file'])
            atomic_json_save(dict(objective=100.,obj_bound=90.,mip_gap=.1,is_optimal=False,
                schedule=[dict(machine_id=i%3,slot=i,family=int(r['family']),job_ids=[int(r['job_id'])]) for i,r in enumerate(raw)]),
                args.milp_results/'jobs-10_instance-001_milp.json')
            with contextlib.redirect_stdout(io.StringIO()):blocks=small_milp.execute(args)
            self.assertEqual(sum(b['failed_runs'] for b in blocks),0)
            for row in read(args.output_dir/'run_metrics.csv'):
                self.assertEqual(row['milp_reference_type'],'time_limited_incumbent')
                self.assertEqual(float(row['ipm_count'])+float(row['pm_count']),0.)
                self.assertAlmostEqual(float(row['gap_to_milp_reference_percent']),float(row['total_tardiness'])-100.)


if __name__=='__main__':unittest.main()
