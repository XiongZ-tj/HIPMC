"""Entry-point assembly tests only; never run formal experiments."""
import ast
import contextlib
import io
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from comparison_runners.revision_workflow import run as workflow
from params_analysis import train_revision as grid


def built(action, *flags):
    args, extra = workflow.parse_args([action, *flags])
    return workflow.build_commands(args, extra)


def value(command, flag):
    return command[command.index(flag) + 1]


class WorkflowTests(unittest.TestCase):
    def test_every_action_targets_existing_script_and_supported_flags(self):
        delegates = {'evaluate_learned_methods.py': 'comparison_runners/evaluate_drl_baselines.py'}
        for action in workflow.ACTIONS:
            for command in built(action):
                with self.subTest(action=action, script=command[1]):
                    path = Path(command[1])
                    self.assertTrue(path.is_file())
                    source = workflow.ROOT / delegates[path.name] if path.name in delegates else path
                    tree = ast.parse(source.read_text(encoding='utf-8'))
                    allowed = {arg.value for call in ast.walk(tree) if isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute) and call.func.attr == 'add_argument'
                        for arg in call.args if isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str) and arg.value.startswith('--')}
                    # BooleanOptionalAction produces these negative forms.
                    allowed |= {'--no-' + x[2:] for x in allowed}
                    requested = {x.split('=')[0] for x in command[2:] if x.startswith('--')}
                    self.assertFalse(requested - allowed, requested - allowed)
                    self.assertNotIn('--gamma-one', command)

    def test_reference_is_one_existing_grid_cell_and_main_grid_reuses_root(self):
        reference = built('train-reference')[0]
        full = built('train-grid')[0]
        self.assertEqual(value(reference, '--output-dir'), value(full, '--output-dir'))
        self.assertEqual(value(reference, '--episodes'), '500')
        self.assertEqual(reference[reference.index('--training-seeds') + 1:reference.index('--device')], ['101','202','303'])
        with patch('sys.argv', ['grid', *reference[2:]]):
            args = grid.parse_args()
        self.assertEqual(len(grid.commands(args)), 1)
        with patch('sys.argv', ['grid', *full[2:]]):
            args = grid.parse_args()
        self.assertEqual(len(grid.commands(args)), 12)

    def test_aliases_and_selected_model_sources_are_consistent(self):
        command = built('evaluate-generated', '--methods', 'hipmc', 'hdrl')[0]
        self.assertEqual(command[command.index('--methods') + 1:command.index('--training-seeds')],
                         ['ours_categorical_baa', 'hdrl_option'])
        for action in ('evaluate-generated','evaluate-extended','evaluate-ablations',
                       ):
            command = built(action)[0]
            self.assertEqual(value(command, '--model-root'), workflow.GRID)
            self.assertEqual(value(command, '--model-layout'), 'grid')
        self.assertEqual(value(built('evaluate-extended')[0], '--model-budget'), '3000')
        self.assertEqual(value(built('train-industrial','--mode','finetune')[0], '--source-model-root'), workflow.GRID)
        self.assertEqual(value(built('train-industrial','--mode','finetune')[0], '--episodes'), '200')

    def test_one_scene_default_explicit_other_scene_and_forwarded_native_options(self):
        command = built('evaluate-nonlearning','--methods','ma','--phase','plan','--scenes','S3','--ma-threads','4')[0]
        self.assertEqual(value(command, '--scenes'), 'S3')
        self.assertEqual(value(command, '--phase'), 'plan')
        self.assertEqual(value(command, '--ma-threads'), '4')
        self.assertEqual(value(command, '--plan-dir'), workflow.PLANS)
        ablations=built('evaluate-ablations')[0]
        index=ablations.index('--scenes')
        self.assertEqual(ablations[index+1:index+7],['S0','S1','S2','S3','S4','S5'])
        self.assertNotIn('evaluate-maintenance-control', workflow.ACTIONS)

    def test_industrial_nonlearners_have_separate_statistics_source_directories(self):
        commands = built('evaluate-industrial-nonlearning','--methods','sdr','composite')
        self.assertEqual(len(commands), 2)
        for command, method in zip(commands, ('sdr','composite')):
            self.assertEqual(command[command.index('--methods') + 1:command.index('--phase')], [method])
            self.assertEqual(value(command, '--output-dir'), 'runs/revision_industrial_' + method)

    def test_dry_run_has_no_subprocess_or_output(self):
        with tempfile.TemporaryDirectory(prefix='revision_workflow_dry_') as temp:
            target = Path(temp) / 'must-not-be-created'
            args, extra = workflow.parse_args(['train-reference','--dry-run','--output-dir',str(target)])
            with patch.object(workflow.subprocess,'run') as child, contextlib.redirect_stdout(io.StringIO()):
                commands = workflow.execute(args, extra)
            child.assert_not_called()
            self.assertEqual(len(commands),1)
            self.assertFalse(target.exists())

    def test_actual_execution_only_selected_task_with_workspace_cwd(self):
        args, extra = workflow.parse_args(['evaluate-generated','--methods','hipmc'])
        with patch.object(workflow.subprocess,'run') as child, contextlib.redirect_stdout(io.StringIO()):
            workflow.execute(args, extra)
        child.assert_called_once()
        self.assertEqual(child.call_args.kwargs['cwd'],workflow.ROOT)
        self.assertTrue(child.call_args.kwargs['check'])

    def test_absolute_entry_can_be_previewed_from_other_directory(self):
        with tempfile.TemporaryDirectory(prefix='revision_workflow_cwd_') as temp:
            result=subprocess.run([sys.executable,str(Path(workflow.__file__)), 'train-reference', '--dry-run'],
                cwd=temp,check=True,capture_output=True,text=True)
            self.assertIn(workflow.REFERENCE,result.stdout)
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_all_current_training_entries_share_ten_validation_instances(self):
        from comparison_runners import train_drl_baselines
        from comparison_runners.revision_special import train_industrial
        from idle_maintenance_experiments import train_revision as idle
        for module, flags in ((grid,['--reward-scale-file','unused.json']),
                              (train_industrial,['--hyperparameter-file','unused.json']),
                              (idle,['--hyperparameter-file','unused.json']),
                              (train_drl_baselines,[])):
            with patch('sys.argv',['test',*flags]):
                self.assertEqual(module.parse_args().validation_instances,10)
        for action in ('train-reference','train-grid','train-industrial','train-idle'):
            self.assertEqual(value(built(action)[0],'--validation-instances'),'10')

    def test_old_thirty_case_pool_uses_same_ten_case_prefix_without_rewrite(self):
        from comparison_runners.drl_common import ensure_validation_manifest
        with tempfile.TemporaryDirectory(prefix='revision_validation_budget_') as temp:
            path=Path(temp)/'validation.csv'
            original=ensure_validation_manifest(path,30)
            contents=path.read_bytes()
            reduced=ensure_validation_manifest(path,10)
            self.assertEqual(len(reduced.rows),10)
            self.assertEqual(reduced.rows,original.rows[:10])
            self.assertEqual(path.read_bytes(),contents)
            self.assertEqual(reduced,ensure_validation_manifest(path,10))

    def test_industrial_ten_cases_are_shared_and_keep_all_load_profiles(self):
        from comparison_runners.industrial_case.training_schedule import schedules
        validation=[]
        for seed in (101,202,303):
            _,cases,loads=schedules(seed,500,10)
            self.assertEqual(len(cases),10)
            self.assertEqual([sum(loads[c['instance_seed']]==level for c in cases)
                              for level in (.9,1.,1.1)],[4,3,3])
            validation.append(cases)
        self.assertTrue(all(cases==validation[0] for cases in validation))


if __name__ == '__main__':
    unittest.main()
