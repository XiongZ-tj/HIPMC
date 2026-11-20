"""Original-budget tuning, frozen shared weights, and isolated output contracts."""
import contextlib
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from comparison_runners.composite_priority import retune_revision as runner


class RetuningTests(unittest.TestCase):
    def test_original_budget_explicit(self):
        args=runner.parse_args(['tune','--dry-run'])
        _,cmd,_=runner.command(args)
        expected={'--optimization-seeds':'101','--search-instances':'30','--selection-instances':'30',
                  '--search-replications':'3','--selection-replications':'10',
                  '--random-candidates':'128','--local-per-step':'16','--finalists':'10'}
        for option,value in expected.items():self.assertEqual(cmd[cmd.index(option)+1],value)

    def test_all_children_accept_flags_and_share_selected_weights(self):
        weights=[]
        with tempfile.TemporaryDirectory() as tmp:
            for action in runner.ACTIONS:
                args=runner.parse_args([action,'--output-root',tmp,'--dry-run'])
                root,cmd,w=runner.command(args)
                module=importlib.import_module('.'.join(Path(cmd[1]).relative_to(runner.ROOT).with_suffix('').parts))
                with patch('sys.argv',[cmd[1],*cmd[2:]]):module.parse_args()
                self.assertTrue(Path(cmd[cmd.index('--output-dir')+1]).is_relative_to(root))
                if action!='tune':
                    self.assertIn('--no-save-job-records',cmd)
                    weights.append(cmd[cmd.index('--weights-file')+1])
            self.assertEqual(len(set(weights)),1)

    def test_dry_run_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'not-created'
            with contextlib.redirect_stdout(io.StringIO()),patch.object(runner.subprocess,'Popen') as process:
                for action in runner.ACTIONS:
                    runner.execute(runner.parse_args([action,'--output-root',str(out),'--dry-run']))
                process.assert_not_called()
            self.assertFalse(out.exists())

    def test_evaluation_requires_fresh_tuning(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=runner.parse_args(['evaluate-generated','--output-root',tmp])
            with self.assertRaises(FileNotFoundError):runner.execute(args)

    def test_replaced_selected_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'tuning').mkdir();(root/'inputs').mkdir()
            (root/'tuning/tuned_composite_priority_weights.json').write_text('{"weights":[1,0]}')
            (root/'inputs/frozen_selected_weights.json').write_text('{"weights":[0,1]}')
            with self.assertRaisesRegex(ValueError,'Selected weights changed'):
                runner.execute(runner.parse_args(['evaluate-generated','--output-root',tmp]))


if __name__=='__main__':unittest.main()
