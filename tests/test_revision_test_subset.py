"""Prefix selection, result reuse and extension-only experiment tests."""
import csv
import io
import json
import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from comparison_runners.revision_workflow import run as workflow
from comparison_runners import evaluate_drl_baselines as drl
from idle_maintenance_experiments import train_revision as idle_train
from idle_maintenance_experiments import evaluate_revision as idle_eval
from result_analysis.ingest import EpisodeStore
from experiment.revision_protocol import TEST_INSTANCES_PER_CONDITION


class PrefixTests(unittest.TestCase):
    def test_original_manifest_retains_seeds(self):
        from experiment.config import E1_INSTANCE_COUNT
        self.assertEqual(E1_INSTANCE_COUNT,20)
        path=workflow.ROOT/'experiment_manifests/e1_formal_comparison_cases.csv'
        with path.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
        chosen=[r for r in rows if int(r['instance_index'])<TEST_INSTANCES_PER_CONDITION]
        self.assertEqual(len(rows),5400)
        self.assertEqual(len(chosen),2700)
        self.assertEqual(len({(r['condition_id'],r['instance_seed']) for r in chosen}),270)
        self.assertEqual(len({r['condition_id'] for r in chosen}),27)
        for row in chosen:self.assertIn(row,rows)

    def test_workflow_and_idle_defaults(self):
        for action in ('evaluate-generated','evaluate-extended','evaluate-nonlearning',
                       'evaluate-ablations','evaluate-idle','summarize-results'):
            args,extra=workflow.parse_args([action])
            command=workflow.build_commands(args,extra)[0]
            self.assertEqual(command[command.index('--instances')+1],'10')
        args,extra=workflow.parse_args(['evaluate-industrial-nonlearning','--methods','sdr','composite'])
        for command in workflow.build_commands(args,extra):self.assertNotIn('--instances',command)
        with patch('sys.argv',['test']):self.assertEqual(drl.parse_args().instances,10)
        with patch('sys.argv',['test','--hyperparameter-file','unused.json']):
            args=idle_train.parse_args();args.dry_run=True
            with contextlib.redirect_stdout(io.StringIO()):plan=idle_train.execute(args)
        self.assertEqual(plan['tasks'],3)
        with patch('sys.argv',['test']):args=idle_eval.parse_args()
        args.dry_run=True
        with contextlib.redirect_stdout(io.StringIO()):plan=idle_eval.execute(args)
        self.assertEqual(plan['runs'],15600)
        self.assertEqual(plan['protocols'],['idle_opportunity'])

    def test_existing_twenty_instance_results_filtered_without_touching_industrial(self):
        with tempfile.TemporaryDirectory(prefix='revision_prefix_test_') as temp:
            root=Path(temp);path=root/'raw.csv'
            rows=[dict(condition_id='c',instance_seed=str(i),simulation_seed='7',
                       method='ours_categorical_baa',training_seed='101',total_tardiness=i)
                  for i in range(20)]
            with path.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            allowed={('c',str(i),'7') for i in range(10)}
            store=EpisodeStore(root/'test.sqlite')
            try:
                store.ingest(dict(path=str(path),domain='generated'),root,allowed_cases=allowed)
                store.ingest(dict(path=str(path),domain='industrial'),root,allowed_cases=allowed)
                frame=store.aggregate()
                self.assertEqual(len(frame[frame.domain=='generated']),10)
                self.assertEqual(len(frame[frame.domain=='industrial']),20)
                self.assertEqual(len(path.read_text().splitlines()),21)
            finally:store.close()

    def test_idle_reference_uses_main_hipmc(self):
        spec=json.loads((workflow.ROOT/'experiment_manifests/revision_analysis_sources.json').read_text())
        self.assertEqual(spec['references']['idle'],'HIPMC')
        self.assertEqual(spec['reference_families']['idle'],'main')
        source=next(s for s in spec['sources'] if s['family']=='idle')
        self.assertEqual(source['protocols'],['idle_opportunity'])

    def test_formal_entries_disable_job_records(self):
        for action in ('evaluate-generated','evaluate-nonlearning','evaluate-industrial',
                       'evaluate-industrial-nonlearning','evaluate-small-milp','evaluate-small-drl',
                       'evaluate-ablations','evaluate-idle'):
            args,extra=workflow.parse_args([action])
            for command in workflow.build_commands(args,extra):
                self.assertIn('--no-save-job-records',command)
                self.assertNotIn('--save-job-records',command)


if __name__=='__main__':unittest.main()
