"""Temporary CPU tuning, frozen-rule replay, and deterministic search tests."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from comparison_runners.composite_priority import tune
from comparison_runners.composite_priority.rule import Controller,Environment,select_candidate,load_weights
from comparison_runners import evaluate_revision_nonlearning as runner


class CompositeTests(unittest.TestCase):
    def test_formal_budget_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory(prefix='composite_plan_cpu_') as temp:
            with patch('sys.argv',['tune']):args=tune.parse_args()
            args.output_dir=Path(temp)/'not_created';args.dry_run=True
            with contextlib.redirect_stdout(io.StringIO()):plan=tune.execute(args)
            self.assertEqual(plan['optimization_seeds'],[101])
            self.assertEqual(plan['nominal_simulations'],20910)
            self.assertFalse(args.output_dir.exists())

    def test_partial_candidate_resume_keeps_completed_cases(self):
        with tempfile.TemporaryDirectory(prefix='composite_resume_cpu_') as temp:
            root=Path(temp)
            cases={'search':[dict(instance_seed=610000000,simulation_seed=630000000+r,
                instance_index=0,replication_index=r,job_file='unused.csv') for r in range(2)]}
            weights=np.ones(6)/6
            evaluator=tune.Evaluator(root,cases)
            def fake(task):
                row,w=task
                return dict(**row,weight_id=tune.weight_id(w),total_tardiness=10+int(row['replication_index']))
            def interrupted(task):
                if int(task[0]['replication_index'])==1:raise KeyboardInterrupt()
                return fake(task)
            with patch.object(tune,'simulate',side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):evaluator(weights,'search')
            with patch.object(tune,'simulate',side_effect=fake) as simulate:
                result=evaluator(weights,'search')
            self.assertEqual(simulate.call_count,1)
            self.assertEqual(result['mean_ttd'],10.5)
            self.assertEqual(result['runs'],2)

    def test_score_ties_and_maintenance_action_indices(self):
        from types import SimpleNamespace
        features=np.array([[.8,.1,0,0,0,0],[.1,.9,0,0,0,0]])
        self.assertEqual(select_candidate(features,[1,0,0,0,0,0],[(2,),(1,)]),0)
        self.assertEqual(select_candidate(features,[0,1,0,0,0,0],[(2,),(1,)]),1)
        self.assertEqual(select_candidate(np.ones((2,6)),np.ones(6)/6,[(9,),(2,)]),1)
        controller=Controller(np.ones(6)/6)
        observation=SimpleNamespace(proposal_features=features)
        self.assertEqual(controller.select_action(observation,machine_state=2),2)
        self.assertEqual(controller.select_action(observation,machine_state=3),3)

    def test_retained_batch_obeys_threshold_before_resuming(self):
        from types import SimpleNamespace
        from methods.baselines.sdr.shared import RuleController
        controller=Controller(np.ones(6)/6)
        threshold=RuleController('CH','MMU','Threshold',202)
        observation=SimpleNamespace(proposal_features=np.ones((1,6)),
                                    requester_index=0,action_mask=np.ones(3,dtype=bool))
        for state,expected in [(0,0),(1,0),(2,1),(3,2)]:
            with self.subTest(state=state):
                actual=controller.select_action(observation,machine_state=state,
                                                retained_batch=True)
                self.assertEqual(actual,expected)
                self.assertEqual(actual,threshold.select_action(observation,
                    requester_class=0,machine_state=state,clock=100))

    def test_retained_batch_bypasses_stale_candidate_scores(self):
        from types import SimpleNamespace
        controller=Controller(np.ones(6)/6)
        controller.proposals=[]
        observation=SimpleNamespace(proposal_features=np.ones((1,6)))
        self.assertEqual(controller.select_action(observation,machine_state=1,
                                                 retained_batch=True),0)

    def test_repaired_machine_request_applies_threshold_maintenance(self):
        for state,expected in [(2,1),(3,2)]:
            with self.subTest(state=state):
                env=Environment([1.,30,150,2.5,610000000],[2,2,1],
                    processing_times=tune.CONDITION['processing_times'],
                    controller=Controller(np.ones(6)/6),device='cpu')
                machine=env.machines[0]
                proposal=env.controller.build_proposals(env._available_jobs(),
                    machine.capacity,env.clock,env.job_cluster_pt)[0]
                machine.batch_ids=list(proposal.job_ids)
                machine.state=state
                with patch.object(env,'_proposal_for_reserved',return_value=proposal), \
                        patch.object(env,'_start_action') as start:
                    self.assertTrue(env._request(machine,epsilon=0.,training=False))
                context=start.call_args.args[2]
                self.assertEqual(context.mma_action,expected)


    def test_search_simplex_reproducible_and_anchors_retained(self):
        def evaluate(w,split):
            self.assertTrue(np.all(np.asarray(w)>=0))
            self.assertAlmostEqual(sum(w),1.)
            return dict(weights=list(w),mean_ttd=float(np.dot(w,np.arange(6))),weight_id=tune.weight_id(w))
        with contextlib.redirect_stdout(io.StringIO()):
            history,best=tune.search(101,evaluate,4,2)
            other,_=tune.search(101,evaluate,4,2)
        self.assertEqual(history,other)
        self.assertEqual(len(history),7+4+8)
        self.assertEqual(best['mean_ttd'],0.)
        chosen=tune.finalists(history)
        self.assertEqual(len(chosen),10)
        self.assertIn('equal_weight',[r['stage'] for r in chosen])
        self.assertIn('single_feature',[r['stage'] for r in chosen])

    def test_feature_identity_and_complete_candidate_pool(self):
        from methods.hierarchical.candidates import build_proposals
        env=Environment([1.,30,150,2.5,610000000],[2,2,1],
            processing_times=tune.CONDITION['processing_times'],controller=Controller(np.ones(6)/6),device='cpu')
        machine=env.machines[0]
        proposals=env.controller.build_proposals(env._available_jobs(),machine.capacity,env.clock,env.job_cluster_pt)
        reference=build_proposals(env._available_jobs(),machine.capacity,env.clock,env.job_cluster_pt)
        self.assertEqual([p.job_ids for p in proposals],[p.job_ids for p in reference])
        for proposal in proposals:
            np.testing.assert_array_equal(env._control_proposal_features(machine,proposal),env._proposal_features(machine,proposal))
        self.assertEqual(env._joint_observation(machine,proposals).proposal_features.shape[1],6)

    def test_cpu_tuning_resume_parallel_and_scene_deployment(self):
        with tempfile.TemporaryDirectory(prefix='composite_cpu_') as temp:
            root=Path(temp)
            with patch('sys.argv',['tune']):args=tune.parse_args()
            args.output_dir=root/'tuning';args.optimization_seeds=[101]
            args.search_instances=args.selection_instances=1
            args.search_replications=args.selection_replications=1
            args.random_candidates=2;args.local_per_step=1;args.finalists=3
            with contextlib.redirect_stdout(io.StringIO()):artifact=tune.execute(args)
            self.assertAlmostEqual(sum(artifact['weights']),1.)
            self.assertEqual(artifact['selection_runs'],1)
            searches=tune.read(next((args.output_dir/'data').glob('search_cases*')))
            selection=tune.read(next((args.output_dir/'data').glob('selection_cases*')))
            self.assertNotEqual(searches[0]['instance_seed'],selection[0]['instance_seed'])
            with patch.object(tune,'simulate',side_effect=AssertionError('Resume must use saved cases')):
                with contextlib.redirect_stdout(io.StringIO()):again=tune.execute(args)
            self.assertEqual(artifact,again)
            # Spawn-based Windows workers exercise the actual simulator too.
            args.output_dir=root/'parallel';args.workers=2
            with contextlib.redirect_stdout(io.StringIO()):parallel=tune.execute(args)
            self.assertEqual(artifact['weights'],parallel['weights'])
            self.assertAlmostEqual(artifact['selection_mean_ttd'],parallel['selection_mean_ttd'])
            with patch('sys.argv',['evaluate']):evaluation=runner.parse_args()
            evaluation.methods=['composite'];evaluation.output_dir=root/'evaluation'
            evaluation.weights_file=args.output_dir/'tuned_composite_priority_weights.json'
            evaluation.condition_limit=1;evaluation.instances=1;evaluation.replications=1
            evaluation.save_job_records=True
            with contextlib.redirect_stdout(io.StringIO()):blocks=runner.execute(evaluation)
            self.assertEqual(len(blocks),6)
            self.assertEqual(sum(b['failed_runs'] for b in blocks),0)
            with (evaluation.output_dir/'run_metrics.csv').open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),6)
            self.assertTrue(all(r['composite_weights_json'] for r in rows))


if __name__=='__main__':unittest.main()
