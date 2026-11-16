"""Frozen-model component replacement, physical invariance, and six-scene CPU replay."""
import contextlib
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ablation_experiments import evaluate_revision as runner
from ablation_experiments.candidates import formation_proposals
from ablation_experiments.variants import ABLATION_VARIANTS,variant_spec
from comparison_runners.composite_priority.rule import weight_artifact,select_candidate
from experiment.artifacts import atomic_json_save
from methods.hierarchical.config import HierarchyConfig
from methods.hierarchical.environment import HierarchicalSchedulingEnv
from methods.hierarchical.types import Job
from methods.hierarchical.candidates import build_proposals
from methods.baselines.sdr.rules import form_batches


def parameters():return [1.,4,20,2.5,101]


class RevisionAblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(101)
        self.config=HierarchyConfig(hidden_dim=16,calibrated_reward_scale=1.,
            reward_scale_artifact_sha256='0'*64,gamma_baa=.95,gamma_mma=.95)
        self.source=HierarchicalSchedulingEnv(parameters(),[2,2,1],
            processing_times=(190,158,202,128,104),config=self.config,device='cpu')
        self.state=dict(config=self.config.to_dict(),mappo=self.source.mappo.inference_state())

    def environment(self,variant):
        return runner.build_environment(variant,parameters(),config=self.config,device='cpu',
            model_state=self.state,weights=np.array([.1,.2,.3,.1,.2,.1]))

    def test_dry_run_has_complete_matrix_without_models_or_weights(self):
        with patch('sys.argv',['evaluate']):args=runner.parse_args()
        args.dry_run=True
        with contextlib.redirect_stdout(io.StringIO()):plan=runner.execute(args)
        self.assertEqual(len(plan['variants']),9)
        self.assertEqual(plan['learned_runs'],9*3*6*5*10*10)
        self.assertNotIn('controls',plan)

    def test_changed_packing_keeps_one_batch_per_family_and_actual_features(self):
        jobs=[Job(0,0,0,35,10),Job(1,0,0,20,20),Job(2,0,0,20,30),
              Job(3,0,1,25,10),Job(4,0,1,15,20)]
        for rule in ('EDD-J','FFD'):
            actual=formation_proposals(jobs,40,0,(190,158),rule)
            expected=form_batches(jobs,40,0,(190,158),rule)
            self.assertEqual([(p.family,set(p.job_ids)) for p in actual],[(f,set(ids)) for f,ids in expected])
            selected={j for _,ids in expected for j in ids}
            canonical=build_proposals([j for j in jobs if j.job_id in selected],40,0,(190,158))
            for a,b in zip(actual,canonical):np.testing.assert_array_equal(a.base_criteria,b.base_criteria)
            self.assertEqual(set(actual[0].job_ids),{0})
        self.assertEqual(set(build_proposals(jobs,40,0,(190,158))[0].job_ids),{1,2})

    def test_fixed_components_are_not_called_and_parameters_do_not_change(self):
        for variant in ABLATION_VARIANTS:
            with self.subTest(variant=variant):
                env=self.environment(variant);spec=variant_spec(variant)
                before=[p.clone() for m in runner.frozen_modules(env) for p in m.parameters()]
                if spec.baa_rule:
                    env.mappo.select_baa=lambda *a,**k:self.fail('Replaced BAA must not be called')
                if spec.mma_rule:
                    env.mappo.select_mma=lambda *a,**k:self.fail('Replaced MMA must not be called')
                with patch.object(env.mappo.optimizer,'step',side_effect=AssertionError('No optimizer updates')):
                    metrics=env.run_episode(training=False)
                self.assertEqual(metrics['unfinished_jobs'],0)
                after=[p for m in runner.frozen_modules(env) for p in m.parameters()]
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,after)))
                self.assertTrue(all(not p.requires_grad for p in after))
                self.assertTrue(all(len(v)==0 for v in env.mappo.rollouts.values()))
                if spec.mma_rule=='CM':
                    self.assertEqual(sum(m.ipm_count+m.pm_count for m in env.machines),0)

    def test_composite_selection_uses_identical_six_features(self):
        env=self.environment('tuned_composite_batch_selection')
        machine=env.machines[0]
        proposals=env._build_candidate_proposals(env._available_jobs(),machine.capacity)
        observation=env._baa_observation(machine,proposals)
        expected=select_candidate(observation.proposal_features,env.composite_weights,[p.job_ids for p in proposals])
        self.assertEqual(env._fixed_baa_selection(observation).executed_action,expected)

    def test_reference_matches_full_policy_execution(self):
        env=self.environment(runner.REFERENCE)
        self.source.mappo.load_state_dict(self.state['mappo'],training=False)
        for item in (self.source,env):
            item.reset(instance_seed=101,simulation_seed=202,policy_seed=303)
        reference=self.source.run_episode(training=False)
        actual=env.run_episode(training=False)
        for name in ('total_tardiness','makespan','batch_fill_ratio'):
            self.assertEqual(reference[name],actual[name])
        self.assertFalse(hasattr(env, "coordination_records"))

    def test_actual_model_loading_all_variants_and_scenes(self):
        with tempfile.TemporaryDirectory(prefix='revision_ablation_cpu_') as temp:
            root=Path(temp)
            model=root/'models/ours_categorical_baa/seed-101/budgets/episodes-500/hierarchical_inference.pt'
            self.source.save_inference(model)
            weights=root/'weights.json';atomic_json_save(weight_artifact([.1,.2,.3,.1,.2,.1]),weights)
            with patch('sys.argv',['evaluate']):args=runner.parse_args()
            args.model_root=root/'models';args.weights_file=weights
            args.output_dir=root/'results';args.training_seeds=[101]
            args.variants=[*ABLATION_VARIANTS,runner.REFERENCE]
            args.condition_limit=1;args.instances=args.replications=1;args.save_job_records=True
            with contextlib.redirect_stdout(io.StringIO()):blocks=runner.execute(args)
            self.assertEqual(len(blocks),10*6)
            self.assertEqual(sum(b['failed_runs'] for b in blocks),0)
            with (args.output_dir/'run_metrics.csv').open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),60)
            cm=[r for r in rows if r['ablation_variant']=='corrective_only_maintenance_policy']
            self.assertTrue(all(float(r['ipm_count'])+float(r['pm_count'])==0 for r in cm))
            with patch.object(runner,'apply_revision_scene',side_effect=AssertionError('Finished blocks must be reused')):
                with contextlib.redirect_stdout(io.StringIO()):again=runner.execute(args)
            self.assertEqual(len(again),60)


if __name__=='__main__':unittest.main()
