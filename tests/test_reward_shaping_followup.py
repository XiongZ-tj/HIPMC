"""Reward boundary, terminal correction, and isolated follow-up command tests."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
from pathlib import Path
import unittest

from methods.hierarchical.config import HierarchyConfig
from methods.hierarchical.environment import HierarchicalSchedulingEnv
from comparison_runners.training_budgets import configure_learning
from comparison_runners import revision_followup as runner


class RewardShapingTests(unittest.TestCase):
    def reward(self, previous, following, terminal=False, agent='BAA', gamma=.9,
               formulation='discount_consistent_potential'):
        state=SimpleNamespace(config=HierarchyConfig(reward_formulation=formulation,
            gamma_baa=gamma,gamma_mma=gamma),_potential=lambda:following)
        return HierarchicalSchedulingEnv._transition_reward_raw(state,previous,
            terminated=terminal,agent=agent)

    def test_default_reward_unchanged(self):
        for terminal in (False,True):
            self.assertEqual(self.reward(-100,-120,terminal,formulation='potential_difference'),-20)

    def test_nonterminal_discount_for_both_agents(self):
        for agent in ('BAA','MMA'):
            self.assertEqual(self.reward(-100,-120,agent=agent),-8)

    def test_terminal_reward_retains_full_tardiness(self):
        for agent in ('BAA','MMA'):
            self.assertEqual(self.reward(-100,-120,True,agent),-20)

    def test_discounted_return_has_terminal_correction(self):
        phi=[-10,-12,-17,-20];gamma=.9
        rewards=[self.reward(phi[i],phi[i+1],i==2,gamma=gamma) for i in range(3)]
        total=sum(gamma**i*r for i,r in enumerate(rewards))
        self.assertAlmostEqual(total,-phi[0]+gamma**2*phi[-1])

    def test_gamma_one_matches_original(self):
        for terminal in (False,True):
            self.assertEqual(self.reward(-100,-120,terminal,gamma=1),-20)

    def test_invalid_formulation_rejected(self):
        with self.assertRaises(ValueError):HierarchyConfig(reward_formulation='unknown')

    def test_episode_closes_baa_and_mma_with_terminal_correction(self):
        config=HierarchyConfig(reward_formulation='discount_consistent_potential',
            gamma_baa=.9,gamma_mma=.9,calibrated_reward_scale=1.,
            reward_scale_artifact_sha256='0'*64)
        env=HierarchicalSchedulingEnv([1.,2,10,2.5,42],[1,1,1],
            processing_times=(190,158,202,128,104),config=config,device='cpu')
        calls=[]
        original=env._transition_reward_raw
        def checked(previous,*,terminated,agent):
            following=env._potential()
            value=original(previous,terminated=terminated,agent=agent)
            gamma=1. if terminated else .9
            self.assertAlmostEqual(value,gamma*following-previous)
            calls.append((agent,terminated))
            return value
        with patch.object(env,'_transition_reward_raw',side_effect=checked):
            metrics=env.run_episode(training=True)
        self.assertIn(('BAA',True),calls)
        self.assertIn(('MMA',True),calls)
        self.assertIn(('BAA',False),calls)
        self.assertGreaterEqual(metrics['total_tardiness'],0)
        effective=configure_learning(env,{'reward_formulation':'discount_consistent_potential','gamma':.9})
        self.assertEqual(effective['reward_formulation'],config.reward_formulation)

    def test_commands_use_isolated_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for action in runner.ACTIONS:
                flags=[action,'--output-root',tmp,'--dry-run']
                args=runner.parse_args(flags)
                root,command,_=runner.commands(args)
                self.assertEqual(root,Path(tmp).resolve())
                self.assertTrue(str(command[command.index('--output-dir')+1]).startswith(str(root)))
                if action=='train-shaping':
                    self.assertIn('discount_consistent_potential',command)
                if action.startswith('evaluate-'):
                    self.assertIn('--no-save-job-records',command)

    def test_training_entry_persists_reward_contract_and_rejects_mixed_resume(self):
        import contextlib
        import io
        import json
        import sys
        from comparison_runners import train_drl_baselines as train
        from experiment.artifacts import atomic_json_save
        from tests.test_training_budgets_and_scene_evaluation import environment
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scale=root/'scale.json'
            atomic_json_save({'scale':{'value':1.},'contract_sha256':'test'},scale)
            argv=['train','--methods','ours_categorical_baa','--training-seeds','101',
                '--episodes','2','--validation-instances','2','--validation-interval','2',
                '--checkpoint-interval','2','--training-manifest',str(root/'train.csv'),
                '--validation-manifest',str(root/'validation.csv'),
                '--reward-scale-file',str(scale),'--output-dir',str(root/'models'),
                '--learning-rate','0.0001','--discount-factor','0.90','--device','cpu',
                '--reward-formulation','discount_consistent_potential']
            with patch.object(sys,'argv',argv),patch.object(train,'build_drl_environment',
                    side_effect=lambda method,*a,**kw:(environment(method),'cpu')), \
                    contextlib.redirect_stdout(io.StringIO()):
                train.main()
            run=root/'models/ours_categorical_baa/seed-101'
            metadata=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['hyperparameters']['reward_formulation'],
                             'discount_consistent_potential')
            self.assertEqual(metadata['hyperparameters']['gamma_baa'],.90)
            self.assertTrue((run/'budgets/episodes-2/hierarchical_inference.pt').exists())
            argv[-1]='potential_difference'
            with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'different reward formulation'):
                    train.main()


if __name__=='__main__':unittest.main()
