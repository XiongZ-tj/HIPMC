"""Fixed-gamma regression tests; durations are physical metadata only."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from methods.hierarchical.agents import transition_discount
from methods.baselines.common import compute_smdp_gae
from methods.baselines.ddqnb.shared import DDQNController
from methods.baselines.hdrl_option.shared import HDRLController
from ablation_experiments.tests.test_ablation_environment import small_environment


class FixedDiscountTests(unittest.TestCase):
    def test_helper_has_no_duration_or_scale_dependence(self):
        for gamma in (.9,.95,.99,1.):
            self.assertEqual(transition_discount(gamma),gamma)

    def test_common_gae_fixed_gamma_and_episode_boundary(self):
        for gamma in (.9,.95,.99,1.):
            for times,scale in (([0.,0.,0.],0.),([1.,500.,10000.],100.)):
                advantages,returns=compute_smdp_gae(torch.tensor([1.,2.,3.]),
                    torch.zeros(3),torch.zeros(3),times,[False,True,True],
                    gamma=gamma,gae_lambda=1.,time_scale=scale)
                torch.testing.assert_close(returns,torch.tensor([1.+gamma*2.,2.,3.]))
                torch.testing.assert_close(advantages,returns)

    def test_hipmc_all_heads_use_fixed_gamma(self):
        agent=small_environment('mmu_batch_selection').mappo
        agent.gae_lambda=1.
        agent.gamma_baa=.9;agent.gamma_mma=.95
        for head in range(4):
            gamma=.9 if head==0 else .95
            results=[]
            for durations in ((0.,0.),(1000.,50000.)):
                transitions=[SimpleNamespace(reward=r,old_value=0.,next_state=None,
                    terminated=done,trajectory_id=0,elapsed_time=t)
                    for r,done,t in zip((1.,2.),(False,True),durations)]
                _,returns=agent._advantages_for_head(head,transitions)
                results.append(returns)
            torch.testing.assert_close(results[0],torch.tensor([1.+gamma*2.,2.]))
            torch.testing.assert_close(*results)

    def test_dqn_targets_ignore_elapsed_time(self):
        for cls in (DDQNController,HDRLController):
            first=cls(device='cpu',batch_size=4)
            second=cls(device='cpu',batch_size=4)
            second.load_state_dict(first.state_dict(),training=True)
            state=np.zeros(first.input_dim,dtype=np.float32);state[-5:]=1.
            for agent,elapsed in ((first,0.),(second,10000.)):
                buffers=[agent.buffer] if hasattr(agent,'buffer') else agent.buffers.values()
                for buffer in buffers:
                    for _ in range(4):
                        buffer.append((state.copy(),0,1.,state.copy(),False,elapsed))
            a,b=first.update(),second.update()
            if isinstance(a,dict):
                for key in a:self.assertAlmostEqual(a[key],b[key],places=6)
            else:self.assertAlmostEqual(a,b,places=6)


if __name__=='__main__':unittest.main()
