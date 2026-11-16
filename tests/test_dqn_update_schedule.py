"""CPU tests of replay-update and target-network synchronization intervals."""
import contextlib
import copy
import importlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from methods.baselines.ddqnb.shared import DDQNController
from methods.baselines.hdrl_option.shared import HDRLController


def add(controller, count, key=None):
    state=np.zeros(controller.input_dim,dtype=np.float32)
    state[-5:]=1.
    for _ in range(count):
        transition=SimpleNamespace(reward=1.,next_state=None,terminated=True,elapsed_time=1.)
        controller._pending=(state,0,np.ones(5,dtype=bool)) if key is None else [(key,state,0)]
        # A synthetic replay transition tests the scheduler without fabricating
        # extra environment actions or depending on a particular simulation.
        with patch.object(controller.episode_transitions,'add'):
            controller.observe(transition)


class DQNUpdateSchedule(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(101)

    def test_ddqn_warmup_and_fractional_credit(self):
        controller=DDQNController(device='cpu')
        add(controller,31)
        self.assertEqual(controller.learn_from_replay(),[])
        add(controller,4)
        records=controller.learn_from_replay()
        self.assertEqual([r['train_steps'] for r in records],[1,2,3,4])
        self.assertEqual(controller.replay_update_credit,3)
        self.assertEqual(controller.learn_from_replay(),[])
        add(controller,5)
        self.assertEqual(len(controller.learn_from_replay()),1)

    def test_full_replay_still_earns_updates(self):
        controller=DDQNController(device='cpu',buffer_size=32)
        add(controller,80)
        self.assertEqual(len(controller.buffer),32)
        self.assertEqual(len(controller.learn_from_replay()),10)
        add(controller,16)
        self.assertEqual(len(controller.learn_from_replay()),2)

    def test_hdrl_heads_have_independent_sample_and_target_clocks(self):
        controller=HDRLController(device='cpu')
        for key,count in [('option',800),('production',640),('maintenance',160)]:
            add(controller,count,key)
        records=controller.learn_from_replay()
        self.assertEqual(controller.train_steps,dict(option=100,production=80,maintenance=20))
        self.assertEqual({key:sum(r['target_synced'] for r in records if r['component']==key)
                          for key in controller.nets},dict(option=2,production=1,maintenance=0))
        self.assertEqual(controller.learn_from_replay(),[])
        add(controller,16,'option')
        controller.learn_from_replay()
        self.assertEqual(controller.train_steps,dict(option=102,production=80,maintenance=20))

    def test_target_matches_online_only_at_optimizer_sync(self):
        controller=DDQNController(device='cpu')
        add(controller,400)
        controller.replay_update_credit=392
        records=controller.learn_from_replay()
        self.assertEqual(controller.train_steps,49)
        self.assertFalse(any(r['target_synced'] for r in records))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(controller.q1.parameters(),controller.q2.parameters())))
        controller.replay_update_credit=8
        self.assertTrue(controller.learn_from_replay()[0]['target_synced'])
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(controller.q1.parameters(),controller.q2.parameters())))

    def test_credit_and_optimizer_resume_exactly(self):
        for cls in (DDQNController,HDRLController):
            source=cls(device='cpu')
            keys=[None] if cls is DDQNController else list(source.nets)
            for key in keys:add(source,43,key)
            source.learn_from_replay()
            state=copy.deepcopy(source.state_dict())
            restored=cls(device='cpu');restored.load_state_dict(state,training=True)
            for controller in (source,restored):
                for key in keys:add(controller,13,key)
                controller.learn_from_replay()
            self.assertEqual(source.train_steps,restored.train_steps)
            self.assertEqual(source.replay_update_credit,restored.replay_update_credit)
            pairs=[(source.q1,restored.q1)] if cls is DDQNController else [(source.nets[k]['q1'],restored.nets[k]['q1']) for k in keys]
            self.assertTrue(all(torch.equal(a,b) for x,y in pairs for a,b in zip(x.parameters(),y.parameters())))

    def test_episode_end_training_records_updates_and_synchronizes(self):
        for method in ('ddqnb','hdrl_option'):
            module=importlib.import_module('methods.baselines.'+method+'.shared')
            env=module.env([1.,4,20,2.5,101],[2,2,1],device='cpu',
                           calibrated_reward_scale=1.,reward_scale_artifact_sha256='0'*64)
            env.max_episode=12;env.validation_instances=1;env.validation_interval=6
            # Small test-only period exercises repeated target synchronization
            # without running a formal 500-episode experiment.
            env.controller.target_sync_steps=5
            env.controller.batch_size=4
            with tempfile.TemporaryDirectory(prefix='dqn_clock_cpu_') as temp,contextlib.redirect_stdout(io.StringIO()):
                env.main(temp)
                updates=pd.read_csv(Path(temp)/'Training_Updates.csv')
                self.assertGreater(updates.target_synced.sum(),0)
                self.assertEqual(len(env.training_metrics),12)
                self.assertTrue(all(m['terminated'] for m in env.training_metrics))
                steps=env.controller.train_steps
                if isinstance(steps,int):self.assertEqual(len(updates),steps)
                else:
                    for key,count in steps.items():self.assertEqual(len(updates[updates.component==key]),count)
                before=copy.deepcopy(env.controller.replay_update_credit)
                env._validation_score()
                self.assertEqual(env.controller.replay_update_credit,before)


if __name__=='__main__':unittest.main()
