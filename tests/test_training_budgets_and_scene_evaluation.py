"""CPU end-to-end tests using temporary models, not formal experiments."""
import importlib
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import contextlib
import io
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiment.artifacts import atomic_json_save
from experiment.config import ConditionConfig,iter_full_factorial
from experiment.evaluation import EvaluationProtocol,seed_simulation
from comparison_runners.common_e1 import ensure_e1_case_manifest,e1_condition_id
from comparison_runners.drl_common import ensure_training_manifest,checkpoint_name
from comparison_runners.frozen_e1 import FrozenE1Dataset,DEFAULT_FROZEN_E1_DATASET
from comparison_runners.training_budgets import (run_training_budget,configure_learning,
                                               budget_model_dir)
from comparison_runners.evaluate_drl_baselines import evaluate_block

METHODS=('ours_categorical_baa','ddqnb','psa_mappo','hdrl_option')


def environment(method):
    module=importlib.import_module('methods.hierarchical' if method.startswith('ours')
                                   else 'methods.baselines.'+method+'.shared')
    env=module.env([1.,4,20,2.5,101],[2,2,1],device='cpu',
                  calibrated_reward_scale=1.,reward_scale_artifact_sha256='0'*64)
    env.validation_interval=2;env.validation_instances=2;env.checkpoint_interval=2
    configure_learning(env,dict(learning_rate=1e-4,clip_ratio=.2,gamma=.95))
    if hasattr(env,'mappo'):
        env.config=replace(env.config,rollout_episodes=2)
    return env


def tensors(agent):
    if hasattr(agent,'mappo'):
        modules=[agent.mappo.baa_actor,agent.mappo.mma_actors,agent.mappo.critic]
    elif hasattr(agent.controller,'nets'):
        modules=[net[key] for net in agent.controller.nets.values() for key in ('q1','q2')]
    elif hasattr(agent.controller,'q1'):
        modules=[agent.controller.q1,agent.controller.q2]
    else:modules=[agent.controller.actors,agent.controller.central_critic]
    return [p.detach().clone() for m in modules for p in m.parameters()]


class BudgetTests(unittest.TestCase):
    def test_full_scene_evaluation_plan_without_running_models(self):
        from comparison_runners import evaluate_drl_baselines as runner
        with tempfile.TemporaryDirectory(prefix='scene_plan_cpu_') as tmp:
            root=Path(tmp)
            argv=['evaluate','--dry-run','--case-manifest',str(root/'cases.csv'),
                  '--output-dir',str(root/'results'),'--model-root',str(root/'models')]
            captured=io.StringIO()
            with patch.object(sys,'argv',argv),contextlib.redirect_stdout(captured):
                runner.main()
            plan=json.loads(captured.getvalue())
            self.assertEqual(plan['blocks'],4*3*(27+5*5))
            self.assertEqual(plan['runs'],4*3*(27+5*5)*10*10)

    def test_training_entry_continues_existing_budget(self):
        from comparison_runners import train_drl_baselines as runner
        with tempfile.TemporaryDirectory(prefix='training_entry_cpu_') as tmp:
            root=Path(tmp)
            scale=root/'scale.json'
            atomic_json_save({'scale':{'value':1.},'contract_sha256':'test'},scale)
            for episodes in (2,4):
                argv=['train','--methods','ddqnb','--training-seeds','101',
                    '--episodes',str(episodes),'--validation-instances','2',
                    '--validation-interval','2','--checkpoint-interval','2',
                    '--training-manifest',str(root/'train.csv'),
                    '--validation-manifest',str(root/'validation.csv'),
                    '--reward-scale-file',str(scale),'--output-dir',str(root/'models'),
                    '--learning-rate','0.0001','--discount-factor','0.95','--device','cpu']
                with patch.object(sys,'argv',argv),patch.object(runner,'build_drl_environment',
                    side_effect=lambda method,*a,**kw:(environment(method),'cpu')):
                    runner.main()
            run=root/'models/ddqnb/seed-101'
            self.assertEqual(json.loads((run/'run_complete.json').read_text())['episodes'],4)
            self.assertTrue((run/'budgets/episodes-2/ddqnb_model.pth').exists())
            self.assertTrue((run/'budgets/episodes-4/ddqnb_model.pth').exists())

    def test_extended_entropy_holds_final_value(self):
        from methods.hierarchical.config import HierarchyConfig
        cfg=HierarchyConfig()
        self.assertEqual(cfg.mma_entropy_weight_at(100,500),cfg.mma_entropy_weight)
        for episode in (350,500,1000,3000):
            self.assertEqual(cfg.mma_entropy_weight_at(episode,500),cfg.mma_entropy_final_weight)

    def test_manifest_extension_preserves_original_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'training.csv'
            before=ensure_training_manifest(p,[101],2)
            after=ensure_training_manifest(p,[101,202],4)
            original=[{k:str(v) for k,v in x.items()} for x in before.rows]
            observed=[{k:str(v) for k,v in x.items()} for x in after.rows[:2]]
            self.assertEqual(original,observed)
            self.assertEqual(len(after.rows),8)

    def test_checkpoint_continuation_and_budget_snapshots(self):
        with tempfile.TemporaryDirectory(prefix='budget_scene_cpu_') as tmp:
            root=Path(tmp)
            for method in METHODS:
                with self.subTest(method=method):
                    seed_simulation(101)
                    direct=environment(method)
                    direct.max_episode=4
                    if hasattr(direct,'training_horizon'):direct.training_horizon=2
                    direct.main(root/'direct'/method)
                    expected=tensors(direct)
                    seed_simulation(101)
                    split=environment(method)
                    target=root/'split'/method/'seed-101'
                    run_training_budget(split,method,target,2,reference_episodes=2)
                    model_files={p.name:p.read_bytes() for p in (target/'budgets/episodes-2').iterdir() if p.is_file()}
                    restarted=environment(method)
                    run_training_budget(restarted,method,target,4,reference_episodes=2)
                    self.assertEqual(restarted.total_episode,4)
                    for a,b in zip(expected,tensors(restarted)):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)
                    for name,data in model_files.items():
                        self.assertEqual((target/'budgets/episodes-2'/name).read_bytes(),data)
                    self.assertTrue((target/'budgets/episodes-4/budget_complete.json').exists())
                    if hasattr(restarted,'training_horizon'):
                        self.assertEqual(restarted.training_horizon,2)

    def test_frozen_evaluation_four_methods_six_scenes(self):
        with tempfile.TemporaryDirectory(prefix='frozen_scene_cpu_') as tmp:
            root=Path(tmp)
            data=FrozenE1Dataset(DEFAULT_FROZEN_E1_DATASET)
            condition=next(iter_full_factorial())
            manifest=ensure_e1_case_manifest(root/'cases.csv',list(iter_full_factorial()),EvaluationProtocol(
                instance_count=20,replications=10,instance_seed_start=20000000,simulation_seed_start=30000000))
            cases=manifest.cases_by_condition[e1_condition_id(condition)][:1]
            for method in METHODS:
                env=environment(method)
                directory=budget_model_dir(root/'models',method,101,500)
                directory.mkdir(parents=True)
                if method.startswith('ours'):env.save_inference(directory/'hierarchical_inference.pt')
                elif method=='psa_mappo':torch.save(env.controller.inference_state(),directory/'psa_mappo_model.pth')
                else:env.save_inference(directory) if hasattr(env,'save_inference') else torch.save(
                    env.controller.q1.state_dict(),directory/'ddqnb_model.pth')
                atomic_json_save(dict(reward_normalization=dict(value=1.,artifact_sha256='0'*64),
                    hyperparameters=dict(learning_rate=1e-4,clip_ratio=.2,gamma_baa=.95)),directory/'run_config.json')
                for scene in ('S0','S1','S2','S3','S4','S5'):
                    result=evaluate_block(output=root/'evaluation',method=method,training_seed=101,
                        condition=condition,cases=cases,case_manifest_sha='',suite_configuration_sha='',
                        source_sha='',model_dir=directory,device='cpu',save_job_records=True,
                        log_every=0,block_number=1,total_blocks=1,frozen_dataset=data,scene_id=scene)
                    self.assertEqual(result['failed_runs'],0)
                    self.assertEqual(result['completed_runs'],1)
                    again=evaluate_block(output=root/'evaluation',method=method,training_seed=101,
                        condition=condition,cases=cases,case_manifest_sha='',suite_configuration_sha='',
                        source_sha='',model_dir=directory,device='cpu',save_job_records=True,
                        log_every=0,block_number=1,total_blocks=1,frozen_dataset=data,scene_id=scene)
                    self.assertTrue(again['skipped'])


if __name__=='__main__':unittest.main()
