import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import replace
from tempfile import TemporaryDirectory
from unittest.mock import patch
import csv
import json
import unittest
import numpy as np
import torch

from experiment.instances import JobRecord
from methods.hierarchical.agents import PolicySelection
from methods.hierarchical.config import HierarchyConfig
from methods.hierarchical.environment import HierarchicalSchedulingEnv
from methods.hierarchical.types import Job, MachinePhase
from idle_maintenance_experiments.environment import IdleMaintenanceEnv

REVISION = Path(__file__).resolve().parents[3]


def make_env(protocol="idle_opportunity"):
    env = IdleMaintenanceEnv([1., 0, 2, 2., 101], [1, 1, 1],
        processing_times=(20, 35), device="cpu", maintenance_protocol=protocol,
        config=HierarchyConfig(calibrated_reward_scale=1., reward_scale_artifact_sha256="a"*64))
    env.reset(instance_seed=101, simulation_seed=101, policy_seed=101,
        job_records=[JobRecord(0, 10., 0, 10, 50.), JobRecord(1, 20., 0, 10, 60.)])
    return env


def scripted_policy(env, empty_action=3):
    def select(class_id, observation, **kwargs):
        action = empty_action if observation.maintenance_features[-1] == 0 else 0
        if not observation.action_mask[action]:
            action = 3 if observation.maintenance_features[-1] == 0 else 0
        probabilities = np.zeros(4, dtype=np.float32)
        probabilities[action] = 1
        return PolicySelection(action, action, 0., 0., 0., probabilities=probabilities)
    env.mappo.select_mma = select


class IdleMaintenanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_standard_hipmc_architecture_is_unchanged(self):
        env = HierarchicalSchedulingEnv([1, 0, 2, 2, 101], [1, 1, 1],
                                       processing_times=(20, 35), device="cpu")
        self.assertEqual(env.mappo.mma_actors[0].model[-1].out_features, 3)
        self.assertEqual(env.maintenance_dim, 5)

    def test_four_action_heads_are_owned_by_optimizer(self):
        env = make_env()
        ids = {id(p) for group in env.mappo.optimizer.param_groups for p in group["params"]}
        self.assertTrue(all(id(p) in ids for p in env.mappo.mma_actors.parameters()))
        self.assertEqual(env.mappo.mma_actors[0].model[-1].out_features, 4)

    def test_empty_healthy_mask_is_wait_only(self):
        env = make_env()
        observation = env._empty_observation(env.machines[0])
        np.testing.assert_array_equal(observation.action_mask, [False, False, False, True])
        np.testing.assert_array_equal(observation.batch_features, np.zeros(6))
        self.assertEqual(observation.maintenance_features.shape, (6,))
        self.assertEqual(observation.maintenance_features[-1], 0)

    def test_empty_degraded_mask_allows_wait_and_both_maintenance_types(self):
        env = make_env()
        env.machines[0].state = 1
        np.testing.assert_array_equal(env._empty_observation(env.machines[0]).action_mask,
                                      [False, True, True, True])

    def test_queue_with_batch_never_allows_wait(self):
        env = make_env()
        env.clock = 10
        from methods.hierarchical.candidates import build_proposals
        proposal = build_proposals(env._available_jobs(), 60, 10, (20,35))[0]
        observation = env._mma_observation(env.machines[0], proposal)
        np.testing.assert_array_equal(observation.action_mask, [True, False, False, False])
        self.assertEqual(observation.maintenance_features[-1], 1)
        env.machines[0].state = 1
        np.testing.assert_array_equal(env._mma_observation(env.machines[0], proposal).action_mask,
                                      [True, True, True, False])

    def test_empty_request_does_not_call_baa_or_create_baa_samples(self):
        env = make_env()
        scripted_policy(env)
        env._episode_training = True
        env.mappo.select_baa = lambda *a,**k: self.fail("BAA called at empty queue")
        self.assertTrue(env._request(env.machines[0], epsilon=0, training=True))
        self.assertFalse(env._request(env.machines[0], epsilon=0, training=True))
        self.assertIsNone(env._pending_baa)
        self.assertEqual(env.mappo.rollouts[0], [])
        self.assertEqual(env.clock, 0)
        self.assertEqual(env.machines[0].state, 0)
        env._advance_with_repairs()
        self.assertEqual(env.clock, 10)


    def test_waits_have_positive_event_durations_and_correct_terminal_flags(self):
        env = make_env()
        scripted_policy(env)
        metrics = env.run_episode(training=True)
        waits = [tr for ts in env.mappo.rollouts.values() for tr in ts if int(tr.action) == 3]
        self.assertTrue(waits)
        self.assertTrue(all(tr.elapsed_time > 0 for tr in waits))
        self.assertTrue(any(tr.terminated for tr in waits))
        self.assertEqual(metrics["makespan"], 40.)
        self.assertEqual(metrics["idle_ipm_count"]+metrics["idle_pm_count"], 0)
        self.assertTrue(all(not env.mappo._is_actor_choice(1, tr) for tr in waits))

    def test_empty_maintenance_stops_reporting_at_last_job_completion(self):
        env = make_env()
        env.machines[0].state = 1
        scripted_policy(env, empty_action=2)
        metrics = env.run_episode(training=True)
        self.assertEqual(metrics["makespan"], 40.)
        self.assertEqual(metrics["maintenance_time_booked"], 60.)
        self.assertEqual(metrics["maintenance_time"], 40.)
        self.assertEqual(metrics["idle_maintenance_actual_time"], 40.)
        self.assertEqual(metrics["idle_maintenance_arrival_overlap_count"], 1)
        self.assertEqual(env.machines[0].planned_maintenance_time, 60.)
        self.assertEqual(env._machine_responsibility_metrics()["responsibility_maintenance_time_total"], 40.)

    def test_empty_ipm_and_degraded_wait_both_execute(self):
        for action in (1,3):
            env = make_env()
            env.machines[0].state = 1
            scripted_policy(env, empty_action=action)
            metrics = env.run_episode(training=True)
            self.assertTrue(metrics["terminated"])
            if action == 1:
                self.assertGreater(metrics["idle_ipm_count"], 0)
            else:
                self.assertGreater(metrics["idle_wait_count"], 0)
                self.assertEqual(env.machines[0].state, 1)


    def test_simultaneous_events_are_settled_before_new_decisions(self):
        env = make_env()
        env.jobs = [Job(0,0,0,10,40,reserved_by=0), Job(1,0,0,10,40,reserved_by=1)]
        for i in (0,1):
            m = env.machines[i]
            m.phase = MachinePhase.PROCESSING
            m.batch_ids = (i,)
            m.remaining_work = 10
            m.last_event_at = 0
            env._schedule(m, 10, "complete")
        env._advance_with_repairs()
        self.assertEqual(env.clock, 10)
        self.assertTrue(all(j.finished for j in env.jobs))
        self.assertTrue(all(m.phase is MachinePhase.IDLE for m in env.machines))

    def test_empty_inputs_do_not_depend_on_future_orders_or_generator_metadata(self):
        env = make_env()
        before = env._empty_observation(env.machines[0])
        env.jobs.append(Job(2,1e6,1,20,2e6))
        env.total_jobs = 9999
        env.target_load = 99
        env.due_date_factor = 999
        env._instance_load = None
        after = env._empty_observation(env.machines[0])
        for name in before.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(before,name), getattr(after,name))

    def test_training_update_validation_and_checkpoints_use_variant(self):
        for protocol in ("idle_opportunity",):
            torch.manual_seed(101)
            env = make_env(protocol)
            metrics = env.train_episode(instance_seed=101, simulation_seed=101, policy_seed=101)
            self.assertTrue(metrics["terminated"])
            self.assertTrue(np.isfinite(env.mappo.update()))
            env.validation_instances = 2
            self.assertTrue(np.isfinite(env._validation_score()))
            with TemporaryDirectory(prefix="q5-test-", dir=REVISION) as temp:
                path = Path(temp)/"inference.pt"
                env.save_inference(path)
                env.load_inference(path)
                path = Path(temp)/"checkpoint.pt"
                env.save_checkpoint(path)
                env.load_checkpoint(path, load_training=True)
                env._save_training_logs(Path(temp))
                self.assertTrue((Path(temp)/"Training_Episodes.csv").is_file())

    def test_short_training_main_keeps_variant_validation_and_logs(self):
        env = make_env()
        env.max_episode = env.training_horizon = 2
        env.validation_instances = 2
        env.validation_interval = env.checkpoint_interval = 1
        with TemporaryDirectory(prefix="q5-main-test-", dir=REVISION) as temp:
            env.main(Path(temp))
            self.assertEqual(env.total_episode, 2)
            self.assertEqual(env.validation_metrics[0]["episode"], 0)
            self.assertTrue((Path(temp)/"hierarchical_inference.pt").exists())
            self.assertTrue((Path(temp)/"Training_Episodes.csv").exists())
            env.load_inference(Path(temp)/"hierarchical_inference.pt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
