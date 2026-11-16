"""CPU regression tests for the shared-feature learning baselines."""
import unittest
import numpy as np
from experiment.shared_simulator import build_shared_simulator
from methods.baselines.ddqnb.shared import DDQNController
from methods.baselines.hdrl_option.shared import HDRLController
from methods.baselines.psa_mappo.shared import PSAMAPPOController
from methods.baselines.learning_features import MACHINE_DIM, fixed_mask
from methods.hierarchical.candidates import build_proposals


def simulator(controller, jobs=20):
    return build_shared_simulator(job_parameters=[1.,4,jobs,2.5,90123],
        machine_parameters=[2,2,1], processing_times=(190,158,202,128,104),
        controller=controller, device='cpu')


class RevisedBaselines(unittest.TestCase):
    def test_industrial_orders_and_nonzero_rewards(self):
        from pathlib import Path
        import csv
        from experiment.instances import load_job_records_csv
        from comparison_runners.industrial_case.case_common import adapt_simulator
        root=Path(__file__).resolve().parents[1]
        data=root/'datasets/industrial_case_small'
        with (data/'manifest.csv').open(encoding='utf-8',newline='') as stream:
            rows=list(csv.DictReader(stream))
        for size in (10,20):
            row=next(r for r in rows if int(r['n_jobs'])==size)
            records=load_job_records_csv(data/row['job_file'])
            for controller in self.controllers():
                env=build_shared_simulator(job_parameters=[1.,size//5,size,1.,90123],
                    machine_parameters=[1,1,1],processing_times=(128,190,104,293,202,279),
                    controller=controller,device='cpu')
                adapt_simulator(env)
                from types import MethodType
                from methods.hierarchical.types import Job
                env._jobs_from_records=MethodType(lambda self, values: [
                    Job(job_id=r.job_id,arrival_time=r.arrival_time,family=r.family,
                        size=r.size,due_time=r.due_time)
                    for r in sorted(values,key=lambda x:x.job_id)],env)
                env.reset(instance_seed=90123,simulation_seed=123,policy_seed=456,
                          job_records=records)
                metrics=env.run_episode(training=True,epsilon=.5)
                self.assertEqual(metrics['unfinished_jobs'],0)
                self.assertGreater(metrics['total_tardiness'],0)
                self.assertIsNotNone(controller.update())

    def test_training_wrappers_and_saved_models(self):
        import importlib
        import tempfile
        from pathlib import Path
        for method in ('ddqnb','psa_mappo','hdrl_option'):
            module=importlib.import_module('methods.baselines.'+method+'.shared')
            env=module.env([1.,4,20,2.5,101],[2,2,1],device='cpu',
                           calibrated_reward_scale=1.,reward_scale_artifact_sha256='0'*64)
            env.max_episode=4
            env.validation_instances=2
            env.validation_interval=2
            env.checkpoint_interval=2
            if hasattr(env.controller,'batch_size'):env.controller.batch_size=4
            with tempfile.TemporaryDirectory(prefix='baseline_cpu_smoke_') as directory:
                env.main(directory)
                self.assertEqual(len(env.training_metrics),4)
                self.assertEqual(len(env.validation_metrics),3)
                env.load_inference_model(Path(directory))

    def controllers(self):
        return [DDQNController(device='cpu',batch_size=4),
                HDRLController(device='cpu',batch_size=4),
                PSAMAPPOController(device='cpu',update_epochs=1,batch_size=8)]

    def test_shared_features(self):
        for controller in self.controllers():
            env=simulator(controller)
            machine=env.machines[0]
            proposals=controller.build_proposals(env._available_jobs(),machine.capacity,
                                                env.clock,env.job_cluster_pt)
            obs=env._joint_observation(machine,proposals)
            self.assertEqual(obs.machine_features.shape,(5,MACHINE_DIM))
            self.assertEqual(obs.proposal_features.shape,(3,6))
            pool={p.job_ids:p for p in build_proposals(env._available_jobs(),
                  machine.capacity,env.clock,env.job_cluster_pt)}
            for index,p in enumerate(proposals):
                np.testing.assert_allclose(obs.proposal_features[index],
                    env._proposal_features(machine,pool[p.job_ids]))
            for job in env.jobs:
                if job.arrival_time>env.clock:
                    job.due_time+=100000
                    job.arrival_time+=100000
            other=env._joint_observation(machine,proposals)
            np.testing.assert_array_equal(obs.machine_features,other.machine_features)
            np.testing.assert_array_equal(obs.proposal_features,other.proposal_features)

    def test_training_and_validation_isolation(self):
        for controller in self.controllers():
            env=simulator(controller)
            metrics=env.run_episode(training=True,epsilon=0.5)
            self.assertEqual(metrics['unfinished_jobs'],0)
            if isinstance(controller,PSAMAPPOController):
                self.assertEqual(len(controller.rollout),metrics['decisions'])
            self.assertIsNotNone(controller.update())
            def size():
                if hasattr(controller,'buffer'):return len(controller.buffer)
                if hasattr(controller,'buffers'):return tuple(len(b) for b in controller.buffers.values())
                return len(controller.rollout)
            before=size()
            env.reset(instance_seed=90124,simulation_seed=123,policy_seed=456)
            metrics=env.run_episode(training=False)
            self.assertEqual(metrics['unfinished_jobs'],0)
            self.assertEqual(before,size())

    def test_retained_mask_mapping(self):
        from methods.hierarchical.types import JointObservation
        obs=JointObservation(np.zeros((1,6)),np.zeros((5,MACHINE_DIM)),0,
                             np.array([True,False,True]))
        np.testing.assert_array_equal(fixed_mask(obs),[True,False,False,False,True])
        for controller in [DDQNController(device='cpu'),HDRLController(device='cpu')]:
            for _ in range(20):
                action=controller.select_action(obs,training=True,epsilon=1.,
                                                 rng=np.random.default_rng())
                self.assertIn(action,[0,2])


if __name__=='__main__':unittest.main()
