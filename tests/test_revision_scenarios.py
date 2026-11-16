"Small CPU tests for the shared scene/configuration layer."
import importlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

from experiment.revision_protocol import TRAINING_SEEDS, MAIN_HYPERPARAMETERS
from experiment.revision_scenarios import (REVISION_SCENES, iter_revision_conditions,
    apply_revision_scene, machine_stream_seed)
from experiment.shared_simulator import build_revision_simulator,build_shared_simulator
from methods.hierarchical.config import HierarchyConfig
from experiment.instances import load_job_records_csv


class RevisionScenes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        path=next((root/'datasets/e1_formal_instances/jobs').rglob('instance-000.csv'))
        cls.records=load_job_records_csv(path)
        cls.small=cls.records[:20]
        # Use a generated small episode for fast execution, preserving its
        # records across scenes; the full frozen file is tested separately.
        e=build_shared_simulator(job_parameters=[1.,4,20,2.5,123],
            machine_parameters=[2,2,1],processing_times=(190,158,202,128,104),device='cpu')
        cls.small=e.jobs

    def build(self,scene='S0',controller=None,records=None):
        return build_revision_simulator(scene=scene,job_records=self.small if records is None else records,
            job_parameters=[1.,4,20,2.5,123] if records is None else [1.,20,len(records),2.5,123],
            processing_times=(190,158,202,128,104),controller=controller,device='cpu',
            instance_seed=123,simulation_seed=456,policy_seed=789)

    def test_protocol_and_matrix(self):
        self.assertEqual(TRAINING_SEEDS,(101,202,303))
        self.assertIsNone(MAIN_HYPERPARAMETERS)
        self.assertEqual(len(list(iter_revision_conditions())),52)
        self.assertEqual([s.total_capacity for s in REVISION_SCENES.values()],
                         [380,380,380,380,440,400])
        json.dumps({k:v.parameters() for k,v in REVISION_SCENES.items()})

    def test_all_physical_parameters_and_noncompounding(self):
        base=HierarchyConfig()
        for scene in REVISION_SCENES.values():
            cfg=scene.physics(base)
            for field,factor in [('degradation_rate_ranges',scene.degradation_factor),
                ('terminal_degradation_rates',scene.degradation_factor),
                ('failure_rate_ranges',scene.failure_factor),('ipm_times',scene.maintenance_factor),
                ('pm_times',scene.maintenance_factor),('minimal_repair_times',scene.maintenance_factor),
                ('corrective_time_ranges',scene.maintenance_factor)]:
                np.testing.assert_allclose(getattr(cfg,field),np.asarray(getattr(base,field))*factor)
            self.assertEqual(cfg.ipm_recovery_rates,base.ipm_recovery_rates)
            self.assertEqual(cfg.failure_type_proportions,base.failure_type_proportions)
            self.assertEqual(cfg.production_rate_range,base.production_rate_range)
        env=self.build('S1')
        first=env.config
        for scene in ('S1','S3','S1'):
            apply_revision_scene(env,scene,job_records=self.small,simulation_seed=456)
        self.assertEqual(first,env.config)
        apply_revision_scene(env,'S0',job_records=self.small,simulation_seed=456)
        self.assertEqual(env.config,base)

    def test_frozen_orders_are_not_regenerated(self):
        expected=[(j.job_id,j.arrival_time,j.family,j.size,j.due_time) for j in self.records]
        env=self.build(records=self.records)
        for scene in REVISION_SCENES.values():
            with patch.object(env,'_new_jobs',side_effect=AssertionError('Do not regenerate frozen orders')):
                apply_revision_scene(env,scene,job_records=self.records,simulation_seed=456)
            self.assertEqual([(j.job_id,j.arrival_time,j.family,j.size,j.due_time) for j in env.jobs],expected)
            self.assertEqual(tuple(m.capacity for m in env.machines),
                tuple(c for c,n in zip(scene.machine_capacities,scene.machine_counts) for _ in range(n)))

    def test_small_episodes_all_scenes_and_learning_interfaces(self):
        from methods.baselines.ddqnb.shared import DDQNController
        from methods.baselines.hdrl_option.shared import HDRLController
        from methods.baselines.psa_mappo.shared import PSAMAPPOController
        from methods.baselines.sdr.shared import RuleController
        for scene in REVISION_SCENES:
            controllers=[None,DDQNController(device='cpu'),HDRLController(device='cpu'),
                PSAMAPPOController(device='cpu'),RuleController('CH','MMU','Threshold',202)]
            for controller in controllers:
                with self.subTest(scene=scene,controller=type(controller).__name__):
                    env=self.build(scene,controller)
                    metrics=env.run_episode(training=False)
                    self.assertEqual(metrics['unfinished_jobs'],0)
                    self.assertTrue(np.isfinite(metrics['total_tardiness']))

    def test_existing_rng_reproducibility_and_stream_mapping(self):
        env=self.build('S3')
        a=env.run_episode(training=False)
        apply_revision_scene(env,'S3',job_records=self.small,simulation_seed=456,policy_seed=789)
        b=env.run_episode(training=False)
        self.assertEqual(a['total_tardiness'],b['total_tardiness'])
        self.assertEqual(machine_stream_seed(456,2,0),machine_stream_seed(456,2,0))
        self.assertNotEqual(machine_stream_seed(456,2,0),machine_stream_seed(456,2,1))

    def test_training_and_evaluation_entry_defaults(self):
        for name in ('comparison_runners.train_drl_baselines',
                     'comparison_runners.evaluate_drl_baselines',
                     'comparison_runners.revision_special.train_industrial',
                     'comparison_runners.revision_special.evaluate_industrial'):
            module=importlib.import_module(name)
            argv=[name]
            if name.endswith('train_industrial'):
                argv+=['--hyperparameter-file','selection.json','--reward-scale-file','scale.json']
            with patch.object(sys,'argv',argv):args=module.parse_args()
            self.assertEqual(tuple(args.training_seeds),TRAINING_SEEDS)
            self.assertEqual(tuple(args.methods),('ours_categorical_baa','ddqnb','psa_mappo','hdrl_option'))
            if hasattr(args,'episodes'):
                self.assertEqual((args.episodes,args.validation_instances,args.validation_interval),
                                 (None if name.endswith('train_industrial') else 500,10,10))


if __name__=='__main__':unittest.main()
