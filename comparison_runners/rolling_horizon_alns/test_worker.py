"""Small-instance smoke tests for the persistent optimization worker."""
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from comparison_runners.rolling_horizon_alns.controller import Controller, build
from comparison_runners.rolling_horizon_alns.environment import Environment


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='rh_alns_')
        cls.executable=build(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_worker_coverage_and_reproducibility(self):
        controller=Controller(self.executable,200)
        try:
            snapshot=dict(clock=0., multiplier=1.,machines=[(0,0,60,1,0.,0),(1,1,80,1,0.,0)],
                          jobs=[(j,j%2,20,200.+j*10,100.) for j in range(8)])
            plans=[]
            for _ in range(2):
                controller.begin_episode(123)
                controller.update(snapshot)
                plans.append(controller.plan[:])
                self.assertEqual(sorted(j for _,_,ids in controller.plan for j in ids),list(range(8)))
                self.assertLessEqual(controller.trace[-1]['predicted_ttd'],controller.trace[-1]['initial_ttd'])
                self.assertLessEqual(controller.trace[-1]['evaluations'],200)
            self.assertEqual(*plans)
        finally:controller.close()

    def test_shared_simulation(self):
        controller=Controller(self.executable,100)
        try:
            env=Environment([1.,8,40,2.5,712345], [2,2,1],processing_times=(190,158,202,128,104),controller=controller,device='cpu')
            env.reset(instance_seed=712345,simulation_seed=456,policy_seed=789)
            metrics=env.run_episode(training=False)
            self.assertEqual(metrics['unfinished_jobs'],0)
            submitted=[r for r in controller.trace if r['submitted_action']!='']
            self.assertTrue(submitted)
            for r in submitted:
                self.assertEqual(r['planned_action'],r['submitted_action'])
                self.assertEqual(r['planned_job_ids'],r['submitted_job_ids'])
                self.assertEqual(r['work_conserving_reassignments'],0)
                self.assertEqual(r['masked_action_corrections'],0)
        finally:controller.close()

    def test_recovery_cache_preserves_original_integration(self):
        source=Path(self.temp.name)/'recovery_probe.cpp'
        planner=Path(__file__).with_name('planner.cpp').resolve().as_posix()
        source.write_text('#define RH_ALNS_LIBRARY\n#include "'+planner+'"\n'
            'int main(){rh::Search s;for(int c=0;c<3;++c)for(int k=0;k<=60;++k)'
            '{double h=k/20.;if(s.recovered_health(c,h)!=expected_ipm_state(c,h))return 1;}}',encoding='utf-8')
        target=source.with_suffix('.exe')
        import shutil
        subprocess.run([shutil.which('g++') or shutil.which('clang++'),'-std=c++20',
                        '-O3','-pthread',str(source),'-o',str(target)],check=True)
        subprocess.run([str(target)],check=True)

    def test_repair_budget_exhaustion_preserves_all_jobs(self):
        controller=Controller(self.executable,8000)
        try:
            snapshot=dict(clock=0.,multiplier=1.,
                machines=[(0,0,60,1,0.,1),(1,1,80,1,0.,2),(2,2,100,1,0.,3)],
                jobs=[(j,j%5,10+j%21,100.+j*3,104.+j%5*20) for j in range(40)])
            for budget in (100,1000,2000,8000):
                controller.budget=budget
                controller.begin_episode(123)
                controller.update(snapshot,requester_id=0)
                self.assertEqual(sorted(j for _,_,ids in controller.plan for j in ids),list(range(40)))
                self.assertLessEqual(controller.trace[-1]['predicted_ttd'],controller.trace[-1]['initial_ttd'])
        finally:controller.close()

    def test_future_jobs_and_event_times_excluded(self):
        controller=Controller(self.executable,100)
        try:
            env=Environment([1.,8,40,2.5,712345],[2,2,1],processing_times=(190,158,202,128,104),controller=controller,device='cpu')
            before=env._snapshot()
            for job in env.jobs:
                if job.arrival_time>env.clock:
                    job.due_time+=100000
                    job.arrival_time+=100000
            for machine in env.machines:machine.scheduled_at+=100000
            self.assertEqual(before,env._snapshot())
        finally:controller.close()


if __name__=='__main__':unittest.main(verbosity=2)
