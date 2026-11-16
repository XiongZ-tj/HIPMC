"""Native scene parameters and reduced offline/online replay, CPU only."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from comparison_runners.native_revision_scene import compile_scene
from comparison_runners.evaluate_revision_nonlearning import execute,parse_args,ROOT
from experiment.revision_scenarios import REVISION_SCENES


class NativeSceneTests(unittest.TestCase):
    def test_actual_compiled_constants_and_all_baseline_replay(self):
        with tempfile.TemporaryDirectory(prefix='native_scene_cpu_') as tmp:
            tmp=Path(tmp)
            with patch('sys.argv',['test']):args=parse_args()
            args.methods=['sdr','ma','rh_alns']
            args.output_dir=tmp/'results';args.plan_dir=tmp/'plans'
            args.condition_limit=1;args.instances=1;args.replications=1
            args.ma_population=4;args.ma_generations=1
            args.ma_refinement_generations=0;args.ma_local_search_evaluations=0
            args.rh_evaluations=100
            # Inspect the actual constexpr arrays through a compiled probe,
            # not just the JSON parameter export.
            from comparison_runners import evaluate_revision_nonlearning as runner
            original=runner.compile_scene
            def compiled(scene,directory,**kwargs):
                binaries=original(scene,directory,**kwargs)
                cfg=scene.physics()
                source=Path(directory)/scene.scene_id/'offline_memetic_scene.cpp'
                probe=source.with_name('probe.cpp')
                arrays={'kMachineClass':[c for c,n in enumerate(scene.machine_counts) for _ in range(n)],
                    'kMachineCapacity':[c for c,n in zip(scene.machine_capacities,scene.machine_counts) for _ in range(n)],
                    'kIpmTime':cfg.ipm_times,'kPmTime':cfg.pm_times,'kMinimalRepairTime':cfg.minimal_repair_times,
                    'kCorrectiveRange':cfg.corrective_time_ranges,'kDegradationRange':cfg.degradation_rate_ranges,
                    'kTerminalDegradationRate':cfg.terminal_degradation_rates,'kFailureRange':cfg.failure_rate_ranges,
                    'kIpmRecoveryRate':cfg.ipm_recovery_rates,'kFailureProportion':cfg.failure_type_proportions}
                probe.write_text('#define main unused_probe_main\n#include "'+source.as_posix()+
                    '"\n#undef main\n#include <type_traits>\ntemplate<class T> void emit(const T& x) {'
                    'if constexpr(std::is_arithmetic_v<T>) std::cout<<x<<" "; else for(const auto& y:x)emit(y);}'
                    '\nint main(){std::cout<<std::setprecision(17);'+''.join('emit('+name+');' for name in arrays)+'}',encoding='utf-8')
                binary=probe.with_suffix('.exe')
                subprocess.run([shutil.which('g++'),'-std=c++20','-O0','-pthread',str(probe),'-o',str(binary)],check=True)
                observed=np.fromstring(subprocess.check_output([str(binary)],text=True),sep=' ')
                expected=np.concatenate([np.asarray(value).reshape(-1) for value in arrays.values()])
                np.testing.assert_allclose(observed,expected,rtol=1e-14,atol=0)
                return binaries
            with patch.object(runner,'compile_scene',side_effect=compiled):blocks=execute(args)
            self.assertEqual(len(blocks),6*20)
            self.assertEqual(sum(b['failed_runs'] for b in blocks),0)
            with (args.output_dir/'run_metrics.csv').open(encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),120)
            self.assertEqual({r['scene_id'] for r in rows},set(REVISION_SCENES))
            for row in rows:
                # Scene subsets retain the seed assigned to that original condition.
                with (ROOT/'experiment_manifests/e1_formal_comparison_cases.csv').open(encoding='utf-8-sig',newline='') as original:
                    expected=next(r for r in csv.DictReader(original) if r['condition_id']==row['condition_id'] and int(r['instance_index'])==0 and int(r['replication_index'])==0)
                self.assertEqual(row['simulation_seed'],expected['simulation_seed'])
                self.assertEqual(float(row['unfinished_jobs']),0.)
                if row['method']=='ma':
                    self.assertNotEqual(row['ma_offline_ttd'],'')
                    self.assertAlmostEqual(float(row['ma_execution_delta']),
                        float(row['total_tardiness'])-float(row['ma_offline_ttd']))
            for scene in REVISION_SCENES:
                self.assertEqual(len(list((args.plan_dir/scene).glob('*/plan_manifest.csv'))),1)
            with (args.output_dir/'reoptimization.csv').open(encoding='utf-8') as stream:
                traces=list(csv.DictReader(stream))
            self.assertTrue(traces)
            for trace in traces:
                self.assertEqual(int(trace['work_conserving_reassignments']),0)
                self.assertEqual(int(trace['masked_action_corrections']),0)
                if trace['submitted_action']!='':
                    self.assertEqual(trace['planned_action'],trace['submitted_action'])
                    self.assertEqual(trace['planned_job_ids'],trace['submitted_job_ids'])
            # Restart skips finished online blocks and reuses existing plans.
            args.phase='evaluate';args.methods=['sdr','ma']
            restarted=execute(args)
            self.assertEqual(len(restarted),6*19)
            with (args.output_dir/'run_metrics.csv').open(encoding='utf-8') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),114)


if __name__=='__main__':unittest.main()
