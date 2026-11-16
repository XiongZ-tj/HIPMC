"""Reduced native MA/RH search and all industrial non-learning controllers."""
import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from comparison_runners.revision_special import evaluate_industrial_nonlearning as runner
from comparison_runners import evaluate_revision_nonlearning as base
from comparison_runners.composite_priority.rule import weight_artifact
from experiment.artifacts import atomic_json_save


class IndustrialNativeTests(unittest.TestCase):
    def test_compiled_three_machine_six_family_physics_and_21_controllers(self):
        with tempfile.TemporaryDirectory(prefix='revision_industrial_native_cpu_') as temp:
            root=Path(temp)
            with patch('sys.argv',['test']):args=runner.parse_args()
            args.output_dir=root/'results';args.plan_dir=root/'plans';args.weights_file=root/'weights.json'
            atomic_json_save(weight_artifact([.1,.2,.3,.1,.2,.1]),args.weights_file)
            args.instance_limit=args.replications=1;args.rh_evaluations=100
            args.ma_population=4;args.ma_generations=1;args.ma_refinement_generations=0;args.ma_local_search_evaluations=0
            original=base.compile_scene
            def compiled(scene,directory,**kwargs):
                binaries=original(scene,directory,**kwargs)
                source=Path(directory)/scene.scene_id/'offline_memetic_scene.cpp'
                probe=source.with_name('probe.cpp');binary=probe.with_suffix('.exe')
                probe.write_text('#define main unused_main\n#include "'+source.as_posix()+'"\n#undef main\n'
                    'int main(){std::cout<<kMachineCount<<" "<<kFamilyCount<<" ";'
                    'for(auto x:kMachineClass)std::cout<<x<<" ";'
                    'for(auto x:kMachineCapacity)std::cout<<x<<" ";'
                    'for(auto x:kProcessingTime)std::cout<<x<<" ";}',encoding='utf-8')
                subprocess.run([shutil.which('g++'),'-std=c++20','-O0','-pthread',str(probe),'-o',str(binary)],check=True)
                np.testing.assert_array_equal(np.fromstring(subprocess.check_output([str(binary)],text=True),sep=' '),
                    [3,6,0,1,2,120,160,200,128,190,104,293,202,279])
                return binaries
            with patch.object(base,'compile_scene',side_effect=compiled),contextlib.redirect_stdout(io.StringIO()):blocks=runner.execute(args)
            self.assertEqual(len(blocks),21)
            self.assertEqual(sum(b['failed_runs'] for b in blocks),0)
            rows=base.read(args.output_dir/'run_metrics.csv')
            self.assertTrue(all(float(r['job_count'])==89 for r in rows))
            self.assertTrue(all(float(r['effective_nominal_load'])==.9 for r in rows))
            self.assertEqual(len(list(args.plan_dir.glob('industrial/*/plan_manifest.csv'))),1)
            rh=base.read(args.output_dir/'reoptimization.csv')
            self.assertTrue(rh)
            self.assertTrue(all(r['planned_action']==r['submitted_action'] for r in rh if r['submitted_action']!=''))


if __name__=='__main__':unittest.main()
