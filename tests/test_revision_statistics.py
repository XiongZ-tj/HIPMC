"""Statistical unit, zero handling, seed retention and streaming ingestion tests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiment.artifacts import atomic_csv_save,atomic_json_save
from result_analysis.statistics import paired_comparison,holm,method_summary,mean_interval
from result_analysis.ingest import EpisodeStore
from result_analysis import summarize
from comparison_runners.composite_priority.rule import DEFAULT_WEIGHTS_FILE,load_weights


def frame(values,seeds=('101',),scenes=('S0',)):
    return pd.DataFrame([dict(seed=seed,scene=scene,condition='c',instance=f'c::{i}',total_tardiness=float(v))
        for seed,series in zip(seeds,values) for scene in scenes for i,v in enumerate(series)])


def raw(method,value,seed='101',scene='S0',domain='generated',mode=None,sim=0):
    row=dict(method=method,training_seed=seed,condition_id='c',instance_seed=100,simulation_seed=sim,
             total_tardiness=value,scene_id=scene,status='completed',total_jobs=150,
             nominal_finite_load=1.,due_date_factor=2.5)
    if mode:row['deployment_mode']=mode
    return row


class StatisticsTests(unittest.TestCase):
    def test_formal_weights_exactly_match_returned_selection(self):
        np.testing.assert_array_equal(load_weights(),[0.1839922814969892,0.5344768156279985,
            0.11566533514227574,0.06316480176245559,0.10270076597028086,0.0])
        self.assertAlmostEqual(sum(load_weights()),1.)
        self.assertAlmostEqual(json.loads(DEFAULT_WEIGHTS_FILE.read_text(encoding='utf-8'))['selection_mean_ttd'],875.8929325106077)

    def test_relative_denominator_is_comparator_and_not_mean_of_percentages(self):
        result=paired_comparison(frame([[1,100]]),frame([[2,100]]),draws=64)
        self.assertAlmostEqual(result['relative_improvement_percent'],100/102)
        self.assertNotAlmostEqual(result['relative_improvement_percent'],25.)

    def test_zeros_and_equal_data(self):
        result=paired_comparison(frame([[0,0,10,20]]),frame([[0,5,0,40]]),draws=64)
        self.assertEqual(result['zero_zero'],1)
        self.assertEqual(result['reference_zero_comparator_positive'],1)
        self.assertEqual(result['comparator_zero_reference_positive'],1)
        self.assertEqual(result['positive_pairs'],1)
        self.assertAlmostEqual(result['geometric_mean_reference_over_comparator'],.5)
        same=paired_comparison(frame([[0,0]]),frame([[0,0]]),draws=64)
        self.assertEqual(same['p_value'],1.)
        self.assertEqual(same['signed_rank_biserial'],0.)
        self.assertIsNone(same['relative_improvement_percent'])

    def test_303_retained_and_seed_resampling_global_across_instances(self):
        data=frame([[0]*100,[0]*100,[300]*100],seeds=('101','202','303'))
        self.assertEqual(method_summary(data,'total_tardiness')['mean'],100.)
        interval=mean_interval(data,draws=1024,seed=5)
        self.assertGreaterEqual(interval['mean_ci_high'],200.)
        self.assertEqual(interval['mean_ci_low'],0.)

    def test_shared_instances_not_multiplied_by_six_scenes(self):
        a=frame([[10,20]],scenes=tuple(f'S{i}' for i in range(6)))
        b=frame([[20,40]],scenes=tuple(f'S{i}' for i in range(6)))
        result=paired_comparison(a,b,draws=64)
        self.assertEqual(result['paired_instances'],2)
        self.assertEqual(result['reference_mean'],15.)

    def test_partial_scene_means_are_matched_before_pooling(self):
        a=frame([[10]],scenes=('S0',));extra=a.copy();extra['scene']='S1';extra['total_tardiness']=10000
        result=paired_comparison(pd.concat([a,extra]),frame([[20]]),draws=32)
        self.assertEqual(result['reference_mean'],10.)
        self.assertEqual(result['paired_instance_scene_cells'],1)
        unavailable=paired_comparison(pd.concat([a,extra]),frame([[20]]),draws=32,required_scenes=['S0','S1'])
        self.assertEqual(unavailable['paired_instances'],0)

    def test_holm_and_macro_weighting(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
        data=frame([[1000,0,0,0]])
        data.loc[0,['condition','instance']]=['a','a::0']
        self.assertEqual(method_summary(data,'total_tardiness')['mean'],500.)

    def test_ingestion_deduplicates_and_exposes_conflicts(self):
        with tempfile.TemporaryDirectory(prefix='statistics_ingest_') as temp:
            root=Path(temp);path=root/'raw.csv'
            atomic_csv_save([raw('ours_categorical_baa',10,sim=r) for r in range(10)],path)
            store=EpisodeStore(root/'data.sqlite')
            try:
                source=dict(path=str(path),domain='generated')
                store.ingest(source,root);store.ingest(source,root)
                result=store.aggregate().iloc[0]
                self.assertEqual(result.completed_replications,10)
                self.assertEqual(result.duplicate_rows,10)
                atomic_csv_save([raw('ours_categorical_baa',99,sim=0)],path)
                store.ingest(source,root);result=store.aggregate().iloc[0]
                self.assertEqual(result.conflicted_replications,1)
                self.assertEqual(result.completed_replications,9)
                self.assertEqual(result.total_tardiness,10.)
            finally:store.close()

    def test_common_industrial_baseline_not_fabricated_as_three_training_modes(self):
        with tempfile.TemporaryDirectory(prefix='statistics_modes_') as temp:
            root=Path(temp);path=root/'raw.csv'
            rows=[raw('ours_categorical_baa',100,seed=s,domain='industrial',mode=m)
                  for s in ('101','202','303') for m in ('zero_shot','scratch','finetune')]
            rows.append(raw('composite',200,seed='',domain='industrial'))
            atomic_csv_save(rows,path)
            spec=root/'sources.json';atomic_json_save(dict(sources=[dict(path=str(path),domain='industrial')]),spec)
            with patch('sys.argv',['summary']):args=summarize.parse_args()
            args.input_spec=spec;args.output_dir=root/'out';args.bootstrap_draws=32;args.expected_replications=1
            with contextlib.redirect_stdout(io.StringIO()):tables=summarize.execute(args)
            summary=tables['method_summary']
            baseline=summary[summary.method=='Tuned composite priority']
            self.assertEqual(set(baseline['mode']),{'common'})
            tests=tables['paired_tests']
            self.assertEqual(set(tests['mode']),{'zero_shot','scratch','finetune'})
            self.assertTrue((tests.relative_improvement_percent==50.).all())



if __name__=='__main__':unittest.main()
