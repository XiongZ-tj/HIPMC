"""Budget selection, optimizer summaries and unavailable-data behavior."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from experiment.artifacts import atomic_csv_save,atomic_json_save
from result_analysis.training import summarize_run
from result_analysis.optimization import summarize as summarize_ma
from result_analysis.timing import summarize_native
from result_analysis import summarize
from result_analysis.ingest import EpisodeStore
from comparison_runners.evaluate_revision_nonlearning import read


class ResultSummaryTests(unittest.TestCase):
    def test_training_budget_uses_snapshot_not_later_best_or_cost(self):
        with tempfile.TemporaryDirectory(prefix='training_statistics_') as temp:
            root=Path(temp);snapshot=root/'budgets/episodes-2'
            atomic_json_save(dict(method='ours_categorical_baa',training_seed=303,
                hyperparameters=dict(learning_rate=2e-4,gamma_baa=.95,clip_ratio=.2)),root/'run_config.json')
            atomic_csv_save([dict(episode=1,total_tardiness=9),dict(episode=2,total_tardiness=5)],snapshot/'Training_Episodes.csv')
            atomic_csv_save([dict(episode=0,mean_tardiness=100/150,total_tardiness_mean=100),
                dict(episode=1,mean_tardiness=40/150,total_tardiness_mean=40),
                dict(episode=2,mean_tardiness=60/150,total_tardiness_mean=60)],snapshot/'Training_Validation.csv')
            atomic_csv_save([dict(episode=4,total_tardiness=999)],root/'Training_Episodes.csv')
            atomic_json_save(dict(episodes=4,elapsed_seconds=999),root/'run_complete.json')
            atomic_json_save(dict(training_validation_wall_seconds=12.5),snapshot/'training_budget_runtime.json')
            result,_=summarize_run(root,budget=2,late_window=1)
            self.assertEqual(result['training_seed'],303)
            self.assertEqual(result['late_window_training_ttd'],5)
            self.assertEqual(result['selected_episode'],1)
            self.assertEqual(result['selected_validation_ttd'],40)
            self.assertEqual(result['training_validation_wall_seconds'],12.5)

    def test_native_ma_search_stages_and_source_curves(self):
        with tempfile.TemporaryDirectory(prefix='ma_statistics_') as temp:
            root=Path(temp)
            atomic_csv_save([dict(plan_file='plan.csv',condition_id='c',instance_index=0,instance_seed=1,
                objective_total_tardiness=8,generations=1,search_evaluations=20,runtime_seconds=.5,
                stop_reason='maximum_generations',last_improvement_generation=1,convergence_file='curve.csv')],root/'plan_manifest.csv')
            atomic_csv_save([dict(initial_objective_total_tardiness=20,baseline_best_ttd=12)],root/'plan.csv')
            atomic_csv_save([dict(search_evaluations=10),dict(search_evaluations=20)],root/'curve.csv')
            result=summarize_ma(root)[0]
            self.assertEqual(result['initial_offline_ttd'],20)
            self.assertEqual(result['pre_refinement_offline_ttd'],12)
            self.assertEqual(result['final_offline_ttd'],8)
            self.assertEqual(result['convergence_points'],2)

    def test_rolling_gain_and_p95_are_within_snapshot_and_exact(self):
        with tempfile.TemporaryDirectory(prefix='rh_statistics_') as temp:
            root=Path(temp);path=root/'trace.csv'
            atomic_csv_save([dict(scene_id='S0',condition_id='c',seconds=t,initial_ttd=10,predicted_ttd=8,evaluations=8000)
                            for t in (1,2,100)],path)
            summarize_native(path,root/'out');row=read(root/'out/rolling_optimizer_timing.csv')[0]
            self.assertEqual(float(row['mean_within_replan_predicted_gain']),2.)
            self.assertAlmostEqual(float(row['p95_replan_seconds']),float(np.quantile([1,2,100],.95)))

    def test_missing_sources_produce_availability_not_fake_results(self):
        with tempfile.TemporaryDirectory(prefix='missing_statistics_') as temp:
            root=Path(temp);spec=root/'spec.json'
            atomic_json_save(dict(sources=[dict(path=str(root/'not_returned'),domain='generated')]),spec)
            with patch('sys.argv',['summary']):args=summarize.parse_args()
            args.input_spec=spec;args.output_dir=root/'out';args.bootstrap_draws=8
            with contextlib.redirect_stdout(io.StringIO()):result=summarize.execute(args)
            self.assertFalse(result)
            self.assertEqual(read(root/'out/source_availability.csv')[0]['status'],'not_available')

    def test_incomplete_instance_does_not_remove_seed303_elsewhere(self):
        with tempfile.TemporaryDirectory(prefix='incomplete_statistics_') as temp:
            root=Path(temp);path=root/'raw.csv'
            rows=[dict(method='ours_categorical_baa',training_seed=seed,condition_id='c',instance_seed=i,
                simulation_seed=1,total_tardiness=300 if seed==303 else 0,status='completed')
                for seed in (101,202,303) for i in (1,2)]
            rows[-2]['status']='failed';rows[-2]['total_tardiness']=''
            atomic_csv_save(rows,path);store=EpisodeStore(root/'data.sqlite')
            try:store.ingest(dict(path=str(path),domain='generated'),root);data=store.aggregate()
            finally:store.close()
            inference=summarize.complete_inference_cells(data,1)
            self.assertTrue(inference.loc[inference.instance=='c::1','total_tardiness'].isna().all())
            self.assertEqual(float(inference.loc[(inference.instance=='c::2')&(inference.seed=='303'),'total_tardiness'].iloc[0]),300.)


if __name__=='__main__':unittest.main()
