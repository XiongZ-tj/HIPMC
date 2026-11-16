"Fixed pressure-stratum selection, paired seeds, runner counts and reuse."
import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiment.revision_selection import GENERALIZATION_SCENES,REPRESENTATIVE_CONDITION_IDS,selected_conditions
from comparison_runners import evaluate_revision_nonlearning as native
from comparison_runners import evaluate_drl_baselines as drl
from result_analysis.ingest import EpisodeStore
from experiment.artifacts import atomic_csv_save


class RepresentativeTests(unittest.TestCase):
    def test_components_use_s0_subset_reference_and_pool_all_six_scenes(self):
        from experiment.revision_selection import COMPONENT_SCENES
        from result_analysis.summarize import compute_tables
        chosen=REPRESENTATIVE_CONDITION_IDS[0]
        other='rhoF-0p90_due-2_jobs-100_initial-20'
        def row(c,s,method,value):
            return dict(condition_id=c,scene_id=s,instance_seed='1',simulation_seed='2',
                        method=method,training_seed='101',total_tardiness=value)
        main=[row(c,s,'ours_categorical_baa',10 if c==chosen else 10000)
              for s in COMPONENT_SCENES for c in (chosen,other)]
        variants=[row(chosen,s,'mmu_batch_selection',12) for s in COMPONENT_SCENES]
        with tempfile.TemporaryDirectory(prefix='component_six_scenes_') as tmp:
            root=Path(tmp);atomic_csv_save(main,root/'main.csv');atomic_csv_save(variants,root/'components.csv')
            store=EpisodeStore(root/'store.sqlite')
            try:
                for family in ('main','components'):
                    store.ingest(dict(path=str(root/(family+'.csv')),domain='generated',family=family),
                                 root,condition_scope=True)
                tables=compute_tables(store.aggregate(),{'reference_families':{'components':'main'}},draws=8)
                summary=tables['method_summary']
                main_s0=summary[(summary.family=='main')&(summary.scope=='S0')&(summary.metric=='total_tardiness')]
                self.assertEqual(float(main_s0['mean'].iloc[0]),5005.)
                reference=summary[(summary.family=='components')&(summary.method=='HIPMC')&(summary.metric=='total_tardiness')]
                self.assertEqual(set(reference.scope),set(COMPONENT_SCENES)|{'all_scenes'})
                self.assertTrue((reference['mean']==10).all())
                self.assertEqual(int(reference.loc[reference.scope=='all_scenes','scenes'].iloc[0]),6)
                comparisons=tables['paired_tests']
                comparisons=comparisons[comparisons.family=='components']
                self.assertEqual(set(comparisons.scope),set(COMPONENT_SCENES)|{'all_scenes'})
                for value in comparisons.relative_improvement_percent:self.assertAlmostEqual(value,100/6)
            finally:store.close()

    def test_frozen_instances_and_original_seeds(self):
        rows=native.read(native.ROOT/'experiment_manifests/e1_formal_comparison_cases.csv')
        ids=list(dict.fromkeys(r['condition_id'] for r in rows))
        self.assertEqual(len(selected_conditions(ids,'S0')),27)
        self.assertEqual(set(selected_conditions(ids,'S1')),set(REPRESENTATIVE_CONDITION_IDS))
        for scene in GENERALIZATION_SCENES:
            chosen=[r for r in rows if r['condition_id'] in selected_conditions(ids,scene) and int(r['instance_index'])<10]
            self.assertEqual(len(chosen),500)
            self.assertEqual(len({(r['condition_id'],r['instance_seed']) for r in chosen}),50)
            self.assertTrue(all(r in rows for r in chosen))

    def test_native_mixed_scene_counts_and_ma_plan_subset(self):
        with patch('sys.argv',['test']):args=native.parse_args()
        args.methods=['ma','rh_alns'];args.scenes=['S0','S1'];args.dry_run=True
        with contextlib.redirect_stdout(io.StringIO()):plan=native.execute(args)
        self.assertEqual(plan['offline_plans'],320)
        self.assertEqual(plan['online_runs'],6400)
        self.assertEqual(len(plan['conditions_by_scene']['S0']),27)
        self.assertEqual(len(plan['conditions_by_scene']['S1']),5)
        args.scenes=list(GENERALIZATION_SCENES)
        with contextlib.redirect_stdout(io.StringIO()):plan=native.execute(args)
        self.assertEqual(plan['offline_plans'],250)
        self.assertEqual(plan['online_runs'],5000)

    def test_drl_formal_generalization_dry_run(self):
        import json
        with patch('sys.argv',['test','--scenes',*GENERALIZATION_SCENES,'--dry-run']):
            captured=io.StringIO()
            with contextlib.redirect_stdout(captured):drl.main()
        plan=json.loads(captured.getvalue())
        self.assertEqual(plan['blocks'],300)
        self.assertEqual(plan['runs'],30000)

    def test_historical_full_grid_filtered_without_deleting_raw_rows(self):
        selected=REPRESENTATIVE_CONDITION_IDS[0]
        other='rhoF-0p90_due-2_jobs-100_initial-20'
        rows=[dict(condition_id=c,scene_id=s,instance_seed='1',simulation_seed='2',
                   method='ours_categorical_baa',training_seed='101',total_tardiness=10)
              for c in (selected,other) for s in ('S0','S1','S2')]
        with tempfile.TemporaryDirectory(prefix='representative_selection_') as tmp:
            root=Path(tmp);path=root/'runs.csv';atomic_csv_save(rows,path)
            store=EpisodeStore(root/'observations.sqlite')
            try:
                for domain,family in [('generated','main'),('generated','components'),('industrial','main')]:
                    store.ingest(dict(path=str(path),domain=domain,family=family),root,condition_scope=True)
                data=store.aggregate()
                self.assertEqual(len(data[(data.domain=='generated')&(data.family=='main')]),4)
                self.assertEqual(len(data[(data.domain=='generated')&(data.family=='components')]),3)
                self.assertEqual(len(data[data.domain=='industrial']),6)
                self.assertEqual(len(native.read(path)),6)
                from result_analysis.summarize import compute_tables
                tables=compute_tables(data[data.domain=='generated'],{},draws=8)
                pooled=tables['method_summary']
                pooled=pooled[(pooled.family=='main')&(pooled.scope=='all_scenes')]
                self.assertEqual(set(pooled.scenes),{2})
            finally:store.close()


if __name__=='__main__':unittest.main()
