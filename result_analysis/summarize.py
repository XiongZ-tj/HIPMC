"Aggregate selected experiment sources without modifying raw observations."
import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment.artifacts import atomic_csv_save,atomic_json_save
from experiment.revision_protocol import TRAINING_SEEDS, TEST_INSTANCES_PER_CONDITION
from result_analysis.ingest import EpisodeStore,METRICS,paths_for
from result_analysis.statistics import paired_comparison,method_summary,mean_interval,holm


def write_frame(frame,path):
    rows=frame.astype(object).where(pd.notna(frame),None).to_dict('records')
    atomic_csv_save(rows,path,fieldnames=list(frame.columns))


def subset_mode(frame,mode):
    return frame[frame['mode'].isin([mode,'common'])]


def compute_tables(data,spec,*,draws=2000,seed=47291,tolerance=1e-9):
    summaries=[];conditions=[];factors=[];tests=[]
    for (domain,family,mode),base in data.groupby(['domain','family','mode'],dropna=False):
        if mode=='common':continue
        group=data[(data.domain==domain)&(data.family==family)]
        group=subset_mode(group,mode)
        ref_name=spec.get('references',{}).get(family,'HIPMC')
        reference=group[group.method==ref_name]
        if reference.empty:
            reference=subset_mode(data[(data.domain==domain)&(data.family==spec.get('reference_families',{}).get(family,'main'))&(data.method==ref_name)],mode)
        if family=='components':
            # Reuse existing HIPMC runs, but not the extra 22 S0 conditions.
            reference=reference[reference.condition.isin(group.condition.unique())]
        methods=sorted(group.method.unique())
        if not reference.empty and ref_name not in methods:methods=[ref_name,*methods]
        scenes=sorted(base.scene.unique())
        pooled=[s for s in scenes if s!='S0'] if domain=='generated' and family!='components' and any(s!='S0' for s in scenes) else scenes
        for scope,scene_names in [*((s,[s]) for s in scenes),('all_scenes',pooled)]:
            identity=dict(domain=domain,family=family,mode=mode,scope=scope,scenes=len(scene_names))
            ref=reference[reference.scene.isin(scene_names)]
            for method in methods:
                frame=ref if method==ref_name else group[(group.method==method)&group.scene.isin(scene_names)]
                if frame.empty:continue
                if len(scene_names)>1:
                    valid=frame[frame.total_tardiness.notna()].groupby('instance').scene.nunique()
                    frame=frame[frame.instance.isin(valid[valid==len(scene_names)].index)]
                    if frame.empty:continue
                common_only=bool((frame['mode']=='common').all())
                if not common_only:
                    for metric in METRICS:
                        if frame[metric].notna().any():summaries.append({**identity,'method':method,**method_summary(frame,metric),
                            **(mean_interval(frame,draws=draws,seed=seed) if metric=='total_tardiness' else {})})
                if not ref.empty and method!=ref_name:
                    tests.append({**identity,'reference':ref_name,'comparator':method,
                        **paired_comparison(ref,frame,draws=draws,seed=seed,tolerance=tolerance,required_scenes=scene_names)})
                if scope!='all_scenes' and not common_only:
                    for condition,part in frame.groupby('condition'):
                        for metric in METRICS:
                            if part[metric].notna().any():conditions.append({**identity,'method':method,'condition':condition,**method_summary(part,metric)})
                    if domain=='generated':
                        for factor in ('rho','due','jobs'):
                            for value,part in frame.groupby(factor):
                                factors.append({**identity,'method':method,'factor':factor,'level':value,
                                                **method_summary(part,'total_tardiness')})
    # Non-learning industrial methods have one common result, not three trained modes.
    for (domain,family,scene,method),frame in data[data['mode']=='common'].groupby(['domain','family','scene','method']):
        for metric in METRICS:
            if frame[metric].notna().any():summaries.append(dict(domain=domain,family=family,mode='common',scope=scene,
                scenes=1,method=method,**method_summary(frame,metric),
                **(mean_interval(frame,draws=draws,seed=seed) if metric=='total_tardiness' else {})))
        for condition,part in frame.groupby('condition'):
            for metric in METRICS:
                if part[metric].notna().any():conditions.append(dict(domain=domain,family=family,mode='common',scope=scene,
                    scenes=1,method=method,condition=condition,**method_summary(part,metric)))
    tests=pd.DataFrame(tests)
    if not tests.empty and 'p_value' in tests:
        tests['p_holm_within_scope']=tests.groupby(['domain','family','mode','scope'])['p_value'].transform(lambda s:holm(s.to_numpy()))
        tests['scope_family']=tests.scope.map(lambda s:'across_scenes' if s=='all_scenes' else 'per_scene')
        tests['p_holm_across_scene_family']=tests.groupby(['domain','family','mode','scope_family'])['p_value'].transform(lambda s:holm(s.to_numpy()))
    return dict(method_summary=pd.DataFrame(summaries),condition_summary=pd.DataFrame(conditions),
                factor_summary=pd.DataFrame(factors),paired_tests=tests)


def complete_inference_cells(data,expected_replications=10):
    """Mark incomplete instance and model-seed cells unavailable for comparison."""
    result=data.copy()
    keys=['domain','family','mode','scene','condition','instance','method']
    global_seeds={key:set(part.seed) for key,part in data.groupby(['domain','family','mode','method'])}
    for key,part in data.groupby(keys):
        observed=global_seeds[(key[0],key[1],key[2],key[6])]
        expected={'deterministic'} if observed=={'deterministic'} else observed|set(map(str,TRAINING_SEEDS))
        complete=set(part.loc[part.completed_replications==expected_replications,'seed'])
        if not expected.issubset(complete):result.loc[part.index,list(METRICS)]=float('nan')
    return result


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-spec',type=Path,default=ROOT/'experiment_manifests/revision_analysis_sources.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/revision_result_statistics')
    p.add_argument('--bootstrap-draws',type=int,default=2000)
    p.add_argument('--statistical-seed',type=int,default=47291)
    p.add_argument('--tie-tolerance',type=float,default=1e-9)
    p.add_argument('--expected-replications',type=int,default=10)
    p.add_argument('--instances',type=int,default=TEST_INSTANCES_PER_CONDITION,
                   help='Generated-only prefix; industrial instances remain unchanged.')
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args()


def execute(args):
    spec_path=args.input_spec if args.input_spec.is_absolute() else ROOT/args.input_spec
    spec=json.loads(spec_path.read_text(encoding='utf-8'))
    allowed_cases=None
    subset=spec.get('generated_test_selection')
    if subset:
        manifest=Path(subset['case_manifest'])
        manifest=manifest if manifest.is_absolute() else ROOT/manifest
        with manifest.open(encoding='utf-8-sig',newline='') as stream:
            allowed_cases={(r['condition_id'],r['instance_seed'],r['simulation_seed'])
                for r in csv.DictReader(stream) if int(r['instance_index'])<args.instances}
    if args.dry_run:
        plan=[dict(**s,available_csv_files=len(paths_for(s,ROOT))) for s in spec['sources']]
        print(json.dumps(dict(sources=plan,bootstrap_draws=args.bootstrap_draws),indent=2));return plan
    out=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='revision_statistics_') as temp:
        store=EpisodeStore(Path(temp)/'episodes.sqlite')
        try:
            for source in spec['sources']:
                store.ingest(source,ROOT,allowed_cases=allowed_cases,
                             condition_scope=spec.get('generated_condition_selection'))
                print(f'[RESULT-SUMMARY] {store.sources[-1]}',flush=True)
            data=store.aggregate()
            write_frame(pd.DataFrame(store.sources),out/'source_availability.csv')
        finally:store.close()
    write_frame(data,out/'instance_model_seed_means.csv')
    coverage=data[[c for c in ('domain','family','mode','scene','condition','instance','method','seed',
        'observed_replications','completed_replications','failed_replications','conflicted_replications','duplicate_rows') if c in data]].copy()
    if not coverage.empty:
        coverage['expected_replications']=args.expected_replications
        coverage['complete_replications']=coverage.completed_replications==args.expected_replications
        keys=['domain','family','mode','scene','condition','instance','method']
        completeness=[]
        for key,part in coverage.groupby(keys):
            observed=set(part.seed);expected={'deterministic'} if observed=={'deterministic'} else observed|set(map(str,TRAINING_SEEDS))
            completed=set(part.loc[part.complete_replications,'seed'])
            completeness.append(dict(zip(keys,key),observed_model_seeds=json.dumps(sorted(observed)),
                missing_complete_model_seeds=json.dumps(sorted(expected-completed)),complete_model_seeds=not bool(expected-completed)))
        write_frame(pd.DataFrame(completeness),out/'model_seed_coverage.csv')
    write_frame(coverage,out/'coverage.csv')
    inference=complete_inference_cells(data,args.expected_replications) if not data.empty else data
    tables=compute_tables(inference,spec,draws=args.bootstrap_draws,seed=args.statistical_seed,tolerance=args.tie_tolerance) if not data.empty else {}
    for name,table in tables.items():write_frame(table,out/(name+'.csv'))
    from result_analysis.followups import followup_tables
    for name,table in followup_tables(inference,draws=args.bootstrap_draws,seed=args.statistical_seed,tolerance=args.tie_tolerance).items():
        write_frame(table,out/(name+'.csv'))
    atomic_json_save(dict(input_spec=str(spec_path),source_availability=store.sources,
        model_seed_policy='all observed model seeds retained, including 303',expected_protocol_model_seeds=list(TRAINING_SEEDS),
        generated_test_instances_per_condition=args.instances if subset else None,
        generated_test_selection=subset,
        generated_condition_selection=spec.get('generated_condition_selection'),
        component_reference='reuse existing HIPMC results on matching scenes and component condition subset; S0 excludes the other 22 conditions',
        statistical_unit='structural order instance after averaging online simulation seeds',
        cross_scene_unit='same original instance clustered across scenes',
        industrial_population='arrival realizations conditional on the shared observed order set',
        bootstrap='condition-stratified instance resampling and global model-seed resampling; paired model labels when matched',
        bootstrap_draws=args.bootstrap_draws,statistical_seed=args.statistical_seed,
        relative_denominator='comparator',zero_policy='no epsilon; positive-pair geometric ratio plus explicit zero counts',
        uncertainty='model_seed_std and instance_std are distinct; percentile bootstrap intervals are not those standard deviations',
        tests='two-sided instance-level Wilcoxon on model-averaged outcomes; signed rank-biserial positive favors reference',
        multiplicity='Holm within each domain/family/mode/scope, plus across the per-scene family',
        tie_tolerance=args.tie_tolerance,incomplete_data='raw partial means retained; primary inference uses complete prescribed replications/model seeds and matched instance-scene cells; no whole-seed or outlier filtering',
        conflicts='conflicting substantive duplicate outcomes unavailable; identical duplicates not additional replications'),out/'statistical_protocol.json')
    from result_analysis.section5_tables import export_section5_tables
    export_section5_tables(ROOT,out,spec,instances=args.instances)
    return tables


if __name__=='__main__':execute(parse_args())
