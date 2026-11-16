"""Training/validation evidence tables; no test-based parameter selection."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from result_analysis.labels import method_name
from result_analysis.summarize import write_frame
from experiment.artifacts import atomic_json_save


def read_optional(path):
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def summarize_run(directory,budget=500,late_window=50):
    config=json.loads((directory/'run_config.json').read_text(encoding='utf-8'))
    snapshot=directory/'budgets'/f'episodes-{budget}'
    logs=snapshot if (snapshot/'Training_Episodes.csv').exists() else directory
    episodes=read_optional(logs/'Training_Episodes.csv')
    validation=read_optional(logs/'Training_Validation.csv')
    if 'episode' in episodes:episodes=episodes[episodes.episode<=budget].sort_values('episode')
    if 'episode' in validation:validation=validation[validation.episode<=budget].sort_values('episode')
    hp=config.get('hyperparameters',{})
    row=dict(source_directory=str(directory),method=method_name(config,{}),training_seed=config.get('training_seed'),
        domain='industrial' if 'industrial' in config.get('experiment','') else 'generated',
        mode=config.get('mode','standard'),budget=budget,learning_rate=hp.get('learning_rate'),
        gamma=hp.get('gamma_baa',hp.get('gamma')),clip_ratio=hp.get('clip_ratio'),
        configuration_id=next((p.name for p in directory.parents if p.name.startswith('lr-')),''),
        episodes_observed=len(episodes),complete_training_budget=len(episodes)>=budget)
    if not episodes.empty and 'total_tardiness' in episodes:
        values=pd.to_numeric(episodes.total_tardiness,errors='coerce')
        row.update(first_episode_ttd=float(values.iloc[0]),first_window_training_ttd=float(values.head(late_window).mean()),
            late_window_training_ttd=float(values.tail(late_window).mean()),
            late_window_training_std=float(values.tail(late_window).std(ddof=1)),
            late_window_observed=min(len(values),late_window),training_ttd_reduction=float(values.head(late_window).mean()-values.tail(late_window).mean()))
    if 'mean_tardiness' in validation and validation.mean_tardiness.notna().any():
        best=validation.loc[validation.mean_tardiness.idxmin()]
        row.update(selected_validation_score=float(best.mean_tardiness),selected_episode=int(best.episode),
            initial_validation_score=float(validation.mean_tardiness.iloc[0]),
            selected_validation_ttd=float(best['total_tardiness_mean']) if 'total_tardiness_mean' in best else None,
            validation_evaluations=len(validation))
        threshold=float(best.mean_tardiness)*1.05
        reached=validation[validation.mean_tardiness<=threshold]
        row['first_episode_within_five_percent_of_best_validation']=int(reached.episode.iloc[0]) if len(reached) else None
    runtime=snapshot/'training_budget_runtime.json'
    if runtime.exists():row['training_validation_wall_seconds']=json.loads(runtime.read_text(encoding='utf-8')).get('training_validation_wall_seconds')
    else:
        marker=directory/'run_complete.json';complete=json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else {}
        row['training_validation_wall_seconds']=(complete.get('training_and_validation_wall_seconds',complete.get('elapsed_seconds'))
            if complete.get('episodes')==budget else None)
    diagnostics=[]
    updates=read_optional(logs/'Training_Updates.csv')
    if 'episode' in updates:updates=updates[updates.episode<=budget]
    if not updates.empty:
        grouping=[c for c in ('record_scope','component','head_id') if c in updates]
        groups=updates.groupby(grouping,dropna=False) if grouping else [('all',updates)]
        for key,part in groups:
            descriptor=dict(zip(grouping,key if isinstance(key,tuple) else [key]))
            for metric in ('loss','actor_loss','critic_loss','entropy','approximate_kl','clip_fraction','explained_variance','gradient_norm_before_clip'):
                if metric not in part:continue
                values=pd.to_numeric(part[metric],errors='coerce').dropna()
                if values.empty:continue
                diagnostics.append({**row,**descriptor,'metric':metric,'observations':len(values),'mean':float(values.mean()),
                    'p95':float(values.quantile(.95)),'minimum':float(values.min()),
                    'fraction_below_minus_one':float((values< -1).mean()) if metric=='explained_variance' else None})
    return row,diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-roots',type=Path,nargs='+',required=True)
    p.add_argument('--budget',type=int,default=500)
    p.add_argument('--late-window',type=int,default=50)
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/revision_training_statistics')
    args=p.parse_args();rows=[];diagnostics=[]
    for root in args.training_roots:
        root=root if root.is_absolute() else ROOT/root
        for path in sorted(root.rglob('run_config.json')):
            if 'budgets' in path.relative_to(root).parts:continue
            row,detail=summarize_run(path.parent,args.budget,args.late_window)
            rows.append(row);diagnostics.extend(detail)
    output=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
    data=pd.DataFrame(rows);write_frame(data,output/'training_by_seed.csv')
    write_frame(pd.DataFrame(diagnostics),output/'optimizer_diagnostics.csv')
    summary=[]
    if not data.empty:
        keys=[k for k in ('domain','mode','method','configuration_id','learning_rate','gamma','clip_ratio') if k in data]
        for key,part in data.groupby(keys,dropna=False):
            item=dict(zip(keys,key));item.update(model_seeds=part.training_seed.nunique(),
                observed_training_seeds=json.dumps(sorted(part.training_seed.dropna().astype(int).unique().tolist())),
                complete_runs=int(part.complete_training_budget.sum()),runs=len(part))
            for metric in ('first_episode_ttd','late_window_training_ttd','late_window_training_std',
                'selected_validation_ttd','selected_validation_score','selected_episode',
                'first_episode_within_five_percent_of_best_validation','training_validation_wall_seconds'):
                if metric in part:
                    values=pd.to_numeric(part[metric],errors='coerce').dropna()
                    item[metric+'_mean']=float(values.mean()) if len(values) else None
                    item[metric+'_model_seed_std']=float(values.std(ddof=1)) if len(values)>1 else None
            summary.append(item)
    write_frame(pd.DataFrame(summary),output/'configuration_summary.csv')
    atomic_json_save(dict(budget=args.budget,late_window=args.late_window,
        selection='no automatic configuration choice; training stability, validation and cost shown separately',
        threshold_definition='first validation episode within 5% of its observed minimum, not a claim of permanent convergence',
        validation_selection='first occurrence of the minimum mean job tardiness within the declared budget',
        model_seed_policy='retain every observed seed including 303',
        runtime_missing='unavailable, never inferred from a later 3000-episode total'),output/'training_statistics_protocol.json')


if __name__=='__main__':main()
