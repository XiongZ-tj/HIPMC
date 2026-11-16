"""Summarize offline MA search evidence, without treating the proxy as a bound."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment.artifacts import atomic_csv_save,atomic_json_save


def summarize(root,instances=None):
    rows=[]
    for manifest in sorted(Path(root).rglob('plan_manifest.csv')):
        with manifest.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                from experiment.revision_selection import GENERALIZATION_SCENES,selected_conditions
                scene=next((p for p in manifest.relative_to(root).parts if p in GENERALIZATION_SCENES),'S0')
                if not selected_conditions([row['condition_id']],scene):continue
                if instances is not None and int(row['instance_index'])>=instances:continue
                plan=manifest.parent/row['plan_file']
                with plan.open(encoding='utf-8-sig',newline='') as source:metadata=next(csv.DictReader(source))
                convergence=manifest.parent/row['convergence_file']
                first=last=None;count=0
                with convergence.open(encoding='utf-8-sig',newline='') as source:
                    for point in csv.DictReader(source):
                        if first is None:first=point
                        last=point;count+=1
                rows.append(dict(plan_set=str(manifest.parent.relative_to(root)),condition_id=row['condition_id'],
                    instance_index=row['instance_index'],instance_seed=row['instance_seed'],
                    initial_offline_ttd=float(metadata['initial_objective_total_tardiness']),
                    pre_refinement_offline_ttd=float(metadata['baseline_best_ttd']),
                    final_offline_ttd=float(row['objective_total_tardiness']),
                    generations=int(row['generations']),search_evaluations=int(row['search_evaluations']),
                    optimization_seconds=float(row['runtime_seconds']),stop_reason=row['stop_reason'],
                    last_improvement_generation=int(row['last_improvement_generation']),
                    convergence_points=count,convergence_first_evaluation=int(first['search_evaluations']) if first else None,
                    convergence_last_evaluation=int(last['search_evaluations']) if last else None,
                    plan_file=str(plan),convergence_file=str(convergence)))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-root',type=Path,required=True)
    p.add_argument('--instances',type=int,help='Generated plans: first N original indices; omit for industrial.')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/revision_ma_search_statistics')
    args=p.parse_args();rows=summarize(args.plan_root,args.instances);atomic_csv_save(rows,args.output_dir/'ma_search_by_instance.csv')
    groups=defaultdict(list)
    for row in rows:groups[(row['plan_set'],row['condition_id'])].append(row)
    summary=[]
    for (plan_set,condition),items in groups.items():
        initial=float(np.mean([r['initial_offline_ttd'] for r in items]));final=float(np.mean([r['final_offline_ttd'] for r in items]))
        summary.append(dict(plan_set=plan_set,condition=condition,instances=len(items),initial_mean_offline_ttd=initial,
            pre_refinement_mean_offline_ttd=float(np.mean([r['pre_refinement_offline_ttd'] for r in items])),final_mean_offline_ttd=final,
            relative_search_improvement_percent=100*(initial-final)/initial if initial>0 else None,
            mean_optimization_seconds=float(np.mean([r['optimization_seconds'] for r in items])),
            maximum_generation_stops=sum(r['stop_reason']=='maximum_generations' for r in items)))
    atomic_csv_save(summary,args.output_dir/'ma_search_by_condition.csv')
    atomic_json_save(dict(offline_objective='expected-physics surrogate, neither stochastic optimum nor lower bound',
        plan_sets='different saved budget/scene namespaces kept separate; no best-run selection',
        convergence='source curves remain in their original CSVs; instance table links them'),args.output_dir/'ma_statistics_protocol.json')


if __name__=='__main__':main()
