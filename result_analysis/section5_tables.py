"Summarize selected training results and online computation times."
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment.artifacts import atomic_csv_save, atomic_json_save
from .labels import METHODS


def export_section5_tables(workspace, output, spec, *, instances=10):
    workspace, output = Path(workspace), Path(output)
    sources = [s for s in spec.get('sources', [])
               if s.get('domain') == 'generated' and s.get('family', 'main') == 'main']
    if not sources:
        return
    selected_file = workspace / 'experiment_manifests/selected_generated_hyperparameters.json'
    training_file = workspace / 'runs/revision_hyperparameter_statistics/configuration_summary.csv'
    selection = json.loads(selected_file.read_text(encoding='utf-8')) if selected_file.exists() else None
    if selection is not None:
        atomic_json_save(selection, output / 'selected_hyperparameters.json')
        if training_file.exists():
            training = pd.read_csv(training_file)
            selected = []
            for method, item in selection['methods'].items():
                matched = training[(training.method == METHODS.get(method, method)) &
                                   (training.configuration_id == item['configuration_id'])]
                selected.extend({**row, 'selection_rationale': item.get('rationale', '')}
                                for row in matched.to_dict('records'))
            if selected:
                atomic_csv_save(selected, output / 'selected_configuration_summary.csv')

    cells, seen = {}, set()
    unavailable = 0
    for source in sources:
        root = Path(source['path'])
        root = root if root.is_absolute() else workspace / root
        paths = sorted((root / 'partitions/run_metrics').glob('*.csv')) if root.is_dir() else [root] if root.is_file() else []
        for path in paths:
            with path.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream):
                    method = row.get('method', '')
                    if method not in ('ours_categorical_baa', 'ddqnb', 'psa_mappo', 'hdrl_option'):
                        continue
                    if row.get('scene_id', 'S0') != 'S0' or int(row['instance_index']) >= instances:
                        continue
                    if int(row.get('replication_index', 0)) >= 10:
                        continue
                    identity = (method, row['training_seed'], row['condition_id'],
                                row['instance_seed'], row['simulation_seed'])
                    if identity in seen:
                        continue
                    seen.add(identity)
                    try:
                        latency, decisions = float(row['decision_latency_mean']), int(row['decisions'])
                        wall = float(row['episode_wall_time_seconds'])
                    except (KeyError, ValueError, TypeError):
                        unavailable += 1
                        continue
                    if row.get('status') != 'completed' or not np.isfinite([latency, wall]).all() or decisions <= 0:
                        unavailable += 1
                        continue
                    key = (METHODS[method], row['training_seed'], int(row['job_count']))
                    cell = cells.setdefault(key, dict(episodes=0, decisions=0, total_decision_seconds=0., episode_wall_seconds=0.))
                    cell['episodes'] += 1
                    cell['decisions'] += decisions
                    cell['total_decision_seconds'] += latency * decisions
                    cell['episode_wall_seconds'] += wall
    if not cells:
        return
    per_seed = [dict(method=m, training_seed=int(s), jobs=j, **v,
                    mean_decision_milliseconds=1000 * v['total_decision_seconds'] / v['decisions'],
                    mean_episode_decision_seconds=v['total_decision_seconds'] / v['episodes'],
                    mean_episode_wall_seconds=v['episode_wall_seconds'] / v['episodes'])
                for (m, s, j), v in sorted(cells.items())]
    atomic_csv_save(per_seed, output / 's0_inference_timing_by_size_seed.csv')
    summary = []
    for (method, jobs), group in pd.DataFrame(per_seed).groupby(['method', 'jobs']):
        entry = dict(method=method, jobs=int(jobs), training_seeds=len(group), episodes=int(group.episodes.sum()))
        for metric in ('mean_decision_milliseconds', 'mean_episode_decision_seconds', 'mean_episode_wall_seconds'):
            entry[metric + '_mean'] = float(group[metric].mean())
            entry[metric + '_model_seed_std'] = float(group[metric].std(ddof=1)) if len(group) > 1 else None
        summary.append(entry)
    atomic_csv_save(summary, output / 's0_inference_timing_by_size.csv')
    atomic_json_save(dict(scene='S0', instances_per_condition=instances, simulation_seeds_per_instance=10,
        source_paths=[s['path'] for s in sources], unavailable_timing_episodes=unavailable,
        per_episode_decision_seconds='decision_latency_mean * decisions before averaging replications',
        mean_decision_milliseconds='1000 * sum(episode_decision_seconds) / sum(decisions) within each model seed and job size',
        aggregation='mean and sample standard deviation across the observed model seeds',
        wall_time='Includes simulator advancement and is not a substitute for decision computation time'),
        output / 's0_inference_timing_protocol.json')
