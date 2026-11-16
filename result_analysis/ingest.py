"Stream raw episode CSVs to a temporary deduplicated SQLite table."
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from experiment.metrics import EpisodeMetrics
from .labels import method_name,mode_name

METRICS=tuple(k for k in EpisodeMetrics.__dataclass_fields__ if k!='terminated')+(
    'episode_wall_time_seconds','batch_count','mean_jobs_per_batch','terminal_degradation_failures',
    'ma_offline_ttd','ma_execution_delta','ma_execution_degradation_percent','ma_optimization_seconds',
    'idle_decisions',
    'idle_wait_count','idle_ipm_count','idle_pm_count','idle_maintenance_actual_time',
    'idle_maintenance_arrival_overlap_count','gap_to_milp_reference_percent')
KEYS=('domain','family','mode','scene','condition','instance','method','seed','simulation')
META=('rho','due','jobs')


def numeric(value):
    try:
        result=float(value)
        return result if np.isfinite(result) else None
    except (ValueError,TypeError):return None


def paths_for(source,workspace):
    path=Path(source['path']);path=path if path.is_absolute() else workspace/path
    if path.is_dir():return sorted((path/'partitions'/'run_metrics').glob('*.csv'))
    return [path] if path.is_file() else []


class EpisodeStore:
    def __init__(self,path):
        self.connection=sqlite3.connect(path)
        self.connection.execute('PRAGMA journal_mode=OFF')
        declarations=[f'"{k}" TEXT' for k in KEYS]+[f'"{m}" REAL' for m in (*META,*METRICS)]
        declarations+=['status TEXT','signature TEXT','source_path TEXT','duplicates INTEGER DEFAULT 0','conflicted INTEGER DEFAULT 0']
        self.connection.execute('CREATE TABLE observations ('+','.join(declarations)+',PRIMARY KEY ('+','.join(f'"{k}"' for k in KEYS)+'))')
        self.sources=[]

    def ingest(self,source,workspace,allowed_cases=None,condition_scope=None):
        columns=(*KEYS,*META,*METRICS,'status','signature','source_path')
        sql='INSERT INTO observations ('+','.join(f'"{c}"' for c in columns)+') VALUES ('+','.join('?' for _ in columns)+')'
        sql+=' ON CONFLICT ('+','.join(f'"{k}"' for k in KEYS)+') DO UPDATE SET duplicates=observations.duplicates+1,conflicted=CASE WHEN observations.signature!=excluded.signature THEN 1 ELSE observations.conflicted END'
        files=paths_for(source,workspace);count=0;excluded=0;batch=[]
        for path in files:
            with path.open(encoding='utf-8-sig',newline='') as stream:
                for row in csv.DictReader(stream):
                    if source.get('variants') and row.get('ablation_variant') not in source['variants']:
                        excluded+=1;continue
                    if condition_scope and source['domain']=='generated':
                        from experiment.revision_selection import include_generated_result
                        if not include_generated_result(row.get('condition_id'),row.get('scene_id') or 'S0',source.get('family','main')):
                            excluded+=1;continue
                    if source.get('protocols') and row.get('protocol') not in source['protocols']:
                        excluded+=1;continue
                    if source['domain']=='generated' and allowed_cases is not None and (
                            str(row.get('condition_id')), str(row.get('instance_seed')), str(row.get('simulation_seed'))) not in allowed_cases:
                        excluded+=1;continue
                    domain=source['domain'];family=source.get('family','main')
                    condition=str(row.get('condition_id') or 'jobs-'+str(row.get('n_jobs','industrial')))
                    instance=str(row.get('instance_seed') or row.get('arrival_seed') or row.get('instance_index') or row.get('instance_number'))
                    seed=str(row.get('training_seed') or 'deterministic')
                    simulation=str(row.get('simulation_seed') or row.get('replication_index'))
                    identity=(domain,family,mode_name({**source,**row},domain),row.get('scene_id') or ('S0' if domain=='generated' else 'industrial'),
                              condition,condition+'::'+instance,method_name(row,source),seed,simulation)
                    meta=(numeric(row.get('nominal_finite_load',row.get('target_load'))),
                        numeric(row.get('due_date_factor')) if domain=='generated' else None,
                        numeric(row.get('total_jobs',row.get('n_jobs',row.get('job_count')))))
                    values=tuple(numeric(row.get(k)) for k in METRICS)
                    status=row.get('status') or 'completed'
                    if values[METRICS.index('total_tardiness')] is None:status='unavailable'
                    substantive=[v for k,v in zip(METRICS,values) if 'latency' not in k and 'seconds' not in k]
                    signature=hashlib.sha256(json.dumps([status,substantive]).encode()).hexdigest()
                    batch.append((*identity,*meta,*values,status,signature,str(path)));count+=1
                    if len(batch)>=2000:self.connection.executemany(sql,batch);batch=[]
        if batch:self.connection.executemany(sql,batch)
        self.connection.commit()
        self.sources.append(dict(path=source['path'],domain=source['domain'],family=source.get('family','main'),
                                 files=len(files),raw_rows=count,excluded_by_selection=excluded,status='loaded' if files else 'not_available'))

    def aggregate(self):
        group=KEYS[:-1]
        expressions=[*('"'+k+'"' for k in group),*('MAX("'+k+'") AS "'+k+'"' for k in META),
            'COUNT(*) AS observed_replications',
            "SUM(CASE WHEN status='completed' AND conflicted=0 THEN 1 ELSE 0 END) AS completed_replications",
            "SUM(CASE WHEN status!='completed' THEN 1 ELSE 0 END) AS failed_replications",
            'SUM(conflicted) AS conflicted_replications','SUM(duplicates) AS duplicate_rows']
        expressions += [f'AVG(CASE WHEN status=\'completed\' AND conflicted=0 THEN "{m}" END) AS "{m}"' for m in METRICS]
        return pd.read_sql_query('SELECT '+','.join(expressions)+' FROM observations GROUP BY '+','.join('"'+k+'"' for k in group),self.connection)

    def close(self):self.connection.close()
