"""Summarize native rolling-horizon optimization durations and search gains."""
import argparse
import csv
from pathlib import Path
import sqlite3
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment.artifacts import atomic_csv_save,atomic_json_save




def summarize_native(path,output,instances=None):
    result=[]
    with tempfile.TemporaryDirectory(prefix='native_optimizer_timing_') as temp:
        db=sqlite3.connect(Path(temp)/'replans.sqlite')
        db.execute('CREATE TABLE replans(scene TEXT,condition TEXT,seconds REAL,gain REAL,evaluations REAL)')
        batch=[]
        source=Path(path)
        paths=sorted(source.glob('*.csv')) if source.is_dir() else [source]
        from experiment.revision_selection import selected_conditions
        for source_file in paths:
            with source_file.open(encoding='utf-8-sig',newline='') as stream:
                for row in csv.DictReader(stream):
                    if instances is not None and int(row['instance_index'])>=instances:continue
                    if not selected_conditions([row.get('condition_id','')],row.get('scene_id','industrial')):continue
                    batch.append((row.get('scene_id','industrial'),row.get('condition_id',''),float(row['seconds']),
                        float(row['initial_ttd'])-float(row['predicted_ttd']),float(row['evaluations'])))
                    if len(batch)>=4000:db.executemany('INSERT INTO replans VALUES (?,?,?,?,?)',batch);batch=[]
        if batch:db.executemany('INSERT INTO replans VALUES (?,?,?,?,?)',batch)
        db.execute('CREATE INDEX lookup ON replans(scene,condition,seconds)');db.commit()
        for scene,condition,n,mean,total,gain,evaluations in db.execute('SELECT scene,condition,COUNT(*),AVG(seconds),SUM(seconds),AVG(gain),AVG(evaluations) FROM replans GROUP BY scene,condition'):
            position=.95*(n-1);lo=int(np.floor(position));hi=int(np.ceil(position))
            query='SELECT seconds FROM replans WHERE scene=? AND condition=? ORDER BY seconds LIMIT 1 OFFSET ?'
            lower=db.execute(query,(scene,condition,lo)).fetchone()[0];upper=db.execute(query,(scene,condition,hi)).fetchone()[0]
            result.append(dict(scene=scene,condition=condition,replans=n,mean_replan_seconds=mean,
                p95_replan_seconds=lower+(position-lo)*(upper-lower),total_replan_seconds=total,
                mean_within_replan_predicted_gain=gain,mean_evaluations=evaluations,
                objective_scope='same known-task snapshot within each replan, not global episode convergence'))
        db.close()
    atomic_csv_save(result,Path(output)/'rolling_optimizer_timing.csv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rh-trace',type=Path,required=True)
    p.add_argument('--instances',type=int,help='Generated RH trace: first N original instance indices.')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/revision_timing_statistics')
    args=p.parse_args();out=args.output_dir
    if args.rh_trace:summarize_native(args.rh_trace,out,args.instances)
    atomic_json_save(dict(units='seconds', percentile='linear interpolation of reoptimization durations'),
        out/'timing_statistics_protocol.json')


if __name__=='__main__':main()
