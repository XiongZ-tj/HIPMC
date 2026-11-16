"Map stored method identifiers to canonical reporting labels."
METHODS={'ours':'HIPMC','ours_categorical_baa':'HIPMC','full_hipmc':'HIPMC',
    'ddqnb':'DDQNB','psa_mappo':'PSA-MAPPO','hdrl_option':'HDRL','ma':'MA','rh_alns':'RH-ALNS',
    'composite':'Tuned composite priority',
    'idle_opportunity':'HIPMC with idle-opportunity maintenance',
    'mmu_batch_selection':'maximum-utilization-based batch selection',
    'eddb_batch_selection':'earliest-due-date batch selection',
    'mast_batch_selection':'minimum-average-slack batch selection',
    'periodic_maintenance_policy':'periodic-maintenance policy',
    'health_state_threshold_maintenance_policy':'health-state-threshold maintenance policy',
    'tuned_composite_batch_selection':'tuned composite batch-priority selection',
    'edd_job_batch_formation':'earliest-due-date job-based batch formation',
    'ffd_batch_formation':'first-fit-decreasing batch formation',
    'corrective_only_maintenance_policy':'corrective-only maintenance policy'}


def method_name(row,source):
    raw=str(source.get('method') or row.get('ablation_variant') or row.get('protocol') or
            row.get('method') or row.get('method_id') or '')
    if raw.startswith('DRL-'):raw=raw[4:].rsplit('-seed',1)[0].lower()
    if raw.startswith('SDR-'):
        name=raw[4:].replace('Threshold','TH').replace('Period','PERIOD').replace('EDD-J','EDDJ').replace('EDD-B','EDDB')
    elif raw in METHODS:name=METHODS[raw]
    else:name=raw
    return name+source.get('method_suffix','')


def mode_name(row,domain):
    value=row.get('deployment_mode') or row.get('mode')
    if not value:return 'standard' if domain=='generated' else 'common'
    return {'from_scratch':'scratch','finetune_from_base':'finetune','base_case_zero_shot':'zero_shot'}.get(value,value)
