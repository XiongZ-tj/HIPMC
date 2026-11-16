"""Paired budget/mode follow-ups, distinct from the main method comparisons."""
import pandas as pd
from .statistics import paired_comparison,holm


def followup_tables(data,**kwargs):
    if data.empty:return {}
    budget=[];deployment=[]
    for (domain,mode,scene,method),extended in data[data.family=='extended3000'].groupby(['domain','mode','scene','method']):
        standard=data[(data.family=='main')&(data.domain==domain)&(data['mode']==mode)&(data.scene==scene)&(data.method==method)]
        if not standard.empty:budget.append(dict(domain=domain,mode=mode,scene=scene,method=method,
            reference='3000-episode validation-selected model',comparator='500-episode validation-selected model',
            **paired_comparison(extended,standard,**kwargs)))
    for (domain,scene,method),part in data[(data.domain=='industrial')&(data.family=='main')].groupby(['domain','scene','method']):
        for reference,comparator in (('scratch','zero_shot'),('finetune','zero_shot'),('finetune','scratch')):
            a=part[part['mode']==reference];b=part[part['mode']==comparator]
            if not a.empty and not b.empty:deployment.append(dict(domain=domain,scene=scene,method=method,
                reference_mode=reference,comparator_mode=comparator,**paired_comparison(a,b,**kwargs)))
    result={}
    for name,rows in [('training_budget_comparisons',budget),('industrial_mode_comparisons',deployment)]:
        frame=pd.DataFrame(rows)
        if not frame.empty and 'p_value' in frame:frame['p_holm_followup_family']=holm(frame.p_value)
        result[name]=frame
    return result
