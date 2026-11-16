"Paired instance inference with global model-seed and instance resampling."
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata,wilcoxon


def holm(pvalues):
    values=np.asarray(pvalues,dtype=float);out=np.full(len(values),np.nan)
    finite=np.flatnonzero(np.isfinite(values));order=finite[np.argsort(values[finite])]
    maximum=0.
    for position,index in enumerate(order):
        maximum=max(maximum,(len(order)-position)*values[index]);out[index]=min(1.,maximum)
    return out


def signed_rank(difference,tolerance=1e-9):
    d=np.asarray(difference,dtype=float).copy();d[np.abs(d)<=tolerance]=0.
    nonzero=d[d!=0]
    if not len(nonzero):return 1.,0.
    ranks=rankdata(np.abs(nonzero),method='average')
    effect=float((ranks[nonzero>0].sum()-ranks[nonzero<0].sum())/ranks.sum())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        p=float(wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto').pvalue)
    return p,effect


def matrix(frame,metric='total_tardiness'):
    # Repeated scenes first collapse within the same seed/order instance.
    return frame.pivot_table(index='seed',columns='instance',values=metric,aggfunc='mean',dropna=False)


def instance_weights(instances):
    strata=np.array([str(i).split('::')[0] for i in instances]);unique=np.unique(strata)
    return np.array([1./(len(unique)*np.sum(strata==s)) for s in strata]),strata


def bootstrap_pair(a,b,instances,*,draws=2000,seed=47291):
    weights,strata=instance_weights(instances);rng=np.random.default_rng(seed)
    paired=list(a.index)==list(b.index) and len(a.index)>1
    av=a.to_numpy(float);bv=b.to_numpy(float);results=[]
    for start in range(0,draws,128):
        count=min(128,draws-start)
        indices=np.zeros((count,len(instances)),dtype=int)
        for s in np.unique(strata):
            members=np.flatnonzero(strata==s)
            indices[:,members]=rng.choice(members,size=(count,len(members)),replace=True)
        ai=rng.integers(len(av),size=(count,len(av)))
        bi=ai if paired else rng.integers(len(bv),size=(count,len(bv)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            am=np.nanmean(av[ai[:,:,None],indices[:,None,:]],axis=1)
            bm=np.nanmean(bv[bi[:,:,None],indices[:,None,:]],axis=1)
        # Keep a draw unavailable if selected model seeds provide no outcome
        # for a paired instance; do not fill missing outcomes with zeros.
        ma=(am*weights).sum(axis=1);mb=(bm*weights).sum(axis=1)
        relative=np.divide(100*(mb-ma),mb,out=np.full(count,np.nan),where=mb>0)
        results.extend(zip(ma,mb,mb-ma,relative))
    array=np.asarray(results,float);output={}
    for i,name in enumerate(('reference_mean','comparator_mean','absolute_improvement','relative_improvement_percent')):
        valid=array[:,i][np.isfinite(array[:,i])]
        output[name+'_ci_low']=float(np.quantile(valid,.025)) if len(valid) else None
        output[name+'_ci_high']=float(np.quantile(valid,.975)) if len(valid) else None
    output['bootstrap_valid_draws']=int(np.isfinite(array[:,2]).sum())
    output['model_seed_resampling']='paired_global' if paired else 'independent_global_or_deterministic'
    return output


def paired_comparison(reference,comparator,*,draws=2000,seed=47291,tolerance=1e-9,required_scenes=None):
    source_a=matrix(reference);source_b=matrix(comparator)
    cells=reference.loc[reference.total_tardiness.notna(),['scene','instance']].drop_duplicates().merge(
        comparator.loc[comparator.total_tardiness.notna(),['scene','instance']].drop_duplicates())
    if required_scenes:
        complete=cells.groupby('instance').scene.nunique()
        cells=cells[cells.instance.isin(complete[complete==len(required_scenes)].index)]
    reference=reference.merge(cells);comparator=comparator.merge(cells)
    a=matrix(reference);b=matrix(comparator)
    common=sorted(set(a.columns)&set(b.columns))
    common=[i for i in common if a[i].notna().any() and b[i].notna().any()]
    result=dict(reference_instances=len(source_a.columns),comparator_instances=len(source_b.columns),paired_instances=len(common),
        reference_model_seeds=len(source_a.index),comparator_model_seeds=len(source_b.index),paired_instance_scene_cells=len(cells))
    if not common:return result
    a=a[common];b=b[common];weights,_=instance_weights(common)
    av=a.mean(axis=0).to_numpy(float);bv=b.mean(axis=0).to_numpy(float);d=bv-av
    am=float(av@weights);bm=float(bv@weights)
    p,effect=signed_rank(d,tolerance)
    positive=(av>0)&(bv>0)
    result.update(reference_mean=am,comparator_mean=bm,absolute_improvement=bm-am,
        relative_improvement_percent=100*(bm-am)/bm if bm>0 else None,
        p_value=p,signed_rank_biserial=effect,instance_wins=int((d>tolerance).sum()),
        instance_ties=int((np.abs(d)<=tolerance).sum()),instance_losses=int((d< -tolerance).sum()),
        zero_zero=int(((av==0)&(bv==0)).sum()),reference_zero_comparator_positive=int(((av==0)&(bv>0)).sum()),
        comparator_zero_reference_positive=int(((bv==0)&(av>0)).sum()),positive_pairs=int(positive.sum()),
        geometric_ratio_coverage=float(positive.mean()),
        geometric_mean_reference_over_comparator=float(np.exp(np.mean(np.log(av[positive]/bv[positive])))) if positive.any() else None)
    result.update(bootstrap_pair(a,b,common,draws=draws,seed=seed))
    conditions={str(i).split('::')[0] for i in common}
    wins=ties=losses=0
    for condition in conditions:
        value=float(np.mean([delta for i,delta in zip(common,d) if str(i).split('::')[0]==condition]))
        wins+=value>tolerance;losses+=value< -tolerance;ties+=abs(value)<=tolerance
    result.update(condition_wins=int(wins),condition_ties=int(ties),condition_losses=int(losses))
    return result


def method_summary(frame,metric):
    valid=frame.dropna(subset=[metric])
    if valid.empty:return dict(metric=metric,mean=None,model_seed_std=None,instance_std=None,model_seeds=0,instances=0)
    # Equal weight per condition, then per instance and model seed.
    cells=valid.groupby(['seed','condition','instance'],dropna=False)[metric].mean().reset_index()
    per_seed=cells.groupby(['seed','condition'])[metric].mean().groupby('seed').mean()
    per_instance=cells.groupby('instance')[metric].mean()
    per_condition=cells.groupby(['condition','instance'])[metric].mean().groupby('condition').mean()
    return dict(metric=metric,mean=float(per_condition.mean()),
        model_seed_std=float(per_seed.std(ddof=1)) if len(per_seed)>1 else None,
        instance_std=float(per_instance.std(ddof=1)) if len(per_instance)>1 else None,
        model_seeds=len(per_seed),instances=len(per_instance),conditions=len(per_condition),
        zero_instance_fraction=float((per_instance==0).mean()))


def mean_interval(frame,*,draws=2000,seed=47291):
    a=matrix(frame)
    a=a.loc[:,a.notna().any(axis=0)]
    if a.empty:return dict(mean_ci_low=None,mean_ci_high=None)
    values=bootstrap_pair(a,a,list(a.columns),draws=draws,seed=seed)
    return dict(mean_ci_low=values['reference_mean_ci_low'],mean_ci_high=values['reference_mean_ci_high'])
