import json
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SOURCE = ROOT/'source/results/CIFAR10/runs'
RUNS = {m: next(SOURCE.glob('rev13_bc_rec_'+m+'_*')) for m in ('audit','unmasked','masked')}
RUNS['old_bc'] = next((REPO/'results/CIFAR10/runs').glob('rev13_trusted_lr015_bc_s7_*'))
WINDOWS = {'early': (31,90), 'middle': (91,200), 'late': (201,300), 'tail30': (271,300)}

def window(df, bounds):
    return df[df['round'].between(*bounds)]

def histogram_stats(counts):
    a = np.zeros((40,2), dtype=float)
    for idx, row in counts.iterrows():
        a[int(idx)] = row[['correct','wrong']].to_numpy()
    good,bad = a.sum(0)
    numerator = (a[:,1]*(np.cumsum(a[:,0])-a[:,0]/2)).sum()
    return dict(correct=float(good), wrong=float(bad),
                auroc=float(numerator/(good*bad)) if good*bad else None,
                good_error=float(np.dot(a[:,0], (np.arange(40)+.5)/10)/good) if good else None,
                bad_error=float(np.dot(a[:,1], (np.arange(40)+.5)/10)/bad) if bad else None,
                bin0_fraction=float(a[0].sum()/max(good+bad,1)))

def grouped_histogram(frame, group):
    result=[]
    for key, part in frame.groupby(group, observed=True):
        if not isinstance(key, tuple): key=(key,)
        hist=part.groupby('error_bin')[['correct','wrong']].sum()
        result.append(dict(zip(group,key), **histogram_stats(hist)))
    return result

summary={'commit':subprocess.check_output(['git','rev-parse','origin/codex/bc-reconstruction-prototypes'],cwd=REPO,text=True).strip(),
         'runs':{},'comparisons':{}}
series={}
metrics={}; dynamics={}
for mode,path in RUNS.items():
    m=pd.read_csv(path/'metrics.csv'); d=pd.read_csv(path/'dynamics.csv')
    metrics[mode]=m; dynamics[mode]=d
    a=pd.read_csv(path/'acc.csv')
    cfg=json.loads((path/'config.json').read_text())
    assert list(m['round'])==list(range(1,301)), (mode,'metrics incomplete')
    assert list(d['round'])==list(range(1,301)), (mode,'dynamics incomplete')
    assert np.allclose(a.acc,m.acc,atol=1e-9), (mode,'acc disagreement')
    info=dict(path=str(path.relative_to(REPO)),config=cfg,rounds=len(m),best=100*m.acc.max(),
              best_round=int(m.loc[m.acc.idxmax(),'round']),final=100*m.acc.iloc[-1],
              tail30=100*m.acc.tail(30).mean(),tail10=100*m.acc.tail(10).mean(),
              accounting_error=float((d.loss_reconstructed-d.loss_observed).abs().max()), windows={})
    for name,bounds in WINDOWS.items():
        q=window(m,bounds); v=window(d,bounds)
        info['windows'][name]=dict(acc=100*q.acc.mean(),
            a_precision=100*q.a_correct.sum()/q.a_total.sum(), b_precision=100*q.b_correct.sum()/max(q.b_total.sum(),1),
            a_fraction=q.cnt_a.sum()/q.cnt_u.sum(),
            feature=v.L_feature_effective.mean(), feature_std=v.bc_feature_std.mean(),
            authority=v.trust_authority_mean.mean(),
            teacher_b_precision=100*np.average(v.teacher_b_precision,weights=v.teacher_b_visits) if v.teacher_b_visits.sum() else None,
            rec_objective=v.rec_objective.mean() if 'rec_objective' in v else None)
    summary['runs'][mode]=info
    series[mode]=dict(round=m['round'].tolist(),acc=(100*m.acc).round(6).tolist(),
                      feature=d.L_feature_effective.tolist(),std=d.bc_feature_std.tolist())
    print(mode, {k:info[k] for k in ('rounds','best','best_round','final','tail30','tail10','accounting_error')}, flush=True)

for mode in ('unmasked','masked','old_bc'):
    fields=list(set(metrics[mode].columns)&set(metrics['audit'].columns))
    delta=(metrics[mode][fields]-metrics['audit'][fields]).abs()
    summary['comparisons'][mode]=dict(prefix30_exact=bool((delta.iloc[:30]==0).all().all()),
        whole_metrics_exact=bool((delta==0).all().all()),
        first_difference=int(np.flatnonzero((delta>0).any(axis=1))[0])+1 if (delta>0).any().any() else None,
        tail30_delta_pp=summary['runs'][mode]['tail30']-summary['runs']['audit']['tail30'])
print('comparisons',summary['comparisons'],flush=True)

for mode,path in RUNS.items():
    if mode=='old_bc':continue
    info=summary['runs'][mode]
    ref=pd.read_csv(path/'reconstruction_reference.csv')
    assert not ref.duplicated(['round','client','epoch','cls']).any(), 'duplicate reference rows'
    info['reference']={}
    for name,bounds in WINDOWS.items():
        q=window(ref,bounds)
        grouped=q.groupby('cls')[['query_count','query_correct','own_distance_sum','margin_sum']].sum()
        def geo(v):
            n=float(v.query_count)
            return dict(n=n,acc=float(100*v.query_correct/n),distance=float(v.own_distance_sum/n),margin=float(v.margin_sum/n))
        info['reference'][name]=dict(overall=geo(grouped.sum()),classes={str(k):geo(v) for k,v in grouped.iterrows()})
    perround=ref.groupby('round')[['query_count','query_correct','own_distance_sum','margin_sum']].sum()
    series[mode]['reference']={'round':perround.index.tolist(),
        'acc':(100*perround.query_correct/perround.query_count).tolist(),
        'distance':(perround.own_distance_sum/perround.query_count).tolist(),
        'margin':(perround.margin_sum/perround.query_count).tolist()}
    print(mode,'geometry',info['reference']['tail30']['overall'],flush=True)
    audit=pd.read_csv(path/'reconstruction_audit.csv')
    key=['round','client','score','predicted_class','bucket','confidence_band','error_bin']
    assert not audit.duplicated(key).any(), 'duplicate histogram rows'
    visits=audit.assign(visits=audit.correct+audit.wrong).groupby(['round','score']).visits.sum().unstack()
    expected=dynamics[mode].set_index('round').loc[visits.index,'u_visits']
    assert (visits.eq(expected,axis=0)).all().all(), 'audit count mismatch'
    info['histogram_rows']=len(audit)
    info['audit']={}
    for name,bounds in WINDOWS.items():
        q=window(audit,bounds)
        overall=grouped_histogram(q,['score'])
        buckets=grouped_histogram(q,['score','bucket'])
        strata=grouped_histogram(q,['score','predicted_class','bucket','confidence_band'])
        info['audit'][name]=dict(overall=overall,buckets=buckets,strata=strata)
    print(mode,'AUROC',info['audit']['tail30']['overall'],flush=True)
    print(mode,'bucketAUROC',info['audit']['tail30']['buckets'],flush=True)
    del audit
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

(ROOT/'series.json').write_text(json.dumps(series,ensure_ascii=False,allow_nan=False),encoding='utf-8')
print('Saved',ROOT/'summary.json',flush=True)
