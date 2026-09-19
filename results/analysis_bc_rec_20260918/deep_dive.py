import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
s=json.loads((ROOT/'summary.json').read_text())
out={'stages':{},'audit_risk':{}}
stages={'31_60':(31,60),'61_90':(61,90),'91_150':(91,150),'151_200':(151,200),'201_270':(201,270),'271_300':(271,300)}
for mode,info in s['runs'].items():
    path=REPO/info['path']
    m=pd.read_csv(path/'metrics.csv');d=pd.read_csv(path/'dynamics.csv')
    u=pd.read_csv(path/'updates.csv');c=pd.read_csv(path/'client_updates.csv')
    out['stages'][mode]={}
    for name,(lo,hi) in stages.items():
        a=m[m['round'].between(lo,hi)];b=d[d['round'].between(lo,hi)];v=u[u['round'].between(lo,hi)];cl=c[c['round'].between(lo,hi)]
        out['stages'][mode][name]=dict(acc=float(100*a.acc.mean()),
            sup=float(b.L_sup_effective.mean()),a=float(b.L_A_effective.mean()),b=float(b.L_B_effective.mean()),
            proto=float(b.L_proto_effective.mean()),feature=float(b.L_feature_effective.mean()),
            feature_fraction=float((b.L_feature_effective/b.loss_observed).mean()),
            local_update=float(v.params_local_mean_norm.mean()),fused_update=float(v.params_global_norm.mean()),
            relative_disagreement=float(v.params_disagreement_relative.mean()),retained=float(v.params_retained_ratio.mean()),
            negative_client_fraction=float((cl.params_cos_to_fused<0).mean()),
            min_acc=float(100*a.acc.min()),max_acc=float(100*a.acc.max()))
    print(mode,'stages',out['stages'][mode],flush=True)

path=REPO/s['runs']['audit']['path']
a=pd.read_csv(path/'reconstruction_audit.csv')
a=a[a['round'].between(271,300)]
out['audit_risk']['within_client_class_confidence']={}
for score in ('cross_view','teacher_self'):
    for bucket in ('A','B','C'):
        q=a[(a.score==score)&(a.bucket==bucket)]
        favorable=denominator=total_n=0.
        for _, group in q.groupby(['client','predicted_class','confidence_band']):
            hist=group.groupby('error_bin')[['correct','wrong']].sum().sort_index()
            good=hist.correct.to_numpy();bad=hist.wrong.to_numpy()
            favorable+=float(np.dot(bad,np.cumsum(good)-good/2))
            denominator+=float(good.sum()*bad.sum())
            total_n+=float(good.sum()+bad.sum())
        out['audit_risk']['within_client_class_confidence'][score+'_'+bucket]=dict(
            pair_weighted_auroc=favorable/denominator if denominator else None,visits=total_n)

out['audit_risk']['thresholds']=[]
for score in ('cross_view','teacher_self'):
    for cls in ('all',3,5,6,8):
        q=a[(a.score==score)&(a.bucket=='A')]
        if cls!='all':q=q[(q.predicted_class==cls)&(q.confidence_band==3)]
        g=float(q.correct.sum());b=float(q.wrong.sum())
        for threshold in (.2,.3,.4,.5):
            selected=q[q.error_bin>=round(10*threshold)]
            rg=float(selected.correct.sum());rb=float(selected.wrong.sum())
            out['audit_risk']['thresholds'].append(dict(score=score,predicted_class=cls,
                confidence='>=0.95' if cls=='all' else '>=0.99',threshold=threshold,
                correct=g,wrong=b,removed_correct=rg,removed_wrong=rb,
                removed_wrong_precision=rb/(rg+rb) if rg+rb else None,
                wrong_recall=rb/b if b else None,correct_loss_rate=rg/g if g else None,
                correct_per_wrong=rg/rb if rb else None))
print('within client AUROC',out['audit_risk']['within_client_class_confidence'],flush=True)
print('threshold .4',[x for x in out['audit_risk']['thresholds'] if x['score']=='cross_view' and x['threshold']==.4],flush=True)
(ROOT/'deep_dive.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
