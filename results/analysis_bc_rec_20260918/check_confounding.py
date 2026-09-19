import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
s=json.loads((ROOT/'summary.json').read_text())
out={}

def conditional(q, groups):
    # Grouped sort/bin cumulative counts implements exact tied-score Mann-Whitney
    # on quantized scores. Comparisons only use correct/wrong from the same group.
    v=q.groupby(groups+['error_bin'],as_index=False)[['correct','wrong']].sum().sort_values(groups+['error_bin'])
    v['lower_good']=v.groupby(groups).correct.cumsum()-v.correct
    v['favorable']=v.wrong*(v.lower_good+.5*v.correct)
    z=v.groupby(groups)[['correct','wrong','favorable']].sum()
    z=z[(z.correct>0)&(z.wrong>0)]
    pairs=z.correct*z.wrong
    auc=z.favorable/pairs
    return dict(pair_weighted_auc=float(z.favorable.sum()/pairs.sum()),
                macro_auc=float(auc.mean()),visit_weighted_auc=float(np.average(auc,weights=z.correct+z.wrong)),
                paired_groups=len(z),groups_below_half=int((auc<.5).sum()),
                eligible_visits=float((z.correct+z.wrong).sum()))

for mode in ('audit','unmasked','masked'):
    a=pd.read_csv(REPO/s['runs'][mode]['path']/'reconstruction_audit.csv')
    out[mode]={}
    for name,(lo,hi) in {'middle':(91,200),'late':(201,300),'tail30':(271,300)}.items():
        w=a[a['round'].between(lo,hi)]
        out[mode][name]={}
        for score in ('cross_view','teacher_self'):
            q=w[(w.score==score)&(w.bucket=='A')]
            scores={}
            for label,groups in [('class_conf',['predicted_class','confidence_band']),
                                 ('client_class_conf',['client','predicted_class','confidence_band']),
                                 ('round_client_class_conf',['round','client','predicted_class','confidence_band'])]:
                scores[label]=conditional(q,groups)
            if name=='tail30':
                scores['by_class']={str(cls):conditional(q[q.predicted_class==cls],['client','confidence_band']) for cls in range(10)}
            out[mode][name][score]=scores
        print(mode,name,{k:{g:round(v['pair_weighted_auc'],5) for g,v in data.items() if g!='by_class'} for k,data in out[mode][name].items()},flush=True)
    print(mode,'A class-controlled cross_view', {k:round(v['pair_weighted_auc'],3) for k,v in out[mode]['tail30']['cross_view']['by_class'].items()},flush=True)
(ROOT/'confounding.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
