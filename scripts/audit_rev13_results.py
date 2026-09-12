import csv,json,collections,statistics as st
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/analysis_rev13'
summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8')); result={}
for name in ['reference','high_lr','trusted']:
    folder=Path(summary[name]['folder']); windows={}
    for lo,hi in [(136,200),(201,270),(271,300)]:
        groups=collections.defaultdict(lambda:collections.defaultdict(float)); strata=collections.defaultdict(lambda:collections.defaultdict(float))
        with (folder/'geometry_audit.csv').open() as f:
            for r in csv.DictReader(f):
                if not lo<=int(r['round'])<=hi: continue
                key=(int(r['round']),int(r['client']),int(r['pred_class']),r['s_lo'],r['s_hi'])
                for target in [groups[r['geometry_group']],strata[key]]:
                    for k in ['visits','correct','a_visits','a_correct','a_weight','a_wrong_weight']: target[k]+=float(r[k])
        for g in groups.values():
            g['precision']=g['correct']/g['visits'] if g['visits'] else None
            g['a_precision']=g['a_correct']/g['a_visits'] if g['a_visits'] else None
        diffs=[]; weights=[]
        for s in strata.values():
            c=s['a_correct']; w=s['a_visits']-c
            if min(c,w)<5: continue
            diffs.append((s['a_weight']-s['a_wrong_weight'])/c-s['a_wrong_weight']/w); weights.append(min(c,w))
        windows[f'{lo}-{hi}']={'geometry':dict(groups),'matched_A_weight_gap':sum(d*w for d,w in zip(diffs,weights))/sum(weights) if weights else None,'matched_strata':len(diffs),'matched_positive_fraction':sum(w for d,w in zip(diffs,weights) if d>0)/sum(weights) if weights else None}
    result[name]=windows
historical=[]
for p in (ROOT/'results/CIFAR10/runs').glob('*/acc.csv'):
    with p.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    try: vals=[float(r['acc']) for r in rows]
    except (KeyError,ValueError): continue
    historical.append({'run':p.parent.name,'rounds':len(vals),'best':max(vals),'last30':st.mean(vals[-30:])})
result['historical']=sorted(historical,key=lambda x:x['best'],reverse=True)
(OUT/'audit_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
