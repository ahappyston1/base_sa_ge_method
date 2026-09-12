"""Reproduce REV13 comparison using only the Python standard library."""
import csv, json, math, statistics as st, re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'analysis_rev13'
OUT.mkdir(exist_ok=True)
def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k, v in row.items():
            try: row[k] = float(v)
            except (ValueError, TypeError): pass
    return rows
def avg(xs):
    xs=[x for x in xs if math.isfinite(x)]
    return st.mean(xs) if xs else None
def div(a,b): return a/b if b else None
def window(rows,lo,hi): return [r for r in rows if lo<=r['round']<=hi]
def sums(rows,k): return sum(r[k] for r in rows)
summary={}; data={}
for name in ['reference','high_lr','trusted']:
    folder=next((ROOT/'results/CIFAR10/runs').glob('rev13_'+name+'_s7_*'))
    tables={f:read(folder/(f+'.csv')) for f in ['acc','metrics','dynamics','updates','client_updates']}
    data[name]=tables
    cfg=json.loads((folder/'config.json').read_text())
    acc=tables['acc']; vals=[r['acc'] for r in acc]
    assert [int(r['round']) for r in acc]==list(range(1,301))
    for f in ['metrics','dynamics','updates']:
        assert [r['round'] for r in tables[f]]==[r['round'] for r in acc]
    assert max(abs(a['acc']-m['acc']) for a,m in zip(acc,tables['metrics']))<1e-8
    log=(folder/'train.log').read_text(encoding='utf-8')
    times=[datetime.strptime(line[:23],'%Y-%m-%d %H:%M:%S,%f') for line in log.splitlines() if re.search(r' - INFO - Round \d+/300',line)]
    result={'folder':str(folder),'config':cfg,'best':max(vals),'best_round':vals.index(max(vals))+1,'final':vals[-1], 'last10':avg(vals[-10:]),'last30':avg(vals[-30:]),'last30_sd':st.pstdev(vals[-30:]),'last30_range':[min(vals[-30:]),max(vals[-30:])], 'hours_round1_to300':(times[-1]-times[0]).total_seconds()/3600,'first_reach':{str(t):next((int(r['round']) for r in acc if r['acc']>=t),None) for t in [.7,.75,.8,.82,.84,.85]},'phase_ranges':{},'windows':{},'steps':sorted(set(r['local_steps'] for r in tables['client_updates']))}
    for ph in [1,2,3]:
        rs=[r['round'] for r in tables['metrics'] if r['phase']==ph]
        result['phase_ranges'][ph]=[min(rs),max(rs)]
    for lo,hi in [(1,60),(61,90),(91,135),(136,200),(201,270),(271,300),(91,300)]:
        m=window(tables['metrics'],lo,hi); d=window(tables['dynamics'],lo,hi); u=window(tables['updates'],lo,hi)
        aa=[r['acc'] for r in m]; deltas=[y-x for x,y in zip(aa,aa[1:])]
        w={'acc':avg(aa),'acc_sd':st.pstdev(aa),'step_abs_pp':100*avg([abs(x) for x in deltas]),'worst_step_pp':100*min(deltas),'drops_gt2pp':sum(x<-.02 for x in deltas)}
        for bucket in ['a','b']:
            w[bucket+'_ratio']=sums(m,'cnt_'+bucket)/sums(m,'cnt_u')
            w[bucket+'_precision']=div(sums(m,bucket+'_correct'),sums(m,bucket+'_total'))
            mass=sum(r[bucket+'_effective_mass_per_u']*r['u_visits'] for r in d)
            wrong=sum(r[bucket+'_wrong_mass_per_u']*r['u_visits'] for r in d)
            w[bucket+'_mass_per_u']=mass/sums(d,'u_visits')
            w[bucket+'_wrong_per_u']=wrong/sums(d,'u_visits')
            w[bucket+'_wrong_share']=div(wrong,mass)
            w[bucket+'_weight_correct']=div(mass-wrong,sums(m,bucket+'_correct'))
            w[bucket+'_weight_wrong']=div(wrong,sums(m,bucket+'_total')-sums(m,bucket+'_correct'))
        w['c_ratio']=sums(m,'cnt_c')/sums(m,'cnt_u')
        w['hce']=sums(m,'hce_hc')/sums(m,'hce_den_hc')
        w['geom_drop']=div(sums(m,'geom_drop'),sums(m,'pass_s'))
        for key in ['trust_authority_mean','trust_valid_fraction','L_sup_effective','L_A_effective','L_B_effective','L_proto_effective']:
            w[key]=avg([r[key] for r in d])
        for key in ['params_global_norm','params_local_mean_norm','params_disagreement_relative','params_retained_ratio','bn_mean_global_norm','bn_var_global_norm']:
            w[key]=avg([r[key] for r in u])
        result['windows'][f'{lo}-{hi}']=w
    if name=='trusted':
        tr=read(folder/'trust_reference.csv')
        result['trust']={'rows':len(tr),'positive_fraction':avg([float(r['trust']>0) for r in tr]),'mean_all':avg([r['trust'] for r in tr]),'mean_positive':avg([r['trust'] for r in tr if r['trust']>0]),'query_median':st.median([r['query_n'] for r in tr]),'per_class':{str(c):{'trust':avg([r['trust'] for r in tr if r['cls']==c]),'positive':avg([float(r['trust']>0) for r in tr if r['cls']==c])} for c in range(10)}}
    summary[name]=result
base=summary['reference']['config']
summary['config_differences']={name:{k:[base.get(k),v] for k,v in summary[name]['config'].items() if v!=base.get(k)} for name in ['high_lr','trusted']}
summary['checks']={'trusted_reference_phase1_max_acc_diff':max(abs(a['acc']-b['acc']) for a,b in zip(data['trusted']['acc'][:90],data['reference']['acc'][:90])), 'client_schedule_equal':all([(r['round'],r['client'],r['local_steps'],r['fedavg_weight']) for r in data[n]['client_updates']]==[(r['round'],r['client'],r['local_steps'],r['fedavg_weight']) for r in data['reference']['client_updates']] for n in ['high_lr','trusted'])}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'series.json').write_text(json.dumps({n:{k:v for k,v in t.items() if k!='client_updates'} for n,t in data.items()},ensure_ascii=False),encoding='utf-8')
for name in ['reference','high_lr','trusted']:
    x=summary[name]
    print(name,json.dumps({k:v for k,v in x.items() if k not in ['windows','config','trust','folder']},ensure_ascii=False))
    for win in ['61-90','91-135','136-200','201-270','271-300']:
        print(win,json.dumps(x['windows'][win]))
print('CONFIG',json.dumps(summary['config_differences']))
print('CHECKS',summary['checks'])
if 'trust' in summary['trusted']: print('TRUST',json.dumps(summary['trusted']['trust']))
