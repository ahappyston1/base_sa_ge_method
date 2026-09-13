"""Audit completed follow-ups and the available partial 500-round snapshot."""
import csv, json, math, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis_rev13/followup'
NAMES = ['reference', 'high_lr', 'trusted', 'trusted_lr015', 'legacy_lr020', 'legacy_lr015_500']
OUT.mkdir(exist_ok=True)

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try: r[k] = float(v)
            except (ValueError, TypeError): pass
    return rows

def mean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return st.mean(xs) if xs else None

def ratio(a, b): return a / b if b else None
def total(rows, key): return sum(r[key] for r in rows)

data, summary, schedules = {}, {}, {}
for name in NAMES:
    folder = next((ROOT / 'results/CIFAR10/runs').glob('rev13_' + name + '_s7_*'))
    tables = {k: read(folder / (k + '.csv')) for k in ['acc', 'metrics', 'dynamics', 'updates', 'client_updates']}
    cfg = json.loads((folder / 'config.json').read_text())
    data[name] = {k: v for k, v in tables.items() if k != 'client_updates'}
    vals = [r['acc'] for r in tables['acc']]
    count = len(vals)
    expected = list(range(1, count + 1))
    for k in ['acc', 'metrics', 'dynamics', 'updates']:
        assert [int(r['round']) for r in tables[k]] == expected, (name, k)
    assert max(abs(a['acc']-b['acc']) for a,b in zip(tables['acc'], tables['metrics'])) < 1e-8
    schedules[name] = [(r['round'], r['client'], r['local_steps'], r['fedavg_weight']) for r in tables['client_updates']]
    s = dict(folder=str(folder), config=cfg, rounds=count, planned=cfg['num_rounds'], complete=count==cfg['num_rounds'],
             best=max(vals), best_round=vals.index(max(vals))+1, final=vals[-1], last10=mean(vals[-10:]),
             last30=mean(vals[-30:]), last30_sd=st.pstdev(vals[-30:]), windows={}, geometry={},
             local_steps=sorted(set(r['local_steps'] for r in tables['client_updates'])),
             loss_reconstruction_max_error=max(abs(r['loss_reconstructed']-r['loss_observed']) for r in tables['dynamics']),
             phases={str(p):[min(r['round'] for r in tables['metrics'] if r['phase']==p),max(r['round'] for r in tables['metrics'] if r['phase']==p)] for p in [1,2,3]})
    windows = [(61,90),(91,135),(136,200),(201,270),(271,300),(301,350),(351,382),(383,412)]
    for lo, hi in windows:
        if hi > count: continue
        subsets = {k:[r for r in tables[k] if lo<=r['round']<=hi] for k in ['metrics','dynamics','updates']}
        m,d,u = (subsets[k] for k in ['metrics','dynamics','updates'])
        a = [r['acc'] for r in m]; delta = [y-x for x,y in zip(a,a[1:])]
        w = dict(acc=mean(a), sd=st.pstdev(a), step_abs_pp=100*mean([abs(x) for x in delta]),
                 worst_step_pp=100*min(delta), drops_gt2pp=sum(x < -.02 for x in delta))
        for b in ['a','b']:
            mass = sum(r[b+'_effective_mass_per_u']*r['u_visits'] for r in d)
            wrong = sum(r[b+'_wrong_mass_per_u']*r['u_visits'] for r in d)
            w.update({b+'_ratio':total(m,'cnt_'+b)/total(m,'cnt_u'),
                      b+'_precision':ratio(total(m,b+'_correct'),total(m,b+'_total')),
                      b+'_mass_per_u':mass/total(d,'u_visits'), b+'_wrong_per_u':wrong/total(d,'u_visits'),
                      b+'_weight_correct':ratio(mass-wrong,total(m,b+'_correct')),
                      b+'_weight_wrong':ratio(wrong,total(m,b+'_total')-total(m,b+'_correct'))})
        w['c_ratio'] = total(m,'cnt_c')/total(m,'cnt_u')
        w['proto_a_dirty'] = ratio(sum(r['proto_a_dirty']*r['proto_a_mass'] for r in m),total(m,'proto_a_mass'))
        w['proto_a_share'] = ratio(total(m,'proto_a_mass'),total(m,'proto_a_mass')+total(m,'proto_lab_mass'))
        for k in ['trust_authority_mean','trust_valid_fraction','L_sup_effective','L_A_effective','L_B_effective','L_proto_effective']:
            w[k] = mean([r[k] for r in d])
        for k in ['params_global_norm','params_local_mean_norm','params_disagreement_relative','params_retained_ratio','bn_mean_global_norm','bn_var_global_norm']:
            w[k] = mean([r[k] for r in u])
        s['windows'][f'{lo}-{hi}'] = w
    groups, strata = defaultdict(lambda:defaultdict(float)), defaultdict(lambda:defaultdict(float))
    with (folder/'geometry_audit.csv').open() as f:
        for r in csv.DictReader(f):
            rd = int(r['round'])
            win = next((f'{lo}-{hi}' for lo,hi in windows if lo<=rd<=hi and lo>=136 and hi<=count),None)
            if win is None: continue
            key = (win, rd, r['client'],r['pred_class'],r['s_lo'],r['s_hi'])
            for target in [groups[(win,r['geometry_group'])],strata[key]]:
                for k in ['visits','correct','a_visits','a_correct','a_weight','a_wrong_weight']: target[k]+=float(r[k])
    for win in s['windows']:
        gg = {g:dict(v) for (w,g),v in groups.items() if w==win}
        for g in gg.values():
            g['precision'] = ratio(g['correct'],g['visits'])
            g['a_precision'] = ratio(g['a_correct'],g['a_visits'])
        num=den=0
        for key,v in strata.items():
            if key[0] != win: continue
            c=v['a_correct']; e=v['a_visits']-c
            if min(c,e)<5: continue
            wt=min(c,e); num+=wt*((v['a_weight']-v['a_wrong_weight'])/c-v['a_wrong_weight']/e);den+=wt
        if gg: s['geometry'][win] = dict(groups=gg,matched_A_weight_gap=ratio(num,den))
    summary[name] = s

summary['checks'] = {
    'config_diff_vs_high_lr': {n:{k:[summary['high_lr']['config'].get(k),v] for k,v in summary[n]['config'].items() if v!=summary['high_lr']['config'].get(k)} for n in NAMES if n!='high_lr'},
    'same_client_schedule_first300':{n:schedules[n][:2400]==schedules['high_lr'] for n in NAMES},
    'trusted015_legacy015_phase1_max_acc_diff':max(abs(a['acc']-b['acc']) for a,b in zip(data['trusted_lr015']['acc'][:90],data['high_lr']['acc'][:90]))}
summary['interaction_pp'] = {k:100*((summary['trusted_lr015'][k]-summary['high_lr'][k])-(summary['trusted'][k]-summary['reference'][k])) for k in ['best','final','last30']}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'series.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
for n in NAMES:
    print(n, json.dumps({k:v for k,v in summary[n].items() if k not in ['config','folder','geometry','windows']}))
    for w in ['136-200','201-270','271-300','383-412']:
        if w in summary[n]['windows']: print(w,json.dumps(summary[n]['windows'][w]))
print('CHECKS',json.dumps(summary['checks']))
print('INTERACTION',summary['interaction_pp'])
