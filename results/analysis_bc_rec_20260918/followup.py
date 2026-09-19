import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
s=json.loads((ROOT/'summary.json').read_text())
configs={}
for mode,info in s['runs'].items():
    cfg=json.loads(json.dumps(info['config']));cfg.pop('run_id',None)
    cfg.get('geometry_controls',{}).pop('bc_reconstruction',None)
    configs[mode]=cfg
s['config_differences_against_audit']={m:{k:[configs['audit'].get(k),c.get(k)] for k in configs['audit'].keys()|c.keys()
                                       if configs['audit'].get(k)!=c.get(k)} for m,c in configs.items()}
for mode,info in s['runs'].items():
    path=REPO/info['path']
    lines=(path/'train.log').read_text(encoding='utf-8').splitlines()
    stamps=[datetime.strptime(l[:23],'%Y-%m-%d %H:%M:%S,%f') for l in lines if ' - INFO - Round ' in l]
    first=datetime.strptime(lines[0][:23],'%Y-%m-%d %H:%M:%S,%f')
    info['elapsed_log_hours']=(stamps[-1]-first).total_seconds()/3600
    geom=pd.read_csv(path/'geometry_audit.csv')
    geom=geom[geom['round'].between(271,300)]
    groups=geom.groupby('geometry_group')[['a_visits','a_correct','a_weight','a_wrong_weight']].sum()
    info['tail_geometry_groups']=groups.to_dict('index')
    classes=geom.groupby('pred_class')[['a_visits','a_correct']].sum()
    info['tail_a_by_class']={str(c):100*float(v.a_correct/v.a_visits) for c,v in classes.iterrows() if v.a_visits}
    print(mode,'hours',info['elapsed_log_hours'],'geometry',info['tail_geometry_groups'],flush=True)
    print(mode,'Aclasses',info['tail_a_by_class'],flush=True)
    if mode=='old_bc': continue
    for win,w in info['audit'].items():
        w['within_stratum_pair_auroc']={}
        for score in ('cross_view','teacher_self'):
            strata=[v for v in w['strata'] if v['score']==score and v['auroc'] is not None]
            weights=[v['correct']*v['wrong'] for v in strata]
            w['within_stratum_pair_auroc'][score]=float(np.average([v['auroc'] for v in strata],weights=weights))
    print(mode,'tail conditional AUROC',info['audit']['tail30']['within_stratum_pair_auroc'],flush=True)

# Common fields in dynamics are higher precision than rounded metrics.
old=pd.read_csv(REPO/s['runs']['old_bc']['path']/'dynamics.csv')
audit=pd.read_csv(REPO/s['runs']['audit']['path']/'dynamics.csv')
common=old.columns.intersection(audit.columns)
s['audit_old_dynamics_exact']=bool(old[common].equals(audit[common]))
print('configdiff',s['config_differences_against_audit'])
print('audit baseline dynamics exact',s['audit_old_dynamics_exact'])
(ROOT/'summary.json').write_text(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
