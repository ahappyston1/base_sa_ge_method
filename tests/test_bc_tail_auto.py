import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tail_auto', ROOT/'scripts/run_bc_tail_auto.py')
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)


def test_exact_comparison_rejects_missing_nonfinite_and_small_differences(tmp_path):
    a={225:dict(acc='0.8528',phase='3',lr='0.0001')}
    b=copy.deepcopy(a)
    assert auto.compare(a,b,225,225,('acc','phase','lr'))['passed']
    b[225]['acc']='0.8528000001'
    assert not auto.compare(a,b,225,225,('acc',))['passed']
    b[225]['acc']='NaN'
    assert not auto.compare(a,b,225,225,('acc',))['passed']
    assert not auto.compare(a,{},225,225,('acc',))['passed']
    p=tmp_path/'a.csv';p.write_text('round,acc\n1,0.8\n1,0.8\n')
    with pytest.raises(ValueError):auto.rows(p)
    p.write_text('round,acc\n1,0.8\n3,0.8\n')
    with pytest.raises(ValueError):auto.rows(p)


@pytest.mark.parametrize('fail', [False,True])
def test_queue_limits_lanes_and_stops_pending_on_failure(tmp_path,fail):
    clock=[0]; calls=[]
    class Child:
        def __init__(self,index):self.index=index;self.pid=100+index
        def poll(self):
            if clock[0]>=2+self.index:
                return 1 if fail and self.index==0 else 0
            return None
    def launch(cmd,**kwargs):
        calls.append((clock[0],cmd));return Child(len(calls)-1)
    def sleep(_):clock[0]+=1
    jobs=[(n,lambda gpu,n=n:[n,str(gpu)]) for n in ['b','a','f']]
    if fail:
        with pytest.raises(RuntimeError):auto.run_jobs(jobs,[0,2],tmp_path,launch,sleep)
        assert len(calls)==2
    else:
        auto.run_jobs(jobs,[0,2],tmp_path,launch,sleep)
        assert calls==[(0,['b','0']),(0,['a','2']),(2,['f','0'])]


@pytest.mark.parametrize('failure', ['none','prefix','replay','checkpoint'])
def test_complete_workflow_gates_all_three_launches(tmp_path,monkeypatch,failure):
    from test_bc_tail import checkpoint
    monkeypatch.setattr(auto,'ROOT',tmp_path)
    (tmp_path/'configs').mkdir()
    shutil.copy(ROOT/'configs/experiment_trusted_lr015_bc.yaml',tmp_path/'configs')
    reference=tmp_path/'reference';reference.mkdir()
    cfg=json.loads((auto.DEFAULT_REFERENCE/'config.json').read_text())
    (reference/'config.json').write_text(json.dumps(cfg))
    def csvs(directory,start,end,corrupt=False):
        directory.mkdir(parents=True,exist_ok=True)
        for file in ['metrics','dynamics']:
            header='round,acc,phase,lr,gate,aux_scale,trusted_weight_gate\n'
            data=''.join(f'{r},{"0.81" if corrupt and r==start else "0.8"},3,0.001,1,1,1\n' for r in range(start,end+1))
            (directory/(file+'.csv')).write_text(header+data)
    csvs(reference,1,300)
    c,_=checkpoint()
    keys=('seed_model','seed_partition','seed_sample','dataset','alpha','num_clients',
          'num_online_clients','num_labeled_per_class','mu','local_epochs','batch_labeled','num_workers')
    c['run_identity']={k:cfg[k] for k in keys}
    c['client_states']={i:copy.deepcopy(c['client_states'][0]) for i in range(20)}
    if failure=='checkpoint':c['round']=300
    monkeypatch.setattr(torch,'load',lambda *a,**kw:c)
    calls=[]
    def run_jobs(jobs,gpus,folder):
        for name,build in jobs:
            calls.append(name);cmd=build(gpus[0]);rid=cmd[cmd.index('--run_id')+1]
            directory=tmp_path/'results/CIFAR10/runs'/(rid+'_a0.1')
            if name=='prefix':start,end=1,225
            elif name=='replay':start,end=226,230
            else:start,end=226,300
            csvs(directory,start,end,corrupt=name==failure)
    monkeypatch.setattr(auto,'run_jobs',run_jobs)
    monkeypatch.setattr(sys,'argv',['auto','--gpus','0','2','--reference-run',str(reference)])
    code=auto.main()
    assert (code==0)==(failure=='none')
    if failure in ['prefix','checkpoint']:assert calls==['prefix']
    elif failure=='replay':assert calls==['prefix','replay']
    else:assert calls==['prefix','replay','breliability','aguard','featurehalf']
    record=json.loads(next((tmp_path/'results/bc_tail_launches').glob('*/status.json')).read_text())
    assert record['status']==('complete' if failure=='none' else 'failed')
