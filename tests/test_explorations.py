import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from exploration import initialize_heads, teacher_objectives, prox_coefficient, proximal_loss, ema
from test_trusted_risk import run_local
from test_trusted_followups import recipe
from training_dynamics import dynamics_row


def test_prox_schedule_and_gradient_direction():
    args=recipe('trusted_lr015_prox')
    assert prox_coefficient(30,args)==0
    assert prox_coefficient(60,args)==.01
    assert prox_coefficient(200,args)==.01
    assert 0<prox_coefficient(225,args)<.01
    assert prox_coefficient(250,args)==0
    model=nn.Linear(2,1,bias=False);anchor={n:p.detach().clone() for n,p in model.named_parameters()}
    assert proximal_loss(model,anchor,.01)==0
    with torch.no_grad():model.weight.add_(1)
    proximal_loss(model,anchor,.01).backward()
    torch.testing.assert_close(model.weight.grad,torch.full_like(model.weight,.01))


def test_heads_rng_isolation_and_stop_gradient():
    rng=torch.get_rng_state().clone();heads=initialize_heads(8,'cpu')
    torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
    target=copy.deepcopy(heads.projector)
    z=torch.randn(4,8,requires_grad=True);zs=torch.randn(4,8,requires_grad=True)
    logits=torch.randn(4,10,requires_grad=True);lt=torch.randn(4,10,requires_grad=True)
    b,f=teacher_objectives(zs,logits,z,lt,heads,target,torch.tensor([0,1,0,0],dtype=torch.bool),
                           torch.tensor([0,0,1,1],dtype=torch.bool),torch.ones(4),1.)
    (b+f).backward()
    assert z.grad is None and lt.grad is None
    assert all(p.grad is None for p in target.parameters())
    assert zs.grad.abs().sum()>0 and logits.grad.abs().sum()>0
    assert any(p.grad is not None for p in heads.parameters())
    zero_b,zero_f=teacher_objectives(zs,logits,z,lt,heads,target,torch.zeros(4,dtype=torch.bool),
                                   torch.zeros(4,dtype=torch.bool),torch.ones(4),1.)
    assert zero_b==zero_f==0


def test_both_prefixes_preserve_early_and_direct_a_objective():
    before,_,_=run_local(1,0,experiment='trusted_lr015_early',round_idx=30)
    for name in ['trusted_lr015_bc','trusted_lr015_prox']:
        after,_,_=run_local(1,0,experiment=name,round_idx=30)
        for k in before:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
        for phase in [1,2,3]:
            _,base,_=run_local(phase,0,True,experiment='trusted_lr015_early')
            _,new,_=run_local(phase,0,True,experiment=name)
            for key in ['cnt_a','cnt_b','cnt_c','L_A','L_proto']:
                assert abs(base['log'][key]-new['log'][key])<1e-7


def test_real_training_loss_accounting_and_hidden_label_isolation():
    for name in ['trusted_lr015_bc','trusted_lr015_prox']:
        for phase in [1,2,3]:
            weights,state,_=run_local(phase,0,experiment=name,a_threshold=.95)
            flipped,_,_=run_local(phase,0,flip_hidden_labels=True,experiment=name,a_threshold=.95)
            for k in weights:
                assert torch.isfinite(weights[k]).all()
                torch.testing.assert_close(weights[k],flipped[k],rtol=0,atol=0)
            args=recipe(name)
            r=45 if phase==1 else 150
            row=dynamics_row(r,phase,[state['log']],args)
            assert abs(row['loss_reconstructed']-row['loss_observed'])<1e-6
            if name.endswith('_bc'):assert 'bc_heads' in state and row['L_feature_effective']>0
            else:assert row['L_prox_effective']>0


def test_queue_uses_first_free_lane_and_waits_for_old_pid(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/run_explorations.py'
    spec=importlib.util.spec_from_file_location('queue_experiments',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    for third in [False,True]:
        calls=[];clock=[0];lives=[]
        class Process:
            def __init__(self,end,pid):self.end=end;self.pid=pid
            def poll(self):return 0 if clock[0]>=self.end else None
        def launch(command,**kwargs):
            index=len(calls);calls.append((clock[0],command))
            p=Process(clock[0]+(4 if index==0 else 2),100+index);lives.append(p);return p
        def sleep(seconds):clock[0]+=1
        def identity(pid):return 'old-starttime' if clock[0]<1 else None
        args=SimpleNamespace(gpus=[0,1],after_pid=123 if third else None,after_gpu=2 if third else None,seed=7,dry_run=False)
        # Unique output root per mocked queue run.
        monkeypatch.setattr(module,'ROOT',tmp_path/str(third))
        assert module.run(args,popen=launch,sleep=sleep,identity=identity)==0
        assert len(calls)==3
        assert calls[2][0]==(1 if third else 2)
        assert calls[2][1][-2]==('2' if third else '1')


def test_bc_auxiliary_state_roundtrip(tmp_path):
    weights,state,_=run_local(3,0,experiment='trusted_lr015_bc',a_threshold=.95)
    path=tmp_path/'client.pt';torch.save(state,path)
    restored=torch.load(path,weights_only=False)
    live,next_state,_=run_local(3,0,experiment='trusted_lr015_bc',a_threshold=.95,round_idx=151,
                                saved_state=state,global_weights=weights)
    resumed,next_restored,_=run_local(3,0,experiment='trusted_lr015_bc',a_threshold=.95,round_idx=151,
                                    saved_state=restored,global_weights=weights)
    for k in live:torch.testing.assert_close(live[k],resumed[k],rtol=0,atol=0)
    for k in next_state['bc_heads']:
        torch.testing.assert_close(next_state['bc_heads'][k],next_restored['bc_heads'][k],rtol=0,atol=0)
