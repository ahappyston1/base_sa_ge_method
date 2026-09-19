import sys
from unittest.mock import patch

import pytest
import torch

import bc_targets
import target_experiments as te
from fl_runner import _geometry_controls
from options import args_parser
from test_trusted_followups import recipe, trajectory
from test_trusted_risk import run_local
from training_dynamics import training_learning_rate, cosine_learning_rate, lr_controls, dynamics_row


def test_lr_boundaries_and_unchanged_phase_schedule():
    base=recipe('bc_targets_labelhead')
    constant=recipe('bc_targets_labelhead_constant')
    step=recipe('bc_targets_labelhead_step')
    tail=recipe('bc_targets_labelhead_step_tail')
    guard=recipe('bc_targets_labelhead_guard')
    for r in range(1,301):
        assert training_learning_rate(r,constant)==.15
        assert training_learning_rate(r,base)==cosine_learning_rate(r,.15,300)
        assert training_learning_rate(r,guard)==training_learning_rate(r,base)
        assert training_learning_rate(r,step)==(.15 if r<=200 else .05)
        if r<=270:assert training_learning_rate(r,step)==training_learning_rate(r,tail)
    rates=[training_learning_rate(r,tail) for r in range(270,301)]
    assert rates[0]==.05 and rates[-1]==.001
    assert all(x>y for x,y in zip(rates,rates[1:]))
    assert training_learning_rate(301,tail)==.001
    assert lr_controls(step)!=lr_controls(tail)!=lr_controls(base)
    assert lr_controls(guard)==lr_controls(base)
    assert lr_controls(constant)!=lr_controls(step)
    assert _geometry_controls(constant)==_geometry_controls(base)
    assert _geometry_controls(step)==_geometry_controls(tail)==_geometry_controls(base)
    assert _geometry_controls(guard)!=_geometry_controls(base)
    assert trajectory(base)==trajectory(constant)==trajectory(step)==trajectory(tail)==trajectory(guard)


@pytest.mark.parametrize('strength',[0.,.5,1.])
@pytest.mark.parametrize('before_state',[1,3])
def test_guard_keeps_correction_and_decouples_protection_from_prototype_feature(strength,before_state):
    pred=torch.zeros(4,dtype=torch.long);a=torch.ones(4,dtype=torch.bool);b=~a
    q=torch.tensor([[.7,.3,0],[.9,.1,0],[.1,.9,0],[.9,.1,0]])
    history=torch.tensor([[.6,.4,0]]).repeat(4,1)
    z=torch.eye(3)[pred]
    ref=dict(centers=torch.eye(3),valid=torch.zeros(3,dtype=torch.bool),trust=torch.zeros(3),
             radius=torch.ones(3),margin_floor=torch.zeros(3))
    original=bc_targets.decide(pred,a,b,q,q,q.max(-1).values,history,a,z,ref)
    assert original['state'].tolist()==[1,1,1,1]
    original['state'].fill_(before_state)
    aq=torch.tensor([[.9,.1,0],[.9,.1,0],[.05,.95,0],[.9,.1,0]])
    ah=aq.clone();ah[3]=torch.tensor([.7,.3,0])
    aux=(aq,a,ah,a,torch.ones(3,dtype=torch.bool))
    old=te.revise(original,'labelhead',pred,a,q,q,history,a,z,ref,aux)
    new=te.revise(original,'labelhead',pred,a,q,q,history,a,z,ref,aux,guard=True)
    assert old['state'].tolist()==[0,0,2,0]
    assert new['state'].tolist()==old['state'].tolist()
    torch.testing.assert_close(old['aux_override'],new['aux_override'])
    torch.testing.assert_close(old['target'][2],new['target'][2],rtol=0,atol=0)
    assert torch.equal(new['feature_a'],old['a_changed'])
    assert new['prototype_blocked'].all()
    logits=torch.zeros(4,3,requires_grad=True)
    la,lb,proto=bc_targets.objectives(logits,pred,a,b,a.float(),b.float(),new,strength)
    hard=torch.nn.functional.cross_entropy(logits,pred,reduction='none')
    logp=logits.log_softmax(-1)
    original_loss=-torch.logsumexp(logp.masked_fill(~original['candidates'],-torch.inf),-1)
    if before_state==3:original_loss=torch.zeros_like(hard)
    soft=torch.nn.functional.kl_div(logp,new['target'],reduction='none').sum(-1)
    expected=.5*hard+.5*original_loss
    expected[2]=soft[2]
    torch.testing.assert_close(la,((1-strength)*hard+strength*expected).mean())
    assert lb.item()==0
    torch.testing.assert_close(proto,torch.full_like(proto,1-strength))
    la.backward()
    assert logits.grad[1,0]<0
    if strength==1:assert logits.grad[2,1]<0
    before={k:v.clone() for k,v in new.items()}
    for truth in (pred,torch.ones_like(pred)):
        stats=te.guard_audit(new,pred,truth,a.float(),.5)
        assert stats['guard_blocked_protected']==3
        assert stats['guard_blocked_correct_mass']+stats['guard_blocked_wrong_mass']==1.5
        assert stats['guard_ce_reduced_correct_mass']+stats['guard_ce_reduced_wrong_mass']==.75
        for k in before:torch.testing.assert_close(new[k],before[k],rtol=0,atol=0)
    diagnostic=bc_targets.audit(new,pred,a,b,pred,a.float(),b.float(),strength)
    released=diagnostic[:,bc_targets.FIELDS.index('a_hard_released_correct_mass')].sum()
    assert released.item()==pytest.approx(strength*2.5)
    assert te.guard_controls()['version']==2


@pytest.mark.parametrize('name',['labelhead_constant','labelhead_step','labelhead_step_tail','labelhead_guard'])
@pytest.mark.parametrize('r',[80,201,271,300])
def test_real_training_lr_accounting_and_hidden_gt(name,r):
    experiment='bc_targets_'+name;phase=1 if r==80 else 3
    w,state,_=run_local(phase,0,experiment=experiment,round_idx=r)
    other,st,_=run_local(phase,0,experiment=experiment,round_idx=r,flip_hidden_labels=True)
    for k in w:
        assert torch.isfinite(w[k]).all()
        torch.testing.assert_close(w[k],other[k],rtol=0,atol=0)
    for history in ['target_history','aux_history']:
        for k in state[history]:torch.testing.assert_close(state[history][k],st[history][k],rtol=0,atol=0)
    args=recipe(experiment)
    assert state['log']['lr']==training_learning_rate(r,args)
    row=dynamics_row(r,phase,[state['log']],args)
    assert abs(row['loss_observed']-row['loss_reconstructed'])<1e-6


@pytest.mark.parametrize('name',['labelhead_constant','labelhead_step','labelhead_step_tail','labelhead_guard'])
def test_resume_at_tail_boundary(name,tmp_path):
    experiment='bc_targets_'+name
    w,st,_=run_local(3,0,experiment=experiment,round_idx=269)
    w,st,_=run_local(3,0,experiment=experiment,round_idx=270,global_weights=w,saved_state=st)
    path=tmp_path/'round270.pt';torch.save((w,st),path)
    loaded,ls=torch.load(path,weights_only=False)
    wa,sa,_=run_local(3,0,experiment=experiment,round_idx=271,global_weights=w,saved_state=st)
    wb,sb,_=run_local(3,0,experiment=experiment,round_idx=271,global_weights=loaded,saved_state=ls)
    for k in wa:torch.testing.assert_close(wa[k],wb[k],rtol=0,atol=0)
    for history in ['target_history','aux_history']:
        for k in sa[history]:torch.testing.assert_close(sa[history][k],sb[history][k],rtol=0,atol=0)


def test_cli_rejects_accidental_method_schedule_combination():
    with patch.object(sys,'argv',['test','--config','configs/experiment_bc_targets_labelhead_guard.yaml','--lr_schedule','step200']):
        with pytest.raises(SystemExit):args_parser()
