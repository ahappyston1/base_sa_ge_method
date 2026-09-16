import copy
import pytest
import torch
import torch.nn.functional as F
from bc_targets import History, decide, objectives, audit, gate
from test_trusted_risk import run_local
from test_trusted_followups import recipe
from training_dynamics import dynamics_row


def test_history_duplicates_stale_and_snapshot():
    h=History([10,20],3,'cpu')
    ids=torch.tensor([10,10,20]);q=torch.tensor([[1.,0,0],[1.,0,0],[0.,1,0]])
    h.observe(ids,q)
    assert not h.lookup(ids,31)[1].any()
    state=h.finish(31)
    assert state['count'].tolist()==[1,1]
    h=History([10,20],3,'cpu',state);h.observe(ids,q)
    state=h.finish(32)
    h=History([10,20],3,'cpu',state)
    assert h.lookup(ids,33)[1].all()
    assert not h.lookup(ids,53)[1].any()
    h.observe(ids,q);state=h.finish(53)
    assert state['count'].tolist()==[1,1]
    with pytest.raises(ValueError):h.finish(53)


def inputs():
    q=torch.full((5,10),.001)
    q[0,0]=.991;q[1,1]=.991
    q[2,1]=.55;q[2,0]=.442
    q[3]=.1;q[4,1]=.991
    pred=torch.tensor([0,0,0,0,1]);a=torch.tensor([True,True,True,False,False]);b=~a
    z=torch.eye(10)[torch.tensor([0,1,1,0,1])]
    ref=dict(centers=torch.eye(10),valid=torch.ones(10,dtype=torch.bool),trust=torch.ones(10),
             radius=torch.full((10,),.1),margin_floor=torch.zeros(10))
    r=decide(pred,a,b,q,q,q.max(-1).values,q,torch.ones(5,dtype=torch.bool),z,ref)
    return r,pred,a,b


def test_states_partial_gradient_and_no_hidden_label_decisions():
    r,pred,a,b=inputs()
    assert r['state'].tolist()==[0,2,1,3,2]
    assert r['candidates'].sum(-1).max()<=3
    logits=torch.randn(5,10,requires_grad=True)
    wa=a.float();wb=b.float()
    la,lb,proto=objectives(logits,pred,a,b,wa,wb,r,1.)
    assert proto.tolist()==[1,0,0,1,1]
    (la+lb).backward()
    assert logits.grad[3].abs().sum()==0
    assert logits.grad[2][~r['candidates'][2]].min()>0
    before={k:v.clone() for k,v in r.items()}
    audit(r,pred,a,b,pred,wa,wb,1.)
    audit(r,pred,a,b,(pred+1)%10,wa,wb,1.)
    for k in r: torch.testing.assert_close(r[k],before[k],rtol=0,atol=0)


@pytest.mark.parametrize('phase,r',[(1,45),(1,80),(3,150)])
def test_actual_training_accounting_gt_isolation_and_prefix(phase,r):
    w,state,_=run_local(phase,0,experiment='bc_targets',round_idx=r)
    altered,_,_=run_local(phase,0,experiment='bc_targets',round_idx=r,flip_hidden_labels=True)
    for k in w:torch.testing.assert_close(w[k],altered[k],rtol=0,atol=0)
    row=dynamics_row(r,phase,[state['log']],recipe('bc_targets'))
    assert abs(row['loss_observed']-row['loss_reconstructed'])<1e-6
    assert state['target_history']['count'].max()==1
    if r==45:
        original,_,_=run_local(phase,0,experiment='trusted_lr015_bc',round_idx=r)
        for k in w:torch.testing.assert_close(w[k],original[k],rtol=0,atol=0)


def test_checkpoint_replay_and_mature_history(tmp_path):
    w,state,_=run_local(3,0,experiment='bc_targets',round_idx=150)
    w,state,_=run_local(3,0,experiment='bc_targets',round_idx=151,global_weights=w,saved_state=state)
    path=tmp_path/'checkpoint.pt';torch.save((w,state),path)
    lw,ls=torch.load(path,weights_only=False)
    a,ast,_=run_local(3,0,experiment='bc_targets',round_idx=152,global_weights=w,saved_state=state)
    b,bst,_=run_local(3,0,experiment='bc_targets',round_idx=152,global_weights=lw,saved_state=ls)
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    for k in ast['target_history']:torch.testing.assert_close(ast['target_history'][k],bst['target_history'][k],rtol=0,atol=0)
    assert ast['target_history']['count'].max()==3
    assert sum(ast['log'].get(f'target_c{c}_history_mature',0) for c in range(10))>0
    broken=copy.deepcopy(state);del broken['target_history']
    with pytest.raises(ValueError,match='history'):
        run_local(3,0,experiment='bc_targets',round_idx=152,global_weights=w,saved_state=broken)
