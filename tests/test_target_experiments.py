import copy
import csv
import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import bc_targets
import target_experiments as te
from Dataset.normalize import to_tensor_normalize
from fl_runner import _geometry_controls
from options import args_parser
from test_trusted_followups import recipe
from test_trusted_multi import Pictures, TinyBackbone
from test_trusted_risk import run_local
from training_dynamics import dynamics_row, lr_controls


def case():
    pred = torch.tensor([0,0,0,0])
    a = torch.ones(4,dtype=torch.bool)
    b = ~a
    q = torch.tensor([[.7,.3,0],[.7,.3,0],[.1,.9,0],[.45,.55,0]])
    hist = q.clone(); mature = a.clone()
    z = torch.eye(3)[torch.tensor([0,1,1,0])]
    ref = dict(centers=torch.eye(3), valid=torch.ones(3,dtype=torch.bool),trust=torch.ones(3),
               radius=torch.full((3,),.1),margin_floor=torch.zeros(3),margin_scale=torch.full((3,),.1))
    result = bc_targets.decide(pred,a,b,q,q,q.max(-1).values,hist,mature,z,ref)
    return result,pred,a,b,q,hist,mature,z,ref


def test_evidence_missing_is_not_a_veto_and_conflict_blocks_protection():
    r,pred,a,b,q,hist,mature,z,ref=case()
    # No geometry: old targets releases all; moderate positive evidence saves
    # only a subset, not all weak/missing-reference samples.
    ref['valid'].zero_()
    r=bc_targets.decide(pred,a,b,q,q,q.max(-1).values,hist,mature,z,ref)
    new=te.revise(r,'evidence',pred,a,q,q,hist,mature,z,ref)
    assert new['state'].tolist()==[0,0,1,1]
    assert new['geometry_unknown'].all()
    ref['valid'].fill_(True)
    new=te.revise(r,'evidence',pred,a,q,q,hist,mature,z,ref)
    assert new['state'][0]==0 and new['state'][1]!=0
    assert new['geometry_conflict'][1]
    imm=te.revise(r,'evidence',pred,a,q,q,hist,torch.zeros_like(mature),z,ref)
    torch.testing.assert_close(imm['state'],r['state'])


def test_head_can_replace_target_but_missing_original_class_abstains():
    r,pred,a,b,q,hist,mature,z,ref=case()
    aq=torch.tensor([[.85,.15,0],[.85,.15,0],[.05,.95,0],[.05,.95,0]])
    aux=(aq,a,aq,a,torch.ones(3,dtype=torch.bool))
    new=te.revise(r,'labelhead',pred,a,q,q,hist,mature,z,ref,aux)
    assert new['state'][0]==0
    assert new['aux_override'][2] and not new['aux_override'][3]
    assert new['target'][2,1]>.9  # No mandatory 50% wrong-label retention.
    torch.testing.assert_close(new['target'].sum(-1),torch.ones(4))
    logits=torch.zeros(4,3,requires_grad=True)
    la,lb,proto=bc_targets.objectives(logits,pred,a,b,a.float(),b.float(),new,1.)
    la.backward()
    assert logits.grad[2,1]<0 and logits.grad[2,0]>0
    assert proto[2]==0 and proto[0]==1
    unavailable=torch.tensor([False,True,True])
    abstain=te.revise(r,'labelhead',pred,a,q,q,hist,mature,z,ref,(aq,a,aq,a,unavailable))
    assert not abstain['aux_override'].any()
    before={k:v.clone() for k,v in new.items()}
    te.audit(new,pred,a,pred,a.float())
    te.audit(new,pred,a,(pred+1)%3,a.float())
    for k in new: torch.testing.assert_close(new[k],before[k],rtol=0,atol=0)


def test_ridge_statistics_equal_pooled_fit_and_validation_counts():
    z=F.normalize(torch.arange(1.,25.).reshape(8,3),dim=-1)
    y=torch.tensor([0,1,0,1,0,1,0,1]);fold=torch.tensor([0,0,1,1,0,0,1,1])
    full=te.sufficient_statistics(z,y,fold,3)
    left=te.sufficient_statistics(z[:4],y[:4],fold[:4],3)
    right=te.sufficient_statistics(z[4:],y[4:],fold[4:],3)
    for key in full: torch.testing.assert_close(full[key],left[key]+right[key])
    expected=te.solve_head(full['gram'].sum(0),full['rhs'].sum(0))
    observed=te.solve_head((left['gram']+right['gram']).sum(0),(left['rhs']+right['rhs']).sum(0))
    torch.testing.assert_close(expected,observed)
    scores=F.one_hot(y,3).double()
    h=te.calibration_histogram(scores,y,torch.tensor([True,True,False]))
    assert h[:,:,0].sum()==8 and h[:,:,1].sum()==8
    assert torch.isinf(te.calibration_thresholds(h)).all()  # Four votes/class insufficient.
    threshold=te.calibration_thresholds(h*5)
    assert torch.isfinite(threshold[:2]).all() and torch.isinf(threshold[2])
    bad=te.calibration_histogram(scores,(y+1)%2,torch.tensor([True,True,False]))
    assert torch.isinf(te.calibration_thresholds(bad*5)).all()


def test_head_fit_preserves_encoder_rng_deduplicates_and_abstains():
    torch.manual_seed(21);model=TinyBackbone().train();data=Pictures()
    state=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    head=te.fit_online_head(model,data,[list(range(16))+[0,0]],10,to_tensor_normalize('CIFAR10'),'cpu')
    assert model.training and head['counts'].sum()==16
    for key in state: torch.testing.assert_close(model.state_dict()[key],state[key],rtol=0,atol=0)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    head['valid'].zero_()
    q,good=te.predict_head(torch.randn(4,8),head)
    assert not good.any() and torch.isfinite(q).all()
    torch.testing.assert_close(q.sum(-1),torch.ones(4))


def test_separation_gradient_duplicate_invariance_single_class_and_bn():
    z1=torch.randn(3,5,requires_grad=True);z2=torch.randn(3,5,requires_grad=True)
    labels=torch.tensor([0,0,1]);ids=torch.tensor([4,5,6]);logits=torch.randn(3,3,requires_grad=True)
    loss=te.separation_loss(z1,z2,labels,ids,logits)
    loss.backward()
    assert z1.grad.abs().sum()>0 and z2.grad.abs().sum()>0 and logits.grad is None
    ix=torch.tensor([0,1,2,0,1])
    dup=te.separation_loss(z1[ix],z2[ix],labels[ix],ids[ix],logits[ix])
    torch.testing.assert_close(loss,dup)
    single=te.separation_loss(z1,z2,torch.zeros(3,dtype=torch.long),ids,logits)
    assert single==0 and torch.isfinite(single)
    model=TinyBackbone().train();bn=copy.deepcopy(model.bn.state_dict())
    with te.no_bn_updates(model): model(torch.randn(4,3,32,32))
    for key in bn: torch.testing.assert_close(model.bn.state_dict()[key],bn[key],rtol=0,atol=0)
    assert model.bn.track_running_stats


@pytest.mark.parametrize('mode',['evidence','labelhead','separation'])
@pytest.mark.parametrize('phase,round_idx',[(1,45),(1,80),(2,100),(3,150)])
def test_training_gradients_hidden_gt_isolation_and_accounting(mode,phase,round_idx):
    name='bc_targets_'+mode
    w,st,_=run_local(phase,0,experiment=name,round_idx=round_idx)
    altered,other,_=run_local(phase,0,experiment=name,round_idx=round_idx,flip_hidden_labels=True)
    for key in w:
        assert torch.isfinite(w[key]).all()
        torch.testing.assert_close(w[key],altered[key],rtol=0,atol=0)
    for history in ['target_history']+(['aux_history'] if mode=='labelhead' else []):
        for key in st[history]:torch.testing.assert_close(st[history][key],other[history][key],rtol=0,atol=0)
    row=dynamics_row(round_idx,phase,[st['log']],recipe(name))
    assert abs(row['loss_observed']-row['loss_reconstructed'])<1e-6
    assert sum(row[f'experiment_a_pred{c}_true{t}'] for c in range(10) for t in range(10))==st['log']['cnt_a']
    if round_idx==45:
        base,_,_=run_local(phase,0,experiment='bc_targets',round_idx=round_idx)
        for key in base:torch.testing.assert_close(w[key],base[key],rtol=0,atol=0)
    if mode=='separation' and round_idx>60:
        assert row['L_separation_effective']>0


@pytest.mark.parametrize('mode',['evidence','labelhead','separation'])
def test_round_boundary_checkpoint_replay_and_controls(mode,tmp_path):
    name='bc_targets_'+mode
    w,state,_=run_local(3,0,experiment=name,round_idx=150)
    w,state,_=run_local(3,0,experiment=name,round_idx=151,global_weights=w,saved_state=state)
    path=tmp_path/'complete.pt';torch.save((w,state),path)
    loaded,ls=torch.load(path,weights_only=False)
    wa,sa,_=run_local(3,0,experiment=name,round_idx=152,global_weights=w,saved_state=state)
    wb,sb,_=run_local(3,0,experiment=name,round_idx=152,global_weights=loaded,saved_state=ls)
    for key in wa:torch.testing.assert_close(wa[key],wb[key],rtol=0,atol=0)
    for history in ['target_history']+(['aux_history'] if mode=='labelhead' else []):
        for key in sa[history]:torch.testing.assert_close(sa[history][key],sb[history][key],rtol=0,atol=0)
    assert lr_controls(recipe(name))==lr_controls(recipe('bc_targets'))
    assert _geometry_controls(recipe(name))!=_geometry_controls(recipe('bc_targets'))
    if mode=='labelhead':
        del state['aux_history']
        with pytest.raises(ValueError,match='auxiliary history'):
            run_local(3,0,experiment=name,round_idx=152,global_weights=w,saved_state=state)


def test_parser_rejects_incompatible_configs():
    with patch.object(sys,'argv',['test','--target_experiment','labelhead']):
        with pytest.raises(SystemExit):args_parser()


def test_head_diagnostics_rollback_does_not_duplicate_rounds(tmp_path):
    head=dict(counts=torch.ones(3)*10,valid=torch.ones(3,dtype=torch.bool),
              threshold=torch.ones(3)*.1,calibration=torch.ones(3,20,2))
    path=tmp_path/'head.csv'
    for r in (31,32,33,32):te.write_head_audit(path,r,head)
    with path.open() as f:rows=list(csv.DictReader(f))
    assert [int(r['round']) for r in rows]==[31]*3+[32]*3


def test_queue_limits_per_gpu_and_failure_stops_pending():
    import threading
    from scripts.run_target_experiments import run_queue,command
    rendezvous=threading.Barrier(2)
    active=set();lock=threading.Lock();seen=[]
    def launch(name,gpu):
        with lock:
            assert gpu not in active
            active.add(gpu);seen.append((name,gpu))
            assert len(active)<=2
        if name in ('evidence','labelhead'):rendezvous.wait(timeout=5)
        with lock:active.remove(gpu)
        return 0
    results,failed=run_queue([0,1],['evidence','labelhead','separation'],launch)
    assert not failed and len(results)==3 and len(seen)==3
    assert command('evidence',1,'unique')[-4:]==['--gpu_id','1','--run_id','unique']
    results,failed=run_queue([0],['evidence','labelhead','separation'],lambda name,gpu:1)
    assert failed and len(results)==1


@pytest.mark.parametrize('mode',['evidence','labelhead','separation'])
def test_real_resnet_training_path(mode):
    from fl_runner import LocalPPFPSL,AdaptiveSchedule
    from Model.resnet import ResNet
    from Dataset.dataset import Indices2Dataset_labeled,Indices2Dataset_unlabeled_fixmatch
    torch.manual_seed(19)
    args=recipe('bc_targets_'+mode)
    args.num_workers=0;args.local_epochs=1;args.batch_size_local_labeled_fixmatch=4
    local=LocalPPFPSL.__new__(LocalPPFPSL)
    local.args=args;local.device=torch.device('cpu')
    local.model=ResNet(resnet_size=8,scaling=4,num_classes=10)
    local.dim=256;local.teacher=copy.deepcopy(local.model)
    local.optimizer=torch.optim.SGD(local.model.parameters(),lr=.15,momentum=.9,weight_decay=1e-4)
    data=Pictures(16)
    ld=Indices2Dataset_labeled(data);ld.load(list(range(8)))
    ud=Indices2Dataset_unlabeled_fixmatch(data);ud.load(list(range(16)))
    head=None
    if mode=='labelhead':
        head=te.fit_online_head(local.model,data,[list(range(8))],10,to_tensor_normalize('CIFAR10'),'cpu')
    initial=copy.deepcopy(local.model.state_dict())
    snap=AdaptiveSchedule(args,300).for_round(150);snap['phase']=3
    w,state,_=local.train_round(args,ld,ud,initial,torch.zeros(10,256),
        torch.zeros(10,dtype=torch.bool),150,None,torch.tensor([4,4]+[0]*8),snap,auxiliary_head=head)
    assert all(torch.isfinite(value).all() for value in w.values())
    assert not torch.equal(w['conv1.weight'],initial['conv1.weight'])
    row=dynamics_row(150,3,[state['log']],args)
    assert abs(row['loss_observed']-row['loss_reconstructed'])<2e-6
