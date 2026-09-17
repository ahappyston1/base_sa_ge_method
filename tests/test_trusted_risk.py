import copy
from contextlib import nullcontext
import random
import numpy as np
import torch
import torch.nn.functional as F
from unittest.mock import patch
from trusted_risk import evidence, fit_calibration, score_a, audit, new_audit
from trusted_geometry import fit_reference
from test_trusted_followups import recipe, local_round
from fl_runner import _geometry_controls, LocalPPFPSL, AdaptiveSchedule
import fl_runner
from test_trusted_multi import Pictures, TinyBackbone
from Dataset.dataset import Indices2Dataset_labeled, Indices2Dataset_unlabeled_fixmatch


def reference():
    return dict(centers=torch.tensor([[1.,0.],[0.,1.],[0.,0.]]),
                valid=torch.tensor([True,True,False]), radius=torch.full((3,),.2),
                distance_scale=torch.full((3,),.1), margin_scale=torch.full((3,),.1),
                trust=torch.full((3,),.8), risk_calibration=dict(count=torch.zeros(3,2,4),
                errors=torch.zeros(3,2,4),threshold=.95))


def test_distance_conflict_missing_and_no_double_penalty():
    z=F.normalize(torch.tensor([[1.,0.],[1.,1.],[0.,1.],[1.,0.]]),dim=1)
    pred=torch.tensor([0,0,0,2]); logits=F.one_hot(pred,3).float()*12
    ref=reference(); result=score_a(z,pred,logits,ref,1.)
    assert result['group'].tolist()==[0,1,2,3]
    w=result['weight']
    assert w[0]==w[3]==1
    assert .85 <= w[1] < 1
    assert .4 <= w[2] < w[1]
    assert torch.equal(score_a(z,pred,logits,ref,0.)['weight'],torch.ones(4))
    assert torch.all(score_a(z,pred,logits,ref,1.,1.)['weight'] >= w)
    ref['trust'][1]=0
    assert evidence(z[2:3],pred[2:3],ref)[0].item()==1  # untrusted rival cannot establish conflict


def test_calibration_only_uses_query_examples_and_stays_bounded():
    y=torch.tensor([0]*4+[1]*4)
    z=F.one_hot(y,2).float()
    ref=fit_reference(z,y,2)
    logits=z*12
    good=fit_calibration(z,logits,y,ref)
    altered=logits.clone();altered[[0,2,4,6]]=altered[[0,2,4,6]].flip(1)
    other=fit_calibration(z,altered,y,ref)
    torch.testing.assert_close(good['count'],other['count'])
    assert good['count'].sum()==4
    assert good['errors'].sum()==0
    # Actual confident wrong query predictions must be recorded as errors.
    altered=logits.flip(1)
    bad=fit_calibration(z,altered,y,ref)
    assert bad['errors'].sum()==4


def test_sparse_calibration_shrinks_and_eval_disagreement_falls_back():
    ref=reference(); z=torch.tensor([[0.,1.]]); pred=torch.tensor([0]); logits=torch.tensor([[12.,0.,0.]])
    initial=score_a(z,pred,logits,ref,1.)['weight']
    ref['risk_calibration']['count'][0,1,2]=1
    one=score_a(z,pred,logits,ref,1.)['weight']
    ref['risk_calibration']['count'][0,1,2]=100
    many=score_a(z,pred,logits,ref,1.)['weight']
    assert initial < one < many < 1
    ref['risk_calibration']['errors'][0,1,2]=100
    assert score_a(z,pred,logits,ref,1.)['weight'] >= .4
    mismatch=score_a(z,pred,torch.tensor([[0.,12.,0.]]),ref,1.)
    torch.testing.assert_close(mismatch['weight'],initial)
    assert mismatch['calibration_n']==0


def test_diagnostic_labels_do_not_change_weights():
    ref=reference();z=torch.tensor([[0.,1.]]);pred=torch.tensor([0]);logits=torch.tensor([[12.,0.,0.]])
    result=score_a(z,pred,logits,ref,1.); before=result['weight'].clone()
    for gt in [torch.tensor([0]),torch.tensor([1])]:
        audit(new_audit(3,'cpu'),result,torch.tensor([True]),pred,gt,torch.tensor([.5]))
        torch.testing.assert_close(result['weight'],before)


def test_original_prefix_and_checkpoint_controls():
    def run(name):
        random.seed(23);np.random.seed(23)
        return local_round(recipe(name),30)
    old,_,_=run('trusted_lr015_early');new,_,_=run('trusted_lr015_risk')
    for k in old: torch.testing.assert_close(old[k],new[k],rtol=0,atol=0)
    assert 'trusted_a_risk' not in _geometry_controls(recipe('trusted_lr015_early'))
    assert _geometry_controls(recipe('trusted_lr015_risk'))['risk_version']==1


def run_local(phase, enabled, frozen=False, flip_hidden_labels=False, experiment='trusted_lr015_risk', round_idx=None, a_threshold=.11, saved_state=None, global_weights=None):
    random.seed(23);np.random.seed(23);torch.manual_seed(19)
    args=recipe(experiment);args.trusted_a_risk=enabled
    args.num_workers=0;args.local_epochs=1;args.batch_size_local_labeled_fixmatch=4
    args.batch_size_local_unlabeled=8;args.tau_warmup=a_threshold;args.eta_B=.10
    local=LocalPPFPSL.__new__(LocalPPFPSL);local.args=args;local.device=torch.device('cpu')
    local.model=TinyBackbone();local.dim=8;local.teacher=copy.deepcopy(local.model)
    local.optimizer=torch.optim.SGD(local.model.parameters(),lr=.15,momentum=.9,weight_decay=1e-4)
    data=Pictures();ld=Indices2Dataset_labeled(data);ld.load(list(range(16)))
    class HiddenLabels(Pictures):
        def __getitem__(self, i):
            image,y=super().__getitem__(i)
            return image, 1-y if flip_hidden_labels else y
    ud=Indices2Dataset_unlabeled_fixmatch(HiddenLabels());ud.load(list(range(32)))
    r=round_idx if round_idx is not None else (45 if phase==1 else 150)
    snap=AdaptiveSchedule(args,300).for_round(r);snap['phase']=phase;snap['aux_scale']=1.
    context=patch.object(local.optimizer,'step',return_value=None) if frozen else nullcontext()
    extra = {}
    if getattr(args, 'target_experiment', 'none') == 'labelhead' and r > 30:
        import target_experiments
        from Dataset.normalize import to_tensor_normalize
        encoder = copy.deepcopy(local.model)
        if global_weights is not None:
            encoder.load_state_dict(global_weights)
        extra['auxiliary_head'] = target_experiments.fit_online_head(encoder,data,[list(range(16))],10,
                                           to_tensor_normalize('CIFAR10'),torch.device('cpu'))
    with context:
        return local.train_round(args,ld,ud,copy.deepcopy(global_weights if global_weights is not None else local.model.state_dict()),torch.zeros(10,8),
                torch.zeros(10,dtype=torch.bool),r,copy.deepcopy(saved_state),torch.tensor([8,8]+[0]*8),snap,**extra)


def test_same_inputs_keep_b_and_proto_objectives_and_routes():
    # Freeze optimizer only here: isolate direct objective changes from later
    # changes in trajectory caused by legitimate A updates.
    for phase in [1,2,3]:
        _,old,_=run_local(phase,0,True);_,new,_=run_local(phase,1,True)
        for key in ['L_B','L_proto','cnt_a','cnt_b','cnt_c']:
            assert abs(old['log'][key]-new['log'][key])<1e-7
        assert new['risk_rows'] and new['risk_calibration_rows']
        assert sum(row['a_visits'] for row in new['risk_rows'])==new['log']['cnt_a']


def test_real_optimizer_smoke_all_phases():
    for phase in [1,2,3]:
        params,state,_=run_local(phase,1)
        assert all(torch.isfinite(t).all() for t in params.values())
        assert np.isfinite(state['log']['loss'])
        assert state['risk_rows']


def test_actual_training_independent_of_hidden_labels():
    for phase in [1,3]:
        original,state,_=run_local(phase,1)
        flipped,other,_=run_local(phase,1,flip_hidden_labels=True)
        for k in original:torch.testing.assert_close(original[k],flipped[k],rtol=0,atol=0)
        assert state['risk_calibration_rows']==other['risk_calibration_rows']


def test_snapshot_preserves_rng_and_models(tmp_path):
    from update_diagnostics import save_update_snapshot
    before={'weight':torch.tensor([1.,2.]),'bn.running_mean':torch.tensor([.2]),
            'bn.num_batches_tracked':torch.tensor(2)}
    local={k:v+1 for k,v in before.items()}
    original=copy.deepcopy(before);rng=torch.get_rng_state().clone()
    path=tmp_path/'updates'/'round_0083.pt'
    save_update_snapshot(str(path),83,before,[local],local,[8],[10],['weight'])
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    for k in before:torch.testing.assert_close(before[k],original[k],rtol=0,atol=0)
    saved=torch.load(path,weights_only=True)
    before['weight'].zero_()
    torch.testing.assert_close(saved['before']['weight'],original['weight'])
    assert saved['client_ids']==[8] and saved['data_counts']==[10]
    assert saved['fused']['bn.num_batches_tracked'].dtype==torch.int64
