import sys
from unittest.mock import patch
import pytest
import torch
from options import args_parser
from trusted_risk import score_a
from test_trusted_risk import reference, run_local
from test_trusted_followups import recipe
from training_dynamics import dynamics_row


def test_actual_configuration_and_reject_combinations():
    args=recipe('trusted_lr015_classrisk')
    assert args.trusted_class_risk==1 and args.trusted_a_risk==0
    assert args.lr_local_training==.15 and args.max_rounds==300
    assert args.bc_teacher==0 and args.mid_prox_mu==0
    for flag in ['--trusted_a_risk','--bc_teacher','--mid_prox_mu']:
        with patch.object(sys,'argv',['test','--config',args.config,flag,'1']):
            with pytest.raises(SystemExit):args_parser()


def test_insufficient_pair_evidence_is_exact_risk_fallback():
    ref=reference()
    ref['risk_calibration']['pairs']=dict(count=torch.zeros(3,3,2,2),errors=torch.zeros(3,3,2,2))
    z=torch.tensor([[0.,1.],[.70710678,.70710678],[1.,0.]])
    pred=torch.tensor([0,0,2]);logits=torch.nn.functional.one_hot(pred,3).float()*12
    old=score_a(z,pred,logits,ref,1.)
    new=score_a(z,pred,logits,ref,1.,class_specific=True)
    torch.testing.assert_close(new['weight'],old['weight'],rtol=0,atol=0)


def conditioned_reference(pair_errors,rest_errors):
    ref=reference();cal=ref['risk_calibration']
    cal['pairs']=dict(count=torch.zeros(3,3,2,2),errors=torch.zeros(3,3,2,2))
    cal['pairs']['count'][0,1,1,1]=20
    cal['pairs']['errors'][0,1,1,1]=pair_errors
    cal['count'][0,1,2]=20
    cal['errors'][0,1,2]=pair_errors
    cal['count'][0,1,0]=20
    cal['errors'][0,1,0]=rest_errors
    return ref


def test_evidence_can_strengthen_or_relax_same_pair_without_flipping_labels():
    z=torch.tensor([[0.,1.]]);pred=torch.tensor([0]);logits=torch.tensor([[12.,0.,0.]])
    for errors,rest,direction in [(20,0,-1),(0,20,1)]:
        ref=conditioned_reference(errors,rest)
        old=score_a(z,pred,logits,ref,1.)
        new=score_a(z,pred,logits,ref,1.,class_specific=True)
        assert direction*(new['weight']-old['weight'])>0
        assert .15<=new['weight']<=1
        assert new['pair']['eligible'].item()
    # Equal conditional risk gives no evidence-based adjustment.
    ref=conditioned_reference(10,10)
    old=score_a(z,pred,logits,ref,1.)
    new=score_a(z,pred,logits,ref,1.,class_specific=True)
    torch.testing.assert_close(new['weight'],old['weight'],rtol=0,atol=0)


def test_pair_calibration_excludes_center_support_and_unconfident_queries():
    from trusted_geometry import fit_reference
    from trusted_risk import fit_calibration
    y=torch.tensor([0]*8+[1]*8);z=torch.nn.functional.one_hot(y,2).float()
    ref=fit_reference(z,y,2)
    logits=torch.nn.functional.one_hot(1-y,2).float()*12
    cal=fit_calibration(z,logits,y,ref,pair_specific=True)
    assert cal['pairs']['count'].sum()==8
    assert cal['pairs']['errors'].sum()==8
    changed=logits.clone();changed[::2]=changed[::2].flip(1)
    other=fit_calibration(z,changed,y,ref,pair_specific=True)
    torch.testing.assert_close(cal['pairs']['count'],other['pairs']['count'],rtol=0,atol=0)
    empty=fit_calibration(z,torch.zeros_like(logits),y,ref,pair_specific=True)
    assert empty['pairs']['count'].sum()==0


def test_classrisk_actual_training_and_hidden_label_isolation():
    for phase in [1,2,3]:
        params,state,_=run_local(phase,0,experiment='trusted_lr015_classrisk')
        other,_,_=run_local(phase,0,experiment='trusted_lr015_classrisk',flip_hidden_labels=True)
        for k in params:
            assert torch.isfinite(params[k]).all()
            torch.testing.assert_close(params[k],other[k],rtol=0,atol=0)
        row=dynamics_row(45 if phase==1 else 150,phase,[state['log']],recipe('trusted_lr015_classrisk'))
        assert abs(row['loss_reconstructed']-row['loss_observed'])<1e-6
        assert state['risk_rows']
        assert 'pair_rows' in state
