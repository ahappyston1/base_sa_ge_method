"""Follow-up recipes preserve the original schedule and only enable requested weights."""
import copy
import contextlib
import io
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from options import args_parser
from training_dynamics import schedule_rounds, trusted_weight_gate, lr_controls, cosine_learning_rate, dynamics_row
from fl_runner import AdaptiveSchedule, LocalPPFPSL, _geometry_controls
import fl_runner
from Dataset.dataset import Indices2Dataset_labeled, Indices2Dataset_unlabeled_fixmatch
from test_trusted_multi import Pictures, TinyBackbone

ROOT=Path(__file__).resolve().parents[1]


def recipe(name):
    with patch.object(sys,'argv',['test','--config',str(ROOT/'configs'/f'experiment_{name}.yaml')]):
        args=args_parser()
    args.num_rounds=args.max_rounds;args.num_classes=10
    return args


def trajectory(args):
    s=AdaptiveSchedule(args,schedule_rounds(args));rows=[]
    with contextlib.redirect_stdout(io.StringIO()):
        for r in range(1,args.num_rounds+1):
            rows.append(s.for_round(r).copy())
            s.update_after_round(r,dict(acc=.8,a_prec=.9,b_prec=.6,a_ratio=.7,b_ratio=.2,c_ratio=.1))
    return rows


def test_tail_preserves_first300_and_never_restarts_lr():
    old=recipe('trusted_lr015');tail=recipe('trusted_lr015_tail500')
    assert lr_controls(old)==lr_controls(tail)
    assert trajectory(old)==trajectory(tail)[:300]
    assert all(cosine_learning_rate(r,.15,schedule_rounds(tail))==.0001 for r in range(300,501))
    assert _geometry_controls(old)==_geometry_controls(tail)


def test_early_does_not_move_phase_or_auxiliary_losses():
    old=recipe('trusted_lr015');early=recipe('trusted_lr015_early')
    assert trajectory(old)==trajectory(early)
    assert lr_controls(old)==lr_controls(early)
    assert trusted_weight_gate(30,0,early)==0
    assert 0<trusted_weight_gate(31,0,early)<.01
    assert abs(trusted_weight_gate(45,0,early)-.5)<1e-10
    assert trusted_weight_gate(60,0,early)==1
    assert trusted_weight_gate(91,.001,early)==1
    assert _geometry_controls(old)!=_geometry_controls(early)


def local_round(args,r):
    torch.manual_seed(19)
    args.num_workers=0;args.local_epochs=1;args.batch_size_local_labeled_fixmatch=4
    args.batch_size_local_unlabeled=8
    local=LocalPPFPSL.__new__(LocalPPFPSL)
    local.args=args;local.device=torch.device('cpu');local.model=TinyBackbone();local.dim=8
    local.teacher=copy.deepcopy(local.model)
    local.optimizer=torch.optim.SGD(local.model.parameters(),lr=.15,momentum=.9,weight_decay=1e-4)
    data=Pictures();ld=Indices2Dataset_labeled(data);ld.load(list(range(16)))
    ud=Indices2Dataset_unlabeled_fixmatch(data);ud.load(list(range(32)))
    state=copy.deepcopy(local.model.state_dict())
    # Use a deterministic synthetic reference score to ensure the early path lowers weights.
    real_score=fl_runner.score_queries
    def score(*pos,**kwargs):
        tg=real_score(*pos,**kwargs)
        tg['weight']=torch.full_like(tg['weight'],.5)
        return tg
    # Route enough A/B to test both loss branches without changing the actual route function.
    args.tau_warmup=.11;args.eta_B=.10
    s=AdaptiveSchedule(args,schedule_rounds(args));snap=s.for_round(r)
    # Inject phase1 with B active, preserving the production early/full phase distinction.
    snap['phase']=1;snap['aux_scale']=1.
    with patch('fl_runner.score_queries',score) as scorer,patch('fl_runner.refresh_reference',wraps=fl_runner.refresh_reference) as ref:
        weights,new_state,_=local.train_round(args,ld,ud,state,torch.zeros(10,8),torch.zeros(10,dtype=torch.bool),r,None,torch.tensor([8,8]+[0]*8),snap)
        return weights,new_state['log'],ref.call_count


def test_actual_local_training_prefix_identical_and_early_reference_active():
    # Same data augmentation RNG and model for the first 30 rounds.
    import random
    import numpy as np
    def run(name,r):
        random.seed(23);np.random.seed(23)
        return local_round(recipe(name),r)
    baseline,log0,n0=run('trusted_lr015',30)
    early,log1,n1=run('trusted_lr015_early',30)
    assert n0==n1==0
    for k in baseline:torch.testing.assert_close(baseline[k],early[k],rtol=0,atol=0)
    base45,logb,_=run('trusted_lr015',45)
    early45,loge,n=run('trusted_lr015_early',45)
    assert n==1
    assert abs(loge['trusted_weight_gate']-.5)<1e-10
    assert loge['L_A']<logb['L_A']
    assert any(not torch.equal(v,early45[k]) for k,v in base45.items() if v.is_floating_point())
    row=dynamics_row(45,1,[loge],recipe('trusted_lr015_early'))
    assert abs(row['loss_reconstructed']-row['loss_observed'])<1e-6
