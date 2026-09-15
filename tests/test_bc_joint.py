import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import bc_joint
from test_trusted_risk import run_local, reference
from test_trusted_followups import recipe
from training_dynamics import dynamics_row
from fl_runner import _geometry_controls


def sample(mode='joint', strength=1.):
    pred = torch.tensor([0, 0, 0, 2, 0])
    q = torch.tensor([[.98,.01,.01],[.01,.98,.01],[.98,.01,.01],
                      [.01,.01,.98],[.98,.01,.01]])
    old = torch.tensor([.4,.4,.4,.4,0.])
    mask = torch.tensor([True,True,True,True,False])
    z = torch.tensor([[1.,0.],[0.,1.],[0.,1.],[1.,0.],[1.,0.]])
    result = bc_joint.decisions(old,mask,pred,q,q,torch.ones(5),z,
                               reference(),strength,strength,mode)
    return result, old, pred, mask


def test_mutual_exclusion_bounds_reference_and_prototype_feedback():
    result, old, pred, mask = sample()
    assert result['recover'].tolist() == [True,False,False,False,False]
    assert result['correct'].tolist() == [False,True,False,False,False]
    assert (result['weight'] >= old).all()
    assert (result['weight'] <= old+.5*(1-old)).all()
    assert result['rho'].max() <= .5
    assert result['proto_factor'].tolist() == [1.,0.,1.,1.,1.]
    assert sample('recover')[0]['rho'].sum() == 0
    torch.testing.assert_close(sample('correct')[0]['weight'], old)
    neutral = sample(strength=0)[0]
    torch.testing.assert_close(neutral['weight'], old)
    assert neutral['rho'].sum() == 0


def test_soft_gradient_and_read_only_audit():
    result, old, pred, mask = sample()
    logits = torch.randn(5,3,requires_grad=True)
    loss = bc_joint.loss(logits,pred,result)
    loss.backward()
    target = (1-result['rho'][:,None])*F.one_hot(pred,3)+result['rho'][:,None]*result['target']
    expected = result['weight'][:,None]*(logits.detach().softmax(-1)-target)/5
    torch.testing.assert_close(logits.grad,expected)
    before = result['weight'].clone()
    first = bc_joint.audit(result,old,mask,pred,pred,old)
    second = bc_joint.audit(result,old,mask,pred,(pred+1)%3,old)
    assert not torch.equal(first,second)
    torch.testing.assert_close(before,result['weight'])


@pytest.mark.parametrize('mode', ['recover','correct','joint'])
@pytest.mark.parametrize('phase,round_idx', [(1,45),(1,80),(3,150)])
def test_real_training_hidden_gt_isolation_and_loss_accounting(mode,phase,round_idx):
    name = 'bc_repair_'+mode
    weights,state,_ = run_local(phase,0,experiment=name,round_idx=round_idx)
    altered,_,_ = run_local(phase,0,experiment=name,round_idx=round_idx,flip_hidden_labels=True)
    for key in weights:
        torch.testing.assert_close(weights[key],altered[key],rtol=0,atol=0)
        assert torch.isfinite(weights[key]).all()
    row = dynamics_row(round_idx,phase,[state['log']],recipe(name))
    assert abs(row['loss_observed']-row['loss_reconstructed']) < 1e-6
    assert 'repair_c0_correction_fix_visits' in row


@pytest.mark.parametrize('mode', ['recover','correct','joint'])
def test_original_warmup_exact_and_checkpoint_isolation(mode):
    name='bc_repair_'+mode
    old,_,_ = run_local(1,0,experiment='trusted_lr015_bc',round_idx=30)
    new,_,_ = run_local(1,0,experiment=name,round_idx=30)
    for key in old: torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
    assert 'bc_a_repair' in _geometry_controls(recipe(name))
    assert 'bc_a_repair' not in _geometry_controls(recipe('trusted_lr015_bc'))
    assert bc_joint.gates(30)==(0.,0.)
    assert bc_joint.gates(120)==(1.,1.)


def test_launcher_dry_run_no_resume_or_tail():
    root=Path(__file__).resolve().parents[1]
    run=subprocess.run([sys.executable,str(root/'scripts/run_bc_repair.py'),
                        '--gpus','0','1','--dry-run'],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    for mode in ('recover','correct','joint'):
        assert f'experiment_bc_repair_{mode}.yaml' in run.stdout
    assert '--resume' not in run.stdout


def test_actual_training_exercises_correction_and_prototype_taper(monkeypatch):
    original = bc_joint.decisions

    def force_teacher(old, mask, pred, q, p, scores, z, ref, rg, cg, mode):
        teacher = F.one_hot((pred+1)%q.shape[1],q.shape[1]).float()
        return original(old,mask,pred,teacher,teacher,torch.ones_like(scores),
                        z,ref,rg,cg,mode)

    monkeypatch.setattr(bc_joint,'decisions',force_teacher)
    _,state,_ = run_local(3,0,experiment='bc_repair_correct',round_idx=150)
    log=state['log']
    selected=sum(log.get(f'repair_c{c}_correction_visits',0) for c in range(10))
    removed=sum(log.get(f'repair_c{c}_prototype_removed_{kind}',0)
                for c in range(10) for kind in ('correct','wrong'))
    assert selected > 0 and removed > 0
    assert log['proto_a_wrong'] == 0
    row=dynamics_row(150,3,[log],recipe('bc_repair_correct'))
    assert abs(row['loss_observed']-row['loss_reconstructed']) < 1e-6
