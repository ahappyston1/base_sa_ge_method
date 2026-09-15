import copy
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from bc_tail import gate, reliability, reweight_b, guard_a, validate_fork
from test_trusted_risk import run_local
from test_trusted_followups import recipe
from training_dynamics import dynamics_row, lr_controls
from fl_runner import _geometry_controls


def test_schedule_and_objective_bounds():
    assert gate(225) == 0 and gate(260) == 1
    w = torch.tensor([.2, .5, .9, .1]); b = torch.tensor([True, True, True, False])
    scores = torch.tensor([0., .4, 1., .9])
    new = reweight_b(w, b, scores, 1.)
    torch.testing.assert_close(new.sum(), (w*b).sum())
    assert (new[b] >= .5*w[b]).all() and (new[b] <= 1.5*w[b]).all()
    assert reweight_b(w, b & False, scores, 1.).sum() == 0
    q = torch.tensor([[.01,.99],[.99,.01],[.3,.7],[.01,.99]])
    p = torch.tensor([[.01,.99],[.99,.01],[.01,.99],[.99,.01]])
    aw, selected = guard_a(w, torch.ones(4,dtype=torch.bool), torch.zeros(4,dtype=torch.long), q, p, 1.)
    assert selected.tolist() == [True,False,False,False]
    torch.testing.assert_close(aw, torch.tensor([.16,.5,.9,.1]))


def test_teacher_scores_stop_grad_and_consistency():
    x = torch.tensor([[8.,0.],[8.,0.]], requires_grad=True)
    y = torch.tensor([[8.,0.],[0.,8.]], requires_grad=True)
    scores, q, p = reliability(x,y)
    assert scores[0] > .99 and scores[1] < .01
    assert not scores.requires_grad and not q.requires_grad


@pytest.mark.parametrize('mode', ['breliability','aguard','featurehalf'])
def test_actual_tail_training_prefix_hidden_labels_and_accounting(mode):
    original, _, _ = run_local(3,0,experiment='trusted_lr015_bc',round_idx=225,a_threshold=.95)
    weights, _, _ = run_local(3,0,experiment='bc_tail_'+mode,round_idx=225,a_threshold=.95)
    for k in weights: torch.testing.assert_close(weights[k], original[k], rtol=0,atol=0)
    weights, state, _ = run_local(3,0,experiment='bc_tail_'+mode,round_idx=260,a_threshold=.95)
    flipped, _, _ = run_local(3,0,experiment='bc_tail_'+mode,round_idx=260,a_threshold=.95,flip_hidden_labels=True)
    for k in weights:
        torch.testing.assert_close(weights[k],flipped[k],rtol=0,atol=0)
        assert torch.isfinite(weights[k]).all()
    args=recipe('bc_tail_'+mode)
    row=dynamics_row(260,3,[state['log']],args)
    assert abs(row['loss_reconstructed']-row['loss_observed']) < 1e-6
    if mode=='breliability':
        assert abs(row['tail_b_old_mass']-row['tail_b_new_mass']) < 1e-4
    assert 'bc_heads' in state


def checkpoint():
    args=recipe('trusted_lr015_bc')
    identity=dict(num_clients=2)
    state={k:{} for k in ('bc_heads','rho_bar','rho_valid','p_loc','loc_valid','inited')}
    c={k:torch.tensor([1]) for k in ('global_model','p_ref','p_ref_valid','np_random_state',
        'torch_rng_state','python_rng_state','numpy_global_rng_state','cuda_rng_state','schedule')}
    c.update(round=225,method_rev=13,geometry_controls=_geometry_controls(args),lr_controls=lr_controls(args),
             client_states={0:copy.deepcopy(state),1:copy.deepcopy(state)},fedavg_acc=[.8]*225,run_identity=identity)
    return c,identity


def test_fork_rejects_wrong_round_partial_state_and_incompatible_recipe(tmp_path):
    c,identity=checkpoint(); source=str(tmp_path/'source'/'round_0225.pt'); dest=str(tmp_path/'new')
    target=dict(c['geometry_controls'],bc_tail={'mode':'aguard'})
    validate_fork(c,target,c['lr_controls'],source,dest,identity)
    for mutation in ('round','head','geometry','workers','history'):
        bad=copy.deepcopy(c)
        if mutation=='round':bad['round']=300
        elif mutation=='head':del bad['client_states'][0]['bc_heads']
        elif mutation=='geometry':bad['geometry_controls']['bc_ema']=.9
        elif mutation=='workers':bad['run_identity']['num_clients']=3
        else:bad['fedavg_acc']=[]
        with pytest.raises(ValueError):validate_fork(bad,target,c['lr_controls'],source,dest,identity)
    with pytest.raises(ValueError):validate_fork(c,target,c['lr_controls'],source,str(tmp_path/'source'),identity)


def test_second_view_preserves_base_inputs_and_rng():
    from Dataset.dataset import Indices2Dataset_unlabeled_fixmatch
    from test_trusted_multi import Pictures
    data=Indices2Dataset_unlabeled_fixmatch(Pictures(),dataset_name='CIFAR10')
    data.load([0,1])
    def sample(second):
        torch.manual_seed(12);random.seed(13);np.random.seed(14)
        data.second_weak=second;out=data[0]
        return out,torch.get_rng_state(),random.getstate(),np.random.get_state()
    a,ta,pa,na=sample(False);b,tb,pb,nb=sample(True)
    for i in [0,1]:torch.testing.assert_close(a[i],b[i],rtol=0,atol=0)
    torch.testing.assert_close(ta,tb,rtol=0,atol=0)
    assert pa==pb and np.array_equal(na[1],nb[1]) and len(b)==5


def test_checkpoint_serialization_replays_next_round(tmp_path):
    """Real CPU optimizer + client head/prototype state through torch.save/load."""
    weights, state, _ = run_local(3,0,experiment='trusted_lr015_bc',round_idx=225,a_threshold=.95)
    path=tmp_path/'round_0225.pt'
    torch.save({'global_model':weights,'client_states':state},path)
    saved=torch.load(path,weights_only=False)
    expected, expected_state, _=run_local(3,0,experiment='trusted_lr015_bc',round_idx=226,
        a_threshold=.95,saved_state=state,global_weights=weights)
    actual, actual_state, _=run_local(3,0,experiment='trusted_lr015_bc',round_idx=226,
        a_threshold=.95,saved_state=saved['client_states'],global_weights=saved['global_model'])
    for key in actual:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    for key in actual_state['bc_heads']:
        torch.testing.assert_close(actual_state['bc_heads'][key],expected_state['bc_heads'][key],rtol=0,atol=0)


def test_cli_dry_run_validates_without_launching_training(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    from bc_tail import run_identity
    args=recipe('trusted_lr015_bc');args.num_labeled=500
    identity=run_identity(args)
    c,_=checkpoint();c['run_identity']=identity
    c['client_states']={i:copy.deepcopy(c['client_states'][0]) for i in range(20)}
    source=tmp_path/'round_0225.pt';torch.save(c,source)
    command=[sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/bc_tail_experiment.py'),
             'breliability','--checkpoint',str(source),'--dry-run']
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--bc_fork 1' in result.stdout and 'experiment_bc_tail_breliability.yaml' in result.stdout
    c['round']=300;torch.save(c,source)
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode!=0 and 'round 225' in result.stderr
