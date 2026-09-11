"""Tensor tests for support/query separation and trust-controlled geometry."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from trusted_geometry import fit_reference, score_queries


def test_reference_uses_support_not_queries_for_centers():
    z = torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.],
                      [-1.,0.],[0.,-1.],[-1.,0.],[0.,-1.]])
    y = torch.tensor([0,0,0,0,1,1,1,1])
    ref = fit_reference(z,y,2)
    assert torch.equal(ref['centers'],torch.tensor([[1.,0.],[-1.,0.]]))
    assert ref['query_n'].tolist() == [2.,2.]


def test_insufficient_competitors_have_zero_authority():
    ref = fit_reference(torch.tensor([[1.,0.]]*4), torch.zeros(4,dtype=torch.long), 2)
    out = score_queries(torch.tensor([[1.,0.]]),torch.tensor([0]),ref,1.)
    assert not out['valid'].item()
    assert out['authority'].item() == 0
    assert out['weight'].item() == 1


def test_gate_zero_and_zero_trust_are_noops():
    z = torch.tensor([[1.,0.]]*12+[[-1.,0.]]*12)
    y = torch.tensor([0]*12+[1]*12)
    ref = fit_reference(z,y,2)
    out = score_queries(z,y,ref,0.)
    assert torch.equal(out['weight'],torch.ones(24))
    ref['trust'].zero_()
    out = score_queries(z,y,ref,1.)
    assert torch.equal(out['weight'],torch.ones(24))


def test_disagreement_reduces_weight_and_age_reduces_authority():
    z = torch.tensor([[1.,0.]]*20+[[-1.,0.]]*20)
    y = torch.tensor([0]*20+[1]*20)
    ref = fit_reference(z,y,2)
    qz = torch.tensor([[1.,0.],[-1.,0.]])
    pred = torch.tensor([0,0])
    fresh = score_queries(qz,pred,ref,1.)
    old = score_queries(qz,pred,ref,1.,1.)
    assert fresh['weight'][0] > fresh['weight'][1]
    assert (old['authority'] < fresh['authority']).all()
    assert (fresh['weight'] > 0).all()
    assert (fresh['weight'] <= 1).all()


def test_inputs_are_not_mutated_and_query_has_no_gt_argument():
    import inspect
    z = torch.tensor([[1.,0.]]*8+[[-1.,0.]]*8)
    y = torch.tensor([0]*8+[1]*8)
    copy = z.clone()
    ref = fit_reference(z,y,2)
    score_queries(z,y,ref,1.)
    assert torch.equal(z,copy)
    assert 'gt' not in inspect.signature(score_queries).parameters


def test_refresh_restores_train_mode_without_bn_or_rng_updates():
    from types import SimpleNamespace
    from trusted_geometry import refresh_reference
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(2)
        def forward(self, x):
            z = self.bn(x)
            return z, z
    model = Model().train()
    rows = [(torch.tensor([1.,0.]),0)]*4 + [(torch.tensor([-1.,0.]),1)]*4
    dataset = SimpleNamespace(client_dataset_original_len=8, indices=list(range(8)),
                              client_dataset=rows*2000)
    mean, var = model.bn.running_mean.clone(), model.bn.running_var.clone()
    rng = torch.get_rng_state().clone()
    ref = refresh_reference(model,dataset,lambda x:x,2,'cpu',batch_size=3)
    assert model.training
    assert torch.equal(model.bn.running_mean,mean)
    assert torch.equal(model.bn.running_var,var)
    assert torch.equal(torch.get_rng_state(),rng)
    assert ref['support_n'].sum().item() == 4
    assert ref['query_n'].sum().item() == 4
