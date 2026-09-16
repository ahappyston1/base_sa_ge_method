import copy
import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import bc_reconstruction as rec
from options import args_parser
from fl_runner import _geometry_controls
from trusted_geometry import fit_reference
from training_dynamics import dynamics_row
from test_trusted_followups import recipe
from test_trusted_risk import run_local


def assert_weights_equal(first, second):
    assert first.keys() == second.keys()
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_rng_mask_and_real_backbone_gradient():
    from Model.resnet import ResNet
    state = torch.get_rng_state().clone()
    head = rec.initialize(256, 64, 'cpu')
    masked = rec.corrupt(torch.ones(4, 256), .25, 91)
    torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
    assert (masked == 0).sum(-1).tolist() == [64]*4
    torch.testing.assert_close(masked, rec.corrupt(torch.ones(4, 256), .25, 91))
    assert not torch.equal(masked, rec.corrupt(torch.ones(4, 256), .25, 92))
    model = ResNet(resnet_size=8, scaling=4, num_classes=10)
    source = F.normalize(model(torch.randn(4, 3, 32, 32))[0], dim=-1)
    teacher = torch.randn(4, 256, requires_grad=True)
    rec.objective(head, source, teacher, torch.tensor([True, False, True, False]), .25, 91).backward()
    assert teacher.grad is None
    assert model.conv1.weight.grad.abs().sum() > 0
    assert model.classifier.weight.grad is None
    assert head[0].weight.grad.abs().sum() > 0


def test_detached_probe_and_whole_batch_denominator():
    head = rec.initialize(8, 3, 'cpu')
    student = torch.randn(4, 8, requires_grad=True)
    teacher = torch.randn(4, 8, requires_grad=True)
    mask = torch.tensor([True, False, False, False])
    error = rec.errors(head, student, teacher, .25, 1)
    loss = rec.objective(head, student, teacher, mask, .25, 1, detached=True)
    torch.testing.assert_close(loss, error[0]/4)
    loss.backward()
    assert student.grad is None and teacher.grad is None
    assert rec.objective(head, student, teacher, mask & False, .25, 1) == 0


@pytest.mark.parametrize('phase', [1, 2, 3])
def test_audit_is_exact_baseline_even_after_activation(phase):
    baseline, base, _ = run_local(phase, 0, experiment='trusted_lr015_bc', a_threshold=.95)
    audit, state, _ = run_local(phase, 0, experiment='bc_rec_audit', a_threshold=.95)
    assert_weights_equal(baseline, audit)
    assert_weights_equal(base['bc_heads'], state['bc_heads'])
    assert base['log']['loss'] == state['log']['loss']
    assert state['rec_rows'] and state['rec_reference_rows']
    initial = rec.initialize(8, 64, 'cpu').state_dict()
    assert any(not torch.equal(initial[k], state['bc_rec_head'][k]) for k in initial)


@pytest.mark.parametrize('mode', ['audit', 'unmasked', 'masked'])
def test_prefix_hidden_labels_objectives_and_resume(mode, tmp_path):
    name = 'bc_rec_' + mode
    base, _, _ = run_local(1, 0, experiment='trusted_lr015_bc', round_idx=30)
    prefix, prefix_state, _ = run_local(1, 0, experiment=name, round_idx=30)
    assert_weights_equal(base, prefix)
    assert prefix_state['rec_rows'] == []
    for phase in (1, 2, 3):
        weights, state, _ = run_local(phase, 0, experiment=name, a_threshold=.95)
        flipped, flipped_state, _ = run_local(phase, 0, experiment=name, a_threshold=.95, flip_hidden_labels=True)
        assert_weights_equal(weights, flipped)
        assert_weights_equal(state['bc_rec_head'], flipped_state['bc_rec_head'])
        assert state['rec_reference_rows'] == flipped_state['rec_reference_rows']
        assert all(torch.isfinite(v).all() for v in weights.values())
        row = dynamics_row(45 if phase == 1 else 150, phase, [state['log']], recipe(name))
        assert abs(row['loss_reconstructed'] - row['loss_observed']) < 1e-6
        assert row['rec_objective'] > 0
        # Each scored visit appears once for each of the two diagnostic scores.
        assert sum(x['correct']+x['wrong'] for x in state['rec_rows']) == 2*state['log']['cnt_u']
    path = tmp_path/'client.pt'
    torch.save(state, path)
    restored = torch.load(path, weights_only=False)
    live, next_state, _ = run_local(3, 0, experiment=name, round_idx=151, a_threshold=.95,
                                  saved_state=state, global_weights=weights)
    resumed, next_restored, _ = run_local(3, 0, experiment=name, round_idx=151, a_threshold=.95,
                                        saved_state=restored, global_weights=weights)
    assert_weights_equal(live, resumed)
    assert_weights_equal(next_state['bc_rec_head'], next_restored['bc_rec_head'])
    assert next_state['rec_rows'] == next_restored['rec_rows']
    del restored['bc_rec_head']
    with pytest.raises(ValueError, match='decoder state'):
        run_local(3, 0, experiment=name, saved_state=restored)


def test_actual_masking_changes_student_and_fixed_inputs_preserve_class_losses():
    a, _, _ = run_local(3, 0, experiment='bc_rec_unmasked', a_threshold=.95)
    b, _, _ = run_local(3, 0, experiment='bc_rec_masked', a_threshold=.95)
    assert any(not torch.equal(a[k], b[k]) for k in a)
    _, base, _ = run_local(3, 0, frozen=True, experiment='trusted_lr015_bc', a_threshold=.95)
    _, new, _ = run_local(3, 0, frozen=True, experiment='bc_rec_masked', a_threshold=.95)
    for key in ('cnt_a', 'cnt_b', 'cnt_c', 'L_A', 'L_B', 'L_proto'):
        assert base['log'][key] == new['log'][key]


def test_error_bins_and_reference_queries():
    buffer = rec.new_audit(2, 'cpu')
    scores = dict(cross_view=torch.tensor([0., 4.]), teacher_self=torch.tensor([1., 2.]))
    rec.accumulate(buffer, scores, torch.tensor([0, 1]), torch.tensor([.96, .8]),
                   torch.tensor([True, False]), torch.tensor([False, True]), torch.tensor([0, 0]))
    rows = rec.audit_rows(buffer)
    assert len(rows) == 4
    assert sum(x['wrong'] for x in rows) == 2
    assert any(x['error_bin'] == rec.BINS-1 and x['wrong'] == 1 for x in rows)
    y = torch.tensor([0]*4 + [1]*4)
    z = F.one_hot(y, 2).float()
    ref = fit_reference(z, y, 2)
    audit = rec.reference_audit(z, y, ref)
    assert sum(x['query_count'] for x in audit) == 4
    assert sum(x['query_correct'] for x in audit) == 4
    assert all(x['own_distance_sum'] == 0 for x in audit)
    assert rec.reference_audit(z[:4], y[:4], fit_reference(z[:4], y[:4], 2)) == []


def test_configs_and_controls():
    base = recipe('trusted_lr015_bc')
    for mode in ('audit', 'unmasked', 'masked'):
        current = recipe('bc_rec_'+mode)
        differences = {k for k in vars(base) if getattr(base, k) != getattr(current, k)}
        assert differences == {'config', 'bc_reconstruction'}
        assert _geometry_controls(base) != _geometry_controls(current)
    for extra in (['--bc_teacher', '0'], ['--bc_targets', '1'], ['--bc_tail', 'aguard'],
                  ['--bc_rec_mask', 'nan'], ['--bc_rec_bottleneck', '256']):
        with patch.object(sys, 'argv', ['test', '--config', 'configs/experiment_bc_rec_masked.yaml']+extra):
            with pytest.raises(SystemExit):
                args_parser()


def test_binned_auroc_and_streaming_summary(tmp_path):
    from scripts.summarize_bc_reconstruction import histogram_summary, summarize
    assert histogram_summary({0: [2, 0], 39: [0, 3]})['binned_error_auroc'] == 1.
    assert histogram_summary({0: [0, 3], 39: [2, 0]})['binned_error_auroc'] == 0.
    assert histogram_summary({10: [2, 3]})['binned_error_auroc'] == .5
    assert histogram_summary({10: [2, 0]})['binned_error_auroc'] is None
    (tmp_path/'reconstruction_audit.csv').write_text(
        'round,score,predicted_class,bucket,confidence_band,error_bin,correct,wrong\n'
        '90,cross_view,0,A,2,39,0,20\n'
        '91,cross_view,0,A,2,1,4,0\n'
        '91,cross_view,0,A,2,30,0,5\n')
    (tmp_path/'reconstruction_reference.csv').write_text(
        'round,cls,query_count,query_correct,own_distance_sum,margin_sum\n91,0,4,3,0.8,0.4\n')
    result = summarize(tmp_path, 91, 300)
    assert len(result['reconstruction']) == 2
    assert all(x['wrong_visits'] == 5 and x['binned_error_auroc'] == 1 for x in result['reconstruction'])
    assert result['reference'][0]['query_accuracy'] == .75
