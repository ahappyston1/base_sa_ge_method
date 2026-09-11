"""Diagnostic bin boundaries and conservation, no change to model inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from geometry_audit import accumulate, new_buffer, EDGES


def test_geometry_audit_conserves_counts_weights_and_boundary_bins():
    s = torch.tensor([.95, .99, .995, 1.0])
    original = s.clone()
    pred = torch.tensor([0, 0, 1, 1])
    gt = torch.tensor([0, 1, 0, 0])
    m = torch.tensor([.2, -.1, 0., .3])
    valid = torch.tensor([True, True, False, True])
    a = torch.tensor([True, False, False, True])
    w = torch.tensor([.5, 0., 0., .8])
    buf = new_buffer(2, 'cpu')
    accumulate(buf, s, pred, gt, m, torch.full_like(m, .1), valid, a, w)
    totals = buf.sum(0)
    assert totals[0].item() == 4
    assert totals[1].item() == 1
    assert totals[3].item() == 2
    assert abs(totals[5].item() - 1.3) < 1e-6
    assert abs(totals[6].item() - .8) < 1e-6
    assert torch.equal(s, original)
    n_bins = len(EDGES) - 1
    assert buf[(0 * n_bins + 3) * 3 + 0, 0].item() == 1  # .95 support
    assert buf[(0 * n_bins + 5) * 3 + 1, 0].item() == 1  # .99 oppose
    assert buf[(1 * n_bins + 6) * 3 + 2, 0].item() == 1  # .995 unknown
    assert buf[(1 * n_bins + 6) * 3 + 0, 0].item() == 1  # 1.0 support
