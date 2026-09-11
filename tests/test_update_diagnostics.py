"""Small CPU tensor checks; no dataset or training required."""
import math
import torch

from update_diagnostics import update_rows


def run(deltas, counts):
    base = {"weight": torch.tensor([10., 10.]),
            "bn.running_mean": torch.tensor([0.]),
            "bn.running_var": torch.tensor([1.]),
            "bn.num_batches_tracked": torch.tensor(0)}
    locals_ = [{k: v.clone() for k, v in base.items()} for _ in deltas]
    for state, delta in zip(locals_, deltas):
        state["weight"] += torch.tensor(delta)
        state["bn.running_mean"] += 2
        state["bn.num_batches_tracked"] += 100
    fused = {k: sum(state[k] * n for state, n in zip(locals_, counts)) / sum(counts)
             for k in base}
    before = [{k: v.clone() for k, v in state.items()} for state in [base, *locals_, fused]]
    rng = torch.get_rng_state().clone()
    summary, rows = update_rows(base, locals_, fused, counts, [4, 7], ["weight"])
    assert torch.equal(rng, torch.get_rng_state())
    for state, snapshot in zip([base, *locals_, fused], before):
        assert all(torch.equal(state[k], snapshot[k]) for k in state)
    return summary, rows


def test_identical_updates_and_bn_separation():
    s, rows = run([[3., 4.], [3., 4.]], [1, 1])
    assert s["params_global_norm"] == 5
    assert s["params_disagreement_rms"] == 0
    assert s["params_retained_ratio"] == 1
    assert rows[0]["params_cos_to_fused"] == 1
    assert s["bn_mean_global_norm"] == 2
    assert math.isnan(s["bn_mean_global_relative"])
    assert not any("tracked" in k for k in s)


def test_opposite_updates_cancel():
    s, rows = run([[3., 4.], [-3., -4.]], [1, 1])
    assert s["params_global_norm"] == 0
    assert s["params_disagreement_rms"] == 5
    assert s["params_disagreement_relative"] == 1
    assert s["params_retained_ratio"] == 0
    assert math.isnan(rows[0]["params_cos_to_fused"])


def test_fedavg_unequal_weights():
    s, rows = run([[3., 4.], [-3., -4.]], [3, 1])
    assert s["params_global_norm"] == 2.5
    assert math.isclose(s["params_disagreement_rms"], math.sqrt(18.75))
    assert s["params_retained_ratio"] == .5
    assert rows[0]["fedavg_weight"] == .75
    assert rows[1]["params_cos_to_fused"] == -1


def test_no_update_is_undefined_direction_not_conflict():
    s, rows = run([[0., 0.], [0., 0.]], [1, 1])
    assert s["params_global_norm"] == 0
    assert math.isnan(s["params_retained_ratio"])
    assert math.isnan(rows[1]["params_cos_to_fused"])
