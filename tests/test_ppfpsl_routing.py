# -*- coding: utf-8 -*-
"""固定模拟输入：ABC 互斥覆盖、无原型不虚高、g=0 对齐 Phase1、g=0.5/P3 突变。不含 GT。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_runner import (  # noqa: E402
    _blend_alpha,
    _geom_margin,
    _margin_reliability,
    _route_phase1,
    _route_phase23,
    _unique_a_mass,
    _unlabeled_ab_losses,
)


def _assert_partition(in_A, in_B, in_C, n):
    s = in_A.int() + in_B.int() + in_C.int()
    assert int(s.sum()) == n and bool((s == 1).all()), "A/B/C 必须互斥且覆盖"


class _Args:
    alpha0 = 0.80
    alpha_min = 0.30
    alpha_max = 0.95
    pp_gamma = 1.0


def test_phase1_exclusive_and_cover():
    s = torch.tensor([0.99, 0.96, 0.80, 0.50, 0.20])
    a, b, c, p = _route_phase1(s, tau=0.95, eta_B=0.60, aux_on=True, a_cap=0.0, b_cap=0.0)
    _assert_partition(a, b, c, 5)
    assert a.tolist() == [True, True, False, False, False]
    assert b.tolist() == [False, False, True, False, False]
    assert c.tolist() == [False, False, False, True, True]
    assert p.equal(a)  # cap=0 时 pass_s 即 A


def test_phase1_aux_off_no_B():
    s = torch.tensor([0.99, 0.80, 0.20])
    a, b, c, _ = _route_phase1(s, 0.95, 0.60, aux_on=False, a_cap=0.0, b_cap=0.0)
    _assert_partition(a, b, c, 3)
    assert not bool(b.any())
    assert c.tolist() == [False, True, True]


def test_no_competitor_not_high_reliability():
    # 仅类 0 有原型；样本全预测类 0。旧实现 max_other=-1e9 → m≈1e9、m̃≈1
    zw = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.2]]), dim=-1)
    yhat = torch.tensor([0, 0])
    p_mix = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 0.0]]), dim=-1)
    mix_valid = torch.tensor([True, False])
    m_i, geom_ok, own_ok = _geom_margin(zw, yhat, p_mix, mix_valid)
    assert bool(own_ok.all())
    assert not bool(geom_ok.any())
    assert torch.allclose(m_i, torch.zeros(2))
    m_tilde = _margin_reliability(m_i, 0.05, 0.08)
    m_tilde = torch.where(geom_ok, m_tilde, torch.full_like(m_tilde, 0.5))
    assert torch.allclose(m_tilde, torch.full((2,), 0.5))
    assert float(m_tilde.max()) < 0.9


def test_no_own_proto_not_high_reliability():
    zw = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0]]), dim=-1)
    yhat = torch.tensor([1])
    p_mix = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 0.0]]), dim=-1)
    mix_valid = torch.tensor([True, False])
    m_i, geom_ok, own_ok = _geom_margin(zw, yhat, p_mix, mix_valid)
    assert not bool(own_ok.item())
    assert not bool(geom_ok.item())
    assert float(m_i.item()) == 0.0


def test_defined_margin_positive():
    zw = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    yhat = torch.tensor([0])
    p_mix = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    mix_valid = torch.tensor([True, True])
    m_i, geom_ok, own_ok = _geom_margin(zw, yhat, p_mix, mix_valid)
    assert bool(geom_ok.item()) and bool(own_ok.item())
    assert float(m_i.item()) > 0.5


def _p23(s, m, mt, r, tau, delta, own, g, eta=0.60, geom_ok=None):
    return _route_phase23(
        s, m, mt, r, tau, delta, own, g, eta, 0.0, 0.0, 0.0, -2.0, geom_ok
    )


def test_g0_phase2_matches_phase1_routing():
    torch.manual_seed(0)
    s = torch.tensor([0.99, 0.96, 0.80, 0.55, 0.10])
    a1, b1, c1, _ = _route_phase1(s, 0.95, 0.60, True, 0.0, 0.0)
    n = s.numel()
    m = torch.zeros(n)  # 未定义间隔
    mt = torch.full((n,), 0.5)
    r = s.clone()
    tau = torch.full((n,), 0.95)
    delta = torch.full((n,), -2.0)
    own = torch.zeros(n, dtype=torch.bool)
    out = _p23(s, m, mt, r, tau, delta, own, g=0.0, geom_ok=torch.zeros(n, dtype=torch.bool))
    _assert_partition(out["in_A"], out["in_B"], out["in_C"], n)
    assert out["in_A"].equal(a1)
    assert out["in_B"].equal(b1)
    assert out["in_C"].equal(c1)


def test_g05_own_ok_becomes_hard():
    s = torch.tensor([0.99, 0.99])
    m = torch.tensor([0.5, 0.5])
    mt = torch.tensor([0.9, 0.9])
    r = s.clone()
    tau = torch.full((2,), 0.95)
    delta = torch.full((2,), 0.0)
    own = torch.tensor([True, False])
    geom = torch.tensor([True, False])
    lo = _p23(s, m, mt, r, tau, delta, own, g=0.49, geom_ok=geom)
    hi = _p23(s, m, mt, r, tau, delta, own, g=0.50, geom_ok=geom)
    assert lo["in_A"].tolist() == [True, True]
    assert hi["in_A"].tolist() == [True, False]
    _assert_partition(hi["in_A"], hi["in_B"], hi["in_C"], 2)


def test_unknown_own_valid_no_competitor_not_killed_by_zero_margin():
    """本类有效、无竞争类：未知几何回退置信度，不因 m=0 且 δ>0 被踢出 A。"""
    s = torch.tensor([0.99])
    m = torch.tensor([0.0])
    mt = torch.tensor([0.5])
    r = s.clone()
    tau = torch.tensor([0.95])
    delta = torch.tensor([0.12])
    own = torch.tensor([True])
    geom = torch.tensor([False])
    for g in (0.0, 0.3, 0.5, 1.0):
        out = _p23(s, m, mt, r, tau, delta, own, g=g, geom_ok=geom)
        _assert_partition(out["in_A"], out["in_B"], out["in_C"], 1)
        assert bool(out["in_A"].item()), f"未知几何不应在 g={g} 因 m=0 掉出 A"
        assert not bool(out["fail_geom"].item())


def test_unknown_own_invalid_across_gates():
    """本类无效：未知几何不虚高；g<0.5 仍可按置信度进 A，g≥0.5 被 own_ok 挡住。"""
    s = torch.tensor([0.99])
    m = torch.tensor([0.0])
    mt = torch.tensor([0.5])
    r = s.clone()
    tau = torch.tensor([0.95])
    delta = torch.tensor([0.12])
    own = torch.tensor([False])
    geom = torch.tensor([False])
    for g in (0.0, 0.3):
        out = _p23(s, m, mt, r, tau, delta, own, g=g, geom_ok=geom)
        assert bool(out["in_A"].item()), f"g={g} 缺本类原型仍应对齐 Phase1 进 A"
    for g in (0.5, 1.0):
        out = _p23(s, m, mt, r, tau, delta, own, g=g, geom_ok=geom)
        _assert_partition(out["in_A"], out["in_B"], out["in_C"], 1)
        assert not bool(out["in_A"].item()), f"g={g} 缺本类原型不应进 A"


def test_g0_phase2_matches_phase1_losses():
    torch.manual_seed(0)
    n, c = 8, 10
    logs = torch.randn(n, c)
    logw = torch.randn(n, c)
    s, yhat = torch.softmax(logw, dim=-1).max(dim=-1)
    a1, b1, _, _ = _route_phase1(s, 0.95, 0.60, True, 0.0, 0.0)
    tau = torch.full((n,), 0.95)
    delta = torch.full((n,), -2.0)
    own = torch.zeros(n, dtype=torch.bool)
    geom = torch.zeros(n, dtype=torch.bool)
    m = torch.zeros(n)
    mt = torch.full((n,), 0.5)
    out = _p23(s, m, mt, s.clone(), tau, delta, own, g=0.0, geom_ok=geom)
    assert out["in_A"].equal(a1) and out["in_B"].equal(b1)
    w1 = a1.float()
    w2 = torch.zeros_like(s)
    w2[out["in_A"]] = 1.0
    assert torch.allclose(w1, w2)
    assert torch.allclose(out["b_score"], s)
    L_A1, L_B1 = _unlabeled_ab_losses(logs, logw, yhat, a1, b1, w1, s, T=1.0)
    L_A2, L_B2 = _unlabeled_ab_losses(
        logs, logw, yhat, out["in_A"], out["in_B"], w2, out["b_score"], T=1.0
    )
    assert torch.allclose(L_A1, L_A2) and torch.allclose(L_B1, L_B2)
    loss1 = L_A1 + L_B1
    loss2 = L_A2 + L_B2
    assert torch.allclose(loss1, loss2)


def test_unique_a_mass_averages_repeats_then_assigns_class():
    ids = torch.tensor([10, 10, 10, 11])
    yhat = torch.tensor([0, 0, 1, 1])
    w = torch.tensor([1.0, 0.4, 0.2, 0.9])
    mass = _unique_a_mass(ids, yhat, w, num_classes=3)
    # id10: class0 mean 0.7, class1 mean 0.2 → 归 0，贡献 0.7
    # id11: class1 mean 0.9
    assert torch.allclose(mass, torch.tensor([0.7, 0.9, 0.0]), atol=1e-6)


def test_phase3_boost_tightens_tau_not_abc_logic():
    nr = torch.tensor([0.0, 1.0])
    tau0, a, boost_c = 0.92, 0.08, 0.15
    tau_p2 = tau0 + a * nr
    tau_p3 = tau0 + a * nr * (1.0 + boost_c * nr)
    assert torch.allclose(tau_p2[0], tau_p3[0])
    assert float(tau_p3[1]) > float(tau_p2[1])


def test_route_functions_have_no_gt_argument():
    import inspect
    assert "gt" not in inspect.signature(_route_phase1).parameters
    assert "y_u_gt" not in inspect.signature(_route_phase23).parameters
    assert "gt" not in inspect.signature(_geom_margin).parameters


def test_blend_alpha_g0_ignores_pressure():
    args = _Args()
    nr = torch.tensor([0.0, 1.0])
    a0 = _blend_alpha(0.0, nr, args)
    assert torch.allclose(a0[0], a0[1])


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} passed")
