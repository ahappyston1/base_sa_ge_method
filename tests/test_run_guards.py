# -*- coding: utf-8 -*-
"""启动守卫：IID 索引列表、空 DataLoader、Auniq 分母定义。"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_runner import (  # noqa: E402
    _as_index_list,
    _assert_clients_can_batch,
    _diag_geom_c,
    _diag_row,
    _proto_write_weight,
    _unique_a_mass,
)


def test_iid_ndarray_can_extend_labeled():
    unl = _as_index_list(np.array([1, 2, 3]))
    lab = _as_index_list(np.array([10, 11]))
    unl.extend(lab)
    assert unl == [1, 2, 3, 10, 11]


def test_empty_unlabeled_raises():
    args = SimpleNamespace(batch_size_local_labeled_fixmatch=128, mu=2)
    lab = [[0, 1, 2]]
    unl = [[0, 1, 2]]  # after merge still < 256
    try:
        _assert_clients_can_batch(lab, unl, args)
    except RuntimeError as e:
        assert "unlabeled=3" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_enough_samples_ok():
    args = SimpleNamespace(batch_size_local_labeled_fixmatch=4, mu=2)
    lab = [list(range(2))]
    unl = [list(range(8))]
    _assert_clients_can_batch(lab, unl, args)


def test_proto_write_default_is_noop():
    # 默认 floor=0, extra=0：权重原样返回、无筛选掩码，保证与旧行为逐位一致
    args = SimpleNamespace(w_min=0.05, proto_conf_floor=0.0, proto_w_extra=0.0)
    base = torch.tensor([1.0, 0.3, 0.7])
    s = torch.tensor([0.99, 0.60, 0.80])
    w, keep = _proto_write_weight(base, s, args)
    assert keep is None
    assert torch.equal(w, base)


def test_proto_write_extra_downweights_low_conf():
    # extra>0：w = base * s^extra，低置信样本被更狠地压低，且不改 CE（此函数只管原型权重）
    args = SimpleNamespace(w_min=0.05, proto_conf_floor=0.0, proto_w_extra=2.0)
    base = torch.tensor([1.0, 1.0])
    s = torch.tensor([1.0, 0.5])
    w, keep = _proto_write_weight(base, s, args)
    assert keep is None
    assert torch.allclose(w, torch.tensor([1.0, 0.25]), atol=1e-6)


def test_proto_write_floor_selects_high_conf_only():
    # floor>0：低于门槛的样本被硬性剔除出原型写入
    args = SimpleNamespace(w_min=0.05, proto_conf_floor=0.7, proto_w_extra=0.0)
    base = torch.tensor([1.0, 1.0, 1.0])
    s = torch.tensor([0.95, 0.60, 0.71])
    w, keep = _proto_write_weight(base, s, args)
    assert keep.tolist() == [True, False, True]


def test_diag_geom_splits_support_oppose_and_c_reasons():
    # 4 个样本，均落在置信带 [0.95,0.99)
    #  s0: geom 支持(m>=delta) 且正确
    #  s1: geom 反对(m<delta) 且错误  -> 反对组更容易错
    #  s2: 进 C，满足置信版 B(s>=eta_B) 但几何缺失(geom_ok=False)
    #  s3: 进 C，满足置信版 B 且有效几何冲突(m<0)，分类器错、原型对
    s = torch.tensor([0.96, 0.97, 0.96, 0.98])
    yhat = torch.tensor([0, 1, 2, 3])
    gt = torch.tensor([0, 0, 2, 2])          # s1 错，s3 错(分类器)
    m = torch.tensor([0.5, -0.5, 0.0, -0.3])
    delta = torch.tensor([0.1, 0.1, 0.1, 0.1])
    geom_ok = torch.tensor([True, True, False, True])
    in_C = torch.tensor([False, False, True, True])
    # 原型：让 s3 的最近原型指向真类 2
    p_mix = torch.eye(4)
    zw = torch.tensor([
        [1.0, 0, 0, 0],
        [0, 1.0, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 1.0, 0],   # s3 特征最像类 2 原型 -> 原型预测=2=gt
    ])
    mix_valid = torch.tensor([True, True, True, True])
    d = _diag_geom_c(s, yhat, gt, m, delta, geom_ok, in_C, eta_B=0.6,
                     zw=zw, p_mix=p_mix, mix_valid=mix_valid, s_lo=0.95, s_hi=0.99)
    # 几何判别覆盖置信带内所有 geom_ok 样本，不受路由影响：
    # 支持组 = {s0}；反对组 = {s1, s3}（均 m<delta），两者都错 -> 反对组 0% 对
    assert d["dg_sup_n"] == 1 and d["dg_sup_cor"] == 1
    assert d["dg_opp_n"] == 2 and d["dg_opp_cor"] == 0
    assert d["dc_wouldB"] == 2
    assert d["dc_wouldB_geommiss"] == 1                    # s2
    assert d["dc_wouldB_conflict"] == 1                    # s3，b_m_thr 默认 0
    assert d["dconf_n"] == 1
    assert d["dconf_clf_cor"] == 0                         # 分类器错
    assert d["dconf_proto_cor"] == 1                       # 原型对
    # Phase2 中途 b_m_thr<0 时，m=-0.3 还没到 B 间隔门槛，不能记成「几何挡住 B」
    d2 = _diag_geom_c(s, yhat, gt, m, delta, geom_ok, in_C, eta_B=0.6,
                      zw=zw, p_mix=p_mix, mix_valid=mix_valid, s_lo=0.95, s_hi=0.99,
                      b_m_thr=-1.0)
    assert d2["dc_wouldB_conflict"] == 0
    assert d2["dconf_n"] == 1                              # 谁更常对仍按 m<0


def test_diag_row_derives_ratios_and_other_bucket():
    agg = {
        "dg_band_n": 100, "dg_band_geom_n": 80,
        "dg_sup_n": 50, "dg_sup_cor": 45,
        "dg_opp_n": 30, "dg_opp_cor": 9,
        "dc_lowconf": 200, "dc_wouldB": 40,
        "dc_wouldB_geommiss": 10, "dc_wouldB_conflict": 6,
        "dconf_n": 6, "dconf_clf_cor": 2, "dconf_proto_cor": 4,
    }
    row = _diag_row(r=5, phase=3, cnt_u=1000, agg=agg)
    assert row["geom_cov"] == 0.8
    assert row["sup_acc"] == 0.9
    assert abs(row["opp_acc"] - 0.3) < 1e-9
    assert row["c_wouldB_other"] == 40 - 10 - 6          # r_i 降级等
    assert abs(row["conflictC_frac"] - 0.006) < 1e-9
    assert row["conflictC_proto_acc"] > row["conflictC_clf_acc"]


def test_unique_a_mass_denom_is_class_a_hits_not_all_visits():
    # 样本 10 访问多次，仅两次进 A（类 0），一次进 A（类 1）；未进 A 的访问不会传入
    ids = torch.tensor([10, 10, 10])
    yhat = torch.tensor([0, 0, 1])
    w = torch.tensor([1.0, 0.4, 0.1])
    mass = _unique_a_mass(ids, yhat, w, num_classes=2)
    # 类0均值 0.7 > 类1均值 0.1 → 归类0，贡献 0.7
    assert torch.allclose(mass, torch.tensor([0.7, 0.0]), atol=1e-6)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} passed")
