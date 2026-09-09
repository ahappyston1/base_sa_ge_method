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
