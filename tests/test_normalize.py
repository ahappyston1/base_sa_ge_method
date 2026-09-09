# -*- coding: utf-8 -*-
"""关闭随机增广后，有标 / 弱 / 强 / 测试四条路径归一化必须一致。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.dataset import Indices2Dataset_labeled, Indices2Dataset_unlabeled_fixmatch
from Dataset.normalize import to_tensor_normalize


class _OneImage:
    def __init__(self, img):
        self.img = img

    def __getitem__(self, idx):
        return self.img, 0

    def __len__(self):
        return 1


def test_four_paths_same_normalize_cifar10():
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    ds = _OneImage(img)
    lab = Indices2Dataset_labeled(ds, dataset_name="CIFAR10", augment=False)
    unl = Indices2Dataset_unlabeled_fixmatch(ds, dataset_name="CIFAR10", augment=False)
    lab.load([0])
    unl.load([0])
    x_lab, _ = lab[0]
    x_w, x_s, _, _ = unl[0]
    x_test = to_tensor_normalize("CIFAR10")(img)
    for a, b in ((x_lab, x_w), (x_w, x_s), (x_s, x_test)):
        assert torch.allclose(a, b, atol=1e-6, rtol=0), "四条路径归一化不一致"


if __name__ == "__main__":
    test_four_paths_same_normalize_cifar10()
    print("ok test_four_paths_same_normalize_cifar10")
