# -*- coding: utf-8 -*-
"""各数据集有标 / 无标弱 / 无标强 / 测试共用的归一化常量。

CIFAR10 沿用原训练 std (约 0.25)，不再使用测试侧约 0.20 的 std。
"""
from __future__ import annotations

from typing import Tuple

from torchvision.transforms import transforms

# dataset -> (mean, std)
NORMALIZE_STATS = {
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    "CIFAR100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "SVHN": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "CINIC10": ((0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)),
}


def normalize_stats(dataset: str) -> Tuple[tuple, tuple]:
    key = str(dataset).upper()
    if key not in NORMALIZE_STATS:
        raise KeyError(f"未知数据集归一化: {dataset}")
    return NORMALIZE_STATS[key]


def normalize_transform(dataset: str):
    mean, std = normalize_stats(dataset)
    return transforms.Normalize(mean, std)


def to_tensor_normalize(dataset: str):
    """无随机增广：ToTensor + Normalize。测试与关闭增广的训练路径共用。"""
    return transforms.Compose([
        transforms.ToTensor(),
        normalize_transform(dataset),
    ])
