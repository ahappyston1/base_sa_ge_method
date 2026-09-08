# -*- coding: utf-8 -*-
"""
CINIC-10：由 CIFAR 与 ImageNet 子集组成的 10 类数据集，文件夹布局与 ImageFolder 一致。

SAGE.py 期望 root 下存在子文件夹 train/、test/（或 valid），本类内部会拼成 root/split。
"""
from torchvision.datasets import ImageFolder
import os


class CINIC10:
    """
    CINIC-10 轻量封装，委托给 torchvision.ImageFolder。

    Args:
        root: 数据集根目录，其下应有 train、valid、test 等子目录。
        split: 'train' / 'valid' / 'test'，与 root 拼接为实际图像路径。
        transform, target_transform: 与 torchvision 含义相同。
    """

    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = os.path.join(root, split)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = ImageFolder(self.root, transform=self.transform, target_transform=self.target_transform)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

