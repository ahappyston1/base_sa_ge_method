#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPFPSL 联邦训练主入口。

推荐用脚本：
  bash scripts/train.sh --dataset CIFAR10 --alpha 0.1 --gpu_id 0
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


if __name__ == '__main__':
    import random

    import numpy as np
    import torch

    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    torch.backends.cudnn.deterministic = True

    from options import args_parser

    import fl_runner

    args = args_parser()
    fl_runner.fixmatch(args.alpha)
