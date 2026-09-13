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
    from options import args_parser
    args = args_parser()
    if args.experiment_engine == 'trusted_multi':
        from trusted_multi_runner import run
        run(args)
    else:
        import fl_runner
        fl_runner.apply_run_seeds(args)
        fl_runner.fixmatch(args.alpha)
