#!/usr/bin/env bash
# 短跑冒烟：5 轮、更少客户端。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exec bash scripts/train.sh --config configs/debug_short.yaml --dataset CIFAR10 --alpha 0.1 "$@"
