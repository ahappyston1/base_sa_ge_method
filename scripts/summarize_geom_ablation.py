#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 geom 消融 run：末 10 轮 Acc、阶段切换附近、diag.csv 关键列。"""
from __future__ import annotations

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "CIFAR10", "runs")
DEFAULT_IDS = ("geom_diag_s7", "geom_cur_s7", "geom_noboost_s7", "geom_nounk_s7")


def _read_csv(path: str):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def _f(row, key, default=None):
    if key not in row or row[key] in ("", None):
        return default
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return default


def _phase_change(rows, to_phase: int):
    for row in rows:
        if int(float(row.get("phase", 0) or 0)) == to_phase:
            return int(float(row["round"])), _f(row, "acc")
    return None, None


def _window(rows, center, half=5):
    if center is None:
        return []
    lo, hi = center - half, center + half
    return [r for r in rows if lo <= int(float(r["round"])) <= hi]


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def summarize(run_id: str) -> None:
    d = os.path.join(RUNS, f"{run_id}_a0.1")
    metrics = _read_csv(os.path.join(d, "metrics.csv"))
    diag = _read_csv(os.path.join(d, "diag.csv"))
    cfg_path = os.path.join(d, "config.json")
    cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf8") as f:
            cfg = json.load(f)

    print(f"\n=== {run_id} ===")
    if not os.path.isdir(d):
        print("  目录不存在")
        return
    gc = cfg.get("geometry_controls", {})
    print(
        f"  geom={gc} diag_geom={cfg.get('diag_geom')} "
        f"rounds_done={len(metrics)}/{cfg.get('num_rounds', '?')}"
    )
    if not metrics:
        print("  尚无 metrics.csv")
        return

    accs = [_f(r, "acc") for r in metrics]
    last10 = [v for v in accs[-10:] if v is not None]
    peak = max((v, int(float(r["round"]))) for r, v in zip(metrics, accs) if v is not None)
    print(
        f"  Acc last10={_fmt(_mean(last10))}  "
        f"peak={_fmt(peak[0])}@{peak[1]}  "
        f"final={_fmt(accs[-1])}@r{int(float(metrics[-1]['round']))}"
    )

    for ph in (2, 3):
        rnd, acc = _phase_change(metrics, ph)
        win = _window(metrics, rnd, 5)
        wacc = _mean([_f(r, "acc") for r in win])
        print(
            f"  Phase→{ph} @ r{rnd if rnd else '—'} acc={_fmt(acc)}  "
            f"±5轮均值={_fmt(wacc)}"
        )

    tail_m = metrics[-10:]
    print(
        f"  A-ratio={_fmt(_mean([_f(r, 'a_ratio') for r in tail_m]))}  "
        f"A-Prec={_fmt(_mean([_f(r, 'a_prec') for r in tail_m]))}  "
        f"B-ratio={_fmt(_mean([_f(r, 'b_ratio') for r in tail_m]))}"
    )

    if not diag:
        print("  无 diag.csv")
        return
    tail_d = diag[-10:]
    keys = (
        "sup_acc",
        "opp_acc",
        "geom_cov",
        "c_lowconf",
        "c_wouldB_geommiss",
        "c_wouldB_conflict",
        "tau_clipped_frac",
    )
    parts = []
    for k in keys:
        vals = [_f(r, k) for r in tail_d]
        nd = 1 if k.startswith("c_") else 4
        parts.append(f"{k}={_fmt(_mean(vals), nd)}")
    print("  diag last10: " + "  ".join(parts))


def main():
    ids = sys.argv[1:] or list(DEFAULT_IDS)
    for rid in ids:
        summarize(rid)


if __name__ == "__main__":
    main()
