# -*- coding: utf-8 -*-
"""metrics.csv 的列顺序与分组：进度在最前，原始计数在最后。"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_runner import _append_metrics_row, _metrics_row  # noqa: E402

SNAP = {"tau0": 0.92, "delta0": 0.10, "eta_B": 0.60, "tau_warmup": 0.95}


def _row(r=1, acc=0.5):
    return _metrics_row(
        r=r, phase=2, gate=0.5, acc=acc, best_acc=acc, best_round=r,
        losses=(1.0, 0.9, 0.05, 0.04, 2.0),
        route=(0.2, 0.3, 0.5, 0.25, 0.1),
        quality=(0.9, 0.7, 0.03),
        proto=(0.1, 0.05, 12.5, 100.0, 20.0, 9.9),
        rel=(0.5, 0.2, 0.8),
        knobs=(SNAP, 1.0, 1.0, 1.0, 0.0, 0.0, 0.1),
        counts=(1000, 200, 300, 500, 250, 25, 200, 180, 300, 210, 6, 200, 40),
    )


def test_leading_columns_are_phase_round_acc_loss():
    keys = list(_row().keys())
    assert keys[:4] == ["phase", "round", "acc", "loss"]


def test_losses_then_routing_then_counts():
    keys = list(_row().keys())
    assert keys[4:8] == ["L_sup", "L_A", "L_B", "L_proto"]
    assert keys[8:11] == ["best_acc", "best_round", "gate"]
    assert keys[11] == "a_ratio"
    # 原始计数在最后一组
    assert keys[-13:] == [
        "cnt_u", "cnt_a", "cnt_b", "cnt_c", "pass_s", "geom_drop",
        "a_total", "a_correct", "b_total", "b_correct",
        "hce_hc", "hce_den_hc", "n_batches",
    ]


def test_ratios_match_raw_counts():
    row = _row()
    assert abs(row["a_ratio"] - row["cnt_a"] / row["cnt_u"]) < 1e-9
    assert abs(row["a_prec"] - row["a_correct"] / row["a_total"]) < 1e-4


def test_append_keeps_single_header():
    with tempfile.TemporaryDirectory() as d:
        p = str(Path(d) / "metrics.csv")
        _append_metrics_row(p, _row(1, 0.5), first_round=True)
        _append_metrics_row(p, _row(2, 0.6), first_round=False)
        with open(p, encoding="utf8") as f:
            rows = list(csv.DictReader(f))
        assert [r["round"] for r in rows] == ["1", "2"]
        assert [r["acc"] for r in rows] == ["0.5", "0.6"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} passed")
