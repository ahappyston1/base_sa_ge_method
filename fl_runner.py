# -*- coding: utf-8 -*-
"""
PPFPSL 联邦半监督训练主循环。

Pressure-aware + Prototype + A/B/C 路由。
当前唯一实现：A/B/C 路由 + 门槛退火 + 自适应阶段/门槛。
"""
from __future__ import annotations
import math
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from torchvision import datasets
from torchvision.transforms import transforms
from options import args_parser
from Dataset.dataset import (
    classify_label,
    show_clients_data_distribution,
    Indices2Dataset_labeled,
    Indices2Dataset_unlabeled_fixmatch,
    partition_train,
    LABELED_LEN_MULTIPLIER,
)
from Dataset.normalize import to_tensor_normalize
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo
import numpy as np
import pandas as pd
from torch import eq, no_grad, nn
from Dataset.CINIC10 import CINIC10
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from Model.resnet import ResNet
from tqdm import tqdm
import copy
import torch
import random
from torch.utils.data import DataLoader, RandomSampler
import logging
import os
import csv
import time
import json
from trusted_geometry import refresh_reference, score_queries
import bc_tail
import bc_targets
import target_experiments
from exploration import (initialize_heads, ema as bc_ema_update, teacher_objectives,
                         ramp as exploration_ramp, prox_coefficient, proximal_loss)
from trusted_risk import (score_a as score_a_risk, audit as audit_a_risk, GROUPS as RISK_GROUPS,
                          new_audit as new_risk_audit, audit_rows as risk_audit_rows)
from training_dynamics import cosine_learning_rate, training_learning_rate, dynamics_row, lr_controls, schedule_rounds, trusted_weight_gate
from update_diagnostics import update_rows, save_update_snapshot
from geometry_audit import accumulate as audit_accumulate, new_buffer as audit_buffer, append_rows as audit_append

worker_num = 4
# 当前唯一实现的 checkpoint 标记；旧 V1/V2 权重没有此字段，不会被误续训
METHOD_REV = 13


def _lerp(g, a, b):
    """g=0 取 a，g=1 取 b；a/b 可以是标量或 Tensor。"""
    return (1.0 - g) * a + g * b


def _run_paths(dataset, alpha, args) -> Dict[str, str]:
    """一次实验的全部产物放在 results/<数据集>/runs/<run_id>_a<alpha>/ 下。

    目录名即 run_id，续训时从 --resume 的父目录名反推，不再靠文件名解析。
    """
    import re

    run_id = str(getattr(args, "run_id", "") or "").strip()
    resume = str(getattr(args, "resume", "") or "").strip()
    if resume:
        resume = os.path.abspath(resume)
        if not run_id:
            # 新布局：runs/<run_id>_a<alpha>/checkpoint.pt
            d = os.path.basename(os.path.dirname(resume))
            m = re.match(r"^(\d{8}_\d{6})", d)
            if not m:  # 旧布局：checkpoints/PPFPSL_a0.1_<run_id>_latest.pt
                m = re.search(r"_(\d{8}_\d{6})(?:_latest)?\.pt$", resume)
            if m:
                run_id = m.group(1)
    if not run_id:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    # %g 归一 alpha：1.0 与 1 得到同一个目录名，便于脚本预先建目录
    run_dir = os.path.join(".", "results", str(dataset), "runs", f"{run_id}_a{float(alpha):g}")
    if getattr(args, 'bc_fork', 0) and os.path.exists(run_dir):
        raise ValueError('BC fork requires a new, nonexistent run directory')
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "acc_path": os.path.join(run_dir, "acc.csv"),
        "metrics_path": os.path.join(run_dir, "metrics.csv"),
        "diag_path": os.path.join(run_dir, "diag.csv"),
        "dynamics_path": os.path.join(run_dir, "dynamics.csv"),
        "geometry_audit_path": os.path.join(run_dir, "geometry_audit.csv"),
        "trust_reference_path": os.path.join(run_dir, "trust_reference.csv"),
        "ckpt_path": os.path.join(run_dir, "checkpoint.pt"),
        "resume_src": resume,
        "log_file": os.path.join(run_dir, "train.log"),
        "tb_dir": os.path.join(run_dir, "tensorboard"),
        "config_path": os.path.join(run_dir, "config.json"),
    }


def apply_run_seeds(args) -> None:
    """模型种子：torch / python / 全局 numpy。划分与在线抽样用独立 RandomState。"""
    if getattr(args, "sample_seed", None) is None:
        args.sample_seed = int(args.seed)
    s = int(args.seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_index_list(indices) -> list:
    """Dirichlet 返回 list，IID homo 返回 ndarray；统一成 Python int 列表以便 extend。"""
    if indices is None:
        return []
    arr = np.asarray(indices).reshape(-1)
    return [int(x) for x in arr.tolist()]


def _assert_clients_can_batch(
    list_lab: list,
    list_unl: list,
    args,
) -> None:
    """drop_last=True 时，每个客户端必须能拿出至少 1 个有标 batch 和 1 个无标 batch。"""
    lab_bs = int(args.batch_size_local_labeled_fixmatch)
    unl_bs = lab_bs * int(args.mu)
    bad = []
    for k, (lab, unl) in enumerate(zip(list_lab, list_unl)):
        n_lab = len(lab)
        n_unl = len(unl)
        lab_loader_len = n_lab * int(LABELED_LEN_MULTIPLIER)
        if n_lab < 1 or lab_loader_len < lab_bs:
            bad.append(
                f"client {k}: labeled={n_lab} (loader_len={lab_loader_len}) < labeled_batch={lab_bs}"
            )
        if n_unl < unl_bs:
            bad.append(
                f"client {k}: unlabeled={n_unl} < unlabeled_batch={unl_bs} (drop_last would be empty)"
            )
    if bad:
        raise RuntimeError(
            "有客户端无法形成本地 batch（drop_last=True）。请减小 batch/μ 或调整划分。\n"
            + "\n".join(bad)
        )


def _metrics_row(
    r: int,
    phase: int,
    gate: float,
    acc: float,
    best_acc: float,
    best_round: int,
    losses: tuple,
    route: tuple,
    quality: tuple,
    proto: tuple,
    rel: tuple,
    knobs: tuple,
    counts: tuple,
) -> "OrderedDict":
    """metrics.csv 的一行。列顺序在这里集中定义，按「先看什么」从左到右排：
    阶段/轮次/acc/loss → 损失分解 → 最优与 gate → 路由占比 → 路由质量
    → 原型 → 可靠性 → 门槛 → 原始计数。
    """
    loss, l_sup, l_a, l_b, l_proto = losses
    a_ratio, b_ratio, c_ratio, m_a, geom_drop_rate = route
    a_prec, b_prec, hce_rate = quality
    p_share, p_dirty, p_a_uniq, p_lab_mass, p_a_mass, lab_mult = proto
    rel_r, rel_m, rel_alpha = rel
    snap, lambda_a_scale, lambda_b_scale, use_a_for_proto, cap_a, cap_b, lr = knobs
    (cnt_u, cnt_a, cnt_b, cnt_c, pass_s, geom_drop,
     a_total, a_correct, b_total, b_correct, hce_hc, hce_den, n_batches) = counts

    row = OrderedDict()
    # 最前面四列：阶段、轮次、精度、总损失
    row["phase"] = int(phase)
    row["round"] = r
    row["acc"] = round(float(acc), 4)
    row["loss"] = round(float(loss), 5)
    # 损失分解，紧跟总损失
    for k, v in (("L_sup", l_sup), ("L_A", l_a), ("L_B", l_b), ("L_proto", l_proto)):
        row[k] = round(float(v), 5)
    # 次要进度：历史最优与几何门控进度
    row["best_acc"] = round(float(best_acc), 4)
    row["best_round"] = int(best_round)
    row["gate"] = round(float(gate), 4)
    # 路由占比：A/B/C 三者互斥且和为 1；m_a 是通过置信门槛的比例
    for k, v in (("a_ratio", a_ratio), ("b_ratio", b_ratio), ("c_ratio", c_ratio),
                 ("m_a", m_a), ("geom_drop_rate", geom_drop_rate)):
        row[k] = round(float(v), 5)
    # 路由质量（用无标 GT，仅诊断，不参与训练与阶段决策）
    for k, v in (("a_prec", a_prec), ("b_prec", b_prec), ("hce_rate", hce_rate)):
        row[k] = round(float(v), 5)
    # 原型：A 相对有标的质量占比、A 中错样本占比、去重后的 A 质量
    for k, v in (("proto_a_share", p_share), ("proto_a_dirty", p_dirty),
                 ("proto_a_unique", p_a_uniq), ("proto_lab_mass", p_lab_mass),
                 ("proto_a_mass", p_a_mass), ("lab_mult", lab_mult)):
        row[k] = round(float(v), 5)
    # 可靠性
    for k, v in (("rel_r", rel_r), ("rel_m", rel_m), ("rel_alpha", rel_alpha)):
        row[k] = round(float(v), 5)
    # 自适应门槛与开关
    for k, v in (("tau0", snap["tau0"]), ("delta0", snap["delta0"]),
                 ("eta_B", snap["eta_B"]), ("tau_warmup", snap["tau_warmup"]),
                 ("cap_a", cap_a), ("cap_b", cap_b),
                 ("lambda_A_scale", lambda_a_scale), ("lambda_B_scale", lambda_b_scale),
                 ("use_a_for_proto", use_a_for_proto), ("lr", lr)):
        row[k] = round(float(v), 5)
    # 原始计数放最后，供核对上面的比例
    for k, v in (("cnt_u", cnt_u), ("cnt_a", cnt_a), ("cnt_b", cnt_b), ("cnt_c", cnt_c),
                 ("pass_s", pass_s), ("geom_drop", geom_drop),
                 ("a_total", a_total), ("a_correct", a_correct),
                 ("b_total", b_total), ("b_correct", b_correct),
                 ("hce_hc", hce_hc), ("hce_den_hc", hce_den), ("n_batches", n_batches)):
        row[k] = int(v)
    return row


def _append_metrics_row(path: str, row: "OrderedDict", first_round: bool) -> None:
    """追加一行；表头不匹配（换了列定义）时重写文件，避免新旧列错位。"""
    fieldnames = list(row.keys())
    header_ok = False
    if os.path.isfile(path) and not first_round:
        with open(path, "r", encoding="utf8") as rf:
            header_ok = rf.readline().strip().split(",") == fieldnames
    mode = "a" if header_ok else "w"
    with open(path, mode, newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            w.writeheader()
        w.writerow(row)


def _diag_row(r: int, phase: int, cnt_u: int, agg: Dict[str, int]) -> "OrderedDict":
    """由各客户端累计的诊断计数换算成轮级只读诊断行（GT 仅离线用）。"""
    def g(k):
        return int(agg.get(k, 0))

    band_n = g("dg_band_n")
    band_geom = g("dg_band_geom_n")
    sup_n = g("dg_sup_n")
    opp_n = g("dg_opp_n")
    wouldB = g("dc_wouldB")
    conflict = g("dc_wouldB_conflict")
    geommiss = g("dc_wouldB_geommiss")
    dconf_n = g("dconf_n")

    def ratio(a, b):
        return round(a / b, 6) if b > 0 else 0.0

    row = OrderedDict()
    row["round"] = r
    row["phase"] = phase
    # 几何判别（置信带内）
    row["geom_cov"] = ratio(band_geom, band_n)   # 该置信带内几何有效覆盖率
    row["band_n"] = band_n
    row["sup_n"] = sup_n
    row["sup_acc"] = ratio(g("dg_sup_cor"), sup_n)   # 几何支持组伪标签正确率
    row["opp_n"] = opp_n
    row["opp_acc"] = ratio(g("dg_opp_cor"), opp_n)   # 几何反对组伪标签正确率
    # C 桶成因（Phase2/3）
    row["c_lowconf"] = g("dc_lowconf")               # 纯低置信，合理留 C
    row["c_wouldB"] = wouldB                          # 满足置信版 B 却进 C
    row["c_wouldB_geommiss"] = geommiss              # 因几何缺失
    row["c_wouldB_conflict"] = conflict              # 因有效几何冲突
    row["c_wouldB_other"] = max(0, wouldB - geommiss - conflict)  # r_i 混合降级等
    # 冲突进 C：分类器 vs 最近原型谁更常对
    row["conflictC_n"] = dconf_n
    row["conflictC_frac"] = ratio(dconf_n, cnt_u)
    row["conflictC_clf_acc"] = ratio(g("dconf_clf_cor"), dconf_n)
    row["conflictC_proto_acc"] = ratio(g("dconf_proto_cor"), dconf_n)
    row["tau_n"] = g("dt_n")
    row["tau_clipped_n"] = g("dt_clipped_n")
    row["tau_clipped_frac"] = ratio(g("dt_clipped_n"), g("dt_n"))
    row["tau_raw_ge1_n"] = g("dt_raw_ge1_n")
    row["tau_effective_ge1_n"] = g("dt_effective_ge1_n")
    return row


def _append_diag_row(path: str, row: "OrderedDict", first_round: bool) -> None:
    """写 diag.csv；与 metrics.csv 同风格，表头不匹配则重写。"""
    fieldnames = list(row.keys())
    header_ok = False
    if os.path.isfile(path) and not first_round:
        with open(path, "r", encoding="utf8") as rf:
            header_ok = rf.readline().strip().split(",") == fieldnames
    mode = "a" if header_ok else "w"
    with open(path, mode, newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            w.writeheader()
        w.writerow(row)


def _write_run_config(path: str, args, extra: Dict[str, Any]) -> None:
    cfg = {
        "method_rev": METHOD_REV,
        "seed_model": int(args.seed),
        "seed_partition": int(getattr(args, "partition_seed", 0)),
        "seed_sample": int(getattr(args, "sample_seed", args.seed)),
        "dataset": args.dataset,
        "alpha": extra.get("alpha"),
        "num_rounds": int(args.num_rounds),
        "schedule_rounds": schedule_rounds(args),
        "num_clients": int(args.num_clients),
        "num_online_clients": int(args.num_online_clients),
        "num_labeled_per_class": int(getattr(args, "num_labeled", 0)),
        "mu": int(args.mu),
        "local_epochs": int(args.local_epochs),
        "batch_labeled": int(args.batch_size_local_labeled_fixmatch),
        # num_workers 改变增广随机流：同 seed 但不同 workers 的 run 不可逐位比较
        "num_workers": int(getattr(args, "num_workers", 0)),
        "pp_adaptive": int(getattr(args, "pp_adaptive", 1)),
        "pp_a_ratio_cap": float(getattr(args, "pp_a_ratio_cap", 0)),
        "pp_a_ratio_cap_p1": float(getattr(args, "pp_a_ratio_cap_p1", 0)),
        "proto_conf_floor": float(getattr(args, "proto_conf_floor", 0)),
        "proto_w_extra": float(getattr(args, "proto_w_extra", 0)),
        "diag_geom": int(getattr(args, "diag_geom", 1)),
        "diag_s_lo": float(getattr(args, "diag_s_lo", 0.95)),
        "diag_s_hi": float(getattr(args, "diag_s_hi", 0.99)),
        "geometry_controls": _geometry_controls(args),
        "lr_controls": lr_controls(args),
        "tau_warmup": float(args.tau_warmup),
        "tau0": float(args.tau0),
        "delta0": float(args.delta0),
        "eta_B": float(args.eta_B),
        "run_id": extra.get("run_id"),
    }
    cfg['bc_targets'] = getattr(args, 'bc_targets', 0)
    cfg['target_experiment'] = getattr(args, 'target_experiment', 'none')
    cfg['labelhead_guard'] = getattr(args, 'labelhead_guard', 0)
    cfg['bc_tail'] = getattr(args, 'bc_tail', 'none')
    cfg['fork_source'] = os.path.abspath(args.resume) if getattr(args, 'bc_fork', 0) else None
    cfg['stop_after_round'] = getattr(args, 'stop_after_round', 0)
    if getattr(args, 'diagnostic_update_rounds', ''):
        cfg['diagnostic_update_rounds'] = args.diagnostic_update_rounds
    with open(path, "w", encoding="utf8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ============= PPFPSL 核心函数 =============

def _geometry_controls(args) -> Dict[str, Any]:
    """Training controls persisted for reproducible server runs."""
    controls = {
        "mode": str(getattr(args, "pp_geom_mode", "legacy")),
        "trust_min_count": int(getattr(args, "pp_trust_min_count", 4)),
        "tau_ceiling": float(getattr(args, "pp_tau_ceiling", 0.99)),
        "complete_gate": int(getattr(args, "pp_complete_gate", 1)),
        "b_conf_rescue": int(getattr(args, "pp_b_conf_rescue", 0)),
        "phase3_geom_boost": float(getattr(args, "pp_phase3_geom_boost", 0.0)),
    }
    if getattr(args, 'trusted_weight_start', 0):
        controls.update(trusted_weight_start=int(args.trusted_weight_start),
                        trusted_weight_end=int(args.trusted_weight_end))
    if getattr(args, 'trusted_a_risk', 0):
        controls.update(trusted_a_risk=1, risk_version=1,
                        risk_distance_cap=args.risk_distance_cap,
                        risk_conflict_cap=args.risk_conflict_cap,
                        risk_prior_count=args.risk_prior_count)
    if getattr(args, 'bc_teacher', 0):
        controls.update(bc_teacher=1, bc_version=1, bc_ema=args.bc_ema,
                        bc_feature_weight=args.bc_feature_weight, bc_ramp=[30,90])
    if getattr(args, 'mid_prox_mu', 0):
        controls.update(mid_prox_mu=args.mid_prox_mu, prox_version=1, prox_schedule=[30,60,200,250])
    if getattr(args, 'bc_targets', 0):
        controls['bc_targets'] = bc_targets.controls()
    if getattr(args, 'target_experiment', 'none') != 'none':
        controls['target_experiment'] = target_experiments.controls(args.target_experiment)
    if getattr(args, 'labelhead_guard', 0):
        controls['labelhead_guard'] = target_experiments.guard_controls()
    if getattr(args, "bc_tail", "none") != "none":
        controls["bc_tail"] = bc_tail.tail_controls(args.bc_tail)
    return controls


def _class_thresholds(nr, tau0, delta0, a, b, boost, ceiling):
    """Bound confidence thresholds; preserve the pressure margin formula."""
    pressure = nr * (1.0 + boost * nr)
    raw_tau = tau0 + a * pressure
    delta = delta0 + b * pressure
    tau = raw_tau.clamp(max=ceiling) if ceiling > 0 else raw_tau
    return tau, delta, raw_tau


def _infer_dim(model: nn.Module, device: torch.device) -> int:
    """推断 backbone 特征维度。"""
    with torch.no_grad():
        z, _ = model(torch.zeros(1, 3, 32, 32, device=device))
    return int(z.shape[-1])


def _warmup_rounds(r_total: int, args) -> int:
    rw = max(0, int(round(args.pp_warmup_ratio * r_total)))
    if rw == 0 and args.pp_warmup_ratio > 0 and r_total > 0:
        rw = 1
    return rw


def training_phase(r: int, r_total: int, args) -> int:
    """返回 1/2/3 表示 warm-up / geometry / pressure-enhanced。"""
    rw = _warmup_rounds(r_total, args)
    rg = max(0, int(round(args.pp_geom_ratio * r_total)))
    if int(getattr(args, "pp_complete_gate", 1)):
        rg = max(rg, max(1, int(round(args.pp_gate_anneal_ratio * r_total))))
    if r <= rw:
        return 1
    if r <= rw + rg:
        return 2
    return 3


def gate_anneal(r: int, r_total: int, args) -> float:
    """Phase2 起将几何门槛从 Phase1 平滑拉到目标值，避免一轮切死。

    返回 g∈[0,1]：0 表示仍用 warmup 门槛（几乎无 margin），1 表示满几何门槛。
    使用余弦缓入，切换后前几轮更接近 Phase1。
    """
    rw = _warmup_rounds(r_total, args)
    anneal = max(1, int(round(getattr(args, "pp_gate_anneal_ratio", 0.15) * r_total)))
    if r <= rw:
        return 0.0
    t = min(1.0, (r - rw) / float(anneal))
    return 0.5 * (1.0 - math.cos(math.pi * t))


def warmup_aux_scale(r: int, r_total: int, args) -> float:
    """Warmup 末尾从 0→1，用于提前加一小段 L_proto 和 B 桶，减轻 Phase1 过弱。"""
    rw = _warmup_rounds(r_total, args)
    tail = float(getattr(args, "pp_warmup_aux_ratio", 0.3))
    if rw <= 0 or tail <= 0:
        return 0.0
    tail_rounds = max(1, int(round(rw * tail)))
    start = rw - tail_rounds
    if r <= start:
        return 0.0
    if r >= rw:
        return 1.0
    return (r - start) / float(tail_rounds)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _margin_reliability(m_i: torch.Tensor, m0: float, mT: float) -> torch.Tensor:
    """把原型间隔映射到 (0,1)。m=m0 时为 0.5；负间隔明显低于 0.5，正间隔才算可靠。"""
    t = max(float(mT), 1e-6)
    return torch.sigmoid((m_i - float(m0)) / t)


def _blend_alpha(g_gate: float, nr: torch.Tensor, args) -> torch.Tensor:
    """α 随 gate 从「信 softmax」过渡到「信原型」，高压类更早信间隔。"""
    a0 = float(args.alpha0)
    amin = float(args.alpha_min)
    amax = float(args.alpha_max)
    g = float(g_gate)
    base = (1.0 - g) * a0 + g * amin
    # 压力大的类降低 α，让 r_i 更看 m
    gamma = float(getattr(args, "pp_gamma", 1.0))
    alpha = base - (gamma * 0.25 * g * nr * (base - amin + 0.08))
    return alpha.clamp(0.20, amax)


class AdaptiveSchedule:
    """服务端调度：动态 A/B 门槛 + 按指标推进阶段。

    关闭 ``pp_adaptive`` 时退回固定比例（warmup/geom/anneal）。
    """

    def __init__(self, args, r_total: int):
        self.args = args
        self.r_total = max(1, int(r_total))
        self.enabled = bool(int(getattr(args, "pp_adaptive", 1)))
        self.phase = 1
        self.phase_enter_round = 1
        self.acc_p1_exit = None
        self.ema_acc = None
        self.ema_a_prec = None
        self.ema_b_prec = None
        self.ema_a_ratio = None
        self.ema_b_ratio = None
        self.ema_c_ratio = None
        self.tau0 = float(args.tau0)
        self.delta0 = float(args.delta0)
        self.eta_B = float(args.eta_B)
        self.tau_warmup = float(args.tau_warmup)
        self.gate = 0.0
        self.aux_scale = 0.0
        self.last_event = ""
        self.beta = float(getattr(args, "pp_adapt_ema", 0.8))
        self.a_ratio_cap = float(getattr(args, "pp_a_ratio_cap_p1", 0.0))
        self.b_ratio_cap = float(getattr(args, "pp_b_ratio_cap", 0.0))
        self.use_a_for_proto = 1.0
        self.lambda_A_scale = 1.0
        self.lambda_B_scale = 1.0
        self._a_prec_now = None

    def snapshot(self) -> Dict[str, float]:
        return {
            "phase": int(self.phase),
            "gate": float(self.gate),
            "aux_scale": float(self.aux_scale),
            "tau0": float(self.tau0),
            "delta0": float(self.delta0),
            "eta_B": float(self.eta_B),
            "tau_warmup": float(self.tau_warmup),
            "a_ratio_cap": float(self.a_ratio_cap),
            "b_ratio_cap": float(self.b_ratio_cap),
            "use_a_for_proto": float(self.use_a_for_proto),
            "lambda_A_scale": float(self.lambda_A_scale),
            "lambda_B_scale": float(self.lambda_B_scale),
        }

    def state_dict(self) -> Dict[str, Any]:
        keys = (
            "phase",
            "phase_enter_round",
            "acc_p1_exit",
            "ema_acc",
            "ema_a_prec",
            "ema_b_prec",
            "ema_a_ratio",
            "ema_b_ratio",
            "ema_c_ratio",
            "tau0",
            "delta0",
            "eta_B",
            "tau_warmup",
            "gate",
            "aux_scale",
            "last_event",
            "a_ratio_cap",
            "b_ratio_cap",
            "use_a_for_proto",
            "lambda_A_scale",
            "lambda_B_scale",
        )
        return {k: getattr(self, k) for k in keys}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _ema(self, name: str, val: Optional[float]) -> Optional[float]:
        if val is None:
            return getattr(self, name)
        cur = getattr(self, name)
        if cur is None:
            setattr(self, name, float(val))
        else:
            setattr(self, name, self.beta * float(cur) + (1.0 - self.beta) * float(val))
        return getattr(self, name)

    def _stayed(self, r: int) -> int:
        return r - int(self.phase_enter_round) + 1

    def for_round(self, r: int) -> Dict[str, float]:
        if not self.enabled:
            ph = training_phase(r, self.r_total, self.args)
            cap_p1 = float(getattr(self.args, "pp_a_ratio_cap_p1", 0.0))
            cap_p2 = float(getattr(self.args, "pp_a_ratio_cap", 0.0))
            g = gate_anneal(r, self.r_total, self.args) if ph == 2 else (0.0 if ph == 1 else 1.0)
            if ph == 1:
                cap = cap_p1
            elif ph == 2:
                cap = _lerp(g, cap_p1, cap_p2)
            else:
                cap = cap_p2
            return {
                "phase": ph,
                "gate": gate_anneal(r, self.r_total, self.args),
                "aux_scale": warmup_aux_scale(r, self.r_total, self.args),
                "tau0": float(self.args.tau0),
                "delta0": float(self.args.delta0),
                "eta_B": float(self.args.eta_B),
                "tau_warmup": float(self.args.tau_warmup),
                "a_ratio_cap": cap,
                "b_ratio_cap": float(getattr(self.args, "pp_b_ratio_cap", 0.0)),
                "use_a_for_proto": 1.0,
                "lambda_A_scale": 1.0,
                "lambda_B_scale": 1.0,
            }
        self._refresh_scales(r)
        return self.snapshot()

    def _refresh_scales(self, r: int) -> None:
        args = self.args
        stayed = self._stayed(r)
        if self.phase == 1:
            min_w = max(1, int(round(args.pp_warmup_min_ratio * self.r_total)))
            tail = max(1, int(round(min_w * args.pp_warmup_aux_ratio)))
            start_aux = max(1, min_w - tail)
            if stayed <= start_aux:
                self.aux_scale = 0.0
            else:
                self.aux_scale = min(1.0, (stayed - start_aux) / float(tail))
            self.gate = 0.0
        elif self.phase == 2:
            self.aux_scale = 1.0
            anneal = max(1, int(round(args.pp_gate_anneal_ratio * self.r_total)))
            t = min(1.0, stayed / float(anneal))
            self.gate = 0.5 * (1.0 - math.cos(math.pi * t))
        else:
            self.aux_scale = 1.0
            self.gate = 1.0
        cap_p1 = float(getattr(args, "pp_a_ratio_cap_p1", 0.0))
        cap_p2 = float(getattr(args, "pp_a_ratio_cap", 0.0))
        if self.phase == 1:
            self.a_ratio_cap = cap_p1
        elif self.phase == 2:
            self.a_ratio_cap = _lerp(self.gate, cap_p1, cap_p2)
        else:
            self.a_ratio_cap = cap_p2

    def update_after_round(self, r: int, metrics: Dict[str, float]) -> None:
        self._ema("ema_acc", metrics.get("acc"))
        self._ema("ema_a_prec", metrics.get("a_prec"))
        if metrics.get("b_total", 0) > 0:
            self._ema("ema_b_prec", metrics.get("b_prec"))
        self._ema("ema_a_ratio", metrics.get("a_ratio"))
        self._ema("ema_b_ratio", metrics.get("b_ratio"))
        self._ema("ema_c_ratio", metrics.get("c_ratio"))
        self._a_prec_now = metrics.get("a_prec")
        if not self.enabled:
            self.phase = training_phase(r, self.r_total, self.args)
            return
        self._adapt_thresholds()
        self._maybe_advance_phase(r)

    def _adapt_thresholds(self) -> None:
        """对齐 81% 实验：不改 τ/η/λ。Phase2 的 A cap 随 gate 从 p1 插到 p2。"""
        args = self.args
        cap_p1 = float(getattr(args, "pp_a_ratio_cap_p1", 0.0))
        cap_p2 = float(getattr(args, "pp_a_ratio_cap", 0.0))
        if self.phase == 1:
            self.a_ratio_cap = cap_p1
        elif self.phase == 2:
            self.a_ratio_cap = _lerp(float(self.gate), cap_p1, cap_p2)
        else:
            self.a_ratio_cap = cap_p2
        self.use_a_for_proto = 1.0
        self.lambda_A_scale = 1.0
        self.lambda_B_scale = 1.0

    def _adapt_eta_b(self, step: float, min_c: float) -> None:
        if self.ema_b_prec is None:
            return
        tgt = float(getattr(self.args, "pp_target_b_prec", 0.75))
        c_ratio = 0.0 if self.ema_c_ratio is None else float(self.ema_c_ratio)
        b_cap = float(getattr(self.args, "pp_b_ratio_cap", 0.0))
        b_ratio = 0.0 if self.ema_b_ratio is None else float(self.ema_b_ratio)
        b_at_cap = b_ratio >= 0.95 * b_cap
        if c_ratio < min_c or (b_at_cap and self.ema_b_prec < tgt - 0.03):
            self.eta_B = _clip(self.eta_B + step, 0.50, 0.88)
            return
        if self.ema_b_prec > tgt + 0.03 and c_ratio > 0.18:
            self.eta_B = _clip(self.eta_B - step, 0.50, 0.88)
        elif self.ema_b_prec < tgt - 0.03:
            self.eta_B = _clip(self.eta_B + step, 0.50, 0.88)

    def _maybe_advance_phase(self, r: int) -> None:
        args = self.args
        min_w = max(1, int(round(args.pp_warmup_min_ratio * self.r_total)))
        max_w = max(min_w, int(round(args.pp_warmup_max_ratio * self.r_total)))
        min_g = max(1, int(round(args.pp_geom_min_ratio * self.r_total)))
        max_g = max(min_g, int(round(args.pp_geom_max_ratio * self.r_total)))
        stayed = self._stayed(r)
        if self.phase == 1:
            ready = stayed >= min_w and self.aux_scale >= 0.8
            if ready or stayed >= max_w:
                self.acc_p1_exit = self.ema_acc
                why = "ready" if ready else "max_warmup"
                self.phase = 2
                self.phase_enter_round = r + 1
                self.last_event = (
                    f"r{r} phase1→2 ({why}, stayed={stayed}, aux={self.aux_scale:.2f})"
                )
                print("[Schedule]", self.last_event)
        elif self.phase == 2:
            exit_ar = float(getattr(args, "pp_phase2_exit_a_ratio", 0.08))
            complete = bool(int(getattr(args, "pp_complete_gate", 1)))
            anneal = max(1, int(round(args.pp_gate_anneal_ratio * self.r_total)))
            gate_ready = (stayed >= anneal and self.gate >= 1.0) if complete else self.gate >= 0.95
            ready = (
                stayed >= min_g
                and gate_ready
                and self.ema_a_ratio is not None
                and self.ema_a_ratio >= exit_ar
            )
            timed_out = stayed >= max_g and (not complete or gate_ready)
            if ready or timed_out:
                why = "ready" if ready else "max_geom"
                self.phase = 3
                self.phase_enter_round = r + 1
                ar = "na" if self.ema_a_ratio is None else f"{self.ema_a_ratio:.3f}"
                self.last_event = f"r{r} phase2→3 ({why}, stayed={stayed}, gate={self.gate:.2f}, A-ratio={ar})"
                print("[Schedule]", self.last_event)


def _cap_bucket(in_A: torch.Tensor, score: torch.Tensor, cap: float) -> torch.Tensor:
    """超过 cap 占比时只保留 score 最高的恰好 k 个（并列按 top-k 索引，不整批放行）。"""
    if cap <= 0 or (not in_A.any()):
        return in_A
    n = int(in_A.numel())
    k = max(1, int(math.floor(cap * n)))
    n_a = int(in_A.sum().item())
    if n_a <= k:
        return in_A
    candidates = torch.nonzero(in_A, as_tuple=True)[0]
    chosen = candidates[torch.topk(score.detach()[candidates], k, largest=True).indices]
    out = torch.zeros_like(in_A)
    out[chosen] = True
    return out


def _geom_margin(
    zw: torch.Tensor,
    yhat: torch.Tensor,
    p_mix: torch.Tensor,
    mix_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """本类间隔 m = sim(ŷ) - max_{c≠ŷ, valid} sim(c)。

    本类原型缺失、或没有任何竞争类原型时，间隔未定义：m=0 且 geom_ok=False。
    禁止把无效位的 -1e9 当成「极大间隔 / 高可靠」。
    """
    n, c = int(zw.size(0)), int(p_mix.size(0))
    sims = zw @ p_mix.T
    sim_pos = sims.gather(1, yhat.unsqueeze(1)).squeeze(1)
    own_ok = mix_valid[yhat]
    other_ok = mix_valid.unsqueeze(0).expand(n, c).clone()
    other_ok.scatter_(1, yhat.unsqueeze(1), False)
    has_other = other_ok.any(dim=1)
    geom_ok = own_ok & has_other
    max_other, _ = sims.masked_fill(~other_ok, -1e9).max(dim=-1)
    m_i = torch.where(geom_ok, sim_pos - max_other, torch.zeros_like(sim_pos))
    return m_i, geom_ok, own_ok


def _route_phase1(
    s_i: torch.Tensor,
    tau: float,
    eta_B: float,
    aux_on: bool,
    a_cap: float,
    b_cap: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase1：只看置信度。返回 in_A, in_B, in_C, pass_s。"""
    pass_s = s_i >= tau
    in_A = _cap_bucket(pass_s, s_i, a_cap)
    if aux_on:
        in_B = _cap_bucket((~in_A) & (s_i >= eta_B), s_i, b_cap)
    else:
        in_B = torch.zeros_like(pass_s)
    in_C = (~in_A) & (~in_B)
    return in_A, in_B, in_C, pass_s


def _route_phase23(
    s_i: torch.Tensor,
    m_i: torch.Tensor,
    m_tilde: torch.Tensor,
    r_i: torch.Tensor,
    tau_i: torch.Tensor,
    delta_i: torch.Tensor,
    own_ok: torch.Tensor,
    g_gate: float,
    eta_B: float,
    a_cap: float,
    b_cap: float,
    b_min_m: float,
    m_floor: float = -2.0,
    geom_ok: Optional[torch.Tensor] = None,
    b_conf_rescue: bool = False,
) -> Dict[str, torch.Tensor]:
    """Phase2/3：置信度 + 随 gate 接入的几何。不含 GT。

    未知几何不参与间隔门槛比较；m̃=0.5进入混合分数和软权重。
    不再为未知几何单独将B分数替换为置信度。
    """
    if geom_ok is None:
        geom_ok = torch.ones_like(s_i, dtype=torch.bool)
    pass_s = s_i >= tau_i
    keep_own = own_ok | (g_gate < 0.5)
    margin_ok = (~geom_ok) | (m_i >= delta_i)
    in_A = pass_s & margin_ok & keep_own
    score_a = _lerp(g_gate, s_i, s_i * m_tilde)
    in_A = _cap_bucket(in_A, score_a, a_cap)
    b_score = _lerp(g_gate, s_i, r_i)
    b_m_thr = _lerp(g_gate, m_floor, b_min_m)
    b_margin_ok = (~geom_ok) | (m_i >= b_m_thr)
    b_score_ok = b_score >= eta_B
    if b_conf_rescue:
        b_score_ok = b_score_ok | (s_i >= eta_B)
    in_B = _cap_bucket((~in_A) & b_score_ok & b_margin_ok, b_score, b_cap)
    in_C = (~in_A) & (~in_B)
    fail_geom = pass_s & geom_ok & ~((m_i >= delta_i) & keep_own)
    return {
        "in_A": in_A,
        "in_B": in_B,
        "in_C": in_C,
        "pass_s": pass_s,
        "fail_geom": fail_geom,
        "score_a": score_a,
        "b_score": b_score,
    }


@torch.no_grad()
def _diag_geom_c(
    s_i: torch.Tensor,
    yhat: torch.Tensor,
    y_gt: torch.Tensor,
    m_i: torch.Tensor,
    delta_i: torch.Tensor,
    geom_ok: torch.Tensor,
    in_C: torch.Tensor,
    eta_B: float,
    zw: torch.Tensor,
    p_mix: torch.Tensor,
    mix_valid: torch.Tensor,
    s_lo: float,
    s_hi: float,
    b_m_thr: float = 0.0,
) -> Dict[str, int]:
    """只读诊断（Phase2/3）。GT 仅用于离线核对，不参与任何训练决策。

    回答两件事：
    1) 同一置信带 [s_lo,s_hi) 内，几何支持 vs 反对两组的伪标签正确率与数量、几何覆盖率
       —— 支持/反对用的是 A 的间隔门槛 delta_i。
    2) C 桶成因：纯低置信 / 满足置信版 B 却进 C。
       - 几何缺失：~geom_ok（中性可靠性参与混合分数，可能导致拒收）
       - 有效几何挡住 B：geom_ok 且 m < b_m_thr（与 _route_phase23 的 B 间隔门槛一致，
         不是写死 m<0；Phase2 中 b_m_thr 随 gate 从 -2 插到 pp_b_min_margin）
       - 其余记入 other（主要是 r_i 混合后 b_score < eta_B）
       冲突进 C 的「谁更常对」仍用 m<0（分类器与原型符号不一致），与 B 门槛分开。
    """
    d: Dict[str, int] = {}
    correct = yhat == y_gt
    band = (s_i >= s_lo) & (s_i < s_hi)
    d["dg_band_n"] = int(band.sum().item())
    bg = band & geom_ok
    d["dg_band_geom_n"] = int(bg.sum().item())
    sup = bg & (m_i >= delta_i)
    opp = bg & (m_i < delta_i)
    d["dg_sup_n"] = int(sup.sum().item())
    d["dg_sup_cor"] = int((sup & correct).sum().item())
    d["dg_opp_n"] = int(opp.sum().item())
    d["dg_opp_cor"] = int((opp & correct).sum().item())

    would_b = in_C & (s_i >= eta_B)
    d["dc_lowconf"] = int((in_C & (s_i < eta_B)).sum().item())
    d["dc_wouldB"] = int(would_b.sum().item())
    d["dc_wouldB_geommiss"] = int((would_b & (~geom_ok)).sum().item())
    d["dc_wouldB_conflict"] = int((would_b & geom_ok & (m_i < b_m_thr)).sum().item())

    conf = in_C & geom_ok & (m_i < 0)
    d["dconf_n"] = int(conf.sum().item())
    d["dconf_clf_cor"] = int((conf & correct).sum().item())
    if conf.any():
        sims = zw @ p_mix.t()
        sims = sims.masked_fill(~mix_valid.unsqueeze(0), float("-inf"))
        proto_pred = sims.argmax(dim=1)
        d["dconf_proto_cor"] = int((conf & (proto_pred == y_gt)).sum().item())
    else:
        d["dconf_proto_cor"] = 0
    return d


_DIAG_KEYS = (
    "dt_n", "dt_clipped_n", "dt_raw_ge1_n", "dt_effective_ge1_n",
    "dg_band_n", "dg_band_geom_n", "dg_sup_n", "dg_sup_cor", "dg_opp_n", "dg_opp_cor",
    "dc_lowconf", "dc_wouldB", "dc_wouldB_geommiss", "dc_wouldB_conflict",
    "dconf_n", "dconf_clf_cor", "dconf_proto_cor",
)


@torch.no_grad()
def _proto_write_weight(base_w: torch.Tensor, s_sel: torch.Tensor, args):
    """写原型的独立权重与筛选，解耦于 CE。

    - proto_w_extra>0：在既有权重上再乘 s^extra，进一步压低低置信样本的原型贡献；
    - proto_conf_floor>0：置信度低于门槛的样本硬性不写原型（返回 keep 掩码）。

    默认（extra=0, floor=0）返回 (base_w, None)，与旧行为逐位一致，且 CE 权重不受影响。
    """
    w = base_w
    extra = float(getattr(args, "proto_w_extra", 0.0))
    if extra > 0.0:
        w = w * s_sel.pow(extra)
    floor = float(getattr(args, "proto_conf_floor", 0.0))
    keep = (s_sel >= floor) if floor > 0.0 else None
    return w, keep


def _unique_a_mass(
    sample_ids: torch.Tensor,
    yhat: torch.Tensor,
    weights: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """服务器 A 侧有效质量：只统计进入 A 的访问。

    对每个样本 ID、每个被命中的类 c：平均权重 = (该类进 A 的权重之和) / (该类进 A 的次数)。
    分母是「进 A 且预测为 c 的次数」，不是该样本的全部无标访问次数。
    再把该样本归到平均权重最高的类，贡献那个平均值。

    有标样本若被并入无标 loader 并进入 A，会同时进入 |L| 与 Auniq（双重加权），
    不是「独立样本只计一次」。本地原型向量仍用训练期累计特征。
    """
    out = torch.zeros(num_classes, dtype=torch.float64)
    if sample_ids.numel() == 0:
        return out.float()
    ids = sample_ids.detach().reshape(-1).cpu()
    cls = yhat.detach().reshape(-1).cpu()
    w = weights.detach().reshape(-1).cpu().to(torch.float64)
    key = ids.to(torch.int64) * int(num_classes) + cls.to(torch.int64)
    uniq, inv = torch.unique(key, return_inverse=True)
    w_sum = torch.zeros(uniq.numel(), dtype=torch.float64)
    n_sum = torch.zeros(uniq.numel(), dtype=torch.float64)
    w_sum.scatter_add_(0, inv, w)
    n_sum.scatter_add_(0, inv, torch.ones_like(w))
    mean_w = w_sum / n_sum.clamp_min(1.0)
    sid = uniq // int(num_classes)
    cc = uniq % int(num_classes)
    order = torch.argsort(sid)
    sid_s = sid[order]
    cc_s = cc[order]
    mw_s = mean_w[order]
    i = 0
    n = int(sid_s.numel())
    while i < n:
        j = i + 1
        while j < n and int(sid_s[j]) == int(sid_s[i]):
            j += 1
        sl = slice(i, j)
        pick = int(torch.argmax(mw_s[sl]).item())
        out[int(cc_s[sl][pick])] += float(mw_s[sl][pick])
        i = j
    return out.float()


def _unlabeled_ab_losses(
    logs: torch.Tensor,
    logw: torch.Tensor,
    yhat: torch.Tensor,
    in_A: torch.Tensor,
    in_B: torch.Tensor,
    w_A: torch.Tensor,
    w_B: torch.Tensor,
    T: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A: 可靠性权重 detach 后乘 CE；B: 弱→强 KL。L_proto 不在此。"""
    ce = F.cross_entropy(logs, yhat, reduction="none")
    L_A = (w_A.detach() * ce).mean()
    L_B = _consistency_kl(logw, logs, in_B, T, weights=w_B)
    return L_A, L_B


def _consistency_kl(
    logw: torch.Tensor,
    logs: torch.Tensor,
    mask: torch.Tensor,
    T: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """弱→强 KL。按无标签整批平均（与 FixMatch 的 mask.mean() 一致），避免 B 再大梯度也不缩。"""
    n = int(logw.size(0))
    if n == 0 or (not mask.any()):
        return logw.new_zeros(())
    p_w = F.softmax(logw[mask].detach() / T, dim=-1)
    p_s = F.log_softmax(logs[mask] / T, dim=-1)
    kl = F.kl_div(p_s, p_w, reduction="none").sum(-1)
    if weights is not None:
        kl = kl * weights[mask].detach()
    return kl.sum() / float(n)


def norm_rho_client(rho_bar: torch.Tensor, proto_valid: torch.Tensor, eps: float) -> torch.Tensor:
    """对 proto_valid 类做 min-max。调用方应传入 rho_valid（已有压力估计），不要用 loc_valid。"""
    c = rho_bar.shape[0]
    out = torch.full((c,), 0.5, device=rho_bar.device, dtype=rho_bar.dtype)
    active = proto_valid.clone()
    if active.sum() < 2:
        return out
    vals = rho_bar[active]
    mn, mx = vals.min(), vals.max()
    if mx - mn < eps:
        out[active] = 0.5
    else:
        out[active] = ((rho_bar[active] - mn) / (mx - mn + eps)).clamp(0, 1)
    return out


@torch.no_grad()
def init_p_loc_from_labeled(
    model: nn.Module,
    data_labeled,
    num_classes: int,
    device: torch.device,
    feat_dim: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """仅用本地有标样本初始化单原型；无标类不初始化。"""
    # 使用原始数据长度，避免遍历2000倍复制的数据
    original_len = getattr(data_labeled, 'client_dataset_original_len', len(data_labeled))
    
    # 创建只遍历原始样本的subset
    from torch.utils.data import Subset
    if original_len < len(data_labeled):
        data_labeled = Subset(data_labeled, list(range(original_len)))
    
    loader = DataLoader(
        data_labeled,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    sums = torch.zeros(num_classes, feat_dim, device=device)
    cnts = torch.zeros(num_classes, device=device)
    model.eval()
    
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z, _ = model(x)
        z = F.normalize(z, dim=-1)
        for c in range(num_classes):
            m = y == c
            if m.any():
                sums[c] = sums[c] + z[m].sum(0)
                cnts[c] = cnts[c] + m.float().sum()
    
    p_loc = torch.zeros_like(sums)
    valid = cnts > 0
    p_loc[valid] = F.normalize(sums[valid], dim=-1)
    model.train()
    return p_loc, valid


def build_p_mix(p_loc: torch.Tensor, loc_valid: torch.Tensor, p_ref: torch.Tensor, ref_valid: torch.Tensor, lam: float) -> torch.Tensor:
    """p_mix = normalize(lambda*p_loc + (1-lambda)*p_ref)，无效侧退化为可用一侧。"""
    device = p_loc.device
    c, d = p_loc.shape
    out = torch.zeros(c, d, device=device)
    for cls in range(c):
        a = p_loc[cls] if loc_valid[cls] else None
        b = p_ref[cls] if ref_valid[cls] else None
        if a is None and b is None:
            continue
        if a is None:
            vec = b
        elif b is None:
            vec = a
        else:
            vec = lam * a + (1.0 - lam) * b
        out[cls] = F.normalize(vec, dim=0, eps=1e-12)
    return out


def prototype_contrastive_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    p_mix: torch.Tensor,
    mix_row_valid: torch.Tensor,
    T: float,
) -> torch.Tensor:
    """L_proto^L：仅 labeled；logits_c = sim(z,p_mix(c))/T。真类原型无效时不参与。"""
    if z.numel() == 0:
        return z.new_zeros(())
    logits = (z @ p_mix.T) / T
    logits = logits.masked_fill(~mix_row_valid.unsqueeze(0), -1e9)
    valid = mix_row_valid[y]
    if valid.sum() == 0:
        return z.new_zeros(())
    return F.cross_entropy(logits[valid], y[valid])


class LocalPPFPSL:
    """PPFPSL 客户端：仅 local_model，无 local_G。"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:%d" % args.gpu_id)
        self.model = ResNet(
            resnet_size=8,
            scaling=4,
            save_activations=False,
            group_norm_num_groups=None,
            freeze_bn=False,
            freeze_bn_affine=False,
            num_classes=args.num_classes,
        ).to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4)
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.dim = _infer_dim(self.model, self.device)

    def _forward_z_logits(self, x: torch.Tensor, net=None) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.model if net is None else net
        z, logits = m(x)
        z = F.normalize(z, dim=-1)
        return z, logits

    def train_round(
        self,
        args,
        data_client_labeled,
        data_client_unlabeled,
        global_params: Dict[str, torch.Tensor],
        p_ref: torch.Tensor,
        p_ref_valid: torch.Tensor,
        round_idx: int,
        client_state: Optional[Dict[str, Any]],
        labeled_counts: torch.Tensor,
        sched: Optional[Dict[str, Any]] = None,
        auxiliary_head: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, torch.Tensor]]:
        """单客户端、单通信轮本地训练。返回: (state_dict, new_client_state, proto_upload)"""
        
        if client_state is None:
            rho_bar = torch.zeros(args.num_classes, device=self.device)
            rho_valid = torch.zeros(args.num_classes, dtype=torch.bool, device=self.device)
            p_loc = torch.zeros(args.num_classes, self.dim, device=self.device)
            loc_valid = torch.zeros(args.num_classes, dtype=torch.bool, device=self.device)
            inited = False
        else:
            rho_bar = client_state["rho_bar"].to(self.device)
            rho_valid = client_state["rho_valid"].to(self.device)
            p_loc = client_state["p_loc"].to(self.device)
            loc_valid = client_state["loc_valid"].to(self.device)
            inited = bool(client_state.get("inited", False))

        self.model.load_state_dict(global_params)
        self.optimizer.state.clear()
        self.model.train()
        bc_enabled = bool(getattr(args, 'bc_teacher', 0))
        bc_gate = exploration_ramp(round_idx,30,90) if bc_enabled else 0.
        tail_mode = getattr(args, "bc_tail", "none")
        tail_gate = bc_tail.gate(round_idx) if tail_mode != "none" else 0.
        targets_enabled = bool(getattr(args,'bc_targets',0))
        target_gate = bc_targets.gate(round_idx) if targets_enabled else 0.
        target_mode = getattr(args, 'target_experiment', 'none')
        # Prototype initialization expects the original single labeled view.
        data_client_labeled.paired_labeled = False
        aux_history = None
        if target_mode == 'labelhead':
            if client_state is not None and 'aux_history' not in client_state:
                raise ValueError('labelhead resume requires complete auxiliary history')
            aux_history = bc_targets.History(data_client_unlabeled.indices, args.num_classes, self.device,
                                            (client_state or {}).get('aux_history'))
            if bc_gate > 0:
                if auxiliary_head is None:
                    raise ValueError('labelhead requires current online labeled head')
                self.aux_encoder = copy.deepcopy(self.model)
                # This project's ResNet.train/eval overrides return None.
                self.aux_encoder.eval()
                for parameter in self.aux_encoder.parameters():
                    parameter.requires_grad_(False)
        history = None
        if targets_enabled:
            if client_state is not None and 'target_history' not in client_state:
                raise ValueError('Target-routing resume requires complete per-client history')
            history = bc_targets.History(data_client_unlabeled.indices,args.num_classes,self.device,
                                         (client_state or {}).get('target_history'))
        data_client_unlabeled.second_weak = (targets_enabled and bc_gate > 0) or (tail_gate > 0 and tail_mode in ("breliability", "aguard"))
        prox_mu = prox_coefficient(round_idx,args) if getattr(args,'mid_prox_mu',0) else 0.
        anchor = {n:p.detach().clone() for n,p in self.model.named_parameters()} if prox_mu else None
        if bc_enabled:
            if not hasattr(self,'bc_heads'):
                self.bc_heads=initialize_heads(self.dim,self.device)
                self.bc_initial=copy.deepcopy(self.bc_heads.state_dict())
                self.optimizer.add_param_group({'params':self.bc_heads.parameters()})
            self.bc_heads.load_state_dict((client_state or {}).get('bc_heads',self.bc_initial))
            self.bc_heads.train()
            self.bc_target=copy.deepcopy(self.bc_heads.projector).eval()
            for p in self.bc_target.parameters():p.requires_grad_(False)
            self.teacher.load_state_dict(global_params)
            self.teacher.eval()
            for p in self.teacher.parameters():p.requires_grad_(False)
        use_teacher = bool(int(getattr(args, "pp_teacher", 0)))
        if use_teacher:
            self.teacher.load_state_dict(global_params)
            self.teacher.eval()

        cosine_lr = training_learning_rate(round_idx, args)
        for pg in self.optimizer.param_groups:
            pg['lr'] = cosine_lr

        if not inited:
            p_loc, loc_valid = init_p_loc_from_labeled(
                self.model,
                data_client_labeled,
                args.num_classes,
                self.device,
                self.dim,
                args.batch_size_local_labeled_fixmatch,
            )
            inited = True

        data_client_labeled.paired_labeled = target_mode == 'separation' and target_gate > 0
        if sched is None:
            phase = training_phase(round_idx, schedule_rounds(args), args)
            g_gate = gate_anneal(round_idx, schedule_rounds(args), args)
            aux_scale = warmup_aux_scale(round_idx, schedule_rounds(args), args)
            tau0_eff = float(args.tau0)
            delta0_eff = float(args.delta0)
            eta_B_eff = float(args.eta_B)
            tau_warmup_eff = float(args.tau_warmup)
            cap_p1 = float(getattr(args, "pp_a_ratio_cap_p1", 0.0))
            cap_p2 = float(getattr(args, "pp_a_ratio_cap", 0.0))
            if phase == 1:
                a_ratio_cap = cap_p1
            elif phase == 2:
                a_ratio_cap = _lerp(g_gate, cap_p1, cap_p2)
            else:
                a_ratio_cap = cap_p2
            b_ratio_cap = float(getattr(args, "pp_b_ratio_cap", 0.0))
            use_a_for_proto = True
            lambda_A_scale = 1.0
            lambda_B_scale = 1.0
        else:
            phase = int(sched["phase"])
            g_gate = float(sched["gate"])
            aux_scale = float(sched["aux_scale"])
            tau0_eff = float(sched["tau0"])
            delta0_eff = float(sched["delta0"])
            eta_B_eff = float(sched["eta_B"])
            tau_warmup_eff = float(sched["tau_warmup"])
            cap_p1 = float(getattr(args, "pp_a_ratio_cap_p1", 0.0))
            cap_p2 = float(getattr(args, "pp_a_ratio_cap", 0.0))
            a_ratio_cap = float(sched.get("a_ratio_cap", cap_p2))
            if phase == 2:
                a_ratio_cap = _lerp(g_gate, cap_p1, cap_p2)
            b_ratio_cap = float(sched.get("b_ratio_cap", getattr(args, "pp_b_ratio_cap", 0.0)))
            use_a_for_proto = float(sched.get("use_a_for_proto", 1.0)) > 0.5
            lambda_A_scale = float(sched.get("lambda_A_scale", 1.0))
            lambda_B_scale = float(sched.get("lambda_B_scale", 1.0))
        rho_bar_in = rho_bar.clone()
        # 只对已有压力估计的类做 min-max；未更新类保持 0.5，避免 rho=0 的 loc_valid 类拉歪 Norm
        norm_rho_in = norm_rho_client(rho_bar_in, rho_valid, args.pp_eps)

        z_labeled_sum = torch.zeros(args.num_classes, self.dim, device=self.device)
        z_labeled_cnt = torch.zeros(args.num_classes, device=self.device)
        z_a_sum = torch.zeros(args.num_classes, self.dim, device=self.device)
        w_a_sum = torch.zeros(args.num_classes, device=self.device)
        intra_num = torch.zeros(args.num_classes, device=self.device)
        intra_den = torch.zeros(args.num_classes, device=self.device)
        a_id_chunks = []
        a_cls_chunks = []
        a_w_chunks = []

        # 增广在 CPU 上是主瓶颈（单进程约 265ms/step，GPU 前反向约 23ms/step）。
        # loader 每客户端重建，故不用 persistent_workers；worker 由 torch 按
        # base_seed+worker_id 播种，随机流仍受 --seed 控制。
        nw = max(0, int(getattr(args, "num_workers", 0)))
        loader_kw = dict(num_workers=nw, pin_memory=True)
        if nw > 0:
            loader_kw["prefetch_factor"] = 4
            # loader 只服务当前客户端，但要跨 local_epochs 复用，避免反复起停 worker
            loader_kw["persistent_workers"] = True
        lab_loader = DataLoader(
            data_client_labeled,
            batch_size=args.batch_size_local_labeled_fixmatch,
            shuffle=True,
            drop_last=True,
            **loader_kw,
        )
        u_loader = DataLoader(
            data_client_unlabeled,
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
            shuffle=True,
            drop_last=True,
            **loader_kw,
        )

        audit_counts = audit_buffer(args.num_classes, self.device) if int(getattr(args, "diag_geom", 1)) and phase >= 2 else None
        log_acc = {
            "loss": 0.0,
            "L_sup": 0.0,
            "L_A": 0.0,
            "L_B": 0.0,
            "L_proto": 0.0,
            "n_batches": 0,
            "cnt_u": 0,
            "cnt_a": 0,
            "cnt_b": 0,
            "cnt_c": 0,
            "hce_all": 0,
            "hce_hc": 0,
            "hce_den_hc": 0,
            "a_correct": 0,
            "a_total": 0,
            "b_correct": 0,
            "b_total": 0,
            "lr": cosine_lr,
            "gate": g_gate,
            "aux_scale": aux_scale,
            "tau0": tau0_eff,
            "delta0": delta0_eff,
            "eta_B": eta_B_eff,
            "tau_warmup": tau_warmup_eff,
            "a_ratio_cap": a_ratio_cap,
            "b_ratio_cap": b_ratio_cap,
            "use_a_for_proto": float(use_a_for_proto),
            "lambda_A_scale": lambda_A_scale,
            "lambda_B_scale": lambda_B_scale,
            "r_mean": 0.0,
            "m_mean": 0.0,
            "alpha_t": 0.0,
            "pass_s": 0,
            "geom_drop": 0,
            "proto_a_wrong": 0.0,
        }

        # 使用原始数据集长度（未放大），避免重复迭代50倍
        real_unlabeled_len = getattr(data_client_unlabeled, 'client_dataset_len', len(data_client_unlabeled))
        local_iter = max(1, int(real_unlabeled_len / args.batch_size_local_labeled_fixmatch))
        

        # iter() 提到 epoch 外：每个 epoch 只有 local_iter 步，与 worker 预取量同量级，
        # 每 epoch 重建迭代器会把已预取的 batch 全部丢掉（实测取数开销多 1/3）。
        # 总步数仍是 local_epochs × local_iter，未改变训练步数。
        it_l = iter(lab_loader)
        it_u = iter(u_loader)
        trust_rows = []
        risk_enabled = bool(getattr(args, 'trusted_a_risk', 0))
        risk_buffer = new_risk_audit(args.num_classes, self.device) if risk_enabled else None
        risk_calibration_rows = []
        trusted_mode = getattr(args, "pp_geom_mode", "legacy") == "trusted" and phase >= 2
        weight_gate = trusted_weight_gate(round_idx, g_gate, args)
        early_weighting = (getattr(args, "pp_geom_mode", "legacy") == "trusted"
                           and getattr(args, "trusted_weight_start", 0) > 0 and weight_gate > 0)
        log_acc["trusted_weight_gate"] = weight_gate if (trusted_mode or early_weighting) else 0.0
        for _local_epoch in range(args.local_epochs):
            trusted_ref = None
            if trusted_mode or early_weighting:
                trusted_ref = refresh_reference(
                    self.model, data_client_labeled, to_tensor_normalize(args.dataset),
                    args.num_classes, self.device, int(getattr(args, "pp_trust_min_count", 4)),
                    **(dict(risk_threshold=tau_warmup_eff, risk_temperature=args.T) if risk_enabled else {}),
                )
                if risk_enabled:
                    cal = trusted_ref['risk_calibration']
                    cal_counts = cal['count'].cpu().tolist()
                    cal_errors = cal['errors'].cpu().tolist()
                    for cls in range(args.num_classes):
                        for band in range(2):
                            for group, group_name in enumerate(RISK_GROUPS):
                                risk_calibration_rows.append(dict(epoch=_local_epoch+1, cls=cls,
                                    confidence_band=band, geometry_group=group_name,
                                    query_count=cal_counts[cls][band][group],
                                    query_errors=cal_errors[cls][band][group]))
                if int(getattr(args, "diag_geom", 1)):
                    for cls in range(args.num_classes):
                        trust_rows.append(dict(epoch=_local_epoch + 1, cls=cls,
                            **{key: float(trusted_ref[key][cls]) for key in
                               ("support_n", "query_n", "votes", "trust", "radius", "margin_floor",
                                "distance_scale", "margin_scale")}))
                log_acc["trust_refreshes"] = log_acc.get("trust_refreshes", 0) + 1
                log_acc["trust_class_sum"] = log_acc.get("trust_class_sum", 0.0) + float(trusted_ref["trust"].sum())
            for _step in range(local_iter):
                try:
                    labeled_batch = next(it_l)
                except StopIteration:
                    it_l = iter(lab_loader)
                    labeled_batch = next(it_l)
                x, y = labeled_batch[:2]
                try:
                    u_batch = next(it_u)
                except StopIteration:
                    it_u = iter(u_loader)
                    u_batch = next(it_u)
                uw, us, y_u_gt, u_ids = u_batch[:4]
                second_weak = u_batch[4].to(self.device) if len(u_batch) == 5 else None

                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                uw = uw.to(self.device, non_blocking=True)
                us = us.to(self.device, non_blocking=True)
                y_u_gt = y_u_gt.to(self.device, non_blocking=True)
                u_ids = u_ids.to(self.device, non_blocking=True)

                z_x, logits_x = self._forward_z_logits(x)
                L_sup = F.cross_entropy(logits_x, y)

                with torch.no_grad():
                    for cls in range(args.num_classes):
                        m = y == cls
                        if m.any():
                            z_labeled_sum[cls] = z_labeled_sum[cls] + z_x[m].sum(0)
                            z_labeled_cnt[cls] = z_labeled_cnt[cls] + m.float().sum()
                            if loc_valid[cls] and phase >= 2:
                                sim_lc = (z_x[m] * p_loc[cls].unsqueeze(0)).sum(-1)
                                intra_num[cls] += (1.0 - sim_lc).sum()
                                intra_den[cls] = intra_den[cls] + m.sum().float()

                p_mix = build_p_mix(p_loc, loc_valid, p_ref, p_ref_valid, args.pp_lambda_mix)
                mix_valid = loc_valid | p_ref_valid

                if args.lambda_proto > 0:
                    proto_centers = trusted_ref["centers"] if trusted_mode else p_mix
                    proto_valid = trusted_ref["valid"] if trusted_mode else mix_valid
                    L_proto = prototype_contrastive_loss(z_x, y, proto_centers, proto_valid, args.pp_proto_T)
                else:
                    L_proto = z_x.new_zeros(())

                zs, logs = self._forward_z_logits(us)
                if use_teacher:
                    with torch.no_grad():
                        zw, logw = self._forward_z_logits(uw, self.teacher)
                else:
                    zw, logw = self._forward_z_logits(uw)

                target_teacher = None
                target_result = None
                auxiliary_prediction = None
                if targets_enabled and bc_gate > 0:
                    with torch.no_grad():
                        zt_new,lt_new = self._forward_z_logits(uw,self.teacher)
                        _,lt2_new = self._forward_z_logits(second_weak,self.teacher)
                        target_scores,target_q,target_p = bc_tail.reliability(lt_new,lt2_new,args.T)
                        hist_q,hist_mature = history.lookup(u_ids,round_idx)
                        history.observe(u_ids,(target_q+target_p)/2)
                        if aux_history is not None:
                            az1, _ = self._forward_z_logits(uw, self.aux_encoder)
                            az2, _ = self._forward_z_logits(second_weak, self.aux_encoder)
                            aq1, ag1 = target_experiments.predict_head(az1, auxiliary_head)
                            aq2, ag2 = target_experiments.predict_head(az2, auxiliary_head)
                            ah, am = aux_history.lookup(u_ids, round_idx)
                            aq = (aq1+aq2)/2
                            ag = ag1 & ag2 & (aq1.argmax(-1) == aq2.argmax(-1))
                            auxiliary_prediction = (aq, ag, ah, am, auxiliary_head['valid'].to(self.device))
                            aux_history.observe(u_ids, aq)
                    target_teacher = (zt_new,lt_new)

                probs = F.softmax(logw / args.T, dim=-1)
                s_i, yhat = probs.max(dim=-1)

                if phase == 1:
                    in_A, in_B, in_C, pass_s = _route_phase1(
                        s_i,
                        tau_warmup_eff,
                        eta_B_eff,
                        aux_scale > 1e-6,
                        a_ratio_cap,
                        b_ratio_cap,
                    )
                    mask = in_A
                    w_A = mask.float()
                    wb_phase1 = s_i
                    if early_weighting:
                        was_training = self.model.training
                        self.model.eval()
                        try:
                            with torch.no_grad():
                                zg, geometry_logits = self._forward_z_logits(uw)
                        finally:
                            self.model.train(was_training)
                        early_tg = score_queries(zg, yhat, trusted_ref, weight_gate, _step / max(1, local_iter))
                        w_A = w_A * early_tg["weight"]
                        if risk_enabled:
                            risk = score_a_risk(zg, yhat, geometry_logits, trusted_ref, weight_gate,
                                _step / max(1, local_iter), args.risk_distance_cap,
                                args.risk_conflict_cap, args.risk_prior_count, args.T)
                            audit_a_risk(risk_buffer, risk, in_A, yhat, y_u_gt, early_tg['weight'])
                            w_A = in_A.float() * risk['weight']
                        wb_phase1 = s_i * early_tg["weight"]
                        log_acc["trust_authority_sum"] = log_acc.get("trust_authority_sum", 0.0) + float(early_tg["authority"].sum())
                        log_acc["trust_valid_visits"] = log_acc.get("trust_valid_visits", 0) + int(early_tg["valid"].sum())
                    L_A, L_B = _unlabeled_ab_losses(
                        logs, logw, yhat, in_A, in_B, w_A, wb_phase1, args.T
                    )
                    if target_gate > 0:
                        target_result = bc_targets.decide(yhat,in_A,in_B,target_q,target_p,target_scores,
                            hist_q,hist_mature,zg,trusted_ref)
                        if target_mode != 'none':
                            target_result = target_experiments.revise(target_result,target_mode,yhat,in_A,
                                target_q,target_p,hist_q,hist_mature,zg,trusted_ref,auxiliary_prediction,
                                guard=bool(getattr(args,'labelhead_guard',0)))
                        L_A,target_lb,target_proto = bc_targets.objectives(logs,yhat,in_A,in_B,
                            w_A,wb_phase1,target_result,target_gate)
                        target_audit = bc_targets.audit(target_result,yhat,in_A,in_B,y_u_gt,
                            w_A,wb_phase1,target_gate).cpu().tolist()
                        for cls,values in enumerate(target_audit):
                            for name,value in zip(bc_targets.FIELDS,values):
                                key=f'target_c{cls}_{name}'
                                log_acc[key]=log_acc.get(key,0.)+value
                    log_acc["r_mean"] += float(s_i.mean().item())
                    log_acc["m_mean"] += 0.0
                    log_acc["alpha_t"] += 1.0
                    log_acc["pass_s"] += int(pass_s.sum().item())
                    log_acc["geom_drop"] += 0
                    with torch.no_grad():
                        if mask.any():
                            correct = (yhat[mask] == y_u_gt[mask]).float()
                            log_acc["a_total"] += mask.sum()
                            log_acc["a_correct"] += correct.sum()
                        if in_B.any():
                            log_acc["b_total"] += in_B.sum()
                            log_acc["b_correct"] += (yhat[in_B] == y_u_gt[in_B]).float().sum()
                        if use_a_for_proto and mask.any():
                            zw_a = zw[mask].detach()
                            cc = yhat[mask]
                            s_sel = s_i[mask]
                            gt_sel = y_u_gt[mask]
                            ids_sel = u_ids[mask]
                            wi = (s_sel ** args.w_gamma).clamp(args.w_min, 1.0)
                            if target_result is not None:
                                wi = wi * target_proto[mask]
                            # 写原型独立于 CE：更严筛选 / 更狠权重，默认无操作
                            wi, keep = _proto_write_weight(wi, s_sel, args)
                            if keep is not None:
                                zw_a = zw_a[keep]
                                cc = cc[keep]
                                wi = wi[keep]
                                gt_sel = gt_sel[keep]
                                ids_sel = ids_sel[keep]
                            if wi.numel() > 0:
                                a_id_chunks.append(ids_sel.detach())
                                a_cls_chunks.append(cc.detach())
                                a_w_chunks.append(wi.detach())
                                log_acc["proto_a_wrong"] += float(
                                    wi[cc != gt_sel].sum().item()
                                )
                                for cls in range(args.num_classes):
                                    mm = cc == cls
                                    if mm.any():
                                        z_a_sum[cls] += (zw_a[mm] * wi[mm].unsqueeze(1)).sum(0)
                                        w_a_sum[cls] += wi[mm].sum()
                    log_acc["cnt_u"] += yhat.numel()
                    log_acc["cnt_a"] += mask.sum()
                    log_acc["cnt_b"] += in_B.sum()
                    log_acc["cnt_c"] += in_C.sum()
                    hce = (yhat != y_u_gt) & mask
                    log_acc["hce_all"] += (yhat != y_u_gt).sum()
                    log_acc["hce_hc"] += hce.sum()
                    log_acc["hce_den_hc"] += (s_i >= args.hce_tau).sum()
                    loss = L_sup + (args.lambda_A * lambda_A_scale) * L_A
                    if aux_scale > 1e-6:
                        loss = loss + aux_scale * args.lambda_B * lambda_B_scale * L_B
                        if args.lambda_proto > 0:
                            loss = loss + aux_scale * args.lambda_proto * L_proto
                else:
                    if trusted_mode:
                        # Same current student, eval-only geometry forward: no BN updates or teacher labels.
                        was_training = self.model.training
                        self.model.eval()
                        try:
                            with torch.no_grad():
                                zg, geometry_logits = self._forward_z_logits(uw)
                        finally:
                            self.model.train(was_training)
                        tg = score_queries(zg, yhat, trusted_ref, weight_gate, _step / max(1, local_iter))
                        in_A, in_B, in_C, pass_s = _route_phase1(
                            s_i, tau_warmup_eff, eta_B_eff, True, a_ratio_cap, b_ratio_cap,
                        )
                        tau_i = torch.full_like(s_i, tau_warmup_eff)
                        raw_tau_i = tau_i
                        # Audit score combines class-relative margin and absolute distance.
                        m_i, geom_ok = tg["score"], tg["valid"]
                        delta_i = torch.zeros_like(s_i)
                        m_tilde = tg["reliability"]
                        alpha_t = 1.0 - tg["authority"]
                        r_i = s_i * tg["weight"]
                        b_score = r_i
                        fail_geom = torch.zeros_like(in_A)  # No geometry hard rejection in this mode.
                        diag_z, diag_p, diag_valid = zg, trusted_ref["centers"], trusted_ref["valid"]
                        log_acc["trust_authority_sum"] = log_acc.get("trust_authority_sum", 0.0) + float(tg["authority"].sum())
                        log_acc["trust_valid_visits"] = log_acc.get("trust_valid_visits", 0) + int(geom_ok.sum())
                    else:
                        nr = norm_rho_in
                        ceiling = float(getattr(args, "pp_tau_ceiling", 0.99))
                        boost = args.pp_phase3_geom_boost if phase == 3 else 0.0
                        tau_c, delta_c, raw_tau_c = _class_thresholds(
                            nr, tau0_eff, delta0_eff, args.pp_a, args.pp_b, boost, ceiling
                        )

                        m_floor = -2.0
                        tau_i = _lerp(g_gate, tau_warmup_eff, tau_c[yhat])
                        delta_i = _lerp(g_gate, m_floor, delta_c[yhat])
                        raw_tau_i = _lerp(g_gate, tau_warmup_eff, raw_tau_c[yhat])

                        m_i, geom_ok, own_ok = _geom_margin(zw, yhat, p_mix, mix_valid)
                        m0 = float(getattr(args, "pp_m0", 0.05))
                        mT = float(getattr(args, "pp_mT", 0.08))
                        m_tilde = _margin_reliability(m_i, m0, mT)
                        m_tilde = torch.where(geom_ok, m_tilde, torch.full_like(m_tilde, 0.5))
                        alpha_t = _blend_alpha(g_gate, nr[yhat], args)
                        r_i = alpha_t * s_i + (1.0 - alpha_t) * m_tilde

                        routed = _route_phase23(
                            s_i,
                            m_i,
                            m_tilde,
                            r_i,
                            tau_i,
                            delta_i,
                            own_ok,
                            g_gate,
                            eta_B_eff,
                            a_ratio_cap,
                            b_ratio_cap,
                            float(getattr(args, "pp_b_min_margin", 0.0)),
                            m_floor,
                            geom_ok,
                            b_conf_rescue=bool(int(getattr(args, "pp_b_conf_rescue", 0))),
                        )
                        in_A, in_B, in_C = routed["in_A"], routed["in_B"], routed["in_C"]
                        pass_s, fail_geom = routed["pass_s"], routed["fail_geom"]
                        b_score = routed["b_score"]

                        diag_z, diag_p, diag_valid = zw, p_mix, mix_valid

                    if int(getattr(args, "diag_geom", 1)):
                        for key, value in {
                            "dt_n": yhat.numel(),
                            "dt_clipped_n": int((raw_tau_i > tau_i).sum().item()),
                            "dt_raw_ge1_n": int((raw_tau_i >= 1.0).sum().item()),
                            "dt_effective_ge1_n": int((tau_i >= 1.0).sum().item()),
                        }.items():
                            log_acc[key] = log_acc.get(key, 0) + value
                        b_m_thr = 0.0 if trusted_mode else _lerp(
                            g_gate, m_floor, float(getattr(args, "pp_b_min_margin", 0.0))
                        )
                        # Candidate mask independent of the rescue switch, for paired diagnostics.
                        score_only = (~in_A) & (s_i >= eta_B_eff) & (b_score < eta_B_eff)
                        score_only = score_only & ((~geom_ok) | (m_i >= b_m_thr))
                        if trusted_mode:
                            score_only = torch.zeros_like(score_only)  # legacy rescue diagnostic only
                        for key, value in {
                            "dyn_b_score_only_n": int(score_only.sum().item()),
                            "dyn_b_score_only_correct": int((score_only & (yhat == y_u_gt)).sum().item()),
                            "dyn_b_score_only_admitted": int((score_only & in_B).sum().item()),
                        }.items():
                            log_acc[key] = log_acc.get(key, 0) + value
                        dd = _diag_geom_c(
                            s_i, yhat, y_u_gt, m_i, delta_i, geom_ok, in_C,
                            eta_B_eff, diag_z, diag_p, diag_valid,
                            float(getattr(args, "diag_s_lo", 0.95)),
                            float(getattr(args, "diag_s_hi", 0.99)),
                            b_m_thr=b_m_thr,
                        )
                        for _k, _v in dd.items():
                            log_acc[_k] = log_acc.get(_k, 0) + _v

                    if trusted_mode:
                        w_full = in_A.float() * tg["weight"]
                        if risk_enabled:
                            risk = score_a_risk(zg, yhat, geometry_logits, trusted_ref, weight_gate,
                                _step / max(1, local_iter), args.risk_distance_cap,
                                args.risk_conflict_cap, args.risk_prior_count, args.T)
                            audit_a_risk(risk_buffer, risk, in_A, yhat, y_u_gt, tg['weight'])
                            w_full = in_A.float() * risk['weight']
                    else:
                        w_geom = (s_i ** args.w_gamma * m_tilde).clamp(args.w_min, 1.0)
                        w_full = torch.zeros_like(s_i)
                        if in_A.any():
                            w_full[in_A] = _lerp(g_gate, torch.ones_like(w_geom[in_A]), w_geom[in_A])
                    # A 的可靠性权重不反传；有标 L_proto 仍通过原型对比更新特征
                    L_A, L_B = _unlabeled_ab_losses(
                        logs, logw, yhat, in_A, in_B, w_full, b_score, args.T
                    )
                    if target_gate > 0:
                        target_result = bc_targets.decide(yhat,in_A,in_B,target_q,target_p,target_scores,
                            hist_q,hist_mature,zg,trusted_ref)
                        if target_mode != 'none':
                            target_result = target_experiments.revise(target_result,target_mode,yhat,in_A,
                                target_q,target_p,hist_q,hist_mature,zg,trusted_ref,auxiliary_prediction,
                                guard=bool(getattr(args,'labelhead_guard',0)))
                        L_A,target_lb,target_proto = bc_targets.objectives(logs,yhat,in_A,in_B,
                            w_full,b_score,target_result,target_gate)
                        target_audit = bc_targets.audit(target_result,yhat,in_A,in_B,y_u_gt,
                            w_full,b_score,target_gate).cpu().tolist()
                        for cls,values in enumerate(target_audit):
                            for name,value in zip(bc_targets.FIELDS,values):
                                key=f'target_c{cls}_{name}'
                                log_acc[key]=log_acc.get(key,0.)+value
                    log_acc["r_mean"] += float(r_i.mean().item())
                    log_acc["m_mean"] += float(m_i.mean().item())
                    log_acc["alpha_t"] += float(alpha_t.mean().item())
                    log_acc["pass_s"] += int(pass_s.sum().item())
                    log_acc["geom_drop"] += int(fail_geom.sum().item())

                    log_acc["cnt_u"] += yhat.numel()
                    log_acc["cnt_a"] += in_A.sum()
                    log_acc["cnt_b"] += in_B.sum()
                    log_acc["cnt_c"] += in_C.sum()
                    log_acc["hce_all"] += (yhat != y_u_gt).sum()
                    log_acc["hce_hc"] += ((yhat != y_u_gt) & (s_i >= args.hce_tau)).sum()
                    log_acc["hce_den_hc"] += (s_i >= args.hce_tau).sum()
                    with torch.no_grad():
                        if in_A.any():
                            correct = (yhat[in_A] == y_u_gt[in_A]).float()
                            log_acc["a_total"] += in_A.sum()
                            log_acc["a_correct"] += correct.sum()
                        if in_B.any():
                            log_acc["b_total"] += in_B.sum()
                            log_acc["b_correct"] += (yhat[in_B] == y_u_gt[in_B]).float().sum()
                    with torch.no_grad():
                        if use_a_for_proto and in_A.any():
                            zw_a = zw[in_A]
                            cc = yhat[in_A]
                            wi = w_full[in_A]
                            if target_result is not None:
                                wi = wi * target_proto[in_A]
                            s_sel = s_i[in_A]
                            gt_sel = y_u_gt[in_A]
                            ids_sel = u_ids[in_A]
                            # 写原型独立于 CE：更严筛选 / 更狠权重，默认无操作
                            wi, keep = _proto_write_weight(wi, s_sel, args)
                            if keep is not None:
                                zw_a = zw_a[keep]
                                cc = cc[keep]
                                wi = wi[keep]
                                gt_sel = gt_sel[keep]
                                ids_sel = ids_sel[keep]
                            if wi.numel() > 0:
                                a_id_chunks.append(ids_sel.detach())
                                a_cls_chunks.append(cc.detach())
                                a_w_chunks.append(wi.detach())
                                log_acc["proto_a_wrong"] += float(
                                    wi[cc != gt_sel].sum().item()
                                )
                                sim_pa = (zw_a * p_loc[cc]).sum(-1)
                                intra_num.scatter_add_(0, cc, wi * (1.0 - sim_pa))
                                intra_den.scatter_add_(0, cc, wi)
                                for cls in range(args.num_classes):
                                    mm = cc == cls
                                    if mm.any():
                                        z_a_sum[cls] += (zw_a[mm] * wi[mm].unsqueeze(1)).sum(0)
                                        w_a_sum[cls] += wi[mm].sum()

                    loss = (
                        L_sup
                        + (args.lambda_A * lambda_A_scale) * L_A
                        + (args.lambda_B * lambda_B_scale) * L_B
                        + args.lambda_proto * L_proto
                    )

                if bc_enabled:
                    teacher_b = L_B
                    feature_effective = logs.new_zeros(())
                    if bc_gate > 0:
                        with torch.no_grad():
                            zt, lt = target_teacher if target_teacher is not None else self._forward_z_logits(uw,self.teacher)
                        bw = wb_phase1 if phase == 1 else b_score
                        if second_weak is not None and not targets_enabled:
                            with torch.no_grad():
                                _, lt2 = self._forward_z_logits(second_weak, self.teacher)
                                scores, tq, tq2 = bc_tail.reliability(lt, lt2, args.T)
                            if tail_mode == 'breliability':
                                old_bw = bw.detach() * in_B
                                bw = bc_tail.reweight_b(bw, in_B, scores, tail_gate)
                                teacher_wrong = tq.argmax(-1) != y_u_gt
                                for key, value in dict(tail_b_old_mass=old_bw.sum(), tail_b_new_mass=bw.sum(),
                                    tail_b_old_wrong=old_bw[teacher_wrong].sum(), tail_b_new_wrong=bw[teacher_wrong].sum()).items():
                                    log_acc[key] = log_acc.get(key, 0.) + float(value)
                                for band, selected in enumerate((scores < .6, (scores >= .6) & (scores < .9), scores >= .9)):
                                    selected = selected & in_B
                                    for suffix, value in [('visits', selected.sum()), ('correct', (selected & ~teacher_wrong).sum())]:
                                        key = f'tail_score_{band}_{suffix}'
                                        log_acc[key] = log_acc.get(key, 0.) + float(value)
                            elif tail_mode == 'aguard':
                                old_aw = w_A if phase == 1 else w_full
                                new_aw, selected = bc_tail.guard_a(old_aw, in_A, yhat, tq, tq2, tail_gate)
                                old_a = L_A
                                L_A = (new_aw * F.cross_entropy(logs, yhat, reduction='none')).mean()
                                loss = loss + args.lambda_A*lambda_A_scale*(L_A-old_a)
                                if phase == 1: w_A = new_aw
                                else: w_full = new_aw
                                for key, value in dict(tail_a_guard_visits=selected.sum(),
                                    tail_a_guard_correct=(selected & (yhat == y_u_gt)).sum(),
                                    tail_a_removed_correct=((old_aw-new_aw)*(yhat == y_u_gt)).sum(),
                                    tail_a_removed_wrong=((old_aw-new_aw)*(yhat != y_u_gt)).sum()).items():
                                    log_acc[key] = log_acc.get(key, 0.) + float(value)
                        teacher_b, feature = teacher_objectives(zs,logs,zt,lt,self.bc_heads,
                            self.bc_target,in_B,in_C,bw,args.T)
                        if target_result is not None:
                            teacher_b = (1-target_gate)*teacher_b + target_gate*target_lb
                            # A samples released from a hard answer keep a gradual feature objective.
                            extra_a = target_result.get('feature_a',target_result['a_changed'])
                            _,extra_feature = teacher_objectives(zs,logs,zt,lt,self.bc_heads,
                                self.bc_target,extra_a,torch.zeros_like(in_C),torch.zeros_like(bw),args.T)
                            feature = feature + target_gate*extra_feature
                        old_b = L_B
                        L_B = (1-bc_gate)*old_b + bc_gate*teacher_b
                        bcoef = args.lambda_B*lambda_B_scale*(aux_scale if phase==1 else 1.)
                        feature_coefficient = args.bc_feature_weight - (.05*tail_gate if tail_mode == "featurehalf" else 0.)
                        feature_effective = bc_gate*feature_coefficient*feature
                        loss = loss + bcoef*(L_B-old_b) + feature_effective
                        # Diagnostic GT is only consumed after targets and loss are computed.
                        log_acc['bc_teacher_b_correct'] = log_acc.get('bc_teacher_b_correct',0.) + float(((lt.argmax(-1)==y_u_gt)&in_B).sum())
                        log_acc['bc_teacher_b_total'] = log_acc.get('bc_teacher_b_total',0.) + float(in_B.sum())
                        with torch.no_grad():
                            teacher_q = F.softmax(lt/args.T, -1)
                            true_prob = teacher_q.gather(1, y_u_gt[:,None]).squeeze(1)
                            for key, value in dict(tail_b_true_prob=(true_prob*in_B).sum(),
                                tail_b_true_nll=(-true_prob.clamp_min(1e-12).log()*in_B).sum()).items():
                                log_acc[key] = log_acc.get(key, 0.) + float(value)
                        log_acc['bc_feature_std'] = log_acc.get('bc_feature_std',0.) + float(zs.detach().std(dim=0,unbiased=False).mean())
                    log_acc['L_feature_effective'] = log_acc.get('L_feature_effective',0.) + float(feature_effective.detach())
                if target_mode == 'separation':
                    separation_effective = logs.new_zeros(())
                    if target_gate > 0:
                        # Extra labeled view must not alter the model/teacher BN buffers.
                        with target_experiments.no_bn_updates(self.model):
                            z_second, _ = self._forward_z_logits(labeled_batch[2].to(self.device))
                        separation = target_experiments.separation_loss(z_x,z_second,y,
                            labeled_batch[3].to(self.device),logits_x)
                        separation_effective = .03*target_gate*separation
                        loss = loss + separation_effective
                        log_acc['separation_active_batches'] = log_acc.get('separation_active_batches',0) + int(y.unique().numel() >= 2)
                        log_acc['separation_unique_labeled'] = log_acc.get('separation_unique_labeled',0) + int(labeled_batch[3].unique().numel())
                    log_acc['L_separation_effective'] = log_acc.get('L_separation_effective',0.) + float(separation_effective.detach())
                if target_mode != 'none':
                    # Read-only counts; never consumed by routing or training schedules.
                    with torch.no_grad():
                        cm = torch.bincount(yhat[in_A]*args.num_classes+y_u_gt[in_A],
                                            minlength=args.num_classes**2).view(args.num_classes,args.num_classes)
                        for cls, values in enumerate(cm.cpu().tolist()):
                            for truth_cls, value in enumerate(values):
                                key = f'experiment_a_pred{cls}_true{truth_cls}'
                                log_acc[key] = log_acc.get(key,0.)+value
                        if target_result is not None:
                            audit_weights = w_A if phase == 1 else w_full
                            extra_audit = target_experiments.audit(target_result,yhat,in_A,y_u_gt,audit_weights)
                            if getattr(args,'labelhead_guard',0):
                                for key,value in target_experiments.guard_audit(target_result,yhat,y_u_gt,audit_weights,target_gate).items():
                                    log_acc[key] = log_acc.get(key,0.)+value
                            for cls, values in enumerate(extra_audit.cpu().tolist()):
                                for name, value in zip(target_experiments.AUDIT_FIELDS,values):
                                    key = f'experiment_c{cls}_{name}'
                                    log_acc[key] = log_acc.get(key,0.)+value
                if getattr(args,'mid_prox_mu',0):
                    prox = proximal_loss(self.model,anchor,prox_mu) if prox_mu else logs.new_zeros(())
                    loss = loss + prox
                    log_acc['L_prox_effective'] = log_acc.get('L_prox_effective',0.) + float(prox.detach())
                if audit_counts is not None:
                    audit_accumulate(audit_counts, s_i, yhat, y_u_gt, m_i, delta_i,
                                     geom_ok, in_A, w_full)
                if int(getattr(args, "diag_geom", 1)):
                    # Read-only: GT never changes weights, routing, or losses.
                    with torch.no_grad():
                        wa_diag = w_A.detach() if phase == 1 else w_full.detach()
                        wb_diag = (bw if tail_gate > 0 and tail_mode == 'breliability' else
                                   (wb_phase1 if phase == 1 else b_score)).detach() * in_B
                        wa_diag = wa_diag * (args.lambda_A * lambda_A_scale)
                        wb_diag = wb_diag * (args.lambda_B * lambda_B_scale)
                        if phase == 1:
                            wb_diag = wb_diag * aux_scale
                        wrong_diag = yhat != y_u_gt
                        for bucket, weights_diag in (("a", wa_diag), ("b", wb_diag)):
                            for suffix, value in (("mass", weights_diag.sum()),
                                                  ("wrong", weights_diag[wrong_diag].sum())):
                                key = "dyn_" + bucket + "_" + suffix
                                log_acc[key] = log_acc.get(key, 0.0) + float(value.item())

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                if bc_enabled:
                    bc_ema_update(self.teacher,self.model,args.bc_ema)
                    bc_ema_update(self.bc_target,self.bc_heads.projector,args.bc_ema)

                log_acc["loss"] += float(loss.detach().item())
                log_acc["L_sup"] += float(L_sup.detach().item())
                log_acc["L_A"] += float(L_A.detach().item()) if torch.is_tensor(L_A) else L_A
                log_acc["L_B"] += float(L_B.detach().item()) if torch.is_tensor(L_B) else L_B
                log_acc["L_proto"] += float(L_proto.detach().item()) if torch.is_tensor(L_proto) else L_proto
                log_acc["n_batches"] += 1

        # persistent_workers 的进程随 loader 释放；显式关掉，避免多客户端叠加
        del it_l, it_u
        del lab_loader, u_loader

        rho_new = rho_bar.clone()
        rho_v_new = rho_valid.clone()
        if phase >= 2:
            active_classes = (z_labeled_cnt + w_a_sum) >= args.pp_min_class_count
            for cls in range(args.num_classes):
                if not loc_valid[cls] or not active_classes[cls]:
                    continue
                idxs = [c2 for c2 in range(args.num_classes) if c2 != cls and loc_valid[c2]]
                if len(idxs) == 0:
                    continue
                if intra_den[cls] < 1e-12:
                    continue
                d_intra = (intra_num[cls] / (intra_den[cls] + 1e-12)).clamp_min(0.0)
                p_c = F.normalize(p_loc[cls], dim=0, eps=1e-12)
                d_inter_list = []
                for c2 in idxs:
                    p2 = F.normalize(p_loc[c2], dim=0, eps=1e-12)
                    d_inter_list.append(1.0 - float((p_c @ p2).item()))
                if len(d_inter_list) == 0:
                    continue
                d_inter = min(d_inter_list)
                rho_t = (d_intra / (d_inter + args.pp_eps)).clamp(0.0, 1e6)
                if rho_v_new[cls]:
                    rho_new[cls] = args.mu_rho * rho_new[cls] + (1.0 - args.mu_rho) * rho_t
                else:
                    rho_new[cls] = rho_t
                    rho_v_new[cls] = True

        p_new = p_loc.clone()
        loc_new = loc_valid.clone()
        for cls in range(args.num_classes):
            num = z_labeled_sum[cls] + z_a_sum[cls]
            den = z_labeled_cnt[cls] + w_a_sum[cls]
            if den < 1e-12:
                continue
            mean_z = F.normalize(num / (den + 1e-12), dim=0, eps=1e-12)
            if loc_valid[cls]:
                p_new[cls] = F.normalize(
                    args.mu_p * p_loc[cls] + (1.0 - args.mu_p) * mean_z, dim=0, eps=1e-12
                )
            else:
                p_new[cls] = mean_z
                loc_new[cls] = True

        if a_id_chunks:
            w_a_unique = _unique_a_mass(
                torch.cat(a_id_chunks),
                torch.cat(a_cls_chunks),
                torch.cat(a_w_chunks),
                args.num_classes,
            ).to(self.device)
        else:
            w_a_unique = torch.zeros(args.num_classes, device=self.device)
        w_agg = labeled_counts.to(self.device) + args.lambda_p * w_a_unique
        # |L| 为独立有标计数；Auniq 含无标 loader 中进 A 的样本（有标并入无标时会双重加权）

        log_final = {}
        for k, v in log_acc.items():
            if k in ("loss", "L_sup", "L_A", "L_B", "L_proto", "r_mean", "m_mean", "alpha_t", "L_feature_effective", "L_prox_effective", "L_separation_effective", "bc_feature_std"):
                log_final[k] = v / max(1, log_acc["n_batches"])
            elif k in (
                "lr",
                "gate",
                "aux_scale",
                "tau0",
                "delta0",
                "eta_B",
                "tau_warmup",
                "a_ratio_cap",
                "b_ratio_cap",
                "use_a_for_proto",
                "lambda_A_scale",
                "lambda_B_scale",
                "trusted_weight_gate",
            ):
                log_final[k] = v
            elif isinstance(v, torch.Tensor):
                log_final[k] = int(v.item())
            else:
                log_final[k] = v
        lab_mass = float(z_labeled_cnt.sum().item())
        a_mass = float(w_a_sum.sum().item())
        lab_uniq = float(labeled_counts.sum().item())
        log_final["proto_lab"] = lab_mass
        log_final["proto_a"] = a_mass
        log_final["proto_a_unique"] = float(w_a_unique.sum().item())
        log_final["lab_mult"] = lab_mass / max(lab_uniq, 1.0)
        log_final["u_seen"] = int(log_acc.get("cnt_u", 0))

        new_state = {
            "rho_bar": rho_new.detach().cpu(),
            "rho_valid": rho_v_new.detach().cpu(),
            "p_loc": p_new.detach().cpu(),
            "loc_valid": loc_new.detach().cpu(),
            "inited": True,
            "log": log_final,
            "geometry_audit": audit_counts.cpu().tolist() if audit_counts is not None else None,
            "trust_rows": trust_rows,
            "risk_rows": risk_audit_rows(risk_buffer) if risk_buffer is not None else [],
            "risk_calibration_rows": risk_calibration_rows,
        }

        proto_upload = {
            "p_loc": p_new.detach(),
            "w_agg": w_agg.detach(),
            "loc_valid": loc_new.detach(),
        }
        if bc_enabled:
            new_state['bc_heads'] = {k:v.detach().cpu().clone() for k,v in self.bc_heads.state_dict().items()}

        if history is not None:
            new_state['target_history'] = history.finish(round_idx)
        if aux_history is not None:
            new_state['aux_history'] = aux_history.finish(round_idx)

        return copy.deepcopy(self.model.state_dict()), new_state, proto_upload


def aggregate_p_ref(
    p_ref: torch.Tensor,
    p_ref_valid: torch.Tensor,
    uploads: list,
    device: torch.device,
    dim: int,
    num_classes: int,
) -> None:
    """服务器端加权聚合参考原型，全无效类保持上一轮。"""
    numer = torch.zeros(num_classes, dim, device=device)
    denom = torch.zeros(num_classes, device=device)
    for up in uploads:
        pl = up["p_loc"]
        w = up["w_agg"]
        lv = up["loc_valid"]
        for c in range(num_classes):
            if lv[c] and w[c] > 0:
                numer[c] += w[c] * pl[c]
                denom[c] += w[c]
    mask = denom > 0
    p_ref[mask] = F.normalize(numer[mask] / denom[mask].unsqueeze(-1), dim=-1, eps=1e-12)
    p_ref_valid[mask] = True


def _labeled_class_counts(dataset, indices: list, num_classes: int):
    """每个客户端本地有标样本的类频（用于原型聚合权重 |L^c|）。"""
    cnt = torch.zeros(num_classes, dtype=torch.float32)
    for idx in indices:
        y = dataset[idx][1]
        if hasattr(y, 'item'):
            y = int(y.item())
        else:
            y = int(y)
        cnt[y] += 1.0
    return cnt


class Global(object):
    """服务端：全局 ResNet + FedAvg + 类参考原型 p_ref。"""

    def __init__(self, args):
        self.gpu_id = args.gpu_id
        self.num_workers = max(0, int(getattr(args, "num_workers", 0)))
        self.model = ResNet(
            resnet_size=8,
            scaling=4,
            save_activations=False,
            group_norm_num_groups=None,
            freeze_bn=False,
            freeze_bn_affine=False,
            num_classes=args.num_classes,
        )
        self.model.cuda(args.gpu_id)
        self.num_classes = args.num_classes
        
        # PPFPSL 原型管理
        with torch.no_grad():
            z0, _ = self.model(torch.zeros(1, 3, 32, 32, device=torch.device('cuda', self.gpu_id)))
        self.feat_dim = int(z0.shape[-1])
        self.p_ref = torch.zeros(args.num_classes, self.feat_dim, device=z0.device)
        self.p_ref_valid = torch.zeros(args.num_classes, dtype=torch.bool, device=z0.device)

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        return fedavg_global_params

    def fedavg_eval(self, fedavg_params, data_test, batch_size_test):
        self.model.load_state_dict(fedavg_params)
        self.model.eval()
        with no_grad():
            test_loader = DataLoader(
                data_test,
                batch_size_test,
                num_workers=min(4, self.num_workers),
                pin_memory=True,
            )
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.cuda(self.gpu_id), labels.cuda(self.gpu_id)
                _, outputs = self.model(images)
                _, predicts = torch.max(outputs, dim=-1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return self.model.state_dict()


def fixmatch(alpha):
    """一次完整联邦训练实验。"""
    args = args_parser()
    apply_run_seeds(args)
    paths = _run_paths(args.dataset, alpha, args)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=paths["log_file"],
    )
    print(
        f'[Run] id={paths["run_id"]}\n'
        f'  目录={paths["run_dir"]}\n'
        f'  acc.csv / metrics.csv / diag.csv / config.json / train.log / checkpoint.pt'
    )
    logging.info(
        'run_id=%s acc=%s metrics=%s ckpt=%s',
        paths["run_id"], paths["acc_path"], paths["metrics_path"], paths["ckpt_path"],
    )

    if args.dataset == 'CIFAR10':
        args.num_classes = 10
        args.num_labeled = 500
        args.num_rounds = 300
        transform_test = to_tensor_normalize("CIFAR10")
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_test)

    elif args.dataset == 'CIFAR100':
        args.num_classes = 100
        args.num_labeled = 50
        args.num_rounds = 500
        transform_test = to_tensor_normalize("CIFAR100")
        data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_test)

    elif args.dataset == 'SVHN':
        args.num_classes = 10
        args.num_labeled = 460
        args.num_rounds = 150
        transform_test = to_tensor_normalize("SVHN")
        data_local_training = datasets.SVHN(args.path_svhn, split='train', download=True, transform=None)
        data_global_test = datasets.SVHN(args.path_svhn, split='test', transform=transform_test, download=True)

    elif args.dataset == 'CINIC10':
        args.num_classes = 10
        args.num_labeled = 900
        args.num_rounds = 400
        transform_test = to_tensor_normalize("CINIC10")
        data_local_training = CINIC10(root=args.path_cinic10, split='train', transform=None)
        data_global_test = CINIC10(root=args.path_cinic10, split='test', transform=transform_test)

    else:
        print(
            f"Error: Unsupported dataset {args.dataset}. Please specify one of the following: CIFAR10, CIFAR100, CINIC10 or SVHN."
        )
        exit(1)

    if getattr(args, 'max_rounds', 0) and args.max_rounds > 0:
        args.num_rounds = args.max_rounds
    if schedule_rounds(args) > args.num_rounds:
        raise ValueError('The baseline schedule horizon cannot exceed the total training rounds')

    _hp = (
        f"[HParams] METHOD_REV={METHOD_REV} adaptive={int(getattr(args,'pp_adaptive',1))} "
        f"capA={getattr(args,'pp_a_ratio_cap_p1',0)}/{getattr(args,'pp_a_ratio_cap',0)} "
        f"capB={getattr(args,'pp_b_ratio_cap',0)} "
        f"warmup=[{getattr(args,'pp_warmup_min_ratio',0)},{getattr(args,'pp_warmup_max_ratio',0)}] "
        f"aux_ratio={getattr(args,'pp_warmup_aux_ratio',0)} "
        f"geom=[{getattr(args,'pp_geom_min_ratio',0)},{getattr(args,'pp_geom_max_ratio',0)}] "
        f"gate_anneal={getattr(args,'pp_gate_anneal_ratio',0)} "
        f"p2_exit_A={getattr(args,'pp_phase2_exit_a_ratio',0)} "
        f"tau_w={args.tau_warmup} tau0={args.tau0} d0={args.delta0} etaB={args.eta_B} "
        f"pp_b={args.pp_b} p3_boost={getattr(args,'pp_phase3_geom_boost',0)} "
        f"pp_a={args.pp_a} geom_controls={_geometry_controls(args)} lr_controls={lr_controls(args)} "
        f"raw_tau_max={args.tau0 + args.pp_a * (1 + args.pp_phase3_geom_boost):.4f} "
        f"teacher={getattr(args,'pp_teacher',0)} mu_p={args.mu_p} "
        f"lA={args.lambda_A} lB={args.lambda_B} lP={args.lambda_proto} "
        f"seed_model={args.seed} seed_partition={getattr(args,'partition_seed',0)} "
        f"seed_sample={getattr(args,'sample_seed', args.seed)} "
        f"workers={getattr(args,'num_workers',0)} "
        f"protoWrite=floor{getattr(args,'proto_conf_floor',0)}/extra{getattr(args,'proto_w_extra',0)} "
        f"diag={int(getattr(args,'diag_geom',1))}"
    )
    print(_hp)
    _write_run_config(
        paths["config_path"],
        args,
        {"alpha": alpha, "run_id": paths["run_id"]},
    )

    print(
        'dataset:{dataset}\n'
        'num_classes:{num_classes}\n'
        'num_labeled:{num_labeled}\n'
        'non_iid:{alpha}\n'
        'mu:{mu}\n'
        'num_rounds:{num_rounds}\n'
        'batch_label:{batch_label}, batch_unlabel:{batch_unlabel}'.format(
            dataset=args.dataset,
            num_classes=args.num_classes,
            num_labeled=args.num_labeled,
            alpha=alpha,
            mu=args.mu,
            num_rounds=args.num_rounds,
            batch_label=args.batch_size_local_labeled,
            batch_unlabel=args.batch_size_local_unlabeled,
        )
    )

    random_state = np.random.RandomState(int(getattr(args, "sample_seed", args.seed)))
    partition_rng = np.random.RandomState(int(getattr(args, "partition_seed", 0)))

    list_label2indices = classify_label(data_local_training, args.num_classes)

    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(
        list_label2indices, args.num_labeled, rng=partition_rng
    )

    if alpha == 0:
        list_client2indices_labeled = clients_indices_homo(
            list_label2indices=list_label2indices_labeled,
            num_classes=args.num_classes,
            num_clients=args.num_clients,
        )
        list_client2indices_unlabeled = clients_indices_homo(
            list_label2indices=list_label2indices_unlabeled,
            num_classes=args.num_classes,
            num_clients=args.num_clients,
        )
    else:
        list_client2indices_labeled = clients_indices(
            list_label2indices=list_label2indices_labeled,
            num_classes=args.num_classes,
            num_clients=args.num_clients,
            non_iid_alpha=alpha,
            seed=int(getattr(args, "partition_seed", 0)),
        )
        list_client2indices_unlabeled = clients_indices(
            list_label2indices=list_label2indices_unlabeled,
            num_classes=args.num_classes,
            num_clients=args.num_clients,
            non_iid_alpha=alpha,
            seed=int(getattr(args, "partition_seed", 0)),
        )

    list_client2indices_labeled = [_as_index_list(x) for x in list_client2indices_labeled]
    list_client2indices_unlabeled = [_as_index_list(x) for x in list_client2indices_unlabeled]

    show_clients_data_distribution(
        data_local_training, list_client2indices_labeled, list_client2indices_unlabeled, args.num_classes
    )

    for client in range(args.num_clients):
        list_client2indices_unlabeled[client].extend(list_client2indices_labeled[client])

    _assert_clients_can_batch(list_client2indices_labeled, list_client2indices_unlabeled, args)

    # FedAvg 权重：原始样本索引并集。有标已被 extend 进无标列表，求和会重复计数；
    # 也不能用 load() 后的 len(dataset)（有标内部复制约 2000 倍）。
    client_n_samples = [
        len(set(list_client2indices_labeled[k]) | set(list_client2indices_unlabeled[k]))
        for k in range(args.num_clients)
    ]

    global_model = Global(args)
    local_ppfpsl = LocalPPFPSL(args)
    client_states = {i: None for i in range(args.num_clients)}
    labeled_counts_per_client = [
        _labeled_class_counts(data_local_training, list_client2indices_labeled[k], args.num_classes)
        for k in range(args.num_clients)
    ]

    total_clients = list(range(args.num_clients))

    tb_writer = None
    if getattr(args, 'use_tensorboard', False):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = paths["tb_dir"]
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_dir)
        except Exception as e:
            print('TensorBoard 不可用，跳过:', e)

    indices2data_labeled = Indices2Dataset_labeled(data_local_training, dataset_name=args.dataset)
    indices2data_unlabeled = Indices2Dataset_unlabeled_fixmatch(
        data_local_training, dataset_name=args.dataset
    )

    ckpt_path = paths["ckpt_path"]
    load_ckpt = paths["resume_src"] or ckpt_path

    schedule = AdaptiveSchedule(args, schedule_rounds(args))
    sched_msg = (
        f"[Schedule] adaptive={int(schedule.enabled)} "
        f"warmup=[{getattr(args, 'pp_warmup_min_ratio', 0.15):.2f},{getattr(args, 'pp_warmup_max_ratio', 0.40):.2f}] "
        f"geom=[{getattr(args, 'pp_geom_min_ratio', 0.12):.2f},{getattr(args, 'pp_geom_max_ratio', 0.35):.2f}] "
        f"capA={getattr(args, 'pp_a_ratio_cap_p1', 0):.2f}/{getattr(args, 'pp_a_ratio_cap', 0):.2f} "
        f"protoWrite=floor{getattr(args, 'proto_conf_floor', 0):.2f}/extra{getattr(args, 'proto_w_extra', 0):.2f} "
        f"geom_rel=Phase2 gate 0→1 over {max(1, int(round(getattr(args, 'pp_gate_anneal_ratio', 0.15) * schedule_rounds(args))))} rounds "
        f"target A-prec={getattr(args, 'pp_target_a_prec', 0.90)} A-ratio={getattr(args, 'pp_target_a_ratio', 0.22)}"
    )
    print(sched_msg)
    logging.info(sched_msg)

    fedavg_acc = []
    start_round = 1
    if paths["resume_src"] and not os.path.isfile(load_ckpt):
        raise FileNotFoundError(load_ckpt)
    if os.path.isfile(load_ckpt):
        ckpt = torch.load(load_ckpt, map_location='cpu', weights_only=False)
        if ckpt.get('method_rev') != METHOD_REV:
            if paths["resume_src"]:
                raise ValueError('Checkpoint METHOD_REV differs; start a fresh run without --resume')
            print(
                f'[Resume] 忽略 {load_ckpt}（method_rev={ckpt.get("method_rev")}，需要 {METHOD_REV}），'
                f'从头训练并写入 {ckpt_path}'
            )
        else:
            if getattr(args, 'bc_fork', 0):
                bc_tail.validate_fork(ckpt, _geometry_controls(args), lr_controls(args),
                                      load_ckpt, paths['run_dir'], bc_tail.run_identity(args))
            if not getattr(args, 'bc_fork', 0) and ckpt.get('geometry_controls') != _geometry_controls(args):
                raise ValueError('Geometry controls differ from checkpoint; restore flags or start a fresh run')
            if ckpt.get('lr_controls') != lr_controls(args):
                raise ValueError('LR controls missing or different in checkpoint; start a fresh run without --resume')
            print(f'[Resume] 从 {load_ckpt} 恢复，保存到 {ckpt_path}')
            global_model.model.load_state_dict(ckpt['global_model'])
            global_model.p_ref = ckpt['p_ref'].to(global_model.p_ref.device)
            global_model.p_ref_valid = ckpt['p_ref_valid'].to(global_model.p_ref.device)
            client_states = ckpt['client_states']
            fedavg_acc = ckpt['fedavg_acc']
            start_round = ckpt['round'] + 1
            random_state.set_state(ckpt['np_random_state'])
            torch.set_rng_state(ckpt['torch_rng_state'])
            if 'python_rng_state' in ckpt:
                random.setstate(ckpt['python_rng_state'])
            if 'numpy_global_rng_state' in ckpt:
                np.random.set_state(ckpt['numpy_global_rng_state'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(ckpt['cuda_rng_state'])
            if ckpt.get('schedule') is not None:
                schedule.load_state_dict(ckpt['schedule'])
            print(f'[Resume] 从第 {start_round} 轮继续，已有 {len(fedavg_acc)} 轮结果，phase={schedule.phase}')

    execution_end = getattr(args, 'stop_after_round', 0) or args.num_rounds
    for r in tqdm(range(start_round, execution_end + 1), desc='Server'):
        snap = schedule.for_round(r)

        dict_global_params = global_model.download_params()

        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        auxiliary_head = None
        if getattr(args, 'target_experiment', 'none') == 'labelhead' and r > 30:
            auxiliary_head = target_experiments.fit_online_head(global_model.model, data_local_training,
                [list_client2indices_labeled[int(c)] for c in online_clients], args.num_classes,
                to_tensor_normalize(args.dataset), next(global_model.model.parameters()).device)
            head_path = os.path.join(paths['run_dir'], 'labelhead_calibration.csv')
            target_experiments.write_head_audit(head_path, r, auxiliary_head)
        list_dicts_local_params = []
        list_nums_local_data = []
        proto_uploads = []
        log_rows = []

        for client in online_clients:
            indices2data_labeled.load(list_client2indices_labeled[client])
            data_client_labeled = indices2data_labeled
            indices2data_unlabeled.load(list_client2indices_unlabeled[client])
            data_client_unlabeled = indices2data_unlabeled

            list_nums_local_data.append(client_n_samples[client])
            
            lc = labeled_counts_per_client[client].to(torch.device('cuda', args.gpu_id))
            local_params, st_new, up = local_ppfpsl.train_round(
                args,
                data_client_labeled,
                data_client_unlabeled,
                copy.deepcopy(dict_global_params),
                global_model.p_ref,
                global_model.p_ref_valid,
                r,
                client_states[client],
                lc,                   
                snap,
                auxiliary_head=auxiliary_head,
            )
            trust_rows = st_new.pop("trust_rows", [])
            for state_key, filename in [('risk_rows', 'risk_audit.csv'),
                                        ('risk_calibration_rows', 'risk_calibration.csv')]:
                risk_rows = st_new.pop(state_key, [])
                if risk_rows:
                    risk_path = os.path.join(os.path.dirname(paths['trust_reference_path']), filename)
                    new_file = not os.path.exists(risk_path)
                    records = [dict(round=r, client=int(client), **row) for row in risk_rows]
                    with open(risk_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=list(records[0]))
                        if new_file:
                            writer.writeheader()
                        writer.writerows(records)
            if trust_rows:
                trust_path = paths["trust_reference_path"]
                new_file = not os.path.exists(trust_path)
                records = [dict(round=r, client=int(client), **row) for row in trust_rows]
                with open(trust_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(records[0]))
                    if new_file:
                        writer.writeheader()
                    writer.writerows(records)
            audit_rows = st_new.pop("geometry_audit", None)
            if audit_rows is not None:
                audit_append(paths["geometry_audit_path"], r, int(snap["phase"]), int(client), audit_rows)
            client_states[client] = st_new
            proto_uploads.append(up)
            list_dicts_local_params.append(copy.deepcopy(local_params))
            log_rows.append(st_new.get('log', {}))

        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
        if str(r) in getattr(args, 'diagnostic_update_rounds', '').split(','):
            save_update_snapshot(os.path.join(paths['run_dir'], 'update_snapshots', f'round_{r:04d}.pt'),
                r, dict_global_params, list_dicts_local_params, fedavg_params, online_clients,
                list_nums_local_data, [name for name, _ in global_model.model.named_parameters()])

        # Capture before fedavg_eval loads new weights into global_state's owner.
        update_summary, update_clients = {}, []
        if int(getattr(args, "diag_geom", 1)):
            update_summary, update_clients = update_rows(
                dict_global_params, list_dicts_local_params, fedavg_params,
                list_nums_local_data, online_clients,
                [name for name, _ in global_model.model.named_parameters()],
            )

        if proto_uploads:
            dev = global_model.p_ref.device
            uploads_device = []
            for up in proto_uploads:
                uploads_device.append(
                    {
                        'p_loc': up['p_loc'].to(dev),
                        'w_agg': up['w_agg'].to(dev),
                        'loc_valid': up['loc_valid'].to(dev),
                    }
                )
            aggregate_p_ref(
                global_model.p_ref,
                global_model.p_ref_valid,
                uploads_device,
                dev,
                global_model.feat_dim,
                args.num_classes,
            )

        global_acc = global_model.fedavg_eval(copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test)
        fedavg_acc.append(global_acc)
        best_acc = max(fedavg_acc)
        best_round = fedavg_acc.index(best_acc) + 1
        ph = int(snap['phase'])
        
        # 计算平均指标
        if log_rows:
            def _mean(key):
                xs = [float(row.get(key, 0)) for row in log_rows if key in row]
                return sum(xs) / max(1, len(xs))
            
            avg_loss = _mean('loss')
            avg_L_sup = _mean('L_sup')
            avg_L_A = _mean('L_A')
            avg_L_B = _mean('L_B')
            avg_L_proto = _mean('L_proto')
            
            cu = sum(int(row.get('cnt_u', 0)) for row in log_rows)
            ca = sum(int(row.get('cnt_a', 0)) for row in log_rows)
            cb = sum(int(row.get('cnt_b', 0)) for row in log_rows)
            cc = sum(int(row.get('cnt_c', 0)) for row in log_rows)
            n_batches = sum(int(row.get('n_batches', 0)) for row in log_rows)
            a_ratio = ca / cu if cu > 0 else 0
            b_ratio = cb / cu if cu > 0 else 0
            c_ratio = 1.0 - a_ratio - b_ratio
            
            atot = sum(int(row.get('a_total', 0)) for row in log_rows)
            acor = sum(int(row.get('a_correct', 0)) for row in log_rows)
            a_prec = acor / atot if atot > 0 else 0
            btot = sum(int(row.get('b_total', 0)) for row in log_rows)
            bcor = sum(int(row.get('b_correct', 0)) for row in log_rows)
            b_prec = bcor / btot if btot > 0 else 0
            # 高置信错误率：s_i≥hce_tau 的样本里伪标签错掉的比例（仅诊断）
            hce_hc = sum(int(row.get('hce_hc', 0)) for row in log_rows)
            hce_den = sum(int(row.get('hce_den_hc', 0)) for row in log_rows)
            hce_rate = hce_hc / hce_den if hce_den > 0 else 0.0
            g_avg = _mean('gate')
            r_mean = _mean('r_mean')
            m_mean = _mean('m_mean')
            alpha_mean = _mean('alpha_t')
            pass_s = sum(int(row.get('pass_s', 0)) for row in log_rows)
            geom_drop = sum(int(row.get('geom_drop', 0)) for row in log_rows)
            ma_ratio = pass_s / cu if cu > 0 else 0.0
            geom_frac = geom_drop / pass_s if pass_s > 0 else 0.0
            cap_a = _mean('a_ratio_cap')
            cap_b = _mean('b_ratio_cap')
            proto_on = float(snap.get('use_a_for_proto', 0.0))
            p_lab = sum(float(row.get('proto_lab', 0)) for row in log_rows)
            p_a = sum(float(row.get('proto_a', 0)) for row in log_rows)
            p_a_u = sum(float(row.get('proto_a_unique', 0)) for row in log_rows)
            p_aw = sum(float(row.get('proto_a_wrong', 0)) for row in log_rows)
            proto_a_share = p_a / (p_lab + p_a) if (p_lab + p_a) > 0 else 0.0
            proto_a_dirty = p_aw / p_a if p_a > 0 else 0.0
            lab_mult = _mean('lab_mult')
            lA = float(snap.get('lambda_A_scale', 1.0))
            lB = float(snap.get('lambda_B_scale', 1.0))
            schedule.update_after_round(
                r,
                {
                    "acc": float(global_acc),
                    "a_prec": float(a_prec),
                    "b_prec": float(b_prec),
                    "b_total": float(btot),
                    "a_ratio": float(a_ratio),
                    "b_ratio": float(b_ratio),
                    "c_ratio": float(c_ratio),
                },
            )
            msg = (
                f'Round {r}/{args.num_rounds} | Acc:{global_acc:.4f} Best:{best_acc:.4f}@{best_round} '
                f'Phase:{ph} gate:{g_avg:.2f} | '
                f'tau0:{snap["tau0"]:.3f} d0:{snap["delta0"]:.3f} etaB:{snap["eta_B"]:.3f} tw:{snap["tau_warmup"]:.3f} '
                f'lA:{lA:.2f} lB:{lB:.2f} protoA:{int(proto_on)} | '
                f'Loss:{avg_loss:.4f} (sup:{avg_L_sup:.3f} A:{avg_L_A:.3f} B:{avg_L_B:.3f} proto:{avg_L_proto:.3f}) | '
                f'Route: A:{a_ratio:.2%} B:{b_ratio:.2%} C:{c_ratio:.2%} capA:{cap_a:.2f} | '
                f'MA:{ma_ratio:.2%} geomDrop:{geom_frac:.2%} | '
                f'A-Prec:{a_prec:.2%} B-Prec:{b_prec:.2%} | '
                f'proto Ashare:{proto_a_share:.2%} Auniq:{p_a_u:.1f} dirty:{proto_a_dirty:.2%} lab×{lab_mult:.1f} | '
                f'rel r:{r_mean:.3f} m:{m_mean:.3f} alpha:{alpha_mean:.3f}'
            )
            if schedule.last_event:
                msg += f' | {schedule.last_event}'
                schedule.last_event = ""
            print(msg)
            logging.info(msg)
        else:
            msg = f'Round {r}/{args.num_rounds} | Acc:{global_acc:.4f} Best:{best_acc:.4f}@{best_round} Phase:{ph}'
            print(msg)
            logging.info(msg)

        if tb_writer is not None and log_rows:
            tb_writer.add_scalar('test/acc', global_acc, r)
            tb_writer.add_scalar('train/loss', avg_loss, r)
            tb_writer.add_scalar('train/L_sup', avg_L_sup, r)
            tb_writer.add_scalar('train/L_A', avg_L_A, r)
            tb_writer.add_scalar('train/L_B', avg_L_B, r)
            tb_writer.add_scalar('train/L_proto', avg_L_proto, r)
            tb_writer.add_scalar('train/phase', float(ph), r)
            tb_writer.add_scalar('routing/a_ratio', a_ratio, r)
            tb_writer.add_scalar('routing/b_ratio', b_ratio, r)
            tb_writer.add_scalar('routing/c_ratio', c_ratio, r)
            tb_writer.add_scalar('routing/m_a', ma_ratio, r)
            tb_writer.add_scalar('routing/geom_drop', geom_frac, r)
            tb_writer.add_scalar('routing/a_ratio_cap', cap_a, r)
            tb_writer.add_scalar('quality/a_precision', a_prec, r)
            tb_writer.add_scalar('quality/b_precision', b_prec, r)
            tb_writer.add_scalar('routing/gate', g_avg, r)
            tb_writer.add_scalar('rel/r_mean', r_mean, r)
            tb_writer.add_scalar('rel/m_mean', m_mean, r)
            tb_writer.add_scalar('rel/alpha_t', alpha_mean, r)
            tb_writer.add_scalar('adapt/tau0', snap['tau0'], r)
            tb_writer.add_scalar('adapt/delta0', snap['delta0'], r)
            tb_writer.add_scalar('adapt/eta_B', snap['eta_B'], r)
            tb_writer.add_scalar('adapt/tau_warmup', snap['tau_warmup'], r)
            tb_writer.add_scalar('adapt/lambda_A_scale', lA, r)
            tb_writer.add_scalar('adapt/lambda_B_scale', lB, r)
            tb_writer.add_scalar('adapt/use_a_for_proto', proto_on, r)
            tb_writer.add_scalar('proto/a_share', proto_a_share, r)
            tb_writer.add_scalar('proto/a_unique', p_a_u, r)
            tb_writer.add_scalar('proto/a_dirty', proto_a_dirty, r)
            tb_writer.add_scalar('proto/lab_mult', lab_mult, r)
            tb_writer.flush()

        # 每轮保存 checkpoint（覆盖同一个文件，断电/中断可续训）
        checkpoint_state = {
            'round': r,
            'global_model': global_model.model.state_dict(),
            'p_ref': global_model.p_ref.cpu(),
            'p_ref_valid': global_model.p_ref_valid.cpu(),
            'client_states': client_states,
            'fedavg_acc': fedavg_acc,
            'np_random_state': random_state.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'python_rng_state': random.getstate(),
            'numpy_global_rng_state': np.random.get_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'schedule': schedule.state_dict(),
            'method_rev': METHOD_REV,
            'geometry_controls': _geometry_controls(args),
            'lr_controls': lr_controls(args),
            'run_id': paths["run_id"],
            'run_identity': bc_tail.run_identity(args),
            'fork_source': paths['resume_src'] if getattr(args, 'bc_fork', 0) else None,
        }
        if getattr(args, 'target_experiment', 'none') == 'labelhead':
            # Audit snapshot; next round refits in the new global feature space.
            checkpoint_state['auxiliary_head'] = auxiliary_head
        temporary = ckpt_path + '.tmp'
        torch.save(checkpoint_state, temporary)
        os.replace(temporary, ckpt_path)
        if r in {int(x) for x in getattr(args, 'save_checkpoint_rounds', '').split(',') if x}:
            snapshot = os.path.join(paths['run_dir'], f'round_{r:04d}.pt')
            torch.save(checkpoint_state, snapshot + '.tmp')
            os.replace(snapshot + '.tmp', snapshot)
        if args.num_rounds > schedule_rounds(args) and r == schedule_rounds(args):
            torch.save(checkpoint_state, os.path.join(paths['run_dir'], 'main_end.pt'))

        acc_df = pd.DataFrame({
            'round': list(range(1, len(fedavg_acc) + 1)),
            'acc': fedavg_acc,
        })
        acc_df.to_csv(paths["acc_path"], index=False, encoding='utf8')

        if log_rows:
            mr = _metrics_row(
                r=r,
                phase=ph,
                gate=g_avg,
                acc=global_acc,
                best_acc=best_acc,
                best_round=best_round,
                losses=(avg_loss, avg_L_sup, avg_L_A, avg_L_B, avg_L_proto),
                route=(a_ratio, b_ratio, c_ratio, ma_ratio, geom_frac),
                quality=(a_prec, b_prec, hce_rate),
                proto=(proto_a_share, proto_a_dirty, p_a_u, p_lab, p_a, lab_mult),
                rel=(r_mean, m_mean, alpha_mean),
                knobs=(snap, lA, lB, proto_on, cap_a, cap_b, _mean('lr')),
                counts=(cu, ca, cb, cc, pass_s, geom_drop,
                        atot, acor, btot, bcor, hce_hc, hce_den, n_batches),
            )
            _append_metrics_row(paths["metrics_path"], mr, first_round=(r == 1))

            if int(getattr(args, "diag_geom", 1)):
                update_context = dict(round=r, phase=ph, gate=g_avg, lr=_mean('lr'),
                                      acc=global_acc,
                                      delta_acc=(global_acc - fedavg_acc[-2]
                                                 if len(fedavg_acc) > 1 else float('nan')))
                _append_diag_row(
                    os.path.join(paths["run_dir"], "updates.csv"),
                    dict(update_context, **update_summary), first_round=(r == 1),
                )
                for i, row in enumerate(update_clients):
                    local_log = log_rows[i]
                    row.update(local_steps=local_log.get('n_batches', 0),
                               a_ratio=local_log.get('cnt_a', 0) / max(local_log.get('cnt_u', 0), 1),
                               a_prec=local_log.get('a_correct', 0) / max(local_log.get('a_total', 0), 1))
                    _append_diag_row(
                        os.path.join(paths["run_dir"], "client_updates.csv"),
                        dict(update_context, **row), first_round=(r == 1 and i == 0),
                    )
                _append_diag_row(
                    paths["dynamics_path"], dynamics_row(r, ph, log_rows, args),
                    first_round=(r == 1),
                )
                diag_agg = {
                    k: sum(int(row.get(k, 0)) for row in log_rows)
                    for k in _DIAG_KEYS
                }
                _append_diag_row(
                    paths["diag_path"],
                    _diag_row(r, ph, cu, diag_agg),
                    first_round=(r == 1),
                )

    if tb_writer is not None:
        tb_writer.close()
