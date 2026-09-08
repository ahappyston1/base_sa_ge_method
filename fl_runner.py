# -*- coding: utf-8 -*-
"""
PPFPSL 联邦半监督训练主循环。

Pressure-aware + Prototype + A/B/C 路由。
当前唯一实现：A/B/C 路由 + 门槛退火 + 自适应阶段/门槛。
"""
from __future__ import annotations
import math
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
)
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

worker_num = 4
# 当前唯一实现的 checkpoint 标记；旧 V1/V2 权重没有此字段，不会被误续训
METHOD_REV = 5


def _run_paths(dataset, alpha, args) -> Dict[str, str]:
    """每次实验用时间戳（或 --run_id）区分输出，避免覆盖正在跑的任务。"""
    run_id = str(getattr(args, "run_id", "") or "").strip()
    resume = str(getattr(args, "resume", "") or "").strip()
    if resume:
        resume = os.path.abspath(resume)
        if not run_id:
            import re
            m = re.search(r"_(\d{8}_\d{6})(?:_latest)?\.pt$", resume)
            if m:
                run_id = m.group(1)
    if not run_id:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    stem = f"PPFPSL_a{alpha}_{run_id}"
    result_dir = f"./results/{dataset}"
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    log_dir = os.path.join(result_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    return {
        "run_id": run_id,
        "stem": stem,
        "acc_path": os.path.join(result_dir, f"{stem}.csv"),
        "metrics_path": os.path.join(result_dir, f"{stem}_metrics.csv"),
        "ckpt_path": os.path.join(ckpt_dir, f"{stem}_latest.pt"),
        "resume_src": resume,
        "log_file": os.path.join(log_dir, f"{stem}.log"),
        "tb_dir": os.path.join(result_dir, "tensorboard", stem),
    }


# ============= PPFPSL 核心函数 =============

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
        self.a_ratio_cap = float(getattr(args, "pp_a_ratio_cap_p1", 0.40))
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
            cap = float(getattr(self.args, "pp_a_ratio_cap_p1", 0.40)) if ph == 1 else float(
                getattr(self.args, "pp_a_ratio_cap", 0.30)
            )
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
        self.a_ratio_cap = float(getattr(args, "pp_a_ratio_cap_p1", 0.40)) if self.phase == 1 else float(
            getattr(args, "pp_a_ratio_cap", 0.30)
        )

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
        """对齐 81% 实验：只按阶段切换 A cap，不改 τ/η/λ。"""
        args = self.args
        self.a_ratio_cap = float(getattr(args, "pp_a_ratio_cap_p1", 0.40)) if self.phase == 1 else float(
            getattr(args, "pp_a_ratio_cap", 0.30)
        )
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
        exit_ap = float(getattr(args, "pp_phase1_exit_a_prec", 0.70))
        exit_acc = float(getattr(args, "pp_phase1_exit_acc", 0.58))
        exit_ar = float(getattr(args, "pp_phase2_exit_a_ratio", 0.08))

        if self.phase == 1:
            ready = (
                stayed >= min_w
                and self.ema_a_prec is not None
                and self.ema_a_prec >= exit_ap
                and self.ema_acc is not None
                and self.ema_acc >= exit_acc
                and self.aux_scale >= 0.8
            )
            if ready or stayed >= max_w:
                self.acc_p1_exit = self.ema_acc
                why = "ready" if ready else "max_warmup"
                self.phase = 2
                self.phase_enter_round = r + 1
                self.last_event = f"r{r} phase1→2 ({why}, stayed={stayed}, acc={self.ema_acc:.3f}, A-prec={self.ema_a_prec:.3f})"
                print("[Schedule]", self.last_event)
        elif self.phase == 2:
            recovered = True
            if self.acc_p1_exit is not None and self.ema_acc is not None:
                recovered = self.ema_acc >= 0.97 * float(self.acc_p1_exit)
            ready = (
                stayed >= min_g
                and self.gate >= 0.95
                and self.ema_a_ratio is not None
                and self.ema_a_ratio >= exit_ar
                and recovered
            )
            if ready or stayed >= max_g:
                why = "ready" if ready else "max_geom"
                self.phase = 3
                self.phase_enter_round = r + 1
                ar = "na" if self.ema_a_ratio is None else f"{self.ema_a_ratio:.3f}"
                self.last_event = f"r{r} phase2→3 ({why}, stayed={stayed}, gate={self.gate:.2f}, A-ratio={ar})"
                print("[Schedule]", self.last_event)


def _cap_bucket(in_A: torch.Tensor, score: torch.Tensor, cap: float) -> torch.Tensor:
    """A 桶硬顶：超过 cap 占比时只保留 score 最高的那部分，其余改走 B/C。"""
    if cap <= 0 or (not in_A.any()):
        return in_A
    n = int(in_A.numel())
    k = max(1, int(math.floor(cap * n)))
    n_a = int(in_A.sum().item())
    if n_a <= k:
        return in_A
    sc = score.detach().clone()
    sc = sc.masked_fill(~in_A, -1e9)
    thresh = torch.topk(sc, k, largest=True).values[-1]
    return in_A & (sc >= thresh)


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
    """客户端内对有效类的 rho_bar 做 min-max 归一化到 [0,1]。"""
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
        self.model.train()
        use_teacher = bool(int(getattr(args, "pp_teacher", 0)))
        if use_teacher:
            self.teacher.load_state_dict(global_params)
            self.teacher.eval()

        # Cosine lr schedule：随通信轮数从 lr_local_training 衰减到 1e-4
        cosine_lr = args.lr_local_training * 0.5 * (
            1.0 + math.cos(math.pi * round_idx / max(1, args.num_rounds))
        )
        cosine_lr = max(cosine_lr, 1e-4)
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

        if sched is None:
            phase = training_phase(round_idx, args.num_rounds, args)
            g_gate = gate_anneal(round_idx, args.num_rounds, args)
            aux_scale = warmup_aux_scale(round_idx, args.num_rounds, args)
            tau0_eff = float(args.tau0)
            delta0_eff = float(args.delta0)
            eta_B_eff = float(args.eta_B)
            tau_warmup_eff = float(args.tau_warmup)
            a_ratio_cap = float(getattr(args, "pp_a_ratio_cap_p1", 0.40)) if phase == 1 else float(
                getattr(args, "pp_a_ratio_cap", 0.30)
            )
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
            a_ratio_cap = float(sched.get("a_ratio_cap", getattr(args, "pp_a_ratio_cap", 0.30)))
            b_ratio_cap = float(sched.get("b_ratio_cap", getattr(args, "pp_b_ratio_cap", 0.0)))
            use_a_for_proto = float(sched.get("use_a_for_proto", 1.0)) > 0.5
            lambda_A_scale = float(sched.get("lambda_A_scale", 1.0))
            lambda_B_scale = float(sched.get("lambda_B_scale", 1.0))
        rho_bar_in = rho_bar.clone()
        norm_rho_in = norm_rho_client(rho_bar_in, loc_valid, args.pp_eps)

        z_labeled_sum = torch.zeros(args.num_classes, self.dim, device=self.device)
        z_labeled_cnt = torch.zeros(args.num_classes, device=self.device)
        z_a_sum = torch.zeros(args.num_classes, self.dim, device=self.device)
        w_a_sum = torch.zeros(args.num_classes, device=self.device)
        intra_num = torch.zeros(args.num_classes, device=self.device)
        intra_den = torch.zeros(args.num_classes, device=self.device)

        lab_loader = DataLoader(
            data_client_labeled,
            batch_size=args.batch_size_local_labeled_fixmatch,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )
        u_loader = DataLoader(
            data_client_unlabeled,
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )

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
        }

        # 使用原始数据集长度（未放大），避免重复迭代50倍
        real_unlabeled_len = getattr(data_client_unlabeled, 'client_dataset_len', len(data_client_unlabeled))
        local_iter = max(1, int(real_unlabeled_len / args.batch_size_local_labeled_fixmatch))
        

        for _local_epoch in range(args.local_epochs):
            it_l = iter(lab_loader)
            it_u = iter(u_loader)
            for _step in range(local_iter):
                try:
                    x, y = next(it_l)
                except StopIteration:
                    it_l = iter(lab_loader)
                    x, y = next(it_l)
                try:
                    uw, us, y_u_gt = next(it_u)
                except StopIteration:
                    it_u = iter(u_loader)
                    uw, us, y_u_gt = next(it_u)

                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                uw = uw.to(self.device, non_blocking=True)
                us = us.to(self.device, non_blocking=True)
                y_u_gt = y_u_gt.to(self.device, non_blocking=True)

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
                    L_proto = prototype_contrastive_loss(z_x, y, p_mix, mix_valid, args.pp_proto_T)
                else:
                    L_proto = z_x.new_zeros(())

                _, logs = self._forward_z_logits(us)
                if use_teacher:
                    with torch.no_grad():
                        zw, logw = self._forward_z_logits(uw, self.teacher)
                else:
                    zw, logw = self._forward_z_logits(uw)

                probs = F.softmax(logw / args.T, dim=-1)
                s_i, yhat = probs.max(dim=-1)

                if phase == 1:
                    mask = _cap_bucket(s_i >= tau_warmup_eff, s_i, a_ratio_cap)
                    ce_u = F.cross_entropy(logs, yhat, reduction="none")
                    L_A = (ce_u * mask.float()).mean()
                    if aux_scale > 1e-6:
                        in_B = _cap_bucket((~mask) & (s_i >= eta_B_eff), s_i, b_ratio_cap)
                        L_B = _consistency_kl(logw, logs, in_B, args.T, weights=s_i)
                    else:
                        in_B = torch.zeros_like(mask)
                        L_B = logw.new_zeros(())
                    log_acc["r_mean"] += float(s_i.mean().item())
                    log_acc["m_mean"] += 0.0
                    log_acc["alpha_t"] += 1.0
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
                            wi = (s_i[mask] ** args.w_gamma).clamp(args.w_min, 1.0)
                            for cls in range(args.num_classes):
                                mm = cc == cls
                                if mm.any():
                                    z_a_sum[cls] += (zw_a[mm] * wi[mm].unsqueeze(1)).sum(0)
                                    w_a_sum[cls] += wi[mm].sum()
                    log_acc["cnt_u"] += yhat.numel()
                    log_acc["cnt_a"] += mask.sum()
                    log_acc["cnt_b"] += in_B.sum()
                    log_acc["cnt_c"] += yhat.numel() - mask.sum() - in_B.sum()
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
                    nr = norm_rho_in
                    tau_c = tau0_eff + args.pp_a * nr
                    delta_c = delta0_eff + args.pp_b * nr
                    if phase == 3:
                        boost = 1.0 + args.pp_phase3_geom_boost * nr
                        tau_c = tau0_eff + args.pp_a * nr * boost
                        delta_c = delta0_eff + args.pp_b * nr * boost

                    # 从 warmup 门槛余弦插值到类自适应门槛；margin 从 0 拉到 delta_c
                    tau_i = (1.0 - g_gate) * tau_warmup_eff + g_gate * tau_c[yhat]
                    delta_i = g_gate * delta_c[yhat]

                    sims = zw @ p_mix.T
                    sim_pos = sims.gather(1, yhat.unsqueeze(1)).squeeze(1)
                    sims_masked = sims.clone()
                    sims_masked[torch.arange(yhat.numel(), device=self.device), yhat] = -1e9
                    sims_masked = sims_masked.masked_fill(~mix_valid.unsqueeze(0), -1e9)
                    max_other, _ = sims_masked.max(dim=-1)
                    m_i = sim_pos - max_other

                    own_ok = loc_valid[yhat] | p_ref_valid[yhat]
                    m0 = float(getattr(args, "pp_m0", 0.05))
                    mT = float(getattr(args, "pp_mT", 0.08))
                    m_tilde = _margin_reliability(m_i, m0, mT)
                    alpha_t = _blend_alpha(g_gate, nr[yhat], args)
                    r_i = alpha_t * s_i + (1.0 - alpha_t) * m_tilde

                    # gate 过半后缺原型的类不能进 A，改由 r_i 决定 B/C
                    in_A = (s_i >= tau_i) & (m_i >= delta_i) & (own_ok | (g_gate < 0.5))
                    in_A = _cap_bucket(in_A, s_i * m_tilde, a_ratio_cap)
                    b_min_m = float(getattr(args, "pp_b_min_margin", 0.0))
                    in_B = _cap_bucket(
                        (~in_A) & (r_i >= eta_B_eff) & (m_i >= b_min_m),
                        r_i,
                        b_ratio_cap,
                    )
                    in_C = (~in_A) & (~in_B)

                    w_full = torch.zeros_like(s_i)
                    if in_A.any():
                        w_full[in_A] = (s_i[in_A] ** args.w_gamma * m_tilde[in_A]).clamp(args.w_min, 1.0)
                    ce_u = F.cross_entropy(logs, yhat, reduction="none")
                    L_A = (w_full * ce_u).mean()
                    L_B = _consistency_kl(logw, logs, in_B, args.T, weights=r_i)
                    log_acc["r_mean"] += float(r_i.mean().item())
                    log_acc["m_mean"] += float(m_i.mean().item())
                    log_acc["alpha_t"] += float(alpha_t.mean().item())

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
                            wi = (s_i[in_A] ** args.w_gamma * m_tilde[in_A]).clamp(args.w_min, 1.0)
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

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                log_acc["loss"] += float(loss.detach().item())
                log_acc["L_sup"] += float(L_sup.detach().item())
                log_acc["L_A"] += float(L_A.detach().item()) if torch.is_tensor(L_A) else L_A
                log_acc["L_B"] += float(L_B.detach().item()) if torch.is_tensor(L_B) else L_B
                log_acc["L_proto"] += float(L_proto.detach().item()) if torch.is_tensor(L_proto) else L_proto
                log_acc["n_batches"] += 1

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

        w_agg = labeled_counts.to(self.device) + args.lambda_p * w_a_sum

        log_final = {}
        for k, v in log_acc.items():
            if k in ("loss", "L_sup", "L_A", "L_B", "L_proto", "r_mean", "m_mean", "alpha_t"):
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
            ):
                log_final[k] = v
            elif isinstance(v, torch.Tensor):
                log_final[k] = int(v.item())
            else:
                log_final[k] = v

        new_state = {
            "rho_bar": rho_new.detach().cpu(),
            "rho_valid": rho_v_new.detach().cpu(),
            "p_loc": p_new.detach().cpu(),
            "loc_valid": loc_new.detach().cpu(),
            "inited": True,
            "log": log_final,
        }

        proto_upload = {
            "p_loc": p_new.detach(),
            "w_agg": w_agg.detach(),
            "loc_valid": loc_new.detach(),
        }

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
            test_loader = DataLoader(data_test, batch_size_test)
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
    paths = _run_paths(args.dataset, alpha, args)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=paths["log_file"],
    )
    print(
        f'[Run] id={paths["run_id"]}\n'
        f'  acc={paths["acc_path"]}\n'
        f'  metrics={paths["metrics_path"]}\n'
        f'  ckpt={paths["ckpt_path"]}\n'
        f'  log={paths["log_file"]}'
    )
    logging.info(
        'run_id=%s acc=%s metrics=%s ckpt=%s',
        paths["run_id"], paths["acc_path"], paths["metrics_path"], paths["ckpt_path"],
    )

    if args.dataset == 'CIFAR10':
        args.num_classes = 10
        args.num_labeled = 500
        args.num_rounds = 300
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_test)

    elif args.dataset == 'CIFAR100':
        args.num_classes = 100
        args.num_labeled = 50
        args.num_rounds = 500
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_test)

    elif args.dataset == 'SVHN':
        args.num_classes = 10
        args.num_labeled = 460
        args.num_rounds = 150
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ]
        )
        data_local_training = datasets.SVHN(args.path_svhn, split='train', download=True, transform=None)
        data_global_test = datasets.SVHN(args.path_svhn, split='test', transform=transform_test, download=True)

    elif args.dataset == 'CINIC10':
        args.num_classes = 10
        args.num_labeled = 900
        args.num_rounds = 400
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)),
            ]
        )
        data_local_training = CINIC10(root=args.path_cinic10, split='train', transform=None)
        data_global_test = CINIC10(root=args.path_cinic10, split='test', transform=transform_test)

    else:
        print(
            f"Error: Unsupported dataset {args.dataset}. Please specify one of the following: CIFAR10, CIFAR100, CINIC10 or SVHN."
        )
        exit(1)

    if getattr(args, 'max_rounds', 0) and args.max_rounds > 0:
        args.num_rounds = args.max_rounds

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

    random_state = np.random.RandomState(args.seed)

    list_label2indices = classify_label(data_local_training, args.num_classes)

    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(list_label2indices, args.num_labeled)

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
            seed=0,
        )
        list_client2indices_unlabeled = clients_indices(
            list_label2indices=list_label2indices_unlabeled,
            num_classes=args.num_classes,
            num_clients=args.num_clients,
            non_iid_alpha=alpha,
            seed=0,
        )

    show_clients_data_distribution(
        data_local_training, list_client2indices_labeled, list_client2indices_unlabeled, args.num_classes
    )

    for client in range(args.num_clients):
        list_client2indices_unlabeled[client].extend(list_client2indices_labeled[client])

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

    indices2data_labeled = Indices2Dataset_labeled(data_local_training)
    indices2data_unlabeled = Indices2Dataset_unlabeled_fixmatch(data_local_training)

    ckpt_path = paths["ckpt_path"]
    load_ckpt = paths["resume_src"] or ckpt_path

    schedule = AdaptiveSchedule(args, args.num_rounds)
    sched_msg = (
        f"[Schedule] adaptive={int(schedule.enabled)} "
        f"warmup=[{getattr(args, 'pp_warmup_min_ratio', 0.15):.2f},{getattr(args, 'pp_warmup_max_ratio', 0.40):.2f}] "
        f"geom=[{getattr(args, 'pp_geom_min_ratio', 0.12):.2f},{getattr(args, 'pp_geom_max_ratio', 0.35):.2f}] "
        f"target A-prec={getattr(args, 'pp_target_a_prec', 0.90)} A-ratio={getattr(args, 'pp_target_a_ratio', 0.22)}"
    )
    print(sched_msg)
    logging.info(sched_msg)

    fedavg_acc = []
    start_round = 1
    if os.path.isfile(load_ckpt):
        ckpt = torch.load(load_ckpt, map_location='cpu')
        if ckpt.get('method_rev') != METHOD_REV:
            print(
                f'[Resume] 忽略 {load_ckpt}（method_rev={ckpt.get("method_rev")}，需要 {METHOD_REV}），'
                f'从头训练并写入 {ckpt_path}'
            )
        else:
            print(f'[Resume] 从 {load_ckpt} 恢复，保存到 {ckpt_path}')
            global_model.model.load_state_dict(ckpt['global_model'])
            global_model.p_ref = ckpt['p_ref'].to(global_model.p_ref.device)
            global_model.p_ref_valid = ckpt['p_ref_valid'].to(global_model.p_ref.device)
            client_states = ckpt['client_states']
            fedavg_acc = ckpt['fedavg_acc']
            start_round = ckpt['round'] + 1
            random_state.set_state(ckpt['np_random_state'])
            torch.set_rng_state(ckpt['torch_rng_state'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(ckpt['cuda_rng_state'])
            if ckpt.get('schedule') is not None:
                schedule.load_state_dict(ckpt['schedule'])
            print(f'[Resume] 从第 {start_round} 轮继续，已有 {len(fedavg_acc)} 轮结果，phase={schedule.phase}')

    for r in tqdm(range(start_round, args.num_rounds + 1), desc='Server'):
        snap = schedule.for_round(r)

        dict_global_params = global_model.download_params()

        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []
        proto_uploads = []
        log_rows = []

        for client in online_clients:
            indices2data_labeled.load(list_client2indices_labeled[client])
            data_client_labeled = indices2data_labeled
            indices2data_unlabeled.load(list_client2indices_unlabeled[client])
            data_client_unlabeled = indices2data_unlabeled

            list_nums_local_data.append(len(data_client_labeled) + len(data_client_unlabeled))
            
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
            )
            client_states[client] = st_new
            proto_uploads.append(up)
            list_dicts_local_params.append(copy.deepcopy(local_params))
            log_rows.append(st_new.get('log', {}))

        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)

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
            a_ratio = ca / cu if cu > 0 else 0
            b_ratio = cb / cu if cu > 0 else 0
            c_ratio = 1.0 - a_ratio - b_ratio
            
            atot = sum(int(row.get('a_total', 0)) for row in log_rows)
            acor = sum(int(row.get('a_correct', 0)) for row in log_rows)
            a_prec = acor / atot if atot > 0 else 0
            btot = sum(int(row.get('b_total', 0)) for row in log_rows)
            bcor = sum(int(row.get('b_correct', 0)) for row in log_rows)
            b_prec = bcor / btot if btot > 0 else 0
            g_avg = _mean('gate')
            r_mean = _mean('r_mean')
            m_mean = _mean('m_mean')
            alpha_mean = _mean('alpha_t')
            proto_a = float(snap.get('use_a_for_proto', 0.0))
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
                f'lA:{lA:.2f} lB:{lB:.2f} protoA:{int(proto_a)} | '
                f'Loss:{avg_loss:.4f} (sup:{avg_L_sup:.3f} A:{avg_L_A:.3f} B:{avg_L_B:.3f} proto:{avg_L_proto:.3f}) | '
                f'Route: A:{a_ratio:.2%} B:{b_ratio:.2%} C:{c_ratio:.2%} | '
                f'A-Prec:{a_prec:.2%} B-Prec:{b_prec:.2%} | '
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
            tb_writer.add_scalar('adapt/use_a_for_proto', proto_a, r)
            tb_writer.flush()

        # 每轮保存 checkpoint（覆盖同一个文件，断电/中断可续训）
        torch.save({
            'round': r,
            'global_model': global_model.model.state_dict(),
            'p_ref': global_model.p_ref.cpu(),
            'p_ref_valid': global_model.p_ref_valid.cpu(),
            'client_states': client_states,
            'fedavg_acc': fedavg_acc,
            'np_random_state': random_state.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'schedule': schedule.state_dict(),
            'method_rev': METHOD_REV,
            'run_id': paths["run_id"],
        }, ckpt_path)

        acc_path = paths["acc_path"]
        acc_num_pseudo_label_csv_index = list(range(1, len(fedavg_acc) + 1))
        acc_num_pseudo_label_csv_df = pd.DataFrame({'acc': fedavg_acc}, index=acc_num_pseudo_label_csv_index)
        acc_num_pseudo_label_csv_df.to_csv(acc_path, encoding='utf8')

        if log_rows:
            mpath = paths["metrics_path"]
            keys = list(log_rows[0].keys())
            mr = {k: _mean(k) for k in keys}
            mr['round'] = r
            mr['acc'] = global_acc
            mr['phase'] = ph
            fieldnames = list(mr.keys())
            header_ok = False
            if os.path.isfile(mpath) and r > 1:
                with open(mpath, 'r', encoding='utf8') as rf:
                    existing = rf.readline().strip().split(',')
                header_ok = existing == fieldnames
            mode = 'a' if header_ok else 'w'
            with open(mpath, mode, newline='', encoding='utf8') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if mode == 'w':
                    w.writeheader()
                w.writerow(mr)

    if tb_writer is not None:
        tb_writer.close()
