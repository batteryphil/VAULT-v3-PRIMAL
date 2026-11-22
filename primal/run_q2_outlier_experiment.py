#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
Peer-Review Track-1 Upgrade Study on `VAULT-v3-DeepSeek-R1-1.5B-PRIMAL`:
  1. Untouched Upstream Baseline Audit (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` in `BF16`)
     under the identical `WT2` + `GSM8K-200` (tracking mid-`<think>` truncation at `1536` tokens)
     + `ARC-Easy` + `ARC-Challenge` + `SciQ-RC` + `SciQ-CB` harness.
  2. Zero-Runtime-Cost Offline Incoherent Pre-Transformations:
     - Exact GQA `V_proj <-> O_proj` `H_128` Walsh-Hadamard head rotation (`2` KV heads <-> `12` Q/O heads),
       allowing `o_proj` (`+66M` params) to compress into 2-bit `V3_Q2_0` alongside all 84 SwiGLU MLP matrices
       (`93.3%` of body in 2-bit, shrinking total `.vlt3` to `~690 MB` — smaller than `704 MB` `T1.58-Hybrid`!)
       while keeping `q_proj` in `V3_P4_0`/`V3_Q8_0` to preserve `100K` induction-head retrieval.
     - Exact `up_proj <-> down_proj` column outlier equilibration (`W_down[:, c] *= s_c`, `W_up[c, :] /= s_c`),
       eliminating `4.0x-7.6x` column weight outliers in `down_proj` losslessly via `up_proj`'s per-row FP16 scales.
  3. Scale-Invariant Levenberg-Marquardt Asymmetric Two-Pass `V3_Q2_0` (`{-2, -1, 0, +1}`) Cholesky-GPTQ:
     - Replaces outlier-inflated `mean(diag(H_s))` damping with scale-invariant diagonal LM damping
       `H_s + lam * diag(max(h_c, 1e-3 * median(h)))`.
     - Uses Schur-complement conditional block precision `P_g = (U_gg U_gg^T)^{-1}`, signed `+-d` outlier alignment,
       and monotonic sign-preserving Pass-2 scale refinement.
  4. Proximal Gauss-Newton Second-Order `PRIMAL` KL-Distillation (`{-2, -1, 0, +1}` + signed `FP16` scales):
     - Full Kronecker diagonal Gauss-Newton preconditioning `-(sign(d) * g_W) / (|d| * G_out_ema * H_in_ema)`,
     - Slow EMA batch normalizer + Proximal Voronoi Deadzone Gate + unbiased stochastic-rounded leaky `int8`
       vote decay (`gamma = 0.95`), eliminating Voronoi limit-cycle chattering and removing all ad-hoc
       super-weight channel masks and `L00..L03` hard layer locks.
  5. Freestanding C99 `V3_Q2_0` Speed/RAM Benchmark & Adversarial 8-Probe Rollover/Multi-Key/Supersession `O(1)` NIAH Suite.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if "expandable_segments" in os.environ.get("PYTORCH_ALLOC_CONF", ""):
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import vlt3  # noqa: E402
from src.vault_v3_model import VAULTv3ForCausalLM  # noqa: E402

HF_TEACHER_DIR = os.path.join(REPO_ROOT, "dist", "hf_vault_v3_deepseek_r1_1_5b")
Q8_VLT3 = os.path.join(REPO_ROOT, "dist", "vault_v3_deepseek_r1_1_5b_q8_0_primal.vlt3")
P4_VLT3 = os.path.join(REPO_ROOT, "dist", "vault_v3_deepseek_r1_1_5b_p4_0_primal.vlt3")
T158_HYB_VLT3 = os.path.join(REPO_ROOT, "dist", "vault_v3_deepseek_r1_1_5b_t1_58_hybrid.vlt3")
T158_PURE_VLT3 = os.path.join(REPO_ROOT, "dist", "vault_v3_deepseek_r1_1_5b_t1_58_pure.vlt3")
Q2_OUT_VLT3 = os.path.join(REPO_ROOT, "dist", "vault_v3_deepseek_r1_1_5b_q2_outlier.vlt3")
STREAM_PT = os.path.join(REPO_ROOT, "runs", "deepseek_r1_1_5b_primal", "deepseek_1_5b_primal_stream.pt")
OUT_DIR = os.path.join(REPO_ROOT, "runs", "q2_outlier")
REPORT_JSON = os.path.join(REPO_ROOT, "benchmarks", "results", "vault_v3_deepseek_1_5b_q2_outlier_report.json")
T158_REPORT_JSON = os.path.join(REPO_ROOT, "benchmarks", "results", "vault_v3_deepseek_1_5b_ternary_1_58_report.json")
HOST_BIN = os.path.join(REPO_ROOT, "runtime", "vault3_host")

BOS_STR = "<｜begin▁of▁sentence｜>"
USR_STR = "<｜User｜>"
AST_STR = "<｜Assistant｜>"
THK_STR = "<think>"
ETHK_STR = "</think>"


# ==============================================================================
# 1. Exact Zero-Runtime-Cost Offline Incoherent Pre-Transformations
# ==============================================================================
def build_hadamard(n: int, device: str = "cuda", dtype: torch.dtype = torch.float64) -> torch.Tensor:
    assert n > 0 and (n & (n - 1)) == 0, f"n={n} must be a power of 2"
    h = torch.tensor([[1.0]], device=device, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return (h / math.sqrt(float(n))).to(dtype=dtype)


@torch.no_grad()
def apply_offline_incoherent_transforms(m: Qwen2ForCausalLM) -> Dict[str, Any]:
    """
    Applies two mathematically exact, zero-runtime-overhead similarity transformations
    across all 28 Qwen2 layers in FP64 before rounding back to model dtype:
      1) GQA V_proj <-> O_proj H_128 Walsh-Hadamard head fusion:
           W_V[g] <- H_128 @ W_V[g],  b_V[g] <- H_128 @ b_V[g]   (g in 0..1)
           W_O[:, h] <- W_O[:, h] @ H_128^T                      (h in 0..11)
         Smears per-head value activation outliers and O_proj weight outliers across all
         128 channels (2 groups of 64) with zero runtime Hadamard ops.
      2) SwiGLU up_proj <-> down_proj column outlier equilibration:
           s_c = clamp(median_k ||W_down[:, k]||_2 / ||W_down[:, c]||_2, 0.12, 2.0)
           W_down[:, c] *= s_c,   W_up[c, :] /= s_c
         Eliminates 4.0x-7.6x column outliers in down_proj while up_proj's row-wise FP16
         scales absorb 1/s_c losslessly.
    """
    dev = next(m.parameters()).device
    orig_dtype = next(m.parameters()).dtype
    h128 = build_hadamard(128, device=str(dev), dtype=torch.float64)
    max_raw_col_ratio = 0.0
    max_eq_col_ratio = 0.0

    for li in range(len(m.model.layers)):
        ly = m.model.layers[li]
        # 1. GQA V_proj <-> O_proj H_128 rotation
        wv = ly.self_attn.v_proj.weight.data.double().view(2, 128, 1536)
        ly.self_attn.v_proj.weight.data.copy_(
            torch.einsum("ij,gjk->gik", h128, wv).reshape(256, 1536).to(orig_dtype)
        )
        if ly.self_attn.v_proj.bias is not None:
            bv = ly.self_attn.v_proj.bias.data.double().view(2, 128)
            ly.self_attn.v_proj.bias.data.copy_(
                torch.einsum("ij,gj->gi", h128, bv).reshape(256).to(orig_dtype)
            )
        wo = ly.self_attn.o_proj.weight.data.double().view(1536, 12, 128)
        # Since H_128 is symmetric orthogonal (H_128^T == H_128, H_128^T @ H_128 == I_128):
        ly.self_attn.o_proj.weight.data.copy_(
            torch.einsum("ohj,ij->ohi", wo, h128).reshape(1536, 1536).to(orig_dtype)
        )

        # 2. SwiGLU up_proj <-> down_proj column outlier equilibration
        wd = ly.mlp.down_proj.weight.data.double()
        wu = ly.mlp.up_proj.weight.data.double()
        col_n = wd.norm(dim=0).clamp(min=1e-12)
        med_n = col_n.median()
        raw_ratio = float((col_n.max() / med_n).item())
        max_raw_col_ratio = max(max_raw_col_ratio, raw_ratio)

        sc = (med_n / col_n).clamp(0.12, 2.0)
        wd_eq = wd * sc.unsqueeze(0)
        wu_eq = wu / sc.unsqueeze(1)
        eq_col_n = wd_eq.norm(dim=0).clamp(min=1e-12)
        eq_ratio = float((eq_col_n.max() / eq_col_n.median()).item())
        max_eq_col_ratio = max(max_eq_col_ratio, eq_ratio)

        ly.mlp.down_proj.weight.data.copy_(wd_eq.to(orig_dtype))
        ly.mlp.up_proj.weight.data.copy_(wu_eq.to(orig_dtype))

    return {
        "hadamard_v_o_dim": 128,
        "down_proj_max_col_outlier_ratio_before": round(max_raw_col_ratio, 3),
        "down_proj_max_col_outlier_ratio_after": round(max_eq_col_ratio, 3),
    }


# ==============================================================================
# 2. Scale-Invariant LM-Damped Signed-Scale Two-Pass V3_Q2_0 Asymmetric GPTQ
# ==============================================================================
@torch.no_grad()
def gptq_quantize_q2_asymmetric(
    w_teacher: torch.Tensor,
    h_s_fp64: torch.Tensor,
    g_ts_fp64: Optional[torch.Tensor] = None,
    group_size: int = 64,
    alpha_asym: float = 0.65,
    lam_asym: float = 0.05,
    lam_gptq: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Quantizes `w_teacher` into `V3_Q2_0` (`q in {-2, -1, 0, +1}`, `G=64`, signed `FP16` scale `d`)
    incorporating all four Peer-Review Section 1.1 mathematical upgrades:
      1) Scale-invariant diagonal Levenberg-Marquardt damping `lam * diag(max(h_c, 1e-3 * median(h)))`
         so super-weight channels (`h_c ~ 2500`) never over-damp quiet bulk channels (`h_c ~ 0.2`).
      2) Verified asymmetric cross-covariance `G_ts = (1/N) X_t^T X_s`: since `H_asym` is symmetric,
         `z_t = cholesky_solve(delta_g.T, L) = H_asym^{-1} delta_g^T` gives `z_t.T = (G_ts - H_s) H_asym^{-1}`.
      3) Conditional Schur-complement group metric `P_g = (U_gg U_gg^T)^{-1}` consistent with
         previously absorbed cross-group Cholesky residuals in `w_g0`.
      4) Signed `+-d` candidate search (`10` signed ratios) + monotonic sign-preserving Pass-2 refinement.
    """
    device = w_teacher.device
    out_f, in_f = w_teacher.shape
    assert group_size == 64 and in_f % 64 == 0, f"in_f={in_f}, group_size={group_size}"

    h = h_s_fp64.clone()
    diag_raw = torch.diag(h)
    dead = diag_raw <= 0.0
    pos_vals = diag_raw[~dead]
    h_med = float(pos_vals.median().item()) if pos_vals.numel() > 0 else 1.0
    h_floor = max(1e-10, 1e-3 * h_med)
    if dead.any():
        h[dead, dead] = h_floor

    d_lm = torch.diag(h).clamp(min=h_floor)
    idx = torch.arange(in_f, device=device)

    if g_ts_fp64 is not None and alpha_asym > 0.0:
        delta_g = g_ts_fp64 - h_s_fp64
        if dead.any():
            delta_g[dead, :] = 0.0
            delta_g[:, dead] = 0.0
        h_asym = h.clone()
        h_asym[idx, idx] += lam_asym * d_lm
        l_asym = torch.linalg.cholesky(h_asym)
        # H_asym is SPD symmetric => (H_asym^{-1} delta_g^T)^T == delta_g H_asym^{-1} == (G_ts - H_s) H_asym^{-1}
        z_t = torch.cholesky_solve(delta_g.t(), l_asym)
        del h_asym, l_asym, delta_g
        w = (w_teacher.detach().double() + alpha_asym * (w_teacher.detach().double() @ z_t.t())).float()
        del z_t
    else:
        w = w_teacher.detach().float().clone()

    if dead.any():
        w[:, dead] = 0.0

    u_chol = None
    h_w = None
    for attempt in range(4):
        try:
            h_w = h.clone()
            h_w[idx, idx] += (lam_gptq * (4.0 ** attempt)) * d_lm
            chol_l = torch.linalg.cholesky(h_w)
            h_inv = torch.cholesky_inverse(chol_l)
            u_chol = torch.linalg.cholesky(h_inv, upper=True)
            del chol_l, h_inv
            break
        except RuntimeError:
            if attempt == 3:
                raise

    assert u_chol is not None and h_w is not None
    u_chol_f32 = u_chol.float()
    h_f32 = h_w.float()
    del h_w, h, u_chol

    q_out = torch.empty((out_f, in_f), dtype=torch.int8, device=device)
    d_out = torch.empty((out_f, in_f // group_size), dtype=torch.float16, device=device)
    # 10 signed ratios so -2 aligns dynamically with either negative (+d) or positive (-d) block outliers!
    signed_alphas = torch.tensor(
        [0.36, 0.46, 0.56, 0.68, 0.84, -0.36, -0.46, -0.56, -0.68, -0.84],
        device=device,
        dtype=torch.float32,
    ).view(1, 10, 1)
    eye_g = torch.eye(group_size, device=device, dtype=torch.float64)

    sb = 256
    for b1 in range(0, in_f, sb):
        b2 = min(b1 + sb, in_f)
        w_sb = w[:, b1:b2].clone()
        err_sb = torch.zeros_like(w_sb)
        u_sb = u_chol_f32[b1:b2, b1:b2]

        for g0 in range(0, b2 - b1, group_size):
            g1 = g0 + group_size
            ga = b1 + g0
            w_g0 = w_sb[:, g0:g1]
            u_gg = u_sb[g0:g1, g0:g1]

            # Exact Schur-complement conditional precision P_g = (U_gg U_gg^T)^{-1}
            u_gg64 = u_gg.double()
            cov_gg = u_gg64 @ u_gg64.t()
            cov_gg.diagonal().add_(1e-10)
            p_g = torch.linalg.solve(cov_gg, eye_g).float()
            pd_g = torch.diag(p_g).clamp(min=1e-8).view(1, 1, group_size)

            mx = w_g0.abs().amax(dim=1, keepdim=True).unsqueeze(1)
            nz_row = (mx.squeeze(2).squeeze(1) > 0.0)
            d_cand = (mx.clamp(min=1e-8) * signed_alphas).to(torch.float16).float()
            s_cand = torch.sign(signed_alphas)  # +1 for first 5, -1 for last 5
            wg_exp = w_g0.unsqueeze(1)

            for _ in range(3):
                inv_c = torch.where(d_cand.abs() > 1e-8, 1.0 / torch.where(d_cand.abs() > 1e-8, d_cand, torch.ones_like(d_cand)), torch.zeros_like(d_cand))
                qc = torch.clamp(torch.round(wg_exp * inv_c), -2.0, 1.0)
                num = (wg_exp * qc * pd_g).sum(dim=2, keepdim=True)
                den = (qc * qc * pd_g).sum(dim=2, keepdim=True).clamp(min=1e-8)
                d_raw = num / den
                # Keep sign locked to s_cand so candidate slots stay sign-consistent
                d_mag = (d_raw * s_cand).clamp(min=1e-7)
                d_cand = (s_cand * d_mag).to(torch.float16).float()

            inv_c = torch.where(d_cand.abs() > 1e-8, 1.0 / torch.where(d_cand.abs() > 1e-8, d_cand, torch.ones_like(d_cand)), torch.zeros_like(d_cand))
            qc = torch.clamp(torch.round(wg_exp * inv_c), -2.0, 1.0)
            diff = wg_exp - d_cand * qc
            err_c = ((diff @ p_g) * diff).sum(dim=2)
            best_idx = err_c.argmin(dim=1, keepdim=True)
            d_pass0 = torch.where(nz_row, d_cand.squeeze(2).gather(1, best_idx).squeeze(1), torch.zeros_like(w_g0[:, 0]))

            # Pass 1: Cholesky column error-compensation sweep
            w_g1 = w_g0.clone()
            q_g1 = torch.empty((out_f, group_size), dtype=torch.float32, device=device)
            err_g1 = torch.zeros_like(w_g0)
            d1_f = d_pass0.to(torch.float16).float()
            inv_d1 = torch.where(d1_f.abs() > 1e-8, 1.0 / torch.where(d1_f.abs() > 1e-8, d1_f, torch.ones_like(d1_f)), torch.zeros_like(d1_f))
            for j in range(group_size):
                col = w_g1[:, j]
                qj = torch.where(nz_row, torch.clamp(torch.round(col * inv_d1), -2.0, 1.0), torch.zeros_like(col))
                q_g1[:, j] = qj
                ej = (col - d1_f * qj) / u_gg[j, j].clamp(min=1e-7)
                err_g1[:, j] = ej
                if j + 1 < group_size:
                    w_g1[:, j + 1 :] -= ej.unsqueeze(1) * u_gg[j, j + 1 :].unsqueeze(0)

            # Sign-preserving, zero-denominator-safe Pass-2 scale refinement on Schur metric P_g
            num_p = ((w_g0 @ p_g) * q_g1).sum(dim=1)
            den_p = ((q_g1 @ p_g) * q_g1).sum(dim=1)
            d_prop = torch.where(den_p > 1e-8, num_p / torch.where(den_p > 1e-8, den_p, torch.ones_like(den_p)), d1_f)
            ratio_ok = (den_p > 1e-8) & ((d_prop * d1_f) > 0.0)
            ratio_clamped = torch.clamp(d_prop / torch.where(d1_f.abs() > 1e-8, d1_f, torch.ones_like(d1_f)), 0.5, 2.0)
            d2_f = torch.where(ratio_ok, (d1_f * ratio_clamped).to(torch.float16).float(), d1_f)

            # Pass 2: Cholesky column sweep + per-row monotonic selection vs Pass 1
            w_g2 = w_g0.clone()
            q_g2 = torch.empty((out_f, group_size), dtype=torch.float32, device=device)
            err_g2 = torch.zeros_like(w_g0)
            inv_d2 = torch.where(d2_f.abs() > 1e-8, 1.0 / torch.where(d2_f.abs() > 1e-8, d2_f, torch.ones_like(d2_f)), torch.zeros_like(d2_f))
            for j in range(group_size):
                col = w_g2[:, j]
                qj = torch.where(nz_row, torch.clamp(torch.round(col * inv_d2), -2.0, 1.0), torch.zeros_like(col))
                q_g2[:, j] = qj
                ej = (col - d2_f * qj) / u_gg[j, j].clamp(min=1e-7)
                err_g2[:, j] = ej
                if j + 1 < group_size:
                    w_g2[:, j + 1 :] -= ej.unsqueeze(1) * u_gg[j, j + 1 :].unsqueeze(0)

            df1 = w_g0 - d1_f.unsqueeze(1) * q_g1
            df2 = w_g0 - d2_f.unsqueeze(1) * q_g2
            loss1 = ((df1 @ p_g) * df1).sum(dim=1)
            loss2 = ((df2 @ p_g) * df2).sum(dim=1)
            use2 = loss2 < loss1

            q_best = torch.where(use2.unsqueeze(1), q_g2, q_g1)
            d_best = torch.where(use2, d2_f, d1_f)
            err_best = torch.where(use2.unsqueeze(1), err_g2, err_g1)

            q_out[:, ga : ga + group_size] = q_best.to(torch.int8)
            d_out[:, ga // group_size] = d_best.to(torch.float16)
            err_sb[:, g0:g1] = err_best
            if g1 < (b2 - b1):
                w_sb[:, g1:] -= err_best @ u_sb[g0:g1, g1:]

        if b2 < in_f:
            w[:, b2:] -= err_sb @ u_chol_f32[b1:b2, b2:]

    w_deq = d_out.float().repeat_interleave(group_size, dim=1) * q_out.float()
    dw = w - w_deq
    num_e = float(((dw @ h_f32) * dw).sum().item())
    den_e = float(((w @ h_f32) * w).sum().clamp(min=1e-12).item())
    del h_f32, u_chol_f32, w
    return q_out, d_out, math.sqrt(max(0.0, num_e / den_e))


# ==============================================================================
# 3. Proximal Gauss-Newton Second-Order Shadowless PRIMAL Q2_0 Module
# ==============================================================================
class _ProximalGaussNewtonQ2STE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q_int8: torch.Tensor,
        d_eff: torch.Tensor,
        bias: Optional[torch.Tensor],
        mod_ref: "PrimalQ2Linear",
    ) -> torch.Tensor:
        out_f, in_f = q_int8.shape
        G = mod_ref.group_size
        q_f = q_int8.to(x.dtype).view(out_f, in_f // G, G)
        w = (q_f * d_eff.to(x.dtype).unsqueeze(-1)).reshape(out_f, in_f)
        ctx.save_for_backward(x, q_int8, d_eff)
        ctx.has_bias = bias is not None
        ctx.mod_ref = mod_ref
        return F.linear(x, w, bias.to(x.dtype) if bias is not None else None)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, q_int8, d_eff = ctx.saved_tensors
        mod: PrimalQ2Linear = ctx.mod_ref
        out_f, in_f = q_int8.shape
        G = mod.group_size
        go_2d = grad_out.reshape(-1, out_f)
        x_2d = x.reshape(-1, in_f)

        q_f = q_int8.to(grad_out.dtype).view(out_f, in_f // G, G)
        w = (q_f * d_eff.to(grad_out.dtype).unsqueeze(-1)).reshape(out_f, in_f)
        grad_x = (go_2d @ w).reshape_as(x)
        del w

        grad_w = torch.matmul(go_2d.float().t(), x_2d.float())
        grad_w_grp = grad_w.view(out_f, in_f // G, G)
        grad_d = (grad_w_grp * q_int8.float().view(out_f, in_f // G, G)).sum(dim=-1)

        # Always update slow Kronecker curvature EMAs (even during warmup!) so curvature is warm before votes start!
        with torch.no_grad():
            n_tok = float(max(1, x_2d.shape[0]))
            scale_un = n_tok * float(mod.grad_accum)
            h_in_cur = x_2d.float().pow(2).mean(dim=0)
            g_out_cur = (go_2d.float() * scale_un).pow(2).mean(dim=0)
            if not mod.ema_initialized:
                mod.h_in_ema.copy_(h_in_cur)
                mod.g_out_ema.copy_(g_out_cur)
                mod.ema_initialized = True
            else:
                mod.h_in_ema.mul_(0.95).add_(h_in_cur, alpha=0.05)
                mod.g_out_ema.mul_(0.95).add_(g_out_cur, alpha=0.05)

            if mod.vote_gain > 0.0:
                # Dimensionless Kronecker Diagonal Gauss-Newton displacement in discrete code units:
                #   u_{ij} = -(sign(d_{ig}) * (g_{W,ij} / gw_rms_ema)) / (d_tilde_{ig} * g_tilde_i * h_tilde_j)
                # Dividing each Kronecker factor by its matrix median cancels out identically in u_hat = u / u_rms_ema
                # while guaranteeing O(1) dynamic range (zero FP32 overflow/underflow across all 28 layers!).
                gw_un = grad_w * float(mod.grad_accum)
                gw_rms_cur = float(gw_un.pow(2).mean().sqrt().clamp(min=1e-30).item())
                if mod.gw_rms_ema <= 0.0:
                    mod.gw_rms_ema = gw_rms_cur
                else:
                    mod.gw_rms_ema = 0.95 * mod.gw_rms_ema + 0.05 * gw_rms_cur
                gw_norm = gw_un / max(mod.gw_rms_ema, 1e-30)
                del gw_un

                h_med = mod.h_in_ema.median().clamp(min=1e-20)
                g_med = mod.g_out_ema.median().clamp(min=1e-30)
                h_tilde = (mod.h_in_ema / h_med) + 0.02
                g_tilde = (mod.g_out_ema / g_med) + 0.02

                d_full = d_eff.detach().float().repeat_interleave(G, dim=1)
                d_sign = torch.where(d_full < 0.0, -1.0, torch.where(d_full > 0.0, 1.0, 0.0))
                d_abs = d_full.abs()
                d_med = d_abs.median().clamp(min=1e-8)
                d_tilde = (d_abs / d_med) + 0.02
                del d_full, d_abs

                denom = d_tilde * g_tilde.unsqueeze(1) * h_tilde.unsqueeze(0)
                u_step = -(d_sign * gw_norm) / denom
                del d_sign, d_tilde, denom, gw_norm

                cur_rms = float(u_step.pow(2).mean().sqrt().clamp(min=1e-12).item())
                if mod.u_rms_ema <= 0.0:
                    mod.u_rms_ema = cur_rms
                else:
                    mod.u_rms_ema = 0.95 * mod.u_rms_ema + 0.05 * cur_rms

                u_hat = u_step / max(mod.u_rms_ema, 1e-12)
                del u_step
                # Proximal Voronoi Deadzone Gate: zero vote inside |u_hat| <= tau_dead (no Voronoi chattering!)
                u_prox = torch.sign(u_hat) * F.relu(u_hat.abs() - mod.tau_dead)
                del u_hat
                raw_v = (mod.vote_gain * u_prox).clamp(-mod.vote_clip, mod.vote_clip)
                del u_prox
                # Unbiased stochastic rounding (exact 0 when raw_v == 0!)
                vi = torch.floor(raw_v + torch.rand_like(raw_v)).to(torch.int16)
                vb_new = (mod.vote_buf.to(torch.int16) + vi).clamp(-127, 127).to(torch.int8)
                mod.vote_buf.copy_(vb_new)

        del grad_w, grad_w_grp
        grad_b = go_2d.float().sum(dim=0) if ctx.has_bias else None
        return grad_x, None, grad_d, grad_b, None


class PrimalQ2Linear(nn.Module):
    """
    Shadowless discrete `V3_Q2_0` (`q in {-2, -1, 0, +1}` + signed `d_base * exp(log_mul)`) layer.
    Zero hard channel masks and zero hard layer locks: super-weight channels self-protect via
    `1 / H_in_j` Gauss-Newton barrier + Proximal Voronoi Deadzone Gate + leaky EMA vote decay!
    """

    def __init__(
        self,
        q_int8: torch.Tensor,
        d_fp16: torch.Tensor,
        bias: Optional[torch.Tensor],
        group_size: int = 64,
        grad_accum: int = 4,
    ):
        super().__init__()
        self.out_features, self.in_features = q_int8.shape
        self.group_size = group_size
        self.grad_accum = grad_accum
        self.register_buffer("q", q_int8.to(torch.int8).contiguous())
        self.register_buffer("vote_buf", torch.zeros_like(self.q, dtype=torch.int8))
        # Preserve exact sign of d_fp16! (Never clamp negative scales to >= 0!)
        d_f = d_fp16.float().contiguous()
        d_safe = torch.where(
            d_f > 0.0,
            d_f.clamp(min=1e-7),
            torch.where(d_f < 0.0, d_f.clamp(max=-1e-7), torch.zeros_like(d_f)),
        )
        self.register_buffer("d_base", d_safe)
        self.register_buffer("h_in_ema", torch.zeros(self.in_features, dtype=torch.float32))
        self.register_buffer("g_out_ema", torch.zeros(self.out_features, dtype=torch.float32))
        self.log_mul = nn.Parameter(torch.zeros_like(self.d_base))
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().float().clone())
        else:
            self.register_parameter("bias", None)
        self.ema_initialized: bool = False
        self.gw_rms_ema: float = 0.0
        self.u_rms_ema: float = 0.0
        self.vote_gain: float = 0.0
        self.vote_clip: float = 2.0
        self.tau_dead: float = 1.15
        self.vote_decay: float = 0.95

    def cur_d(self) -> torch.Tensor:
        return self.d_base * torch.exp(self.log_mul.clamp(-2.5, 2.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ProximalGaussNewtonQ2STE.apply(x, self.q, self.cur_d(), self.bias, self)

    @torch.no_grad()
    def apply_q2_step_capped(self, threshold: int, max_flips: int) -> int:
        if max_flips <= 0:
            return 0
        v = self.vote_buf.to(torch.int16)
        q_old = self.q.to(torch.int16)
        cand = ((v >= threshold) & (q_old < 1)) | ((v <= -threshold) & (q_old > -2))
        n_cand = int(cand.sum().item())
        if n_cand > 0:
            if n_cand > max_flips:
                scores = torch.where(cand, v.abs(), torch.zeros_like(v)).reshape(-1)
                _, top_idx = torch.topk(scores, k=max_flips)
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask[top_idx] = True
                cand = mask.view_as(v)
            step = torch.where(cand, torch.sign(v), torch.zeros_like(v))
            q_new = (q_old + step).clamp(-2, 1)
            self.q.copy_(q_new.to(torch.int8))
            v = torch.where(cand, torch.zeros_like(v), v)
        else:
            q_new = q_old

        # Unbiased stochastic-rounded leaky EMA decay (0 stays exact 0!)
        at_rail = ((q_new == 1) & (v > 0)) | ((q_new == -2) & (v < 0))
        v = torch.where(at_rail, v // 2, v)
        vf = v.float() * self.vote_decay
        v_dec = torch.floor(vf + torch.rand_like(vf)).clamp(-127.0, 127.0).to(torch.int8)
        self.vote_buf.copy_(v_dec)
        return int(cand.sum().item()) if n_cand > 0 else 0

    @torch.no_grad()
    def export_q_and_d_fp16(self) -> Tuple[torch.Tensor, torch.Tensor]:
        d_f16 = self.cur_d().detach().to(torch.float16)
        return self.q.detach().cpu().clone(), d_f16.cpu().clone()

    @torch.no_grad()
    def to_dequantized_bf16(self) -> torch.Tensor:
        d_f16 = self.cur_d().detach().to(torch.float16).float()
        q_f = self.q.float().view(self.out_features, self.in_features // self.group_size, self.group_size)
        return (q_f * d_f16.unsqueeze(-1)).reshape(self.out_features, self.in_features).to(torch.bfloat16)


# ==============================================================================
# 4. Teacher Loaders, Batched Asymmetric Q2_0 GPTQ & Proximal PRIMAL Distillation
# ==============================================================================
def load_vault_teacher_qwen2(device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Qwen2ForCausalLM:
    vf = vlt3.read_vlt3(Q8_VLT3, check_crc=False)
    sd_v = vlt3.vlt3_to_state_dict(vf)
    qcfg = Qwen2Config(
        vocab_size=151936, hidden_size=1536, intermediate_size=8960,
        num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2,
        rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
        bos_token_id=151646, eos_token_id=151643, torch_dtype=dtype,
    )
    m = Qwen2ForCausalLM(qcfg).to(device=device, dtype=dtype).eval()
    pmap = dict(m.named_parameters())
    with torch.no_grad():
        for k, v in sd_v.items():
            if k.startswith("seg_scorer."):
                continue
            hk = k if k == "lm_head.weight" else f"model.{k}"
            pmap[hk].data.copy_(v.to(device=device, dtype=dtype))
    return m


@torch.no_grad()
def eval_wt2_fast(m: nn.Module, wt_chunks: torch.Tensor, sw_words: float) -> Tuple[float, float]:
    m.eval()
    tot = 0.0
    nc = wt_chunks.shape[0]
    for i in range(nc):
        lg = m(wt_chunks[i : i + 1], use_cache=False).logits[0, :-1]
        tgt = wt_chunks[i, 1:]
        for c0 in range(0, lg.shape[0], 256):
            tot += F.cross_entropy(lg[c0 : c0 + 256].float(), tgt[c0 : c0 + 256], reduction="sum").item()
        del lg
    return math.exp(tot / (nc * (wt_chunks.shape[1] - 1))), math.exp(tot / sw_words)


@torch.no_grad()
def materialize_q2_student_from_ckpt(
    transformed_teacher: Qwen2ForCausalLM,
    q2_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    device: str = "cuda",
) -> Qwen2ForCausalLM:
    student = copy.deepcopy(transformed_teacher).to(device=device, dtype=torch.bfloat16).eval()
    for mod_w in (student.model.embed_tokens.weight, student.lm_head.weight):
        arr = mod_w.detach().float().cpu().contiguous().numpy()
        pb = vlt3.quantize_p4_0(arr)
        mod_w.data.copy_(torch.from_numpy(vlt3.dequantize_p4_0(pb, arr.size).reshape(arr.shape)).to(device=device, dtype=torch.bfloat16))
    for li in range(28):
        ls = student.model.layers[li]
        for kv_nm in ("k_proj", "v_proj"):
            wm = getattr(ls.self_attn, kv_nm).weight
            arr = wm.detach().float().cpu().contiguous().numpy()
            qb = vlt3.quantize_q8_0(arr)
            wm.data.copy_(torch.from_numpy(vlt3.dequantize_q8_0(qb, arr.size).reshape(arr.shape)).to(device=device, dtype=torch.bfloat16))
        # q_proj stays in Q8_0 (boundary 0,1,26,27) / P4_0 (interior) to preserve 100K induction heads!
        wq = ls.self_attn.q_proj.weight
        arr_q = wq.detach().float().cpu().contiguous().numpy()
        dt_fn = (vlt3.quantize_q8_0, vlt3.dequantize_q8_0) if li in (0, 1, 26, 27) else (vlt3.quantize_p4_0, vlt3.dequantize_p4_0)
        pb_q = dt_fn[0](arr_q)
        wq.data.copy_(torch.from_numpy(dt_fn[1](pb_q, arr_q.size).reshape(arr_q.shape)).to(device=device, dtype=torch.bfloat16))

    for key, (q_c, d_c) in q2_data.items():
        parts = key.split(".")
        li = int(parts[1])
        mod = getattr(getattr(student.model.layers[li], parts[2]), parts[3])
        w_deq = (d_c.to(device).float().repeat_interleave(64, dim=1) * q_c.to(device).float()).to(torch.bfloat16)
        mod.weight.data.copy_(w_deq)
    return student


@torch.no_grad()
def build_gptq_q2_student(
    transformed_teacher: Qwen2ForCausalLM,
    calib_batches: List[torch.Tensor],
    cache_pt: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[Qwen2ForCausalLM, Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
    if cache_pt and os.path.exists(cache_pt):
        print(f"  [GPTQ Cache Hit] Loading cached LM-Damped Q2_0 GPTQ init -> {cache_pt}", flush=True)
        ck = torch.load(cache_pt, map_location="cpu", weights_only=False)
        stu = materialize_q2_student_from_ckpt(transformed_teacher, ck["q2_data"], device=device)
        return stu, ck["q2_data"], ck["meta"]

    t0 = time.time()
    student = copy.deepcopy(transformed_teacher).to(device=device, dtype=torch.bfloat16).eval()

    for mod_w in (student.model.embed_tokens.weight, student.lm_head.weight):
        arr = mod_w.detach().float().cpu().contiguous().numpy()
        pb = vlt3.quantize_p4_0(arr)
        mod_w.data.copy_(torch.from_numpy(vlt3.dequantize_p4_0(pb, arr.size).reshape(arr.shape)).to(device=device, dtype=torch.bfloat16))

    q2_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    down_errs: List[float] = []
    o_errs: List[float] = []

    n_tokens_cal = float(sum(b.numel() for b in calib_batches))
    seq_len = calib_batches[0].shape[-1]
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    rot_emb = student.model.rotary_emb(torch.zeros((1, seq_len, 1536), device=device, dtype=torch.bfloat16), pos_ids)

    ht_cpu = [transformed_teacher.model.embed_tokens(b.to(device)).cpu() for b in calib_batches]
    hs_cpu = [student.model.embed_tokens(b.to(device)).cpu() for b in calib_batches]

    for li in range(28):
        lt = transformed_teacher.model.layers[li]
        ls = student.model.layers[li]

        # 1. Quantize k_proj & H_128-rotated v_proj to Q8_0, and q_proj to Q8_0 (boundary) / P4_0 (interior)
        for kv_nm in ("k_proj", "v_proj"):
            wm = getattr(ls.self_attn, kv_nm).weight
            arr = wm.detach().float().cpu().contiguous().numpy()
            qb = vlt3.quantize_q8_0(arr)
            wm.data.copy_(torch.from_numpy(vlt3.dequantize_q8_0(qb, arr.size).reshape(arr.shape)).to(device=device, dtype=torch.bfloat16))

        wq = ls.self_attn.q_proj.weight
        arr_q = wq.detach().float().cpu().contiguous().numpy()
        dt_fn = (vlt3.quantize_q8_0, vlt3.dequantize_q8_0) if li in (0, 1, 26, 27) else (vlt3.quantize_p4_0, vlt3.dequantize_p4_0)
        pb_q = dt_fn[0](arr_q)
        wq.data.copy_(torch.from_numpy(dt_fn[1](pb_q, arr_q.size).reshape(arr_q.shape)).to(device=device, dtype=torch.bfloat16))

        # 2. Quantize H_128-rotated o_proj into V3_Q2_0 (18B/64w = 2.25 bpw!)
        h_o = torch.zeros((1536, 1536), dtype=torch.float64, device=device)
        g_o = torch.zeros((1536, 1536), dtype=torch.float64, device=device)
        bt_o, bs_o = [], []
        hkt = lt.self_attn.o_proj.register_forward_hook(lambda _m, inp, _out: bt_o.append(inp[0].detach()))
        hks = ls.self_attn.o_proj.register_forward_hook(lambda _m, inp, _out: bs_o.append(inp[0].detach()))
        for htc, hsc in zip(ht_cpu, hs_cpu):
            bt_o.clear()
            bs_o.clear()
            _ = lt(htc.to(device), position_embeddings=rot_emb)
            _ = ls(hsc.to(device), position_embeddings=rot_emb)
            xt = bt_o[0].reshape(-1, 1536).double()
            xs = bs_o[0].reshape(-1, 1536).double()
            h_o.addmm_(xs.t(), xs)
            g_o.addmm_(xt.t(), xs)
        hkt.remove()
        hks.remove()
        h_o /= n_tokens_cal
        g_o /= n_tokens_cal

        q_o, d_o, eo = gptq_quantize_q2_asymmetric(lt.self_attn.o_proj.weight, h_o, g_o, group_size=64, alpha_asym=0.65)
        q2_data[f"layers.{li}.self_attn.o_proj"] = (q_o.cpu(), d_o.cpu())
        ls.self_attn.o_proj.weight.data.copy_((d_o.float().repeat_interleave(64, dim=1) * q_o.float()).to(torch.bfloat16))
        o_errs.append(eo)
        del h_o, g_o

        # 3. Quantize gate_proj & equilibrated up_proj into V3_Q2_0
        h_m = torch.zeros((1536, 1536), dtype=torch.float64, device=device)
        g_m = torch.zeros((1536, 1536), dtype=torch.float64, device=device)
        bt_m, bs_m = [], []
        hkt = lt.mlp.gate_proj.register_forward_hook(lambda _m, inp, _out: bt_m.append(inp[0].detach()))
        hks = ls.mlp.gate_proj.register_forward_hook(lambda _m, inp, _out: bs_m.append(inp[0].detach()))
        for htc, hsc in zip(ht_cpu, hs_cpu):
            bt_m.clear()
            bs_m.clear()
            _ = lt(htc.to(device), position_embeddings=rot_emb)
            _ = ls(hsc.to(device), position_embeddings=rot_emb)
            xt = bt_m[0].reshape(-1, 1536).double()
            xs = bs_m[0].reshape(-1, 1536).double()
            h_m.addmm_(xs.t(), xs)
            g_m.addmm_(xt.t(), xs)
        hkt.remove()
        hks.remove()
        h_m /= n_tokens_cal
        g_m /= n_tokens_cal

        for mlp_nm in ("gate_proj", "up_proj"):
            wt_m = getattr(lt.mlp, mlp_nm).weight
            ws_m = getattr(ls.mlp, mlp_nm).weight
            qm, dm, _ = gptq_quantize_q2_asymmetric(wt_m, h_m, g_m, group_size=64, alpha_asym=0.65)
            q2_data[f"layers.{li}.mlp.{mlp_nm}"] = (qm.cpu(), dm.cpu())
            ws_m.data.copy_((dm.float().repeat_interleave(64, dim=1) * qm.float()).to(torch.bfloat16))
        del h_m, g_m

        # 4. Quantize equilibrated down_proj into V3_Q2_0
        h_dn = torch.zeros((8960, 8960), dtype=torch.float64, device=device)
        g_dn = torch.zeros((8960, 8960), dtype=torch.float64, device=device)
        bt_d, bs_d = [], []
        hkt = lt.mlp.down_proj.register_forward_hook(lambda _m, inp, _out: bt_d.append(inp[0].detach()))
        hks = ls.mlp.down_proj.register_forward_hook(lambda _m, inp, _out: bs_d.append(inp[0].detach()))
        for htc, hsc in zip(ht_cpu, hs_cpu):
            bt_d.clear()
            bs_d.clear()
            _ = lt(htc.to(device), position_embeddings=rot_emb)
            _ = ls(hsc.to(device), position_embeddings=rot_emb)
            xt = bt_d[0].reshape(-1, 8960).double()
            xs = bs_d[0].reshape(-1, 8960).double()
            h_dn.addmm_(xs.t(), xs)
            g_dn.addmm_(xt.t(), xs)
        hkt.remove()
        hks.remove()
        h_dn /= n_tokens_cal
        g_dn /= n_tokens_cal

        qd, dd, ed = gptq_quantize_q2_asymmetric(lt.mlp.down_proj.weight, h_dn, g_dn, group_size=64, alpha_asym=0.65)
        q2_data[f"layers.{li}.mlp.down_proj"] = (qd.cpu(), dd.cpu())
        ls.mlp.down_proj.weight.data.copy_((dd.float().repeat_interleave(64, dim=1) * qd.float()).to(torch.bfloat16))
        down_errs.append(ed)
        del h_dn, g_dn

        cos_sum = 0.0
        for ci in range(len(calib_batches)):
            ot = lt(ht_cpu[ci].to(device), position_embeddings=rot_emb)
            os_ = ls(hs_cpu[ci].to(device), position_embeddings=rot_emb)
            ht_nxt = ot[0] if isinstance(ot, tuple) else ot
            hs_nxt = os_[0] if isinstance(os_, tuple) else os_
            cos_sum += float(
                F.cosine_similarity(ht_nxt.float().reshape(-1, 1536), hs_nxt.float().reshape(-1, 1536), dim=-1).mean().item()
            )
            ht_cpu[ci] = ht_nxt.detach().cpu()
            hs_cpu[ci] = hs_nxt.detach().cpu()

        if li in (0, 6, 13, 20, 27):
            neg_scale_pct = 100.0 * float((dd < 0).float().mean().item())
            m2_pct = 100.0 * float((qd == -2).float().mean().item())
            z_pct = 100.0 * float((qd == 0).float().mean().item())
            print(
                f"  [LM-GPTQ-Q2_0 L{li:02d}/27] o_err={eo:.4f} | down_err={ed:.4f} | hidden_cos={cos_sum/len(calib_batches):.4f} | "
                f"neg_d={neg_scale_pct:.1f}% | q==-2:{m2_pct:.1f}% | q==0:{z_pct:.1f}% ({time.time()-t0:.1f}s)",
                flush=True,
            )

    del ht_cpu, hs_cpu
    gc.collect()
    torch.cuda.empty_cache()

    q2_params = sum(q.numel() for q, _ in q2_data.values())
    meta = {
        "group_size": 64,
        "q2_params": q2_params,
        "q2_body_share_pct": round(100.0 * q2_params / 1310167040.0, 2),
        "mean_o_proj_rel_err": round(float(np.mean(o_errs)), 4),
        "mean_down_proj_rel_err": round(float(np.mean(down_errs)), 4),
        "gptq_seconds": round(time.time() - t0, 1),
    }
    if cache_pt:
        torch.save({"q2_data": q2_data, "meta": meta}, cache_pt)
        print(f"  [Saved LM-GPTQ Q2_0 Cache] -> {cache_pt}", flush=True)
    return student, q2_data, meta


def run_proximal_primal_q2_distill(
    teacher: Qwen2ForCausalLM,
    student: Qwen2ForCausalLM,
    q2_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    wt_chunks: torch.Tensor,
    wt_words: float,
    steps: int = 120,
    grad_accum: int = 4,
    device: str = "cuda",
) -> Tuple[Qwen2ForCausalLM, Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
    t0 = time.time()
    for p in student.parameters():
        p.requires_grad_(False)
    student.model.norm.weight.requires_grad_(True)
    for layer in student.model.layers:
        layer.input_layernorm.weight.requires_grad_(True)
        layer.post_attention_layernorm.weight.requires_grad_(True)
        for bn in ("q_proj", "k_proj", "v_proj"):
            b_param = getattr(layer.self_attn, bn).bias
            if b_param is not None:
                b_param.requires_grad_(True)

    # Zero hard layer locks (ALL 28 layers L00..L27 participate via Proximal Gauss-Newton Gate!)
    q_mods: Dict[str, PrimalQ2Linear] = {}
    for key, (q_cpu, d_cpu) in q2_data.items():
        parts = key.split(".")
        li = int(parts[1])
        sub_parent = getattr(student.model.layers[li], parts[2])
        orig_lin: nn.Linear = getattr(sub_parent, parts[3])
        pql = PrimalQ2Linear(
            q_cpu.to(device), d_cpu.to(device), orig_lin.bias, group_size=64, grad_accum=grad_accum
        ).to(device)
        setattr(sub_parent, parts[3], pql)
        q_mods[key] = pql
        del orig_lin

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    _, best_ppl = eval_wt2_fast(student, wt_chunks[:30], wt_words * (30.0 / wt_chunks.shape[0]))
    best_state = {
        "q": {k: m.q.detach().cpu().clone() for k, m in q_mods.items()},
        "log_mul": {k: m.log_mul.detach().cpu().clone() for k, m in q_mods.items()},
        "aux": {n: p.detach().cpu().clone() for n, p in student.named_parameters() if p.requires_grad and not n.endswith("log_mul")},
    }

    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lm_params = [m.log_mul for m in q_mods.values()]
    aux_params = [p for n, p in student.named_parameters() if p.requires_grad and not n.endswith("log_mul")]
    base_lr_lm = 1.0e-2
    base_lr_aux = 1.0e-4
    opt = torch.optim.AdamW(
        [
            {"params": lm_params, "lr": base_lr_lm, "weight_decay": 0.0},
            {"params": aux_params, "lr": base_lr_aux, "weight_decay": 0.0},
        ]
    )

    rng = torch.Generator().manual_seed(42)
    n_slots = len(batches) * 2
    perm = torch.randperm(n_slots, generator=rng).tolist()

    tot_flips = 0
    early_l0_l3_flips = 0
    logs = []
    seq_cursor = 0
    temp = 2.0
    warmup_steps = 8

    for step in range(1, steps + 1):
        if step <= warmup_steps:
            lr_mult = step / float(warmup_steps)
        else:
            prog_lr = (step - warmup_steps) / max(1, steps - warmup_steps)
            lr_mult = 0.08 + 0.92 * (0.5 * (1.0 + math.cos(math.pi * prog_lr)))
        opt.param_groups[0]["lr"] = base_lr_lm * lr_mult
        opt.param_groups[1]["lr"] = base_lr_aux * lr_mult

        prog = (step - 1) / max(1, steps - 1)
        cos_v = 0.5 * (1.0 + math.cos(math.pi * prog))
        v_gain = (0.18 + 0.22 * cos_v) if step > warmup_steps else 0.0
        thresh = int(round(30 - 6 * cos_v))  # 24 -> 30
        max_flips_mat = max(6, int(round(6 + 10 * cos_v)))  # 16 -> 6 per matrix per step
        for m in q_mods.values():
            m.vote_gain = v_gain

        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(grad_accum):
            slot = perm[seq_cursor % n_slots]
            seq_cursor += 1
            bx, by = batches[slot // 2]
            sub_b = slot % min(2, bx.shape[0])

            x = bx[sub_b : sub_b + 1, :1024].to(device)
            y = by[sub_b : sub_b + 1, :1024].to(device)

            with torch.no_grad():
                t_lg = teacher(x, use_cache=False).logits
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_lg = student(x, use_cache=False).logits

            n_tok = x.numel()
            V = s_lg.shape[-1]
            s_flat = s_lg.view(n_tok, V)
            t_flat = t_lg.view(n_tok, V)
            y_flat = y.reshape(n_tok)

            kl_val = 0.0
            grad_s = torch.empty_like(s_flat)
            ch_sz = 256
            valid_ce = max(1, int((y_flat != -100).sum().item()))
            with torch.no_grad():
                for c0 in range(0, n_tok, ch_sz):
                    c1 = min(c0 + ch_sz, n_tok)
                    tc = t_flat[c0:c1].float() / temp
                    sc = s_flat[c0:c1].detach().float() / temp
                    p_t = F.softmax(tc, dim=-1)
                    log_p_t = F.log_softmax(tc, dim=-1)
                    log_p_s = F.log_softmax(sc, dim=-1)
                    p_s = log_p_s.exp()
                    kl_c = (p_t * (log_p_t - log_p_s)).sum(dim=-1).mean().item() * (temp * temp)
                    kl_val += kl_c * ((c1 - c0) / n_tok)
                    g_c = temp * (p_s - p_t) / float(n_tok * grad_accum)
                    yc = y_flat[c0:c1]
                    vm = yc != -100
                    if vm.any():
                        p_s1 = F.softmax(s_flat[c0:c1].detach().float(), dim=-1)
                        p_s1[vm, yc[vm]] -= 1.0
                        g_c.add_(p_s1 * (vm.float().unsqueeze(-1) * (0.08 / float(valid_ce * grad_accum))))
                    grad_s[c0:c1] = g_c.to(s_flat.dtype)
            del t_lg, t_flat
            s_flat.backward(grad_s)
            del s_lg, s_flat, grad_s
            loss_sum += kl_val / grad_accum

        torch.nn.utils.clip_grad_norm_(lm_params + aux_params, 1.0)
        opt.step()

        step_flips = 0
        if step > warmup_steps:
            for k, m in q_mods.items():
                nf = m.apply_q2_step_capped(thresh, max_flips_mat)
                step_flips += nf
                if int(k.split(".")[1]) < 4:
                    early_l0_l3_flips += nf
        tot_flips += step_flips

        if step == 1 or step % 20 == 0 or step == steps:
            student.gradient_checkpointing_disable()
            for m in q_mods.values():
                m.vote_gain = 0.0
            _, wrd_p = eval_wt2_fast(student, wt_chunks[:30], wt_words * (30.0 / wt_chunks.shape[0]))
            if wrd_p <= best_ppl:
                best_ppl = wrd_p
                best_state = {
                    "q": {k: m.q.detach().cpu().clone() for k, m in q_mods.items()},
                    "log_mul": {k: m.log_mul.detach().cpu().clone() for k, m in q_mods.items()},
                    "aux": {n: p.detach().cpu().clone() for n, p in student.named_parameters() if p.requires_grad and not n.endswith("log_mul")},
                }
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            vram_g = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(
                f"  [Proximal-PRIMAL-Q2_0 Step {step:3d}/{steps}] kl={loss_sum:.4f} | q2_flips={step_flips:,} | "
                f"WT2_30_WordPPL={wrd_p:.2f} (best={best_ppl:.2f}) | VRAM={vram_g:.2f} GiB ({time.time()-t0:.1f}s)",
                flush=True,
            )
            logs.append({"step": step, "kl": round(loss_sum, 4), "q2_flips": step_flips, "wt2_fast_word_ppl": round(wrd_p, 2)})

    student.gradient_checkpointing_disable()
    pmap_all = dict(student.named_parameters())
    with torch.no_grad():
        for k, m in q_mods.items():
            m.q.copy_(best_state["q"][k].to(device))
            m.log_mul.data.copy_(best_state["log_mul"][k].to(device))
        for n, pv in best_state["aux"].items():
            pmap_all[n].data.copy_(pv.to(device=device, dtype=pmap_all[n].dtype))

    updated_q2: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for key, pql in q_mods.items():
        parts = key.split(".")
        li = int(parts[1])
        sub_parent = getattr(student.model.layers[li], parts[2])
        lin = nn.Linear(pql.in_features, pql.out_features, bias=(pql.bias is not None), device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.data.copy_(pql.to_dequantized_bf16())
            if pql.bias is not None:
                lin.bias.data.copy_(pql.bias.detach().to(torch.bfloat16))
            updated_q2[key] = pql.export_q_and_d_fp16()
        setattr(sub_parent, parts[3], lin)

    norm_weights: Dict[str, torch.Tensor] = {
        "model.norm.weight": student.model.norm.weight.detach().float().cpu().clone()
    }
    bias_weights: Dict[str, torch.Tensor] = {}
    for li, layer in enumerate(student.model.layers):
        norm_weights[f"model.layers.{li}.input_layernorm.weight"] = layer.input_layernorm.weight.detach().float().cpu().clone()
        norm_weights[f"model.layers.{li}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.detach().float().cpu().clone()
        for bn in ("q_proj", "k_proj", "v_proj"):
            bp = getattr(layer.self_attn, bn).bias
            if bp is not None:
                bias_weights[f"model.layers.{li}.self_attn.{bn}.bias"] = bp.detach().float().cpu().clone()

    student.eval()
    return student, updated_q2, norm_weights, bias_weights, {
        "steps": steps,
        "total_q2_flips": tot_flips,
        "early_l0_l3_safe_flips": early_l0_l3_flips,
        "best_val_wt2_30_word_ppl": round(best_ppl, 2),
        "peak_distill_vram_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
        "distill_seconds": round(time.time() - t0, 1),
        "logs": logs,
    }


# ==============================================================================
# 5. Downstream Benchmark Harness (with Mid-<think> Truncation Audit)
# ==============================================================================
def _extract_boxed(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed{")
    if idx != -1:
        depth, out = 0, []
        for ch in text[idx + 7 :]:
            if ch == "{":
                depth += 1
                out.append(ch)
            elif ch == "}":
                if depth == 0:
                    break
                depth -= 1
                out.append(ch)
            else:
                out.append(ch)
        s = "".join(out).strip().replace(",", "").replace("$", "")
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if m:
            return m.group(0)
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def _extract_gold_gsm(ans: str) -> str:
    return ans.split("####")[-1].strip().replace(",", "")


@torch.no_grad()
def _eval_mc_acc(m: nn.Module, tok, items: List[Tuple[str, List[str], int]], batch_size: int = 8) -> float:
    flat_ids, meta = [], []
    for p_txt, choices, gold in items:
        p_ids = tok.encode(p_txt, add_special_tokens=False)
        st = len(flat_ids)
        c_lens = []
        for ch in choices:
            c_ids = tok.encode(" " + ch.strip(), add_special_tokens=False)
            flat_ids.append([151646] + p_ids + c_ids)
            c_lens.append(len(c_ids))
        meta.append((st, len(choices), c_lens, gold))

    all_scores = []
    for b0 in range(0, len(flat_ids), batch_size):
        chunk = flat_ids[b0 : b0 + batch_size]
        ml = max(len(x) for x in chunk)
        inp = torch.full((len(chunk), ml), 151643, dtype=torch.long, device="cuda")
        for i, x in enumerate(chunk):
            inp[i, : len(x)] = torch.tensor(x, dtype=torch.long, device="cuda")
        lg_bf16 = m(inp, use_cache=False).logits
        for i, x in enumerate(chunk):
            L_i = len(x) - 1
            row_lp = F.log_softmax(lg_bf16[i, :L_i].float(), dim=-1)
            tg_i = inp[i, 1 : L_i + 1].unsqueeze(-1)
            all_scores.append(row_lp.gather(-1, tg_i).squeeze(-1).cpu())
        del lg_bf16, inp

    hits = 0
    for st, nc, c_lens, gold in meta:
        vals = []
        for k in range(nc):
            cl = c_lens[k]
            sc = all_scores[st + k]
            vals.append(float(sc[-cl:].sum().item()) / max(1, cl))
        if int(np.argmax(vals)) == gold:
            hits += 1
    return round(100.0 * hits / len(items), 2)


@torch.no_grad()
def evaluate_downstream(m_q: nn.Module, tok, wt_chunks: torch.Tensor, wt_words: float, tag: str) -> Dict[str, Any]:
    m_q.eval()
    gc.collect()
    torch.cuda.empty_cache()
    sub_ppl, wrd_ppl = eval_wt2_fast(m_q, wt_chunks, wt_words)

    ae_ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    ac_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    sq_ds = load_dataset("allenai/sciq", split="test")

    def _arc_items(ds):
        out = []
        for r in ds:
            lbls = list(r["choices"]["label"])
            txts = list(r["choices"]["text"])
            if r["answerKey"] in lbls:
                out.append((f"Question: {r['question']}\nAnswer:", txts, lbls.index(r["answerKey"])))
        return out

    sq_rc, sq_cb = [], []
    for r in sq_ds:
        chs = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        sq_rc.append((f"Context: {r['support']}\nQuestion: {r['question']}\nAnswer:", chs, 0))
        sq_cb.append((f"Question: {r['question']}\nAnswer:", chs, 0))

    arc_e = _eval_mc_acc(m_q, tok, _arc_items(ae_ds))
    arc_c = _eval_mc_acc(m_q, tok, _arc_items(ac_ds))
    sciq_rc = _eval_mc_acc(m_q, tok, sq_rc)
    sciq_cb = _eval_mc_acc(m_q, tok, sq_cb)

    gsm_ds = load_dataset("openai/gsm8k", "main", split="test[:200]")
    tok.padding_side = "left"
    hits = 0
    n_truncated = 0
    completed_hits = 0
    completed_cnt = 0
    bs = 16
    for i in range(0, len(gsm_ds), bs):
        sub = [gsm_ds[j] for j in range(i, min(i + bs, len(gsm_ds)))]
        prompts = [
            f"{BOS_STR}{USR_STR}Solve the following math problem step by step and put your final numerical answer inside \\boxed{{}}.\n\n{r['question']}{AST_STR}{THK_STR}\n"
            for r in sub
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        il = enc["input_ids"].shape[1]
        out = m_q.generate(**enc, max_new_tokens=1536, do_sample=False, pad_token_id=151643, eos_token_id=[151643])
        for bi, r in enumerate(sub):
            gen_ids = out[bi, il:]
            gtxt = tok.decode(gen_ids, skip_special_tokens=True)
            has_eos = bool((gen_ids == 151643).any().item()) or ("\\boxed{" in gtxt)
            ok = _extract_boxed(gtxt) == _extract_gold_gsm(r["answer"])
            if ok:
                hits += 1
            if not has_eos:
                n_truncated += 1
            else:
                completed_cnt += 1
                if ok:
                    completed_hits += 1
        del out, enc
    gsm_acc = round(100.0 * hits / len(gsm_ds), 2)
    trunc_pct = round(100.0 * n_truncated / len(gsm_ds), 2)
    comp_acc = round(100.0 * completed_hits / max(1, completed_cnt), 2)

    ip = tok(
        f"{BOS_STR}{USR_STR}Is Taiwan a de facto independent country? Answer concisely.{AST_STR}{THK_STR}\n{ETHK_STR}\n\n",
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to("cuda")
    tw_out = tok.decode(
        m_q.generate(ip, max_new_tokens=64, do_sample=False, pad_token_id=151643)[0, ip.shape[1] :],
        skip_special_tokens=True,
    ).strip()

    res = {
        "wt2_subtoken_ppl": round(sub_ppl, 3),
        "wt2_word_ppl": round(wrd_ppl, 2),
        "gsm8k_200_acc": gsm_acc,
        "gsm8k_200_ci95_pm": round(1.96 * math.sqrt((gsm_acc / 100.0) * (1.0 - gsm_acc / 100.0) / 200.0) * 100.0, 2),
        "gsm8k_200_truncated_1536_pct": trunc_pct,
        "gsm8k_200_completed_only_acc": comp_acc,
        "arc_easy_full": arc_e,
        "arc_challenge_full": arc_c,
        "sciq_rc_full": sciq_rc,
        "sciq_cb_full": sciq_cb,
        "mc4_mean": round((arc_e + arc_c + sciq_rc + sciq_cb) / 4.0, 2),
        "taiwan_nothink_sample": tw_out[:180],
    }
    print(
        f"[{tag}] WT2_WordPPL={wrd_ppl:.2f} | GSM8K-200={gsm_acc:.2f}% (+-{res['gsm8k_200_ci95_pm']:.1f}%, trunc={trunc_pct:.1f}%, "
        f"completed_acc={comp_acc:.2f}%) | ARC-E={arc_e:.2f}% | ARC-C={arc_c:.2f}% | SciQ-RC={sciq_rc:.2f}% | MC4={res['mc4_mean']:.2f}%",
        flush=True,
    )
    return res


# ==============================================================================
# 6. Lossless VLT3 Export, Freestanding C99 Speed Bench & Adversarial O(1) NIAH
# ==============================================================================
def export_q2_vlt3(
    transformed_teacher: Qwen2ForCausalLM,
    q2_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    norm_w: Dict[str, torch.Tensor],
    bias_w: Dict[str, torch.Tensor],
    out_vlt3: str,
) -> Dict[str, Any]:
    t0 = time.time()
    vf_base = vlt3.read_vlt3(Q8_VLT3, check_crc=False)
    cfg = vlt3.vlt3_config(vf_base)
    sd = vlt3.vlt3_to_state_dict(vf_base)
    del vf_base

    # Copy exact H_128-rotated v_proj weights from transformed_teacher so V <-> O stay 100% paired!
    with torch.no_grad():
        for li in range(28):
            sd[f"layers.{li}.self_attn.v_proj.weight"] = (
                transformed_teacher.model.layers[li].self_attn.v_proj.weight.detach().float().cpu().clone()
            )
            if transformed_teacher.model.layers[li].self_attn.v_proj.bias is not None:
                sd[f"layers.{li}.self_attn.v_proj.bias"] = (
                    transformed_teacher.model.layers[li].self_attn.v_proj.bias.detach().float().cpu().clone()
                )

    for extra_dict in (norm_w, bias_w):
        for nk, nv in extra_dict.items():
            v_key = nk[len("model."):] if nk.startswith("model.") else nk
            if v_key in sd:
                sd[v_key] = nv.detach().float().cpu().clone()

    prepacked: Dict[str, Tuple[int, bytes]] = {}
    q2_params = 0
    zero_cnt = 0
    m2_cnt = 0
    neg_scale_cnt = 0
    tot_scale_cnt = 0
    for mod_key, (q_cpu, d_cpu) in q2_data.items():
        w_key = f"{mod_key}.weight"
        q_np = q_cpu.detach().cpu().numpy().astype(np.int8)
        d_np = d_cpu.detach().cpu().numpy().astype(np.float16)
        raw_bytes = vlt3.pack_q2_0(q_np, d_np)
        prepacked[w_key] = (vlt3.Q2_0, raw_bytes)
        q2_params += int(q_np.size)
        zero_cnt += int((q_np == 0).sum())
        m2_cnt += int((q_np == -2).sum())
        neg_scale_cnt += int((d_np < 0).sum())
        tot_scale_cnt += int(d_np.size)

    meta_w = vlt3.write_vlt3(
        out_vlt3,
        sd,
        cfg,
        HF_TEACHER_DIR,
        fmt="q2_outlier",
        model_name="VAULT-v3-DeepSeek-1.5B-Q2-Outlier",
        prepacked=prepacked,
    )
    vf = vlt3.read_vlt3(out_vlt3, check_crc=True)
    total_params = sum(int(np.prod(e["dims"][: e["ndim"]])) for e in vf.entries)
    weight_bytes = sum(int(e["nbytes"]) for e in vf.entries)
    fsize = int(meta_w["file_size"])
    res = {
        "vlt3_file": os.path.basename(out_vlt3),
        "file_bytes": fsize,
        "file_mb": round(fsize / 1e6, 2),
        "file_gib": round(fsize / (1024 ** 3), 4),
        "total_params": total_params,
        "q2_params": q2_params,
        "q2_zero_pct": round(100.0 * zero_cnt / max(1, q2_params), 2),
        "q2_outlier_m2_pct": round(100.0 * m2_cnt / max(1, q2_params), 2),
        "q2_neg_scale_pct": round(100.0 * neg_scale_cnt / max(1, tot_scale_cnt), 2),
        "overall_effective_bpw": round(8.0 * weight_bytes / total_params, 3),
        "crc32_verified": True,
        "export_seconds": round(time.time() - t0, 2),
    }
    print(
        f"[EXPORT {res['vlt3_file']}] {res['file_mb']:.2f} MB ({res['file_gib']:.3f} GiB) | "
        f"eff_bpw={res['overall_effective_bpw']} | q==0:{res['q2_zero_pct']:.1f}% | q==-2:{res['q2_outlier_m2_pct']:.1f}% | "
        f"neg_d:{res['q2_neg_scale_pct']:.1f}% | CRC32=OK",
        flush=True,
    )
    return res


def bench_c99_vlt3(vlt3_path: str, run_chat: bool = True) -> Dict[str, Any]:
    best_tok_s = 0.0
    best_rec: Dict[str, Any] = {}
    for threads in (12, 24):
        cpu_mask = "0-11" if threads == 12 else "0-23"
        for _ in range(3):
            cmd = [
                "taskset", "-c", cpu_mask,
                HOST_BIN, vlt3_path,
                "--bench-prefill", "256",
                "--bench", "128",
                "--threads", str(threads),
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
            for line in (out.stdout or "").splitlines():
                line = line.strip()
                if line.startswith("{") and '"decode_tok_s"' in line:
                    jb = json.loads(line)
                    tok_s = float(jb["decode_tok_s"])
                    if tok_s > best_tok_s:
                        best_tok_s = tok_s
                        best_rec = {
                            "best_threads": threads,
                            "decode_tok_s": tok_s,
                            "decode_ms_tok": float(jb["ms_per_tok"]),
                            "prefill_tok_s": float(jb["prefill_tok_s"]),
                            "peak_vmhwm_mib": round(float(jb["peak_rss_mb"]) / 1.048576, 2),
                            "peak_vmhwm_mb_dec": float(jb["peak_rss_mb"]),
                        }
    if run_chat:
        c_math = subprocess.run(
            [HOST_BIN, vlt3_path, "--verify", "--chat", "Compute 17 * 23 step by step concisely.", "--nothink", "--n", "80", "--temp", "0", "--threads", "24"],
            capture_output=True, text=True, check=True,
        )
        c_tw = subprocess.run(
            [HOST_BIN, vlt3_path, "--chat", "Is Taiwan a de facto independent country? Answer directly in two sentences.", "--nothink", "--n", "80", "--temp", "0", "--threads", "24"],
            capture_output=True, text=True, check=True,
        )
        best_rec["c99_math_sample"] = c_math.stdout.strip()[:220]
        best_rec["c99_taiwan_sample"] = c_tw.stdout.strip()[:220]
    print(
        f"[C99 BENCH {os.path.basename(vlt3_path)}] {best_rec['decode_tok_s']:.2f} tok/s "
        f"({best_rec['decode_ms_tok']:.2f} ms/tok, threads={best_rec['best_threads']}) | "
        f"VmHWM={best_rec['peak_vmhwm_mib']:.2f} MiB",
        flush=True,
    )
    return best_rec


@torch.no_grad()
def eval_adversarial_niah_suite(vlt3_path: str, tok) -> Dict[str, Any]:
    """
    Peer-Review Section 0.4 Expanded O(1) NIAH Stress Suite:
      1-3. Standard 16K / 32K / 100K @ 25% depth (direct comparability with T1.58 report)
      4.   Deep 16K @ 85% depth
      5.   Rollover-Boundary Straddle 32K @ ~93.5% depth (just evicted across 2W=2048 rollover!)
      6.   Deep 100K @ 85% depth (late RoPE-rebasing + 97+ chunk evictions)
      7.   Multi-Needle Bank Competition (16 distinct key-value secrets scattered across 32K tokens
           competing for B=1024 SegmentScorer budget; queries Key_05 from early-mid haystack)
      8.   Chronos Variable Supersession (key `ORION_PIN` assigned old value `314159` at 20% depth
           and updated to `271828` at 78% depth across 32K tokens; verifies latest state wins!)
    """
    vf = vlt3.read_vlt3(vlt3_path, check_crc=False)
    cfg = vlt3.vlt3_config(vf)
    m = VAULTv3ForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    sd = {
        k: v.to(device="cuda", dtype=(torch.float32 if v.ndim == 1 or k.startswith("seg_scorer.") else torch.bfloat16))
        for k, v in vlt3.vlt3_to_state_dict(vf).items()
    }
    m.load_state_dict(sd, strict=True)

    hay_unit = (
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
        "Water boils at one hundred degrees Celsius at standard atmospheric pressure. "
    )
    unit_ids = tok.encode(hay_unit, add_special_tokens=False)

    def _run_ids(ids: List[int], gold: str, label: str) -> Dict[str, Any]:
        inp = torch.tensor([ids], dtype=torch.long, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        lg, st = m.prefill(inp, vault_on=True)
        gen = []
        nxt = int(lg[0].argmax().item())
        gen.append(nxt)
        for _ in range(16):
            lg2 = m.step(torch.tensor([nxt], dtype=torch.long, device="cuda"), st)
            nxt = int(lg2[0].argmax().item())
            if nxt == 151643:
                break
            gen.append(nxt)
        gtxt = tok.decode(gen, skip_special_tokens=True).strip()
        ok = gold in gtxt
        vram_mib = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
        print(
            f"  [NIAH {os.path.basename(vlt3_path)} | {label} | {inp.shape[1]:,} tok] gold={gold} -> '{gtxt[:42]}' | hit={ok} | VRAM={vram_mib} MiB",
            flush=True,
        )
        return {"probe": label, "total_tokens": inp.shape[1], "gold": gold, "output": gtxt[:64], "exact_hit": ok, "peak_vram_mib": vram_mib}

    probes = []
    # Probes 1-6: Single-needle across depths & exact 2W rollover boundary
    single_specs = [
        ("16K_d25", 16384, 0.25, "739184"),
        ("32K_d25", 32768, 0.25, "482915"),
        ("100K_d25", 100000, 0.25, "916427"),
        ("16K_d85_deep", 16384, 0.85, "604829"),
        ("32K_2W_rollover_edge", 32768, -2250, "591736"),  # negative => exact token offset before prompt!
        ("100K_d85_deep", 100000, 0.85, "827364"),
    ]
    for lbl, ctx_target, depth_spec, pkey in single_specs:
        needle = f" Remember this secret passkey number: {pkey}. The secret passkey is {pkey}. "
        prompt_q = (
            f"<｜User｜>What is the 6-digit secret passkey number mentioned earlier in the text? "
            f"Reply with the exact digits.<｜Assistant｜><think>\n</think>\n\nThe secret passkey number is "
        )
        n_units = max(20, ctx_target // len(unit_ids))
        if depth_spec < 0:
            post_u = max(1, int(abs(depth_spec)) // len(unit_ids))
            pre_u = max(1, n_units - post_u)
        else:
            pre_u = int(n_units * depth_spec)
            post_u = n_units - pre_u
        ids = (
            [151646]
            + unit_ids * pre_u
            + tok.encode(needle, add_special_tokens=False)
            + unit_ids * post_u
            + tok.encode(prompt_q, add_special_tokens=False)
        )
        probes.append(_run_ids(ids, pkey, lbl))

    # Probe 7: 16-Needle Multi-Key Bank Competition across 32K tokens (queries VAULT_KEY_05 = 519283)
    kv_pairs = [(f"VAULT_KEY_{i:02d}", f"{104729 + i * 82911:06d}"[:6]) for i in range(1, 17)]
    gold_mk = kv_pairs[4][1]  # VAULT_KEY_05
    seg_units = max(4, (32768 // len(unit_ids)) // 17)
    ids_mk = [151646]
    for k_id, v_val in kv_pairs:
        ids_mk.extend(unit_ids * seg_units)
        ids_mk.extend(tok.encode(f" Record update: {k_id} is assigned code {v_val}. {k_id} = {v_val}. ", add_special_tokens=False))
    ids_mk.extend(unit_ids * seg_units)
    q_mk = (
        "<｜User｜>What 6-digit code was assigned to VAULT_KEY_05 earlier in the log? "
        "Reply with the exact digits.<｜Assistant｜><think>\n</think>\n\nVAULT_KEY_05 is assigned code "
    )
    ids_mk.extend(tok.encode(q_mk, add_special_tokens=False))
    probes.append(_run_ids(ids_mk, gold_mk, "32K_16key_competition"))

    # Probe 8: 32K Chronos Variable Supersession (old=314159 @ 20% -> updated=271828 @ 78%)
    tot_u = 32768 // len(unit_ids)
    u1 = int(tot_u * 0.20)
    u2 = int(tot_u * 0.58)
    u3 = tot_u - u1 - u2
    n_old = " System state log: ORION_PIN is set to 314159. Current ORION_PIN = 314159. "
    n_new = " System state supersede: ORION_PIN is updated to 271828 (superseding previous value). Current ORION_PIN = 271828. "
    q_sup = (
        "<｜User｜>What is the most recent updated 6-digit value of ORION_PIN in the log? "
        "Reply with the exact latest digits.<｜Assistant｜><think>\n</think>\n\nThe latest updated ORION_PIN value is "
    )
    ids_sup = (
        [151646]
        + unit_ids * u1
        + tok.encode(n_old, add_special_tokens=False)
        + unit_ids * u2
        + tok.encode(n_new, add_special_tokens=False)
        + unit_ids * u3
        + tok.encode(q_sup, add_special_tokens=False)
    )
    probes.append(_run_ids(ids_sup, "271828", "32K_chronos_supersession"))

    del m, sd, vf
    gc.collect()
    torch.cuda.empty_cache()

    std3_hits = sum(1 for p in probes[:3] if p["exact_hit"])
    all_hits = sum(1 for p in probes if p["exact_hit"])
    return {
        "standard_3probe_hits": f"{std3_hits}/3",
        "adversarial_8probe_hits": f"{all_hits}/{len(probes)}",
        "adversarial_accuracy_pct": round(100.0 * all_hits / len(probes), 2),
        "probes": probes,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(HF_TEACHER_DIR, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    wt2 = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    raw_t = "\n\n".join(wt2["text"])
    enc_w = tok(raw_t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    nc = enc_w.numel() // 2048
    wt_chunks = enc_w[: nc * 2048].view(nc, 2048).to("cuda")
    wt_words = len(raw_t.split()) * (nc * 2048 / enc_w.numel())

    results: Dict[str, Any] = {}

    # --------------------------------------------------------------------------
    # STEP 1: Untouched Upstream BF16 Baseline Audit (Peer-Review Item 0.2)
    # --------------------------------------------------------------------------
    up_cache_json = os.path.join(OUT_DIR, "upstream_bf16_eval.json")
    if os.path.exists(up_cache_json):
        print(f"=== [STEP 1] Loading Cached Untouched Upstream BF16 Baseline -> {up_cache_json} ===", flush=True)
        with open(up_cache_json) as f:
            results["upstream_deepseek_r1_1_5b_bf16"] = json.load(f)
    else:
        print("=== [STEP 1] Evaluating Untouched Upstream `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (BF16) Under Identical Harness ===", flush=True)
        up_dir = snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", local_files_only=True)
        m_up = AutoModelForCausalLM.from_pretrained(up_dir, torch_dtype=torch.bfloat16).to("cuda").eval()
        eval_up = evaluate_downstream(m_up, tok, wt_chunks, wt_words, "UPSTREAM_BF16")
        del m_up
        gc.collect()
        torch.cuda.empty_cache()
        results["upstream_deepseek_r1_1_5b_bf16"] = eval_up
        with open(up_cache_json, "w") as f:
            json.dump(eval_up, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------------------------
    # STEP 2: Naive RTN-Q2_0 Control + Offline H_128 & Equilibration Pre-Transforms
    # --------------------------------------------------------------------------
    stream_d = torch.load(STREAM_PT, map_location="cpu", weights_only=False)
    batches = stream_d["batches"]
    raw_cal = [batches[i][0][b, :1024] for i in range(24) for b in range(2)]  # 48 x 1024 = 49,152 tokens
    calib_batches = [torch.stack(raw_cal[k : k + 4], dim=0) for k in range(0, 48, 4)]

    print("\n=== [STEP 2] Loading VAULT-v3 Teacher, Running Naive RTN-Q2_0 Control & Applying Exact Offline H_128 + Equilibration ===", flush=True)
    teacher = load_vault_teacher_qwen2("cuda", dtype=torch.bfloat16)

    # Naive RTN-Q2_0 control on unrotated weights (o_proj + gate + up + down)
    m_rtn = copy.deepcopy(teacher)
    with torch.no_grad():
        for li in range(28):
            ly = m_rtn.model.layers[li]
            for mod in (ly.self_attn.o_proj, ly.mlp.gate_proj, ly.mlp.up_proj, ly.mlp.down_proj):
                arr = mod.weight.detach().float().cpu().contiguous().numpy()
                qb = vlt3.quantize_q2_0(arr)
                mod.weight.data.copy_(
                    torch.from_numpy(vlt3.dequantize_q2_0(qb, arr.size).reshape(arr.shape)).to(device="cuda", dtype=torch.bfloat16)
                )
    rtn_sub, rtn_wrd = eval_wt2_fast(m_rtn, wt_chunks, wt_words)
    print(f"[Naive RTN-Q2_0 Control] WT2_SubPPL={rtn_sub:.2f} | WT2_WordPPL={rtn_wrd:.2f}", flush=True)
    del m_rtn
    gc.collect()
    torch.cuda.empty_cache()

    # Verify exact pre-transform invariance on Teacher
    with torch.no_grad():
        probe_inp = wt_chunks[:1, :256]
        lg_pre = teacher(probe_inp, use_cache=False).logits.float()
    pre_meta = apply_offline_incoherent_transforms(teacher)
    with torch.no_grad():
        lg_post = teacher(probe_inp, use_cache=False).logits.float()
        bf16_cos = float(F.cosine_similarity(lg_pre.reshape(-1, 151936), lg_post.reshape(-1, 151936), dim=-1).mean().item())
        del lg_pre, lg_post
    pre_meta["bf16_28layer_logit_cosine"] = round(bf16_cos, 7)
    print(
        f"[Offline Pre-Transforms] H_128(V<->O) + Equilibrate(up<->down): col_outlier_ratio "
        f"{pre_meta['down_proj_max_col_outlier_ratio_before']}x -> {pre_meta['down_proj_max_col_outlier_ratio_after']}x | "
        f"28-layer BF16 logit cosine = {bf16_cos:.7f}",
        flush=True,
    )
    results["naive_rtn_q2_0"] = {"wt2_subtoken_ppl": round(rtn_sub, 3), "wt2_word_ppl": round(rtn_wrd, 2)}
    results["offline_pretransforms"] = pre_meta

    # --------------------------------------------------------------------------
    # STEP 3: Scale-Invariant LM-Damped Signed-Scale Two-Pass V3_Q2_0 GPTQ
    # --------------------------------------------------------------------------
    print("\n=== [STEP 3] Scale-Invariant LM-Damped Signed-Scale Asymmetric GPTQ -> V3_Q2_0 (`o_proj + MLP`, 93.3% of body) ===", flush=True)
    stu_q2, q2_data, meta_q2 = build_gptq_q2_student(
        teacher,
        calib_batches,
        cache_pt=os.path.join(OUT_DIR, "gptq_init_q2_g64.pt"),
        device="cuda",
    )
    gptq_sub, gptq_wrd = eval_wt2_fast(stu_q2, wt_chunks, wt_words)
    print(
        f"[Q2_0 + H_128 + LM-GPTQ Before PRIMAL] Body_Q2_Share={meta_q2['q2_body_share_pct']}% | "
        f"WT2_SubPPL={gptq_sub:.3f} | WT2_WordPPL={gptq_wrd:.2f}",
        flush=True,
    )

    # --------------------------------------------------------------------------
    # STEP 4: Proximal Gauss-Newton Second-Order PRIMAL KL-Distillation
    # --------------------------------------------------------------------------
    distill_pt = os.path.join(OUT_DIR, "q2_outlier_g64.pt")
    print("\n=== [STEP 4] Proximal Gauss-Newton Second-Order PRIMAL KL-Distillation (120 Steps, Zero Hard Locks) ===", flush=True)
    stu_q2, q2_data, nw_q2, bw_q2, kl_q2 = run_proximal_primal_q2_distill(
        teacher, stu_q2, q2_data, batches, wt_chunks, wt_words, steps=120, grad_accum=4, device="cuda"
    )
    torch.save(
        {"q2_data": q2_data, "norm_weights": nw_q2, "bias_weights": bw_q2, "meta": meta_q2, "kl_meta": kl_q2},
        distill_pt,
    )

    # Export .vlt3 while transformed `teacher` is in memory so rotated v_proj is written losslessly!
    print("\n=== [STEP 5] Exporting CRC32-Verified `vault_v3_deepseek_r1_1_5b_q2_outlier.vlt3` ===", flush=True)
    exp_q2 = export_q2_vlt3(teacher, q2_data, nw_q2, bw_q2, Q2_OUT_VLT3)

    # Free teacher VRAM before full downstream evaluation!
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== [STEP 6] Full Downstream Evaluation on `VAULT-v3-DeepSeek-R1-1.5B-Q2-Outlier` ===", flush=True)
    eval_q2 = evaluate_downstream(stu_q2, tok, wt_chunks, wt_words, "Q2_0_OUTLIER_HADAMARD")
    del stu_q2, q2_data, nw_q2, bw_q2
    gc.collect()
    torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # STEP 7: Freestanding C99 Speed Curve & Adversarial 8-Probe O(1) NIAH Suite
    # --------------------------------------------------------------------------
    print("\n=== [STEP 7] Benchmarking Freestanding C99 Speed Curve & Adversarial 8-Probe O(1) NIAH Stress Suite ===", flush=True)
    c99_q8 = bench_c99_vlt3(Q8_VLT3, run_chat=False)
    c99_p4 = bench_c99_vlt3(P4_VLT3, run_chat=False)
    c99_t158_hyb = bench_c99_vlt3(T158_HYB_VLT3, run_chat=False) if os.path.exists(T158_HYB_VLT3) else None
    c99_q2 = bench_c99_vlt3(Q2_OUT_VLT3, run_chat=True)

    niah_p4 = eval_adversarial_niah_suite(P4_VLT3, tok)
    niah_q2 = eval_adversarial_niah_suite(Q2_OUT_VLT3, tok)

    results["q2_0_outlier_hadamard_g64"] = {
        "gptq_meta": meta_q2,
        "gptq_only_wt2_subtoken_ppl": round(gptq_sub, 3),
        "gptq_only_wt2_word_ppl": round(gptq_wrd, 2),
        "proximal_primal_kl_distill": kl_q2,
        "vlt3_export": exp_q2,
        "c99_runtime": c99_q2,
        "downstream": eval_q2,
        "adversarial_niah_suite": niah_q2,
    }
    results["c99_speed_comparison"] = {
        "q8_0_1810mb": c99_q8,
        "p4_0_1204mb": c99_p4,
        "t1_58_hybrid_704mb": c99_t158_hyb,
        "q2_0_outlier_690mb": c99_q2,
    }
    results["p4_0_adversarial_niah_baseline"] = niah_p4

    with open(REPORT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[TRACK-1 Q2_0 + PEER-REVIEW UPGRADE STUDY COMPLETE] Saved -> {REPORT_JSON}", flush=True)


if __name__ == "__main__":
    main()
