#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
Empirical 1.58-Bit (`{-1, 0, +1}`) Conversion & Second-Order Curvature-Preconditioned
PRIMAL Ternary KL-Distillation Study on `VAULT-v3-DeepSeek-R1-1.5B-PRIMAL`:

  1. Naive RTN-1.58 Control (`G=64` Round-to-Nearest Ternary) — quantifies naive ternary PTQ collapse.
  2. Sequential Asymmetric Sub-Hessian Two-Pass Cholesky-GPTQ (`GPTQ-v2` drift cancellation
     `W_tgt = W_t + alpha * W_t (G_ts - H_s)(H_s + damp*I)^{-1}` + closed-form Sub-Hessian optimal
     FP16 scale `d* = (w_g H_g q_g^T) / (q_g H_g q_g^T)` across all 28 layers):
       - **EXP 1 (`MLP-1.58-Bit Hybrid G=32`, `~677 MB` `.vlt3`):** All 84 SwiGLU MLP matrices
         (`88.2%` of body) in strict `{-1, 0, +1}` (`V3_T1_58_G32`), Attention Q/O & embed/head in
         `V3_P4_0` (boundary Q/O & 6:1 GQA KV in `V3_Q8_0`).
       - **EXP 2 (`Full-Body Pure 1.58-Bit G=64`, `~595 MB` `.vlt3`):** All 140 MLP + Attention Q/O
         matrices (`98.3%` of body) in strict `{-1, 0, +1}` (`V3_T1_58_G64` = `1.875 bpw` w/ scale).
  3. Second-Order Curvature-Preconditioned Shadowless `PRIMAL` Ternary KL-Distillation:
       - Multiplicative log-scale parameterization `d = d_base * exp(log_mul)` eliminates Step-1
         `AdamW` scale shock across heterogeneous matrix fan-ins (`1536` vs `8960`),
       - Diagonal-Hessian preconditioned trit votes `-(g_w / sqrt(E[x_c^2]))` + activation-sink
         channel guard (`h_c < 8 * mean(h_c)`) + `L00..L03` super-weight lock protect `Qwen2`
         `down_proj` super-weights while unlocking true discrete ternary descent past scale-only ceilings.
  4. Full Downstream Curve Evaluation:
       - `WikiText-2` Word PPL, `GSM8K-200` (`<think>`, `1536` tok), `ARC-Easy`, `ARC-Challenge`,
         `SciQ-RC`, `SciQ-CB`, and Uncensored Direct Probes.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import re
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
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import vlt3  # noqa: E402

HF_TEACHER_DIR = os.path.join(REPO_ROOT, "dist", "hf_vault_v3_deepseek_r1_1_5b")
Q8_VLT3 = os.path.join(REPO_ROOT, "dist", "vault_v3_deepseek_r1_1_5b_q8_0_primal.vlt3")
STREAM_PT = os.path.join(REPO_ROOT, "runs", "deepseek_r1_1_5b_primal", "deepseek_1_5b_primal_stream.pt")
OUT_DIR = os.path.join(REPO_ROOT, "runs", "ternary_1_58")
REPORT_JSON = os.path.join(REPO_ROOT, "benchmarks", "results", "vault_v3_deepseek_1_5b_ternary_1_58_report.json")

BOS_STR = "<｜begin▁of▁sentence｜>"
USR_STR = "<｜User｜>"
AST_STR = "<｜Assistant｜>"
THK_STR = "<think>"
ETHK_STR = "</think>"


# ==============================================================================
# 1. Asymmetric Drift-Canceling + Sub-Hessian Two-Pass Cholesky-GPTQ Quantizer
# ==============================================================================
@torch.no_grad()
def optimal_ternary_scales_lloyd(w_grp: torch.Tensor, n_iters: int = 4) -> torch.Tensor:
    mx = w_grp.abs().amax(dim=1, keepdim=True).unsqueeze(1).clamp(min=1e-8)
    alphas = torch.tensor([0.45, 0.60, 0.75, 0.90, 1.05], device=w_grp.device, dtype=torch.float32).view(1, 5, 1)
    d_cand = (mx * alphas).to(torch.float16).float()
    wg_exp = w_grp.unsqueeze(1)
    for _ in range(n_iters):
        qc = torch.clamp(torch.round(wg_exp / d_cand.clamp(min=1e-8)), -1.0, 1.0)
        num = (wg_exp * qc).sum(dim=2, keepdim=True).clamp(min=0.0)
        den = (qc * qc).sum(dim=2, keepdim=True).clamp(min=1.0)
        d_cand = (num / den).to(torch.float16).float().clamp(min=1e-7)
    qc = torch.clamp(torch.round(wg_exp / d_cand), -1.0, 1.0)
    err = (wg_exp - d_cand * qc).pow(2).sum(dim=2)
    best_idx = err.argmin(dim=1, keepdim=True)
    return d_cand.squeeze(2).gather(1, best_idx)


@torch.no_grad()
def gptq_quantize_ternary_asymmetric(
    w_teacher: torch.Tensor,
    h_s_fp64: torch.Tensor,
    g_ts_fp64: Optional[torch.Tensor] = None,
    group_size: int = 64,
    alpha_asym: float = 0.65,
    damp_rel: float = 0.015,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    device = w_teacher.device
    out_f, in_f = w_teacher.shape
    assert in_f % group_size == 0, f"in_f={in_f} not divisible by group_size={group_size}"

    h = h_s_fp64.clone()
    dead = torch.diag(h) <= 0.0
    if dead.any():
        h[dead, dead] = 1.0

    hbar = float(torch.diag(h).mean().clamp(min=1e-8).item())
    idx = torch.arange(in_f, device=device)

    if g_ts_fp64 is not None and alpha_asym > 0.0:
        delta_g = g_ts_fp64 - h_s_fp64
        if dead.any():
            delta_g[dead, :] = 0.0
            delta_g[:, dead] = 0.0
        h_asym = h.clone()
        h_asym[idx, idx] += 0.05 * hbar
        l_asym = torch.linalg.cholesky(h_asym)
        z_t = torch.cholesky_solve(delta_g.t(), l_asym)
        del h_asym, l_asym, delta_g
        w = (w_teacher.detach().double() + alpha_asym * (w_teacher.detach().double() @ z_t.t())).float()
        del z_t
    else:
        w = w_teacher.detach().float().clone()

    if dead.any():
        w[:, dead] = 0.0

    damp = damp_rel * hbar
    u_chol = None
    h_w = None
    for attempt in range(4):
        try:
            h_w = h.clone()
            h_w[idx, idx] += damp * (4.0 ** attempt)
            chol_l = torch.linalg.cholesky(h_w)
            h_inv = torch.cholesky_inverse(chol_l)
            u_chol = torch.linalg.cholesky(h_inv, upper=True).float()
            del chol_l, h_inv
            break
        except RuntimeError:
            if attempt == 3:
                raise

    assert u_chol is not None and h_w is not None
    h_f32 = h_w.float()
    del h_w, h
    h_diag = torch.diag(h_f32).clamp(min=1e-8)

    q_out = torch.empty((out_f, in_f), dtype=torch.int8, device=device)
    d_out = torch.empty((out_f, in_f // group_size), dtype=torch.float16, device=device)
    alphas = torch.tensor([0.45, 0.60, 0.75, 0.90, 1.05, 1.25], device=device, dtype=torch.float32).view(1, 6, 1)

    sb = 256
    for b1 in range(0, in_f, sb):
        b2 = min(b1 + sb, in_f)
        w_sb = w[:, b1:b2].clone()
        err_sb = torch.zeros_like(w_sb)
        u_sb = u_chol[b1:b2, b1:b2]

        for g0 in range(0, b2 - b1, group_size):
            g1 = g0 + group_size
            ga = b1 + g0
            w_g0 = w_sb[:, g0:g1]
            hd_g = h_diag[ga : ga + group_size].view(1, 1, group_size)
            hg = h_f32[ga : ga + group_size, ga : ga + group_size]
            u_gg = u_sb[g0:g1, g0:g1]

            mx = w_g0.abs().amax(dim=1, keepdim=True).unsqueeze(1).clamp(min=1e-8)
            d_cand = (mx * alphas).to(torch.float16).float().clamp(min=1e-7)
            wg_exp = w_g0.unsqueeze(1)
            for _ in range(4):
                qc = torch.clamp(torch.round(wg_exp / d_cand), -1.0, 1.0)
                num = (wg_exp * qc * hd_g).sum(dim=2, keepdim=True).clamp(min=0.0)
                den = (qc * qc * hd_g).sum(dim=2, keepdim=True).clamp(min=1e-8)
                d_cand = (num / den).to(torch.float16).float().clamp(min=1e-7)
            qc = torch.clamp(torch.round(wg_exp / d_cand), -1.0, 1.0)
            diff = wg_exp - d_cand * qc
            err_c = ((diff @ hg) * diff).sum(dim=2)
            best_idx = err_c.argmin(dim=1, keepdim=True)
            d_cur = d_cand.squeeze(2).gather(1, best_idx).squeeze(1)

            q_g = torch.empty((out_f, group_size), dtype=torch.float32, device=device)
            err_g = torch.zeros_like(w_g0)
            for _pass in range(2):
                w_g = w_g0.clone()
                err_g.zero_()
                d_f = d_cur.to(torch.float16).float().clamp(min=1e-7)
                for j in range(group_size):
                    col = w_g[:, j]
                    qj = torch.clamp(torch.round(col / d_f), -1.0, 1.0)
                    q_g[:, j] = qj
                    ej = (col - d_f * qj) / u_gg[j, j].clamp(min=1e-7)
                    err_g[:, j] = ej
                    if j + 1 < group_size:
                        w_g[:, j + 1 :] -= ej.unsqueeze(1) * u_gg[j, j + 1 :].unsqueeze(0)
                if _pass == 0:
                    num_h = ((w_g0 @ hg) * q_g).sum(dim=1).clamp(min=0.0)
                    den_h = ((q_g @ hg) * q_g).sum(dim=1).clamp(min=1e-8)
                    d_cur = (num_h / den_h).to(torch.float16).float().clamp(min=1e-7)

            q_out[:, ga : ga + group_size] = q_g.to(torch.int8)
            d_out[:, ga // group_size] = d_cur.to(torch.float16)
            err_sb[:, g0:g1] = err_g
            if g1 < (b2 - b1):
                w_sb[:, g1:] -= err_g @ u_sb[g0:g1, g1:]

        if b2 < in_f:
            w[:, b2:] -= err_sb @ u_chol[b1:b2, b2:]

    w_deq = d_out.float().repeat_interleave(group_size, dim=1) * q_out.float()
    dw = w - w_deq
    num_e = float(((dw @ h_f32) * dw).sum().item())
    den_e = float(((w @ h_f32) * w).sum().clamp(min=1e-12).item())
    del h_f32, u_chol, w
    return q_out, d_out, math.sqrt(max(0.0, num_e / den_e))


# ==============================================================================
# 2. Second-Order Curvature-Preconditioned Shadowless PRIMAL Ternary Module
# ==============================================================================
class _SecondOrderTernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q_int8: torch.Tensor,
        d_eff: torch.Tensor,
        bias: Optional[torch.Tensor],
        vote_buf: torch.Tensor,
        group_size: int,
        vote_gain: float,
        vote_clip: float,
    ) -> torch.Tensor:
        out_f, in_f = q_int8.shape
        q_f = q_int8.to(x.dtype).view(out_f, in_f // group_size, group_size)
        w = (q_f * d_eff.to(x.dtype).unsqueeze(-1)).reshape(out_f, in_f)
        ctx.save_for_backward(x, q_int8, d_eff)
        ctx.has_bias = bias is not None
        ctx.vote_buf = vote_buf
        ctx.group_size = group_size
        ctx.vote_gain = vote_gain
        ctx.vote_clip = vote_clip
        return F.linear(x, w, bias.to(x.dtype) if bias is not None else None)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, q_int8, d_eff = ctx.saved_tensors
        out_f, in_f = q_int8.shape
        G = ctx.group_size
        go_2d = grad_out.reshape(-1, out_f)
        x_2d = x.reshape(-1, in_f)

        q_f = q_int8.to(grad_out.dtype).view(out_f, in_f // G, G)
        w = (q_f * d_eff.to(grad_out.dtype).unsqueeze(-1)).reshape(out_f, in_f)
        grad_x = (go_2d @ w).reshape_as(x)
        del w

        grad_w = torch.matmul(go_2d.float().t(), x_2d.float())
        grad_w_grp = grad_w.view(out_f, in_f // G, G)
        grad_d = (grad_w_grp * q_int8.float().view(out_f, in_f // G, G)).sum(dim=-1)

        if ctx.vote_gain > 0.0:
            # Diagonal input curvature h_c = E[x_c^2] + 0.02 * mean(h_c):
            # Preconditioning by 1/sqrt(h_c) + sink-channel mask (h_c < 8*mean) prevents
            # Qwen2/DeepSeek super-weight / activation-sink channels from ever hijacking trit flips!
            hc = x_2d.float().pow(2).mean(dim=0, keepdim=True)
            hbar = hc.mean().clamp(min=1e-12)
            hc_reg = hc + 0.02 * hbar
            newton_dir = -(grad_w / hc_reg.sqrt())
            rms = newton_dir.pow(2).mean().sqrt().clamp(min=1e-12)
            sink_ok = (hc < 8.0 * hbar).float()
            raw_v = (ctx.vote_gain * (newton_dir / rms) * sink_ok).clamp(-ctx.vote_clip, ctx.vote_clip)
            vi = torch.floor(raw_v + torch.rand_like(raw_v)).to(torch.int16)
            vb_new = (ctx.vote_buf.to(torch.int16) + vi).clamp(-127, 127).to(torch.int8)
            ctx.vote_buf.copy_(vb_new)
            del newton_dir, sink_ok

        del grad_w, grad_w_grp
        grad_b = go_2d.float().sum(dim=0) if ctx.has_bias else None
        return grad_x, None, grad_d, grad_b, None, None, None, None


class PrimalTernaryLinear(nn.Module):
    def __init__(
        self,
        q_int8: torch.Tensor,
        d_fp16: torch.Tensor,
        bias: Optional[torch.Tensor],
        group_size: int = 64,
        allow_trit_flips: bool = True,
    ):
        super().__init__()
        self.out_features, self.in_features = q_int8.shape
        self.group_size = group_size
        self.allow_trit_flips = allow_trit_flips
        self.register_buffer("q", q_int8.to(torch.int8).contiguous())
        self.register_buffer("vote_buf", torch.zeros_like(self.q, dtype=torch.int8))
        self.register_buffer("d_base", d_fp16.float().clamp(min=1e-7).contiguous())
        self.log_mul = nn.Parameter(torch.zeros_like(self.d_base))
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().float().clone())
        else:
            self.register_parameter("bias", None)
        self.vote_gain: float = 0.0
        self.vote_clip: float = 1.5

    def cur_d(self) -> torch.Tensor:
        return self.d_base * torch.exp(self.log_mul.clamp(-2.5, 2.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vg = self.vote_gain if self.allow_trit_flips else 0.0
        return _SecondOrderTernarySTE.apply(
            x, self.q, self.cur_d(), self.bias, self.vote_buf, self.group_size, vg, self.vote_clip
        )

    @torch.no_grad()
    def apply_ternary_step_capped(self, threshold: int, max_flips: int) -> int:
        if not self.allow_trit_flips or max_flips <= 0:
            return 0
        v = self.vote_buf.to(torch.int16)
        q_old = self.q.to(torch.int16)
        cand = ((v >= threshold) & (q_old < 1)) | ((v <= -threshold) & (q_old > -1))
        n_cand = int(cand.sum().item())
        if n_cand == 0:
            return 0
        if n_cand > max_flips:
            scores = torch.where(cand, v.abs(), torch.zeros_like(v)).reshape(-1)
            _, top_idx = torch.topk(scores, k=max_flips)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[top_idx] = True
            cand = mask.view_as(v)
        step = torch.where(cand, torch.sign(v), torch.zeros_like(v))
        q_new = (q_old + step).clamp(-1, 1)
        self.q.copy_(q_new.to(torch.int8))
        v_rem = torch.where(cand, torch.zeros_like(v), v)
        at_rail = ((q_new == 1) & (v_rem > 0)) | ((q_new == -1) & (v_rem < 0))
        v_rem = torch.where(at_rail, v_rem // 2, v_rem)
        self.vote_buf.copy_(v_rem.to(torch.int8))
        return int(cand.sum().item())

    @torch.no_grad()
    def export_q_and_d_fp16(self) -> Tuple[torch.Tensor, torch.Tensor]:
        d_f16 = self.cur_d().detach().clamp(min=0.0).to(torch.float16)
        return self.q.detach().cpu().clone(), d_f16.cpu().clone()

    @torch.no_grad()
    def to_dequantized_bf16(self) -> torch.Tensor:
        d_f16 = self.cur_d().detach().clamp(min=0.0).to(torch.float16).float()
        q_f = self.q.float().view(self.out_features, self.in_features // self.group_size, self.group_size)
        return (q_f * d_f16.unsqueeze(-1)).reshape(self.out_features, self.in_features).to(torch.bfloat16)


# ==============================================================================
# 3. Teacher Loader, Batched Asymmetric GPTQ Builder & PRIMAL Distillation
# ==============================================================================
def load_teacher_qwen2(device: str = "cuda") -> AutoModelForCausalLM:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    vf = vlt3.read_vlt3(Q8_VLT3, check_crc=False)
    sd_v = vlt3.vlt3_to_state_dict(vf)
    qcfg = Qwen2Config(
        vocab_size=151936, hidden_size=1536, intermediate_size=8960,
        num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2,
        rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
        bos_token_id=151646, eos_token_id=151643, torch_dtype=torch.bfloat16,
    )
    m = Qwen2ForCausalLM(qcfg).to(device=device, dtype=torch.bfloat16).eval()
    pmap = dict(m.named_parameters())
    with torch.no_grad():
        for k, v in sd_v.items():
            if k.startswith("seg_scorer."):
                continue
            hk = k if k == "lm_head.weight" else f"model.{k}"
            pmap[hk].data.copy_(v.to(device=device, dtype=torch.bfloat16))
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
def materialize_student_from_gptq_ckpt(
    teacher: AutoModelForCausalLM,
    ternary_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    group_size: int,
    ternary_attn_qo: bool,
    p4_embed_head: bool = True,
    device: str = "cuda",
) -> AutoModelForCausalLM:
    student = copy.deepcopy(teacher).to(device=device, dtype=torch.bfloat16).eval()
    if p4_embed_head:
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
        if not ternary_attn_qo:
            for qo_nm in ("q_proj", "o_proj"):
                wm = getattr(ls.self_attn, qo_nm).weight
                arr = wm.detach().float().cpu().contiguous().numpy()
                dt_fn = (vlt3.quantize_q8_0, vlt3.dequantize_q8_0) if li in (0, 1, 26, 27) else (vlt3.quantize_p4_0, vlt3.dequantize_p4_0)
                pb = dt_fn[0](arr)
                wm.data.copy_(torch.from_numpy(dt_fn[1](pb, arr.size).reshape(arr.shape)).to(device=device, dtype=torch.bfloat16))
    for key, (q_c, d_c) in ternary_data.items():
        parts = key.split(".")
        li = int(parts[1])
        mod = getattr(getattr(student.model.layers[li], parts[2]), parts[3])
        w_deq = (d_c.to(device).float().repeat_interleave(group_size, dim=1) * q_c.to(device).float()).to(torch.bfloat16)
        mod.weight.data.copy_(w_deq)
    return student


@torch.no_grad()
def build_gptq_ternary_student(
    teacher: AutoModelForCausalLM,
    calib_batches: List[torch.Tensor],
    group_size: int = 64,
    ternary_attn_qo: bool = True,
    p4_embed_head: bool = True,
    cache_pt: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[AutoModelForCausalLM, Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
    if cache_pt and os.path.exists(cache_pt):
        print(f"  [GPTQ Cache Hit] Loading cached Asymmetric GPTQ init -> {cache_pt}", flush=True)
        ck = torch.load(cache_pt, map_location="cpu", weights_only=False)
        stu = materialize_student_from_gptq_ckpt(teacher, ck["ternary_data"], group_size, ternary_attn_qo, p4_embed_head, device)
        return stu, ck["ternary_data"], ck["meta"]

    t0 = time.time()
    student = copy.deepcopy(teacher).to(device=device, dtype=torch.bfloat16).eval()

    if p4_embed_head:
        for mod_w in (student.model.embed_tokens.weight, student.lm_head.weight):
            arr = mod_w.detach().float().cpu().contiguous().numpy()
            pb = vlt3.quantize_p4_0(arr)
            deq = torch.from_numpy(vlt3.dequantize_p4_0(pb, arr.size).reshape(arr.shape)).to(device=device, dtype=torch.bfloat16)
            mod_w.data.copy_(deq)

    ternary_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    layer_errs: List[float] = []

    n_tokens_cal = float(sum(b.numel() for b in calib_batches))
    seq_len = calib_batches[0].shape[-1]
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    rot_emb = student.model.rotary_emb(torch.zeros((1, seq_len, 1536), device=device, dtype=torch.bfloat16), pos_ids)

    ht_cpu = [teacher.model.embed_tokens(b.to(device)).cpu() for b in calib_batches]
    hs_cpu = [student.model.embed_tokens(b.to(device)).cpu() for b in calib_batches]

    for li in range(28):
        lt = teacher.model.layers[li]
        ls = student.model.layers[li]

        for kv_nm in ("k_proj", "v_proj"):
            wm = getattr(ls.self_attn, kv_nm).weight
            arr = wm.detach().float().cpu().contiguous().numpy()
            qb = vlt3.quantize_q8_0(arr)
            wm.data.copy_(torch.from_numpy(vlt3.dequantize_q8_0(qb, arr.size).reshape(arr.shape)).to(device=device, dtype=torch.bfloat16))

        if not ternary_attn_qo:
            for qo_nm in ("q_proj", "o_proj"):
                wm = getattr(ls.self_attn, qo_nm).weight
                arr = wm.detach().float().cpu().contiguous().numpy()
                dt_fn = (vlt3.quantize_q8_0, vlt3.dequantize_q8_0) if li in (0, 1, 26, 27) else (vlt3.quantize_p4_0, vlt3.dequantize_p4_0)
                pb = dt_fn[0](arr)
                wm.data.copy_(torch.from_numpy(dt_fn[1](pb, arr.size).reshape(arr.shape)).to(device=device, dtype=torch.bfloat16))
        else:
            h_q = torch.zeros((1536, 1536), dtype=torch.float64, device=device)
            g_q = torch.zeros((1536, 1536), dtype=torch.float64, device=device)
            for htc, hsc in zip(ht_cpu, hs_cpu):
                xt = lt.input_layernorm(htc.to(device)).reshape(-1, 1536).double()
                xs = ls.input_layernorm(hsc.to(device)).reshape(-1, 1536).double()
                h_q.addmm_(xs.t(), xs)
                g_q.addmm_(xt.t(), xs)
            h_q /= n_tokens_cal
            g_q /= n_tokens_cal

            q_q, d_q, _ = gptq_quantize_ternary_asymmetric(
                lt.self_attn.q_proj.weight, h_q, g_q, group_size=group_size, alpha_asym=0.65
            )
            ternary_data[f"layers.{li}.self_attn.q_proj"] = (q_q.cpu(), d_q.cpu())
            ls.self_attn.q_proj.weight.data.copy_(
                (d_q.float().repeat_interleave(group_size, dim=1) * q_q.float()).to(torch.bfloat16)
            )
            del h_q, g_q

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

            q_o, d_o, _ = gptq_quantize_ternary_asymmetric(
                lt.self_attn.o_proj.weight, h_o, g_o, group_size=group_size, alpha_asym=0.65
            )
            ternary_data[f"layers.{li}.self_attn.o_proj"] = (q_o.cpu(), d_o.cpu())
            ls.self_attn.o_proj.weight.data.copy_(
                (d_o.float().repeat_interleave(group_size, dim=1) * q_o.float()).to(torch.bfloat16)
            )
            del h_o, g_o

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
            qm, dm, _ = gptq_quantize_ternary_asymmetric(wt_m, h_m, g_m, group_size=group_size, alpha_asym=0.65)
            ternary_data[f"layers.{li}.mlp.{mlp_nm}"] = (qm.cpu(), dm.cpu())
            ws_m.data.copy_((dm.float().repeat_interleave(group_size, dim=1) * qm.float()).to(torch.bfloat16))
        del h_m, g_m

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

        qd, dd, ed = gptq_quantize_ternary_asymmetric(
            lt.mlp.down_proj.weight, h_dn, g_dn, group_size=group_size, alpha_asym=0.65
        )
        ternary_data[f"layers.{li}.mlp.down_proj"] = (qd.cpu(), dd.cpu())
        ls.mlp.down_proj.weight.data.copy_((dd.float().repeat_interleave(group_size, dim=1) * qd.float()).to(torch.bfloat16))
        del h_dn, g_dn
        layer_errs.append(ed)

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
            nz_pct = 100.0 * float((qd != 0).float().mean().item())
            print(
                f"  [Asym-GPTQ-T1.58 L{li:02d}/27] down_rel_err={ed:.4f} | hidden_cos={cos_sum/len(calib_batches):.4f} | "
                f"nonzero_trits={nz_pct:.1f}% ({time.time()-t0:.1f}s)",
                flush=True,
            )

    del ht_cpu, hs_cpu
    gc.collect()
    torch.cuda.empty_cache()

    trit_params = sum(q.numel() for q, _ in ternary_data.values())
    blk_bytes = 9 if group_size == 32 else 15
    ternary_bytes = (trit_params // group_size) * blk_bytes
    kv_bytes = (28 * 2 * 256 * 1536 // 32) * 34
    qo_bytes = 0 if ternary_attn_qo else (4 * 2 * 1536 * 1536 // 32) * 34 + (24 * 2 * 1536 * 1536 // 32) * 18
    emb_bytes = 2 * (151936 * 1536 // 32) * (18 if p4_embed_head else 34)
    misc_bytes = int(14.5 * 1024 * 1024)
    tot_mb = (ternary_bytes + kv_bytes + qo_bytes + emb_bytes + misc_bytes) / (1024 ** 2)
    eff_bpw_body = 8.0 * (ternary_bytes + kv_bytes + qo_bytes) / 1310.2e6

    meta = {
        "group_size": group_size,
        "ternary_attn_qo": ternary_attn_qo,
        "ternary_params": trit_params,
        "body_effective_bpw": round(eff_bpw_body, 3),
        "est_vlt3_mb": round(tot_mb, 2),
        "est_vlt3_gib": round(tot_mb / 1024.0, 3),
        "mean_down_rel_err": round(float(np.mean(layer_errs)), 4),
        "gptq_seconds": round(time.time() - t0, 1),
    }
    if cache_pt:
        torch.save({"ternary_data": ternary_data, "meta": meta}, cache_pt)
        print(f"  [Saved GPTQ Init Cache] -> {cache_pt}", flush=True)
    return student, ternary_data, meta


def run_primal_ternary_kl_distill(
    teacher: AutoModelForCausalLM,
    student: AutoModelForCausalLM,
    ternary_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    wt_chunks: torch.Tensor,
    wt_words: float,
    group_size: int = 64,
    steps: int = 100,
    grad_accum: int = 4,
    device: str = "cuda",
) -> Tuple[AutoModelForCausalLM, Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
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

    t_mods: Dict[str, PrimalTernaryLinear] = {}
    for key, (q_cpu, d_cpu) in ternary_data.items():
        parts = key.split(".")
        li = int(parts[1])
        sub_parent = getattr(student.model.layers[li], parts[2])
        orig_lin: nn.Linear = getattr(sub_parent, parts[3])
        # Lock discrete trits in L00..L03 (super-weight / early sink layers) while allowing all scales to adapt!
        ptl = PrimalTernaryLinear(
            q_cpu.to(device), d_cpu.to(device), orig_lin.bias, group_size=group_size, allow_trit_flips=(li >= 4)
        ).to(device)
        setattr(sub_parent, parts[3], ptl)
        t_mods[key] = ptl
        del orig_lin

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    _, best_ppl = eval_wt2_fast(student, wt_chunks[:30], wt_words * (30.0 / wt_chunks.shape[0]))
    best_state = {
        "q": {k: m.q.detach().cpu().clone() for k, m in t_mods.items()},
        "log_mul": {k: m.log_mul.detach().cpu().clone() for k, m in t_mods.items()},
        "aux": {n: p.detach().cpu().clone() for n, p in student.named_parameters() if p.requires_grad and not n.endswith("log_mul")},
    }

    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lm_params = [m.log_mul for m in t_mods.values()]
    aux_params = [p for n, p in student.named_parameters() if p.requires_grad and not n.endswith("log_mul")]
    base_lr_lm = 1.0e-2
    base_lr_aux = 1.0e-4
    opt = torch.optim.AdamW(
        [
            {"params": lm_params, "lr": base_lr_lm, "weight_decay": 0.0},
            {"params": aux_params, "lr": base_lr_aux, "weight_decay": 0.0},
        ]
    )

    # Deterministically shuffle stream sequences so every step sees balanced prose + <think> math/code!
    rng = torch.Generator().manual_seed(42)
    n_slots = len(batches) * 2
    perm = torch.randperm(n_slots, generator=rng).tolist()

    tot_flips = 0
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
        v_gain = (0.10 + 0.14 * cos_v) if step > warmup_steps else 0.0
        thresh = int(round(36 - 8 * cos_v))  # 28 -> 36
        max_flips_mat = max(8, int(round(6 + 10 * cos_v)))  # 16 -> 8 per matrix per step
        for m in t_mods.values():
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

        step_flips = (
            sum(m.apply_ternary_step_capped(thresh, max_flips_mat) for m in t_mods.values())
            if step > warmup_steps
            else 0
        )
        tot_flips += step_flips

        if step == 1 or step % 20 == 0 or step == steps:
            student.gradient_checkpointing_disable()
            for m in t_mods.values():
                m.vote_gain = 0.0
            _, wrd_p = eval_wt2_fast(student, wt_chunks[:30], wt_words * (30.0 / wt_chunks.shape[0]))
            if wrd_p <= best_ppl:
                best_ppl = wrd_p
                best_state = {
                    "q": {k: m.q.detach().cpu().clone() for k, m in t_mods.items()},
                    "log_mul": {k: m.log_mul.detach().cpu().clone() for k, m in t_mods.items()},
                    "aux": {n: p.detach().cpu().clone() for n, p in student.named_parameters() if p.requires_grad and not n.endswith("log_mul")},
                }
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            vram_g = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(
                f"  [PRIMAL-T1.58 Step {step:3d}/{steps}] kl={loss_sum:.4f} | trit_flips={step_flips:,} | "
                f"WT2_30_WordPPL={wrd_p:.2f} (best={best_ppl:.2f}) | VRAM={vram_g:.2f} GiB ({time.time()-t0:.1f}s)",
                flush=True,
            )
            logs.append({"step": step, "kl": round(loss_sum, 4), "trit_flips": step_flips, "wt2_fast_word_ppl": round(wrd_p, 2)})

    student.gradient_checkpointing_disable()
    pmap_all = dict(student.named_parameters())
    with torch.no_grad():
        for k, m in t_mods.items():
            m.q.copy_(best_state["q"][k].to(device))
            m.log_mul.data.copy_(best_state["log_mul"][k].to(device))
        for n, pv in best_state["aux"].items():
            pmap_all[n].data.copy_(pv.to(device=device, dtype=pmap_all[n].dtype))

    updated_ternary: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for key, ptl in t_mods.items():
        parts = key.split(".")
        li = int(parts[1])
        sub_parent = getattr(student.model.layers[li], parts[2])
        lin = nn.Linear(ptl.in_features, ptl.out_features, bias=(ptl.bias is not None), device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.data.copy_(ptl.to_dequantized_bf16())
            if ptl.bias is not None:
                lin.bias.data.copy_(ptl.bias.detach().to(torch.bfloat16))
            updated_ternary[key] = ptl.export_q_and_d_fp16()
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
    return student, updated_ternary, norm_weights, bias_weights, {
        "steps": steps,
        "total_trit_flips": tot_flips,
        "best_val_wt2_30_word_ppl": round(best_ppl, 2),
        "peak_distill_vram_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
        "distill_seconds": round(time.time() - t0, 1),
        "logs": logs,
    }


# ==============================================================================
# 4. Downstream Benchmark Suite (`WT2`, `GSM8K-200`, `ARC-E`, `ARC-C`, `SciQ`)
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
    """Memory-lean multiple-choice log-likelihood scorer (batch_size=8, per-row log_softmax < 150 MiB!)."""
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
def evaluate_downstream(m_q: AutoModelForCausalLM, tok, wt_chunks: torch.Tensor, wt_words: float, tag: str) -> Dict[str, Any]:
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
            gtxt = tok.decode(out[bi, il:], skip_special_tokens=True)
            if _extract_boxed(gtxt) == _extract_gold_gsm(r["answer"]):
                hits += 1
        del out, enc
    gsm_acc = round(100.0 * hits / len(gsm_ds), 2)

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
        "arc_easy_full": arc_e,
        "arc_challenge_full": arc_c,
        "sciq_rc_full": sciq_rc,
        "sciq_cb_full": sciq_cb,
        "mc4_mean": round((arc_e + arc_c + sciq_rc + sciq_cb) / 4.0, 2),
        "taiwan_nothink_sample": tw_out[:180],
    }
    print(
        f"[{tag}] WT2_WordPPL={wrd_ppl:.2f} | GSM8K-200={gsm_acc:.2f}% | ARC-E={arc_e:.2f}% | "
        f"ARC-C={arc_c:.2f}% | SciQ-RC={sciq_rc:.2f}% | SciQ-CB={sciq_cb:.2f}% | MC4={res['mc4_mean']:.2f}%",
        flush=True,
    )
    print(f"  [{tag} Taiwan]: {tw_out[:160]}", flush=True)
    return res


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

    stream_d = torch.load(STREAM_PT, map_location="cpu", weights_only=False)
    batches = stream_d["batches"]
    raw_cal = [batches[i][0][b, :1024] for i in range(24) for b in range(2)]  # 48 x 1024 = 49,152 tokens
    calib_batches = [torch.stack(raw_cal[k : k + 4], dim=0) for k in range(0, 48, 4)]

    print("=== Loading BF16/Q8_0 Teacher (`VAULT-v3-DeepSeek-R1-1.5B-PRIMAL`) ===", flush=True)
    teacher = load_teacher_qwen2("cuda")

    print("\n=== [EXP 0] Naive Round-to-Nearest 1.58-Bit Control (No GPTQ, No PRIMAL) ===", flush=True)
    m_rtn = copy.deepcopy(teacher)
    with torch.no_grad():
        for li in range(28):
            ly = m_rtn.model.layers[li]
            for mod in (ly.self_attn.q_proj, ly.self_attn.o_proj, ly.mlp.gate_proj, ly.mlp.up_proj, ly.mlp.down_proj):
                w = mod.weight.float().view(mod.out_features, -1, 64)
                d = optimal_ternary_scales_lloyd(w.reshape(-1, 64), n_iters=4).view(mod.out_features, -1, 1)
                q = torch.clamp(torch.round(w / d), -1.0, 1.0)
                mod.weight.data.copy_((d * q).reshape_as(mod.weight).to(torch.bfloat16))
    rtn_sub, rtn_wrd = eval_wt2_fast(m_rtn, wt_chunks, wt_words)
    print(f"[RTN-T1.58 Control] WT2_SubPPL={rtn_sub:.2f} | WT2_WordPPL={rtn_wrd:.2f} (Naive Ternary Collapse!)", flush=True)
    del m_rtn
    gc.collect()
    torch.cuda.empty_cache()

    results: Dict[str, Any] = {
        "naive_rtn_1_58_g64": {"wt2_subtoken_ppl": round(rtn_sub, 2), "wt2_word_ppl": round(rtn_wrd, 2)},
    }

    # 1. Variant A: MLP-1.58-Bit (`G=32`, 88.2% of body in `{-1,0,+1}` + `P4_0` Attention, `~677 MB` total!)
    print("\n=== [EXP 1] MLP-1.58-Bit (`G=32`) + P4_0 Attention: Asymmetric Sub-Hessian GPTQ -> Second-Order PRIMAL KL-Distillation ===", flush=True)
    stu_hyb, td_hyb, meta_hyb = build_gptq_ternary_student(
        teacher,
        calib_batches,
        group_size=32,
        ternary_attn_qo=False,
        p4_embed_head=True,
        cache_pt=os.path.join(OUT_DIR, "gptq_init_hyb_g32.pt"),
    )
    hyb_gptq_sub, hyb_gptq_wrd = eval_wt2_fast(stu_hyb, wt_chunks, wt_words)
    print(
        f"[MLP-1.58 Asym-GPTQ Before PRIMAL] Est_Size={meta_hyb['est_vlt3_mb']:.1f} MB ({meta_hyb['body_effective_bpw']} bpw body) | "
        f"WT2_WordPPL={hyb_gptq_wrd:.2f}",
        flush=True,
    )

    stu_hyb, td_hyb, nw_hyb, bw_hyb, kl_hyb = run_primal_ternary_kl_distill(
        teacher, stu_hyb, td_hyb, batches, wt_chunks, wt_words, group_size=32, steps=100, grad_accum=4
    )
    # Save distilled checkpoint BEFORE downstream eval so disk checkpoint is locked in immediately!
    torch.save(
        {"ternary_data": td_hyb, "norm_weights": nw_hyb, "bias_weights": bw_hyb, "meta": meta_hyb},
        os.path.join(OUT_DIR, "mlp_1_58_g32.pt"),
    )
    # Park Teacher on CPU during downstream generation/MC eval to free 3.56 GiB VRAM!
    teacher.to("cpu")
    torch.cuda.empty_cache()
    eval_hyb = evaluate_downstream(stu_hyb, tok, wt_chunks, wt_words, "MLP_1.58_HYBRID_G32")
    results["mlp_1_58_hybrid_g32"] = {
        "meta": meta_hyb,
        "gptq_only_wt2_subtoken_ppl": round(hyb_gptq_sub, 3),
        "gptq_only_wt2_word_ppl": round(hyb_gptq_wrd, 2),
        "primal_kl_distill": kl_hyb,
        "downstream": eval_hyb,
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    del stu_hyb, td_hyb, nw_hyb, bw_hyb
    gc.collect()
    torch.cuda.empty_cache()

    # Bring Teacher back to GPU for EXP 2
    teacher.to("cuda")

    # 2. Variant B: Full-Body Pure 1.58-Bit (`G=64` = `1.875 bpw` w/ scale across ALL `MLP + Attn Q/O`, `~595 MB` total!)
    print("\n=== [EXP 2] Full-Body Pure 1.58-Bit (`G=64`, 1.875 bpw w/ scale, ~595 MB): Asymmetric Sub-Hessian GPTQ -> Second-Order PRIMAL KL-Distillation ===", flush=True)
    stu_pure, td_pure, meta_pure = build_gptq_ternary_student(
        teacher,
        calib_batches,
        group_size=64,
        ternary_attn_qo=True,
        p4_embed_head=True,
        cache_pt=os.path.join(OUT_DIR, "gptq_init_pure_g64.pt"),
    )
    pure_gptq_sub, pure_gptq_wrd = eval_wt2_fast(stu_pure, wt_chunks, wt_words)
    print(
        f"[Full-Body 1.58 Asym-GPTQ Before PRIMAL] Est_Size={meta_pure['est_vlt3_mb']:.1f} MB ({meta_pure['body_effective_bpw']} bpw body) | "
        f"WT2_WordPPL={pure_gptq_wrd:.2f}",
        flush=True,
    )

    stu_pure, td_pure, nw_pure, bw_pure, kl_pure = run_primal_ternary_kl_distill(
        teacher, stu_pure, td_pure, batches, wt_chunks, wt_words, group_size=64, steps=110, grad_accum=4
    )
    torch.save(
        {"ternary_data": td_pure, "norm_weights": nw_pure, "bias_weights": bw_pure, "meta": meta_pure},
        os.path.join(OUT_DIR, "full_body_1_58_g64.pt"),
    )
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    eval_pure = evaluate_downstream(stu_pure, tok, wt_chunks, wt_words, "FULL_BODY_1.58_G64")
    results["full_body_1_58_g64"] = {
        "meta": meta_pure,
        "gptq_only_wt2_subtoken_ppl": round(pure_gptq_sub, 3),
        "gptq_only_wt2_word_ppl": round(pure_gptq_wrd, 2),
        "primal_kl_distill": kl_pure,
        "downstream": eval_pure,
    }
    del stu_pure, td_pure, nw_pure, bw_pure
    gc.collect()
    torch.cuda.empty_cache()

    with open(REPORT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[TERNARY 1.58-BIT STUDY COMPLETE] Saved -> {REPORT_JSON}", flush=True)


if __name__ == "__main__":
    main()
