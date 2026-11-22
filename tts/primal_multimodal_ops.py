# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
Universal Multi-Modal `VAULT + PRIMAL` Discrete Quantization & Training Engine
for `nn.Linear`, `nn.Conv1d`, and `nn.ConvTranspose1d` across TTS, CNN, Flow, and Diffusion (`DiT`) backbones.

Supports:
  - `Q4_0`    : Standard 4-bit linear block quantization (18 bytes / 32 weights = 4.50 bpw)
  - `V3_P4_0` : VAULT Prime-Harmonic 16-entry non-linear LUT (18 bytes / 32 weights = 4.50 bpw)
  - `V3_Q2_0` : VAULT Coalesced Asymmetric Outlier 2-bit `{-2, -1, 0, +1}` (18 bytes / 64 weights = 2.25 bpw)
  - `PrimalUniversalModule`: Shadowless discrete index + signed multiplicative `FP16` scale trainer using
    Adafactor-Kronecker Gauss-Newton preconditioning, Proximal Voronoi Deadzone gating, and `0.95` leaky `int8` votes.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import jiwer
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools import vlt3

P4_LUT_NP = vlt3.P4_LUT_F32.copy()  # [-127, -101, -80, -62, -46, -33, -21, -10, 0, 10, 21, 33, 46, 62, 80, 101]
P4_STEP_NP = np.zeros(16, dtype=np.float32)
for _k in range(16):
    _lo = max(0, _k - 1)
    _hi = min(15, _k + 1)
    P4_STEP_NP[_k] = (P4_LUT_NP[_hi] - P4_LUT_NP[_lo]) / float(_hi - _lo)


def _flat_2d_padded(w: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int, int, int]:
    rows = w.shape[0]
    cols = int(w.numel() // rows)
    pcols = ((cols + block_size - 1) // block_size) * block_size
    w2 = w.detach().float().reshape(rows, cols)
    if pcols > cols:
        w2 = F.pad(w2, (0, pcols - cols), value=0.0)
    return w2, rows, cols, pcols


@torch.no_grad()
def encode_q4_0_tensors(w: torch.Tensor, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor, int]:
    w2, rows, cols, pcols = _flat_2d_padded(w, 32)
    xb = w2.to(device).reshape(-1, 32)
    m = xb.shape[0]
    ratios = [-1.0 / 8.0, -0.98 / 8.0, -0.96 / 8.0, -0.94 / 8.0, -0.92 / 8.0, -0.89 / 8.0, 1.0 / 7.5, 0.96 / 7.5]
    mx = xb.gather(1, xb.abs().argmax(dim=1, keepdim=True)).squeeze(1)
    best_err = torch.full((m,), float("inf"), device=device, dtype=torch.float32)
    best_d = torch.zeros(m, device=device, dtype=torch.float16)
    best_q = torch.full((m, 32), 8, device=device, dtype=torch.int8)
    nz = mx != 0.0
    if nz.any():
        for r in ratios:
            d_f16 = (mx * r).to(torch.float16)
            d_f32 = d_f16.float()
            inv = torch.where(d_f32 != 0.0, 1.0 / torch.where(d_f32 != 0.0, d_f32, torch.ones_like(d_f32)), torch.zeros_like(d_f32))
            qi = torch.clamp(torch.round(xb * inv.unsqueeze(1) + 8.0), 0.0, 15.0).to(torch.int8)
            rec = (qi.float() - 8.0) * d_f32.unsqueeze(1)
            err = (rec - xb).pow(2).sum(dim=1)
            imp = nz & (err < best_err)
            best_err = torch.where(imp, err, best_err)
            best_d = torch.where(imp, d_f16, best_d)
            best_q = torch.where(imp.unsqueeze(1), qi, best_q)
    return best_q.view(rows, pcols).cpu(), best_d.view(rows, pcols // 32).cpu(), pcols


@torch.no_grad()
def encode_p4_0_tensors(w: torch.Tensor, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor, int]:
    w2, rows, cols, pcols = _flat_2d_padded(w, 32)
    xb = w2.to(device).reshape(-1, 32)
    m = xb.shape[0]
    lut_t = torch.tensor(P4_LUT_NP, device=device, dtype=torch.float32)
    mids_t = 0.5 * (lut_t[:-1] + lut_t[1:])
    max_neg, max_pos = 127.0, 101.0
    ratios = [
        -1.00 / max_neg, -0.96 / max_neg, -0.92 / max_neg, -0.88 / max_neg, -0.84 / max_neg, -0.80 / max_neg,
        1.00 / max_pos, 0.96 / max_pos, 0.92 / max_pos, 0.88 / max_pos,
    ]
    mx = xb.gather(1, xb.abs().argmax(dim=1, keepdim=True)).squeeze(1)
    best_err = torch.full((m,), float("inf"), device=device, dtype=torch.float32)
    best_d = torch.zeros(m, device=device, dtype=torch.float16)
    best_q = torch.full((m, 32), 8, device=device, dtype=torch.int8)
    nz = mx != 0.0
    if nz.any():
        for r in ratios:
            d_f16 = (mx * r).to(torch.float16)
            # 2 Alternating Lloyd-Max scale polish steps on the Prime-Harmonic grid!
            for _ in range(2):
                d_f32 = d_f16.float()
                inv = torch.where(d_f32 != 0.0, 1.0 / torch.where(d_f32 != 0.0, d_f32, torch.ones_like(d_f32)), torch.zeros_like(d_f32))
                qi = torch.searchsorted(mids_t, xb * inv.unsqueeze(1))
                lv = lut_t[qi]
                den = (lv * lv).sum(dim=1)
                num = (xb * lv).sum(dim=1)
                d_f16 = torch.where(den > 0.0, num / torch.where(den > 0.0, den, torch.ones_like(den)), d_f32).to(torch.float16)
            d_f32 = d_f16.float()
            inv = torch.where(d_f32 != 0.0, 1.0 / torch.where(d_f32 != 0.0, d_f32, torch.ones_like(d_f32)), torch.zeros_like(d_f32))
            qi = torch.searchsorted(mids_t, xb * inv.unsqueeze(1)).to(torch.int8)
            rec = lut_t[qi.long()] * d_f32.unsqueeze(1)
            err = (rec - xb).pow(2).sum(dim=1)
            imp = nz & (err < best_err)
            best_err = torch.where(imp, err, best_err)
            best_d = torch.where(imp, d_f16, best_d)
            best_q = torch.where(imp.unsqueeze(1), qi, best_q)
    return best_q.view(rows, pcols).cpu(), best_d.view(rows, pcols // 32).cpu(), pcols


@torch.no_grad()
def encode_q2_0_tensors(w: torch.Tensor, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor, int]:
    w2, rows, cols, pcols = _flat_2d_padded(w, 64)
    xb = w2.to(device).reshape(-1, 64)
    m = xb.shape[0]
    ratios = [0.40, 0.50, 0.60, 0.72, 0.88, -0.40, -0.50, -0.60, -0.72, -0.88]
    amax = xb.abs().amax(dim=1)
    best_err = torch.full((m,), float("inf"), device=device, dtype=torch.float32)
    best_d = torch.zeros(m, device=device, dtype=torch.float16)
    best_q = torch.zeros((m, 64), device=device, dtype=torch.int8)
    nz = amax > 0.0
    if nz.any():
        for r in ratios:
            d_f16 = (amax * r).to(torch.float16)
            for _ in range(3):
                d_f32 = d_f16.float()
                inv = torch.where(d_f32 != 0.0, 1.0 / torch.where(d_f32 != 0.0, d_f32, torch.ones_like(d_f32)), torch.zeros_like(d_f32))
                qf = torch.clamp(torch.round(xb * inv.unsqueeze(1)), -2.0, 1.0)
                den = (qf * qf).sum(dim=1)
                num = (xb * qf).sum(dim=1)
                d_f16 = torch.where(den > 0.0, num / torch.where(den > 0.0, den, torch.ones_like(den)), torch.zeros_like(den)).to(torch.float16)
            d_f32 = d_f16.float()
            inv = torch.where(d_f32 != 0.0, 1.0 / torch.where(d_f32 != 0.0, d_f32, torch.ones_like(d_f32)), torch.zeros_like(d_f32))
            qi = torch.clamp(torch.round(xb * inv.unsqueeze(1)), -2.0, 1.0).to(torch.int8)
            err = (qi.float() * d_f32.unsqueeze(1) - xb).pow(2).sum(dim=1)
            imp = nz & (err < best_err)
            best_err = torch.where(imp, err, best_err)
            best_d = torch.where(imp, d_f16, best_d)
            best_q = torch.where(imp.unsqueeze(1), qi, best_q)
    return best_q.view(rows, pcols).cpu(), best_d.view(rows, pcols // 64).cpu(), pcols


def should_quantize_layer(name: str, mod: nn.Module, skip_fn: Optional[Callable[[str, nn.Module], bool]] = None) -> bool:
    if not isinstance(mod, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
        return False
    if skip_fn is not None and skip_fn(name, mod):
        return False
    w = mod.weight
    rows = w.shape[0]
    cols = w.numel() // rows
    # Keep tiny depthwise convs (cols < 16) and tiny scalar/cepstrum/waveform heads (< 4096 params) in FP16
    if cols < 16 or w.numel() < 4096:
        return False
    return True


@torch.no_grad()
def apply_ptq_to_model(
    model: nn.Module,
    fmt: str,
    skip_fn: Optional[Callable[[str, nn.Module], bool]] = None,
    exclude_prefix: str = "",
    device: str = "cuda",
) -> Tuple[nn.Module, Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
    """
    Clones `model`, quantizes all eligible `Linear`, `Conv1d`, and `ConvTranspose1d` weights into
    `fmt` in `{"q4_0", "p4_0", "q2_0"}`, replaces `.weight` in-place with exact dequantized values,
    and returns `(q_model, q_state_dict, size_stats)` with exact serialized byte counts.
    """
    assert fmt in ("q4_0", "p4_0", "q2_0"), f"unknown fmt={fmt}"
    m_out = copy.deepcopy(model)
    orig_dev = next(m_out.parameters()).device
    orig_dtype = next(m_out.parameters()).dtype

    lut_dev = torch.tensor(P4_LUT_NP, device=device, dtype=torch.float32)
    q_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    quant_params = 0
    quant_bytes = 0
    fp16_params = 0
    fp16_bytes = 0

    quant_mod_names = set()
    for name, mod in m_out.named_modules():
        if exclude_prefix and name.startswith(exclude_prefix):
            continue
        if should_quantize_layer(name, mod, skip_fn=skip_fn):
            quant_mod_names.add(name)
            w = mod.weight.detach()
            rows = w.shape[0]
            cols = w.numel() // rows
            if fmt == "q4_0":
                q_c, d_c, pcols = encode_q4_0_tensors(w, device=device)
                rec = (q_c.to(device).float().view(rows, pcols // 32, 32) - 8.0) * d_c.to(device).float().unsqueeze(-1)
                n_blks = (rows * pcols) // 32
            elif fmt == "p4_0":
                q_c, d_c, pcols = encode_p4_0_tensors(w, device=device)
                rec = lut_dev[q_c.to(device).long()].view(rows, pcols // 32, 32) * d_c.to(device).float().unsqueeze(-1)
                n_blks = (rows * pcols) // 32
            else:
                q_c, d_c, pcols = encode_q2_0_tensors(w, device=device)
                rec = q_c.to(device).float().view(rows, pcols // 64, 64) * d_c.to(device).float().unsqueeze(-1)
                n_blks = (rows * pcols) // 64
            w_deq = rec.reshape(rows, pcols)[:, :cols].reshape_as(w).to(device=orig_dev, dtype=orig_dtype)
            mod.weight.data.copy_(w_deq)
            q_data[name] = (q_c, d_c)
            quant_params += w.numel()
            quant_bytes += n_blks * 18

    for pname, p in m_out.named_parameters():
        if exclude_prefix and pname.startswith(exclude_prefix):
            continue
        mname = pname.rsplit(".", 1)[0] if "." in pname else ""
        if pname.endswith(".weight") and mname in quant_mod_names:
            continue
        fp16_params += p.numel()
        fp16_bytes += p.numel() * 2  # stored as FP16 in compact checkpoint

    tot_params = quant_params + fp16_params
    tot_bytes = quant_bytes + fp16_bytes
    fp32_bytes = tot_params * 4
    stats = {
        "format": fmt,
        "active_params": tot_params,
        "quantized_params": quant_params,
        "quantized_share_pct": round(100.0 * quant_params / max(1, tot_params), 2),
        "fp32_mb": round(fp32_bytes / 1e6, 2),
        "packed_bytes": tot_bytes,
        "packed_mb": round(tot_bytes / 1e6, 2),
        "compression_vs_fp32_x": round(fp32_bytes / max(1, tot_bytes), 2),
        "effective_bpw": round(8.0 * tot_bytes / max(1, tot_params), 3),
    }
    return m_out, q_data, stats


# ==============================================================================
# Universal Shadowless PRIMAL Module (`Linear`, `Conv1d`, `ConvTranspose1d`)
# ==============================================================================
class _UniversalProximalWeightSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d_eff: torch.Tensor, mod_ref: "PrimalUniversalModule") -> torch.Tensor:
        ctx.save_for_backward(d_eff)
        ctx.mod_ref = mod_ref
        return mod_ref._reconstruct_weight(d_eff)

    @staticmethod
    def backward(ctx, grad_w_orig: torch.Tensor):
        (d_eff,) = ctx.saved_tensors
        mod: PrimalUniversalModule = ctx.mod_ref
        rows, cols, pcols, G = mod.rows, mod.cols, mod.pcols, mod.group_size

        gw = grad_w_orig.detach().float().reshape(rows, cols)
        if pcols > cols:
            gw = F.pad(gw, (0, pcols - cols), value=0.0)

        code_vals = mod._code_values_float()  # [rows, pcols]
        grad_d = (gw.view(rows, pcols // G, G) * code_vals.view(rows, pcols // G, G)).sum(dim=-1)

        with torch.no_grad():
            gw_un = gw * float(mod.grad_accum)
            gw_sq = gw_un.pow(2)
            r_cur = gw_sq.mean(dim=1)
            c_cur = gw_sq.mean(dim=0)
            if not mod.ema_initialized:
                mod.row_sq_ema.copy_(r_cur)
                mod.col_sq_ema.copy_(c_cur)
                mod.ema_initialized = True
            else:
                mod.row_sq_ema.mul_(0.95).add_(r_cur, alpha=0.05)
                mod.col_sq_ema.mul_(0.95).add_(c_cur, alpha=0.05)
            del gw_sq

            if mod.vote_gain > 0.0:
                gw_rms_cur = float(gw_un.pow(2).mean().sqrt().clamp(min=1e-30).item())
                mod.gw_rms_ema = gw_rms_cur if mod.gw_rms_ema <= 0.0 else (0.95 * mod.gw_rms_ema + 0.05 * gw_rms_cur)
                gw_norm = gw_un / max(mod.gw_rms_ema, 1e-30)

                r_tilde = (mod.row_sq_ema / mod.row_sq_ema.median().clamp(min=1e-30)) + 0.03
                c_tilde = (mod.col_sq_ema / mod.col_sq_ema.median().clamp(min=1e-30)) + 0.03

                d_full = d_eff.detach().float().repeat_interleave(G, dim=1)
                step_c = mod._code_step_sizes_float()  # 1.0 for Q2_0; local LUT delta for P4_0
                dw_signed = d_full * step_c
                dw_sign = torch.where(dw_signed < 0.0, -1.0, torch.where(dw_signed > 0.0, 1.0, 0.0))
                dw_abs = dw_signed.abs()
                dw_tilde = (dw_abs / dw_abs.median().clamp(min=1e-8)) + 0.03

                # Adafactor rank-1 Gauss-Newton code displacement (zero votes on padded tail cols!)
                u_step = -(dw_sign * gw_norm) / (dw_tilde * r_tilde.unsqueeze(1) * c_tilde.unsqueeze(0))
                if pcols > cols:
                    u_step[:, cols:] = 0.0

                cur_rms = float(u_step.pow(2).mean().sqrt().clamp(min=1e-12).item())
                mod.u_rms_ema = cur_rms if mod.u_rms_ema <= 0.0 else (0.95 * mod.u_rms_ema + 0.05 * cur_rms)
                u_hat = u_step / max(mod.u_rms_ema, 1e-12)
                u_prox = torch.sign(u_hat) * F.relu(u_hat.abs() - mod.tau_dead)
                raw_v = (mod.vote_gain * u_prox).clamp(-mod.vote_clip, mod.vote_clip)
                vi = torch.floor(raw_v + torch.rand_like(raw_v)).to(torch.int16)
                mod.vote_buf.copy_((mod.vote_buf.to(torch.int16) + vi).clamp(-127, 127).to(torch.int8))

        return grad_d, None


class PrimalUniversalModule(nn.Module):
    def __init__(
        self,
        orig_mod: nn.Module,
        q_init: torch.Tensor,
        d_init_fp16: torch.Tensor,
        fmt: str,
        grad_accum: int = 1,
    ):
        super().__init__()
        assert fmt in ("p4_0", "q2_0")
        self.fmt = fmt
        self.group_size = 32 if fmt == "p4_0" else 64
        self.q_min = 0 if fmt == "p4_0" else -2
        self.q_max = 15 if fmt == "p4_0" else 1
        self.grad_accum = grad_accum

        w = orig_mod.weight
        self.orig_shape = tuple(w.shape)
        self.compute_dtype = w.dtype
        self.rows = w.shape[0]
        self.cols = int(w.numel() // self.rows)
        self.pcols = q_init.shape[1]

        if isinstance(orig_mod, nn.Linear):
            self.op_kind = "linear"
        elif isinstance(orig_mod, nn.Conv1d):
            self.op_kind = "conv1d"
            self.stride, self.padding, self.dilation, self.groups = (
                orig_mod.stride, orig_mod.padding, orig_mod.dilation, orig_mod.groups
            )
        elif isinstance(orig_mod, nn.ConvTranspose1d):
            self.op_kind = "conv_transpose1d"
            self.stride, self.padding, self.output_padding, self.groups, self.dilation = (
                orig_mod.stride, orig_mod.padding, orig_mod.output_padding, orig_mod.groups, orig_mod.dilation
            )
        else:
            raise TypeError(type(orig_mod))

        self.register_buffer("q", q_init.to(torch.int8).contiguous())
        self.register_buffer("vote_buf", torch.zeros_like(self.q, dtype=torch.int8))
        if fmt == "p4_0":
            self.register_buffer("p4_lut", torch.tensor(P4_LUT_NP, dtype=torch.float32))
            self.register_buffer("p4_step", torch.tensor(P4_STEP_NP, dtype=torch.float32))

        d_f = d_init_fp16.float().contiguous()
        d_safe = torch.where(
            d_f > 0.0, d_f.clamp(min=1e-7), torch.where(d_f < 0.0, d_f.clamp(max=-1e-7), torch.zeros_like(d_f))
        )
        self.register_buffer("d_base", d_safe)
        self.register_buffer("row_sq_ema", torch.zeros(self.rows, dtype=torch.float32))
        self.register_buffer("col_sq_ema", torch.zeros(self.pcols, dtype=torch.float32))
        self.log_mul = nn.Parameter(torch.zeros_like(self.d_base))
        if orig_mod.bias is not None:
            self.bias = nn.Parameter(orig_mod.bias.detach().float().clone())
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

    def _code_values_float(self) -> torch.Tensor:
        if self.fmt == "p4_0":
            return self.p4_lut[self.q.long()]
        return self.q.float()

    def _code_step_sizes_float(self) -> torch.Tensor:
        if self.fmt == "p4_0":
            return self.p4_step[self.q.long()]
        return torch.ones_like(self.q, dtype=torch.float32)

    def _reconstruct_weight(self, d_eff: torch.Tensor) -> torch.Tensor:
        G = self.group_size
        cv = self._code_values_float().view(self.rows, self.pcols // G, G)
        w2 = (cv * d_eff.float().unsqueeze(-1)).reshape(self.rows, self.pcols)[:, : self.cols]
        return w2.reshape(self.orig_shape).to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _UniversalProximalWeightSTE.apply(self.cur_d(), self).to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        if self.op_kind == "linear":
            return F.linear(x, w, b)
        if self.op_kind == "conv1d":
            return F.conv1d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
        return F.conv_transpose1d(x, w, b, self.stride, self.padding, self.output_padding, self.groups, self.dilation)

    @torch.no_grad()
    def apply_discrete_step(self, threshold: int, max_flips: int) -> int:
        if max_flips <= 0:
            return 0
        v = self.vote_buf.to(torch.int16)
        if self.pcols > self.cols:
            v[:, self.cols :] = 0
        q_old = self.q.to(torch.int16)
        cand = ((v >= threshold) & (q_old < self.q_max)) | ((v <= -threshold) & (q_old > self.q_min))
        n_cand = int(cand.sum().item())
        if n_cand > 0:
            if n_cand > max_flips:
                scores = torch.where(cand, v.abs(), torch.zeros_like(v)).reshape(-1)
                _, top_idx = torch.topk(scores, k=max_flips)
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask[top_idx] = True
                cand = mask.view_as(v)
            step = torch.where(cand, torch.sign(v), torch.zeros_like(v))
            q_new = (q_old + step).clamp(self.q_min, self.q_max)
            self.q.copy_(q_new.to(torch.int8))
            v = torch.where(cand, torch.zeros_like(v), v)
        else:
            q_new = q_old

        at_rail = ((q_new == self.q_max) & (v > 0)) | ((q_new == self.q_min) & (v < 0))
        v = torch.where(at_rail, v // 2, v)
        vf = v.float() * self.vote_decay
        self.vote_buf.copy_(torch.floor(vf + torch.rand_like(vf)).clamp(-127.0, 127.0).to(torch.int8))
        return int(cand.sum().item()) if n_cand > 0 else 0

    @torch.no_grad()
    def to_plain_module(self) -> nn.Module:
        d_f16 = self.cur_d().detach().to(torch.float16)
        w_deq = self._reconstruct_weight(d_f16)
        dev = w_deq.device
        if self.op_kind == "linear":
            m = nn.Linear(self.orig_shape[1], self.orig_shape[0], bias=(self.bias is not None), device=dev, dtype=self.compute_dtype)
        elif self.op_kind == "conv1d":
            in_c = self.orig_shape[1] * self.groups
            m = nn.Conv1d(in_c, self.orig_shape[0], self.orig_shape[2], stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups, bias=(self.bias is not None), device=dev, dtype=self.compute_dtype)
        else:
            out_c = self.orig_shape[1] * self.groups
            m = nn.ConvTranspose1d(self.orig_shape[0], out_c, self.orig_shape[2], stride=self.stride, padding=self.padding,
                                   output_padding=self.output_padding, groups=self.groups, dilation=self.dilation,
                                   bias=(self.bias is not None), device=dev, dtype=self.compute_dtype)
        m.weight.data.copy_(w_deq)
        if self.bias is not None:
            m.bias.data.copy_(self.bias.detach().to(self.compute_dtype))
        return m


def _set_submodule(root: nn.Module, path: str, new_mod: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_mod
    else:
        setattr(parent, last, new_mod)


def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    cur = root
    for p in path.split("."):
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    return cur


def install_primal_modules(
    student: nn.Module,
    q_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    fmt: str,
    grad_accum: int = 1,
    device: str = "cuda",
) -> Dict[str, PrimalUniversalModule]:
    for p in student.parameters():
        p.requires_grad_(False)
    # Allow lightweight LayerNorm / RMSNorm / GRN affine gains & biases to co-adapt!
    for mod in student.modules():
        if isinstance(mod, (nn.LayerNorm, nn.GroupNorm)) or mod.__class__.__name__ in ("RMSNorm", "GRN"):
            for p in mod.parameters():
                p.requires_grad_(True)

    p_mods: Dict[str, PrimalUniversalModule] = {}
    for name, (q_c, d_c) in q_data.items():
        orig = _get_submodule(student, name)
        pm = PrimalUniversalModule(orig, q_c.to(device), d_c.to(device), fmt=fmt, grad_accum=grad_accum).to(device)
        _set_submodule(student, name, pm)
        p_mods[name] = pm
    return p_mods


def finalize_primal_modules(student: nn.Module, p_mods: Dict[str, PrimalUniversalModule]) -> None:
    for name, pm in p_mods.items():
        _set_submodule(student, name, pm.to_plain_module())
    student.eval()


# ==============================================================================
# Multi-Resolution STFT Loss & Objective TTS Speech Evaluation Suite
# ==============================================================================
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(256, 512, 1024, 2048)):
        super().__init__()
        self.fft_sizes = fft_sizes

    def forward(self, wav_pred: torch.Tensor, wav_tgt: torch.Tensor) -> torch.Tensor:
        L = min(wav_pred.shape[-1], wav_tgt.shape[-1])
        x = wav_pred[..., :L].float()
        y = wav_tgt[..., :L].float()
        loss = torch.zeros((), device=x.device, dtype=torch.float32)
        for n_fft in self.fft_sizes:
            if L < n_fft:
                continue
            hop = n_fft // 4
            win = torch.hann_window(n_fft, device=x.device, dtype=torch.float32)
            sx = torch.stft(x, n_fft, hop, window=win, return_complex=True).abs().clamp(min=1e-5)
            sy = torch.stft(y, n_fft, hop, window=win, return_complex=True).abs().clamp(min=1e-5)
            sc = torch.norm(sy - sx, p="fro") / (torch.norm(sy, p="fro") + 1e-7)
            mag = F.l1_loss(torch.log(sx), torch.log(sy))
            loss = loss + sc + mag
        return loss / float(len(self.fft_sizes))


def _norm_asr_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@torch.no_grad()
def evaluate_tts_waveforms(
    wavs_pred: List[np.ndarray],
    wavs_ref: List[np.ndarray],
    texts_gold: List[str],
    sr: int,
    whisper_proc,
    whisper_model,
    gen_seconds: float,
) -> Dict[str, Any]:
    """
    Computes:
      1) Whisper-small.en round-trip ASR Word Error Rate (`WER %`) and Character Error Rate (`CER %`)
      2) DTW-aligned 80-band Log-Mel Spectral Distortion (`MCD / Log-Mel dB` vs FP32 teacher)
      3) Multi-Resolution STFT Spectral Convergence (`MR-STFT SC`, aligned via DTW or min-len)
      4) Total synthesized audio duration & Real-Time Speedup (`x_realtime`)
    """
    hyps, refs = [], []
    mel_db_list = []
    sc_list = []
    tot_audio_sec = 0.0

    for wp, wr, txt in zip(wavs_pred, wavs_ref, texts_gold):
        wp = np.asarray(wp, dtype=np.float32).reshape(-1)
        wr = np.asarray(wr, dtype=np.float32).reshape(-1)
        tot_audio_sec += len(wp) / float(sr)

        # 1. Resample to 16 kHz for Whisper-small.en ASR evaluation
        wp_16k = librosa.resample(wp, orig_sr=sr, target_sr=16000) if sr != 16000 else wp
        inp = whisper_proc(wp_16k, sampling_rate=16000, return_tensors="pt").input_features.to(
            device=whisper_model.device, dtype=whisper_model.dtype
        )
        pred_ids = whisper_model.generate(inp, max_new_tokens=128)
        hyp = whisper_proc.batch_decode(pred_ids, skip_special_tokens=True)[0]
        hyps.append(_norm_asr_text(hyp))
        refs.append(_norm_asr_text(txt))

        # 2. 80-Band Log-Mel Spectral Distortion in dB (DTW-aligned across frames for free-running durations)
        mp = librosa.feature.melspectrogram(y=wp, sr=sr, n_fft=1024, hop_length=256, n_mels=80, power=1.0)
        mr = librosa.feature.melspectrogram(y=wr, sr=sr, n_fft=1024, hop_length=256, n_mels=80, power=1.0)
        lp_db = 20.0 * np.log10(np.maximum(mp, 1e-5))
        lr_db = 20.0 * np.log10(np.maximum(mr, 1e-5))
        if lp_db.shape[1] == lr_db.shape[1]:
            mel_db_list.append(float(np.mean(np.abs(lp_db - lr_db))))
        else:
            _, wp_idx = librosa.sequence.dtw(lp_db, lr_db, metric="euclidean")
            diffs = [np.mean(np.abs(lp_db[:, i] - lr_db[:, j])) for i, j in wp_idx]
            mel_db_list.append(float(np.mean(diffs)))

        # 3. STFT Spectral Convergence on overlap window
        L = min(len(wp), len(wr))
        if L >= 1024:
            tp = torch.from_numpy(wp[:L]).cuda()
            tr = torch.from_numpy(wr[:L]).cuda()
            win = torch.hann_window(1024, device="cuda")
            sp = torch.stft(tp, 1024, 256, window=win, return_complex=True).abs()
            sr_t = torch.stft(tr, 1024, 256, window=win, return_complex=True).abs()
            sc_list.append(float((torch.norm(sr_t - sp) / (torch.norm(sr_t) + 1e-7)).item()))

    wer_pct = round(100.0 * jiwer.wer(refs, hyps), 2)
    cer_pct = round(100.0 * jiwer.cer(refs, hyps), 2)
    return {
        "whisper_wer_pct": wer_pct,
        "whisper_cer_pct": cer_pct,
        "log_mel_distortion_db": round(float(np.mean(mel_db_list)), 3),
        "stft_spectral_convergence": round(float(np.mean(sc_list)) if sc_list else 0.0, 4),
        "audio_seconds": round(tot_audio_sec, 2),
        "wall_seconds": round(gen_seconds, 2),
        "x_realtime": round(tot_audio_sec / max(1e-4, gen_seconds), 2),
        "sample_transcript_0": hyps[0] if hyps else "",
    }
