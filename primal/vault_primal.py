# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
VAULT-PRIMAL: Native V3_Q8_0 Shadowless Discrete Training -> Freestanding VLT3 Runtime.

Trains a VAULT v3 GQA Transformer directly on exact V3_Q8_0 blocks (QK=32 int8 weights
+ fp16 block scales) with int8 consensus vote buffers (2.0625 B/param total training state,
zero shadow weights, zero persistent .grad buffers, zero AdamW m/v on 2D weights).

At export time, the trained (d, q) Q8_0 blocks are written directly into a CRC32-verified
VLT3 binary with embedded VTOK tokenizer — ZERO post-training quantization loss and 100%
native execution inside vault/runtime/vault3_host (freestanding C99 + AVX2/FMA).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .discrete_ops import _accumulate_votes, _stochastic_round_int

QK = 32  # Exact VLT3 / GGML Q8_0 block size

# Qwen2 pre-tokenizer regex matching vault/runtime/vault3.c:pretok_match
QWEN2_PRETOK_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


# ==============================================================================
# 1. Native V3_Q8_0 (QK=32) Shadowless Discrete Layers
# ==============================================================================

def _dequant_q8_0(q: torch.Tensor, d: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Materialize float/bf16 weight matrix from exact V3_Q8_0 (q: int8, d: fp16) blocks."""
    out_f, in_f = q.shape
    return (q.view(out_f, -1, QK).to(dtype) * d.to(dtype).unsqueeze(-1)).reshape(out_f, in_f)


class _Q8BlockLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        q: torch.Tensor,
        d: torch.Tensor,
        vote_buffer: torch.Tensor,
        vote_gain: float,
        vote_clip: float,
        scale_aware_votes: bool = False,
    ) -> torch.Tensor:
        w = _dequant_q8_0(q, d, x.dtype)
        b = bias.to(x.dtype) if bias is not None else None
        out = F.linear(x, w, b)
        ctx.save_for_backward(x)
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.q = q
        ctx.d = d
        ctx.vote_buffer = vote_buffer
        ctx.vote_gain = vote_gain
        ctx.vote_clip = vote_clip
        ctx.scale_aware_votes = scale_aware_votes
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        w = _dequant_q8_0(ctx.q, ctx.d, grad_output.dtype)
        grad_x = grad_output @ w
        del w

        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).float().sum(dim=0).to(ctx.bias_dtype)

        if ctx.vote_gain > 0.0:
            x_2d = x.reshape(-1, x.shape[-1])
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_w = go_2d.t() @ x_2d
            if ctx.scale_aware_votes:
                d_f = ctx.d.float().clamp(min=1e-6)
                d_rms = d_f.pow(2).mean().sqrt().clamp(min=1e-6)
                inv_s = (d_rms / d_f).clamp_(0.25, 2.5)
                grad_w = (grad_w.float().view(ctx.q.shape[0], -1, QK) * inv_s.unsqueeze(-1)).reshape(ctx.q.shape)
            # d > 0 everywhere, so dL/dq has the exact same sign as grad_w
            _accumulate_votes(grad_w, ctx.vote_buffer, ctx.vote_gain, ctx.vote_clip)
            del grad_w

        return grad_x, grad_bias, None, None, None, None, None, None


@torch.no_grad()
def _accumulate_votes_chunked(
    grad_w: torch.Tensor,
    vote_buffer: torch.Tensor,
    vote_gain: float,
    vote_clip: float,
    row_chunk: int = 32768,
    d: torch.Tensor | None = None,
    scale_aware: bool = False,
) -> None:
    """
    Memory-safe exact equivalent of _accumulate_votes for huge matrices (e.g. 151936x2048
    Qwen2.5 tied embed_tokens). Computes exact global RMS then processes rows in chunks,
    keeping peak temporary VRAM under ~0.8 GiB instead of ~8.9 GiB.
    """
    n_rows, n_cols = grad_w.shape
    if n_rows <= row_chunk and not scale_aware:
        _accumulate_votes(grad_w, vote_buffer, vote_gain, vote_clip)
        return

    d_rms = None
    if scale_aware and d is not None:
        d_rms = d.float().clamp(min=1e-6).pow(2).mean().sqrt().clamp(min=1e-6)

    sum_sq = torch.zeros((), device=grad_w.device, dtype=torch.float64)
    for r0 in range(0, n_rows, row_chunk):
        r1 = min(r0 + row_chunk, n_rows)
        gc = grad_w[r0:r1].float()
        if d_rms is not None and d is not None:
            inv_s = (d_rms / d[r0:r1].float().clamp(min=1e-6)).clamp_(0.25, 2.5)
            gc = (gc.view(r1 - r0, -1, QK) * inv_s.unsqueeze(-1)).reshape(r1 - r0, n_cols)
        sum_sq = sum_sq + gc.pow(2).sum(dtype=torch.float64)
        del gc
    rms = (sum_sq / max(1, grad_w.numel())).sqrt().float().clamp(min=1e-12)
    scale = -vote_gain / rms

    for r0 in range(0, n_rows, row_chunk):
        r1 = min(r0 + row_chunk, n_rows)
        gc = grad_w[r0:r1].float()
        if d_rms is not None and d is not None:
            inv_s = (d_rms / d[r0:r1].float().clamp(min=1e-6)).clamp_(0.25, 2.5)
            gc = (gc.view(r1 - r0, -1, QK) * inv_s.unsqueeze(-1)).reshape(r1 - r0, n_cols)
        pressure = (gc * scale).clamp_(-vote_clip, vote_clip)
        del gc
        delta = _stochastic_round_int(pressure)
        del pressure
        updated = (vote_buffer[r0:r1].to(torch.int16) + delta).clamp_(-127, 127).to(torch.int8)
        vote_buffer[r0:r1].copy_(updated)
        del delta, updated


class _Q8BlockEmbedFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_anchor: torch.Tensor,
        input_ids: torch.Tensor,
        q: torch.Tensor,
        d: torch.Tensor,
        vote_buffer: torch.Tensor,
        vote_gain: float,
        vote_clip: float,
        out_dtype: torch.dtype,
        scale_aware_votes: bool = False,
    ) -> torch.Tensor:
        flat_ids = input_ids.reshape(-1)
        q_rows = q.index_select(0, flat_ids)
        d_rows = d.index_select(0, flat_ids)
        n_rows, dim = q_rows.shape
        vals = (q_rows.view(n_rows, -1, QK).to(out_dtype) * d_rows.to(out_dtype).unsqueeze(-1)).reshape(n_rows, dim)
        out = vals.reshape(*input_ids.shape, dim)

        ctx.save_for_backward(input_ids)
        ctx.vocab_size = q.shape[0]
        ctx.d = d
        ctx.vote_buffer = vote_buffer
        ctx.vote_gain = vote_gain
        ctx.vote_clip = vote_clip
        ctx.scale_aware_votes = scale_aware_votes
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input_ids,) = ctx.saved_tensors
        if ctx.vote_gain > 0.0:
            flat_ids = input_ids.reshape(-1)
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).float()
            # Sparse unique-row reduction: O(U * D) instead of O(V * D) (75x less VRAM at V=151936)
            uniq_ids, inv = torch.unique(flat_ids, return_inverse=True)
            grad_uniq = torch.zeros(
                uniq_ids.numel(), grad_flat.shape[1],
                device=grad_flat.device, dtype=torch.float32,
            )
            grad_uniq.index_add_(0, inv, grad_flat)
            if ctx.scale_aware_votes:
                d_rms = ctx.d.float().clamp(min=1e-6).pow(2).mean().sqrt().clamp(min=1e-6)
                d_u = ctx.d.index_select(0, uniq_ids).float().clamp(min=1e-6)
                inv_s = (d_rms / d_u).clamp_(0.25, 2.5)
                grad_uniq = (grad_uniq.view(uniq_ids.numel(), -1, QK) * inv_s.unsqueeze(-1)).reshape_as(grad_uniq)
            nz = grad_uniq != 0
            rms = (grad_uniq.pow(2).sum() / nz.sum().clamp(min=1)).sqrt().clamp(min=1e-12)
            pressure = ((-grad_uniq / rms) * ctx.vote_gain).clamp_(-ctx.vote_clip, ctx.vote_clip)
            delta = _stochastic_round_int(pressure) * nz.to(torch.int16)
            cur = ctx.vote_buffer.index_select(0, uniq_ids).to(torch.int16)
            updated = (cur + delta).clamp_(-127, 127).to(torch.int8)
            ctx.vote_buffer.index_copy_(0, uniq_ids, updated)
        return None, None, None, None, None, None, None, None, None


class _Q8BlockMixin:
    q: torch.Tensor
    d: torch.Tensor
    vote_buffer: torch.Tensor
    vote_gain: float
    vote_clip: float
    scale_aware_votes: bool

    @property
    def num_discrete_params(self) -> int:
        return self.q.numel()

    @torch.no_grad()
    def _step_slice(
        self,
        q_s: torch.Tensor,
        v_s: torch.Tensor,
        threshold: int,
        decay: float,
        weight_decay: float,
        max_stride: int,
        error_feedback: bool,
        collect_stats: bool,
    ) -> tuple[int, int, int, int]:
        v16 = v_s.to(torch.int16)
        abs_v = v16.abs()
        fired = abs_v >= threshold
        step_dir = torch.sign(v16)

        if max_stride > 1:
            stride = (abs_v // threshold).clamp(1, max_stride)
            delta_q = torch.where(fired, step_dir * stride, torch.zeros_like(v16))
        else:
            delta_q = torch.where(fired, step_dir, torch.zeros_like(v16))

        q_s.copy_((q_s.to(torch.int16) + delta_q).clamp_(-127, 127).to(torch.int8))

        if error_feedback:
            v_next = torch.where(fired, v16 - delta_q * threshold, v16)
        else:
            v_next = torch.where(fired, torch.zeros_like(v16), v16)

        if weight_decay > 0.0:
            wd_vote = _stochastic_round_int(-(q_s.float() / 127.0) * weight_decay)
            v_next = (v_next + wd_vote).clamp_(-127, 127)

        if decay < 1.0:
            v_next = _stochastic_round_int(v_next.float() * decay).clamp_(-127, 127)

        v_s.copy_(v_next.to(torch.int8))

        if not collect_stats:
            return 0, 0, 0, 0
        return (
            int(fired.sum().item()),
            v16.numel(),
            int((v_next.abs() >= 120).sum().item()),
            int((q_s.abs() >= 127).sum().item()),
        )

    @torch.no_grad()
    def apply_discrete_step(
        self,
        threshold: int = 16,
        decay: float = 0.88,
        weight_decay: float = 0.02,
        max_stride: int = 1,
        error_feedback: bool = True,
        collect_stats: bool = False,
        row_chunk: int = 32768,
    ) -> dict | None:
        # Fully branchless, zero GPU-to-CPU sync (when collect_stats=False), row-chunked for huge vocab matrices
        n_rows = self.q.shape[0]
        if n_rows <= row_chunk:
            nf, nt, ns, nr = self._step_slice(
                self.q, self.vote_buffer, threshold, decay, weight_decay, max_stride, error_feedback, collect_stats
            )
        else:
            nf = nt = ns = nr = 0
            for r0 in range(0, n_rows, row_chunk):
                r1 = min(r0 + row_chunk, n_rows)
                f, t, s, r = self._step_slice(
                    self.q[r0:r1], self.vote_buffer[r0:r1],
                    threshold, decay, weight_decay, max_stride, error_feedback, collect_stats,
                )
                nf += f
                nt += t
                ns += s
                nr += r

        if not collect_stats:
            return None

        return {
            "flips": nf,
            "total": nt,
            "flip_pct": 100.0 * nf / max(1, nt),
            "sat_pct": 100.0 * ns / max(1, nt),
            "rail_pct": 100.0 * nr / max(1, nt),
        }

    @torch.no_grad()
    def rebalance_scales(
        self,
        rail_trigger: int = 120,
        grow_factor: float = 1.15,
        threshold: int = 24,
        error_feedback: bool = True,
        blk_chunk: int = 262144,
    ) -> int:
        """
        Zero-shadow-weight exact block-scale headroom expansion with O(n_crowded) transient VRAM.
        Scans QK=32 blocks in bounded chunks and updates ONLY crowded blocks in-place so huge
        vocab matrices (e.g. 151,936 x 2,048) never materialize multi-GiB float32 copies.
        """
        q_blk = self.q.view(-1, QK)
        d_flat = self.d.view(-1)
        v_blk = self.vote_buffer.view(-1, QK)
        n_blk = q_blk.shape[0]
        total_rebal = 0
        for b0 in range(0, n_blk, blk_chunk):
            b1 = min(n_blk, b0 + blk_chunk)
            qb_sub = q_blk[b0:b1]
            idx_local = (qb_sub.abs().amax(dim=-1) >= rail_trigger).nonzero(as_tuple=False).squeeze(-1)
            n_sub = int(idx_local.numel())
            if n_sub == 0:
                continue
            idx = idx_local + b0
            old_d = d_flat[idx].float()
            new_d = (old_d * grow_factor).clamp(max=65000.0).to(torch.float16)
            ratio = (old_d / new_d.float().clamp(min=1e-6)).unsqueeze(-1)
            q_exact = q_blk[idx].float() * ratio
            q_round = torch.round(q_exact).clamp(-127, 127)
            if error_feedback and threshold > 0:
                v_comp = torch.round((q_exact - q_round) * float(threshold) + v_blk[idx].float() * ratio).clamp(-127, 127).to(torch.int8)
                v_blk[idx] = v_comp
            d_flat[idx] = new_d
            q_blk[idx] = q_round.to(torch.int8)
            total_rebal += n_sub
        return total_rebal

    @torch.no_grad()
    def load_from_float_weight(self, weight: torch.Tensor, headroom: float = 1.05, min_d: float = 1e-5) -> None:
        """
        Initialize exact V3_Q8_0 (d: fp16, q: int8) blocks from a pretrained float/bf16 weight matrix
        with optional scale headroom (e.g. 1.05 -> max |q| ~= 121 so weights can move in both directions
        immediately without hitting the ±127 rail, while keeping ~99.98% of exact BF16 precision).
        """
        out_f, in_f = self.q.shape
        assert weight.shape == (out_f, in_f), f"Shape mismatch: got {weight.shape}, expected {(out_f, in_f)}"
        w = weight.detach().float()
        blk = w.reshape(out_f, in_f // QK, QK)
        blk_max = blk.abs().amax(dim=-1) * float(headroom)
        d_fp16 = (blk_max / 127.0).clamp(min=min_d).to(torch.float16)
        d_f32 = d_fp16.float()
        q_init = torch.round(blk / d_f32.unsqueeze(-1)).clamp(-127, 127).to(torch.int8).reshape(out_f, in_f)
        self.d.copy_(d_fp16.to(self.d.device))
        self.q.copy_(q_init.to(self.q.device))
        self.vote_buffer.zero_()

    @torch.no_grad()
    def pack_v3_q8_0_bytes(self) -> bytes:
        """
        Pack the EXACT trained (d: fp16, q: int8) tensors directly into VLT3 Q8_0 format
        (34 bytes / 32 weights) with ZERO re-quantization / ZERO PTQ loss.
        """
        out_f, in_f = self.q.shape
        n_blk = (out_f * in_f) // QK
        q8_dt = np.dtype([("d", "<f2"), ("q", "i1", (QK,))])
        out = np.empty(n_blk, dtype=q8_dt)
        out["d"] = self.d.detach().cpu().contiguous().numpy().astype("<f2").reshape(n_blk)
        out["q"] = self.q.detach().cpu().contiguous().numpy().astype("i1").reshape(n_blk, QK)
        return out.tobytes()


class PrimeQ8BlockLinear(_Q8BlockMixin, nn.Module):
    """
    Shadowless discrete linear layer storing exact V3_Q8_0 (QK=32) blocks.
    Total weight + optimizer memory: 2.0625 bytes/param (q: 1B, d: 0.0625B, vote: 1B).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_std: float,
        bias: bool = False,
        scale_headroom: float = 4.5,
        min_scale: float = 0.08,
        vote_gain: float = 4.0,
        vote_clip: float = 16.0,
        skip_init: bool = False,
        scale_aware_votes: bool = False,
    ):
        super().__init__()
        assert in_features % QK == 0, f"in_features={in_features} must be a multiple of QK={QK}"
        self.in_features = in_features
        self.out_features = out_features
        self.vote_gain = vote_gain
        self.vote_clip = vote_clip
        self.scale_aware_votes = scale_aware_votes

        if skip_init:
            q_init = torch.zeros(out_features, in_features, dtype=torch.int8)
            d_fp16 = torch.full((out_features, in_features // QK), 1e-4, dtype=torch.float16)
        else:
            with torch.no_grad():
                raw_w = torch.randn(out_features, in_features, dtype=torch.float32) * init_std
                blk = raw_w.view(out_features, in_features // QK, QK)
                # Fixed per-block fp16 scale with generous headroom above initial max/std
                base_max = max(scale_headroom * init_std, min_scale)
                blk_max = torch.maximum(blk.abs().amax(dim=-1) * 1.6, torch.full((out_features, in_features // QK), base_max))
                d_fp16 = (blk_max / 127.0).clamp(min=1e-4).to(torch.float16)
                d_f32 = d_fp16.float()
                q_init = torch.round(blk / d_f32.unsqueeze(-1)).clamp(-127, 127).to(torch.int8).reshape(out_features, in_features)

        self.register_buffer("q", q_init)
        self.register_buffer("d", d_fp16)
        self.register_buffer("vote_buffer", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            w = _dequant_q8_0(self.q, self.d, x.dtype)
            b = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, w, b)
        return _Q8BlockLinearFn.apply(
            x, self.bias, self.q, self.d, self.vote_buffer, self.vote_gain, self.vote_clip, self.scale_aware_votes
        )


class PrimeQ8BlockTiedEmbedding(_Q8BlockMixin, nn.Module):
    """Weight-tied V3_Q8_0 discrete embedding + LM head."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        init_std: float = 0.02,
        scale_headroom: float = 16.0,
        min_scale: float = 0.35,
        vote_gain: float = 4.0,
        vote_clip: float = 16.0,
        skip_init: bool = False,
        scale_aware_votes: bool = False,
    ):
        super().__init__()
        assert dim % QK == 0, f"dim={dim} must be a multiple of QK={QK}"
        self.vocab_size = vocab_size
        self.dim = dim
        self.vote_gain = vote_gain
        self.vote_clip = vote_clip
        self.scale_aware_votes = scale_aware_votes

        self._grad_anchor = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        if skip_init:
            q_init = torch.zeros(vocab_size, dim, dtype=torch.int8)
            d_fp16 = torch.full((vocab_size, dim // QK), 1e-4, dtype=torch.float16)
        else:
            with torch.no_grad():
                raw_w = torch.randn(vocab_size, dim, dtype=torch.float32) * init_std
                blk = raw_w.view(vocab_size, dim // QK, QK)
                base_max = max(scale_headroom * init_std, min_scale)
                blk_max = torch.maximum(blk.abs().amax(dim=-1) * 2.0, torch.full((vocab_size, dim // QK), base_max))
                d_fp16 = (blk_max / 127.0).clamp(min=1e-4).to(torch.float16)
                d_f32 = d_fp16.float()
                q_init = torch.round(blk / d_f32.unsqueeze(-1)).clamp(-127, 127).to(torch.int8).reshape(vocab_size, dim)

        self.register_buffer("q", q_init)
        self.register_buffer("d", d_fp16)
        self.register_buffer("vote_buffer", torch.zeros(vocab_size, dim, dtype=torch.int8))

    def embed(self, input_ids: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if not torch.is_grad_enabled():
            flat = input_ids.reshape(-1)
            q_r = self.q.index_select(0, flat)
            d_r = self.d.index_select(0, flat)
            n_r = q_r.shape[0]
            vals = (q_r.view(n_r, -1, QK).to(out_dtype) * d_r.to(out_dtype).unsqueeze(-1)).reshape(n_r, self.dim)
            return vals.reshape(*input_ids.shape, self.dim)
        return _Q8BlockEmbedFn.apply(
            self._grad_anchor, input_ids, self.q, self.d, self.vote_buffer,
            self.vote_gain, self.vote_clip, out_dtype, self.scale_aware_votes,
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            w = _dequant_q8_0(self.q, self.d, x.dtype)
            return F.linear(x, w)
        return _Q8BlockLinearFn.apply(
            x, None, self.q, self.d, self.vote_buffer, self.vote_gain, self.vote_clip, self.scale_aware_votes
        )

    def chunked_cross_entropy(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        chunk_size: int = 4096,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Exact chunked fused LM-head + cross-entropy with ignore_index masking (zero approximation).
        Avoids materializing (B*T, vocab_size) fp32 logits during long-context / large-vocab training.
        """
        if not torch.is_grad_enabled():
            h2d = x.reshape(-1, x.shape[-1])
            y1d = targets.reshape(-1)
            w = _dequant_q8_0(self.q, self.d, h2d.dtype)
            n_tot = y1d.numel()
            n_valid = (y1d != ignore_index).sum().clamp(min=1).float()
            loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
            for s in range(0, n_tot, chunk_size):
                e = min(s + chunk_size, n_tot)
                lg = F.linear(h2d[s:e], w).float()
                loss_sum = loss_sum + F.cross_entropy(lg, y1d[s:e], ignore_index=ignore_index, reduction="sum")
            return loss_sum / n_valid
        return _Q8ChunkedCrossEntropyFn.apply(
            x, targets, self.q, self.d, self.vote_buffer, self.vote_gain, self.vote_clip,
            chunk_size, ignore_index, self.scale_aware_votes,
        )


class _Q8ChunkedCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        targets: torch.Tensor,
        q: torch.Tensor,
        d: torch.Tensor,
        vote_buffer: torch.Tensor,
        vote_gain: float,
        vote_clip: float,
        chunk_size: int,
        ignore_index: int,
        scale_aware_votes: bool = False,
    ) -> torch.Tensor:
        orig_shape = x.shape
        h2d = x.reshape(-1, orig_shape[-1])
        y1d = targets.reshape(-1)
        n_tot = y1d.numel()
        n_valid = (y1d != ignore_index).sum().clamp(min=1).float()
        inv_n = 1.0 / n_valid

        w = _dequant_q8_0(q, d, h2d.dtype)
        grad_h = torch.empty_like(h2d)
        need_votes = vote_gain > 0.0
        # Accumulate grad_w directly in h2d.dtype (bf16) via in-place addmm_ -> 0 extra temp matrix!
        grad_w = torch.zeros(q.shape[0], q.shape[1], device=h2d.device, dtype=h2d.dtype) if need_votes else None
        loss_sum = torch.zeros((), device=h2d.device, dtype=torch.float32)

        for s in range(0, n_tot, chunk_size):
            e = min(s + chunk_size, n_tot)
            hc = h2d[s:e]
            yc = y1d[s:e]
            vc = yc != ignore_index
            yc_safe = torch.where(vc, yc, torch.zeros_like(yc))

            lg = F.linear(hc, w).float()
            log_p = F.log_softmax(lg, dim=-1)
            nll = -log_p.gather(-1, yc_safe.unsqueeze(-1)).squeeze(-1)
            loss_sum = loss_sum + torch.where(vc, nll, torch.zeros_like(nll)).sum()

            # Exact d(mean CE)/d(logits) = mask * (softmax(lg) - one_hot(y)) / N_valid
            d_lg = log_p.exp_()
            d_lg.scatter_add_(-1, yc_safe.unsqueeze(-1), torch.full((e - s, 1), -1.0, device=h2d.device, dtype=torch.float32))
            d_lg.mul_(vc.float().unsqueeze(-1) * inv_n)
            d_lg_dt = d_lg.to(h2d.dtype)
            grad_h[s:e] = d_lg_dt @ w
            if need_votes:
                grad_w.addmm_(d_lg_dt.t(), hc)
            del lg, log_p, nll, d_lg, d_lg_dt

        del w
        ctx.save_for_backward(grad_h.reshape(orig_shape), grad_w if need_votes else torch.empty(0, device=h2d.device))
        ctx.d = d
        ctx.vote_buffer = vote_buffer
        ctx.vote_gain = vote_gain
        ctx.vote_clip = vote_clip
        ctx.need_votes = need_votes
        ctx.scale_aware_votes = scale_aware_votes
        return loss_sum * inv_n

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        grad_h, grad_w_dt = ctx.saved_tensors
        if ctx.need_votes:
            # Handle sign of grad_loss without doubling grad_w_dt in VRAM (in-place sign mul)
            sgn = torch.sign(grad_loss.float()).to(grad_w_dt.dtype)
            grad_w_dt = grad_w_dt * sgn
            _accumulate_votes_chunked(
                grad_w_dt, ctx.vote_buffer, ctx.vote_gain, ctx.vote_clip,
                d=ctx.d, scale_aware=ctx.scale_aware_votes,
            )
            del grad_w_dt
        return grad_h * grad_loss.to(grad_h.dtype), None, None, None, None, None, None, None, None, None


# ==============================================================================
# 2. Exact VAULT v3 GQA Transformer Backbone (100% VLT3 Schema Compatible)
# ==============================================================================

@dataclass
class VAULTPrimalConfig:
    vocab_size: int = 4096
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    head_dim: int = 64
    d_ff: int = 896
    rms_eps: float = 1e-6
    rope_theta: float = 10_000.0
    tie_embeddings: bool = True
    window: int = 512
    n_sink: int = 4
    seg_budget: int = 0
    seg_layers: tuple[int, ...] = ()

    @classmethod
    def from_qwen_config(cls, qc: dict, window: int = 4096, n_sink: int = 4) -> "VAULTPrimalConfig":
        n_heads = int(qc["num_attention_heads"])
        d_model = int(qc["hidden_size"])
        return cls(
            vocab_size=int(qc["vocab_size"]),
            d_model=d_model,
            n_layers=int(qc["num_hidden_layers"]),
            n_heads=n_heads,
            n_kv_heads=int(qc["num_key_value_heads"]),
            head_dim=d_model // n_heads,
            d_ff=int(qc["intermediate_size"]),
            rms_eps=float(qc.get("rms_norm_eps", 1e-6)),
            rope_theta=float(qc.get("rope_theta", (qc.get("rope_parameters") or {}).get("rope_theta", 1_000_000.0))),
            tie_embeddings=bool(qc.get("tie_word_embeddings", True)),
            window=int(window),
            n_sink=int(n_sink),
        )


class VaultRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * inv * self.weight).to(x.dtype)


class VaultRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
        self.register_buffer("inv_freq", inv_freq.float(), persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float64)
        freqs = torch.outer(t, self.inv_freq.double())
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().float(), persistent=False)
        self.register_buffer("sin_cached", emb.sin().float(), persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype):
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len].to(dtype), self.sin_cached[:seq_len].to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    return (q * c) + (_rotate_half(q) * s), (k * c) + (_rotate_half(k) * s)


class VAULTPrimalAttention(nn.Module):
    def __init__(self, cfg: VAULTPrimalConfig, skip_init: bool = False):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_groups = cfg.n_heads // cfg.n_kv_heads

        std_in = 1.0 / math.sqrt(cfg.d_model)
        std_out = 1.0 / math.sqrt(2.0 * cfg.n_layers * cfg.d_model)

        self.q_proj = PrimeQ8BlockLinear(cfg.d_model, cfg.n_heads * cfg.head_dim, std_in, bias=True, skip_init=skip_init)
        self.k_proj = PrimeQ8BlockLinear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, std_in, bias=True, skip_init=skip_init)
        self.v_proj = PrimeQ8BlockLinear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, std_in, bias=True, skip_init=skip_init)
        self.o_proj = PrimeQ8BlockLinear(cfg.n_heads * cfg.head_dim, cfg.d_model, std_out, bias=False, skip_init=skip_init)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin)

        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class VAULTPrimalMLP(nn.Module):
    def __init__(self, cfg: VAULTPrimalConfig, skip_init: bool = False):
        super().__init__()
        std_in = 1.0 / math.sqrt(cfg.d_model)
        std_out = 1.0 / math.sqrt(2.0 * cfg.n_layers * cfg.d_ff)
        self.gate_proj = PrimeQ8BlockLinear(cfg.d_model, cfg.d_ff, std_in, bias=False, skip_init=skip_init)
        self.up_proj = PrimeQ8BlockLinear(cfg.d_model, cfg.d_ff, std_in, bias=False, skip_init=skip_init)
        self.down_proj = PrimeQ8BlockLinear(cfg.d_ff, cfg.d_model, std_out, bias=False, skip_init=skip_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VAULTPrimalLayer(nn.Module):
    def __init__(self, cfg: VAULTPrimalConfig, skip_init: bool = False):
        super().__init__()
        self.input_layernorm = VaultRMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.self_attn = VAULTPrimalAttention(cfg, skip_init=skip_init)
        self.post_attention_layernorm = VaultRMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.mlp = VAULTPrimalMLP(cfg, skip_init=skip_init)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class VAULTPrimalForCausalLM(nn.Module):
    """
    100% VLT3-schema-compatible GQA Transformer trained shadowlessly in native V3_Q8_0.
    Every module name matches vault/tools/vlt3.py tensor names 1:1.
    """

    def __init__(
        self,
        cfg: VAULTPrimalConfig,
        skip_init: bool = False,
        grad_checkpoint: bool = False,
        ce_chunk_size: int | None = None,
    ):
        super().__init__()
        assert cfg.d_model == cfg.n_heads * cfg.head_dim
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.cfg = cfg
        self.grad_checkpoint = grad_checkpoint
        # Automatically pick compact CE chunk for huge vocabularies (e.g. Qwen2.5 151936)
        self.ce_chunk_size = int(ce_chunk_size) if ce_chunk_size is not None else (512 if cfg.vocab_size >= 65536 else 4096)

        self.embed_tokens = PrimeQ8BlockTiedEmbedding(cfg.vocab_size, cfg.d_model, skip_init=skip_init)
        self.lm_head: PrimeQ8BlockTiedEmbedding | None = (
            None if cfg.tie_embeddings else PrimeQ8BlockTiedEmbedding(cfg.vocab_size, cfg.d_model, skip_init=skip_init)
        )
        self.rope = VaultRotaryEmbedding(cfg.head_dim, max_seq_len=max(2048, 2 * cfg.window), theta=cfg.rope_theta)
        self.layers = nn.ModuleList([VAULTPrimalLayer(cfg, skip_init=skip_init) for _ in range(cfg.n_layers)])
        self.norm = VaultRMSNorm(cfg.d_model, eps=cfg.rms_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = False,
        ignore_index: int = -100,
    ):
        B, T = input_ids.shape
        out_dtype = torch.bfloat16 if torch.is_autocast_enabled() else torch.float32
        x = self.embed_tokens.embed(input_ids, out_dtype=out_dtype)
        cos, sin = self.rope(T, x.dtype)
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for layer in self.layers:
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin)
        x = self.norm(x)
        head_mod = self.embed_tokens if self.lm_head is None else self.lm_head
        if targets is not None and not return_logits:
            loss = head_mod.chunked_cross_entropy(
                x, targets, chunk_size=self.ce_chunk_size, ignore_index=ignore_index
            )
            return None, loss
        logits = head_mod.project(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss

    def count_params(self) -> tuple[int, int]:
        disc = sum(m.q.numel() for m in self.modules() if isinstance(m, _Q8BlockMixin))
        cont = sum(p.numel() for n, p in self.named_parameters() if "_grad_anchor" not in n)
        return disc + cont, disc


@torch.no_grad()
def load_pretrained_qwen2_into_primal_q8(
    model: VAULTPrimalForCausalLM,
    hf_dir: str,
    headroom: float = 1.05,
    min_d: float = 1e-5,
) -> dict:
    """
    Stream Qwen2 / Qwen2.5 safetensors shards directly into a shadowless VAULTPrimalForCausalLM
    (native V3_Q8_0 blocks), never holding more than one tensor in float32 at a time.
    """
    from safetensors import safe_open

    shard_files = sorted(
        os.path.join(hf_dir, fn)
        for fn in os.listdir(hf_dir)
        if fn.endswith(".safetensors")
    )
    assert shard_files, f"No .safetensors files found in {hf_dir}"

    loaded = 0
    total_disc = 0
    t0 = time.time()

    def _assign(name: str, tensor: torch.Tensor) -> bool:
        nonlocal loaded, total_disc
        key = name[6:] if name.startswith("model.") else name
        if key == "embed_tokens.weight":
            model.embed_tokens.load_from_float_weight(tensor, headroom=headroom, min_d=min_d)
            loaded += 1
            total_disc += tensor.numel()
            return True
        if key == "norm.weight":
            model.norm.weight.copy_(tensor.float().to(model.norm.weight.device))
            loaded += 1
            return True
        if key == "lm_head.weight":
            if model.lm_head is not None:
                model.lm_head.load_from_float_weight(tensor, headroom=headroom, min_d=min_d)
                loaded += 1
                total_disc += tensor.numel()
            return True
        if not key.startswith("layers."):
            return False
        parts = key.split(".")
        li = int(parts[1])
        layer = model.layers[li]
        sub = ".".join(parts[2:])
        if sub == "input_layernorm.weight":
            layer.input_layernorm.weight.copy_(tensor.float().to(layer.input_layernorm.weight.device))
        elif sub == "post_attention_layernorm.weight":
            layer.post_attention_layernorm.weight.copy_(tensor.float().to(layer.post_attention_layernorm.weight.device))
        elif sub in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight"):
            proj = getattr(layer.self_attn, parts[3])
            proj.load_from_float_weight(tensor, headroom=headroom, min_d=min_d)
            total_disc += tensor.numel()
        elif sub in ("self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"):
            proj = getattr(layer.self_attn, parts[3])
            proj.bias.copy_(tensor.float().to(proj.bias.device))
        elif sub in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
            proj = getattr(layer.mlp, parts[3])
            proj.load_from_float_weight(tensor, headroom=headroom, min_d=min_d)
            total_disc += tensor.numel()
        else:
            return False
        loaded += 1
        return True

    for sf in shard_files:
        with safe_open(sf, framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                ok = _assign(k, t)
                if not ok:
                    raise KeyError(f"Unrecognized Qwen2 safetensors key: {k}")
                del t

    expected_tensors = (2 if model.lm_head is None else 3) + 12 * model.cfg.n_layers
    assert loaded == expected_tensors, f"Loaded {loaded} tensors, expected {expected_tensors}"
    return {
        "loaded_tensors": loaded,
        "discrete_params": total_disc,
        "headroom": headroom,
        "elapsed_s": round(time.time() - t0, 2),
    }


class Q8StepEngine:
    """SNR-matched cosine / WSD / infinite-stream vote-gate step engine for VAULTPrimalForCausalLM."""

    def __init__(
        self,
        model: VAULTPrimalForCausalLM,
        total_steps: int,
        warmup_steps: int = 80,
        threshold_start: int = 10,
        threshold_end: int = 16,
        vote_gain_start: float = 5.0,
        vote_gain_end: float = 1.2,
        vote_clip: float = 16.0,
        decay_start: float = 0.88,
        decay_end: float = 0.95,
        weight_decay: float = 0.02,
        max_stride: int = 1,
        error_feedback: bool = True,
        aux_lr: float = 1e-3,
        schedule: str = "cosine",
        cooldown_frac: float = 0.15,
        rebalance_every: int = 0,
        rail_trigger: int = 122,
        grow_factor: float = 1.12,
        scale_aware_votes: bool = False,
        grad_accum: int = 1,
    ):
        self.model = model
        self.total_steps = max(1, total_steps)
        self.warmup_steps = warmup_steps
        self.threshold_start = threshold_start
        self.threshold_end = threshold_end
        self.vote_gain_start = vote_gain_start
        self.vote_gain_end = vote_gain_end
        self.vote_clip = vote_clip
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.weight_decay = weight_decay
        self.max_stride = max_stride
        self.error_feedback = error_feedback
        self.schedule = schedule
        self.cooldown_frac = cooldown_frac
        self.rebalance_every = rebalance_every
        self.rail_trigger = rail_trigger
        self.grow_factor = grow_factor
        self.scale_aware_votes = scale_aware_votes
        self.grad_accum = max(1, int(grad_accum))
        self.step_count = 0

        self.discrete_modules: list[_Q8BlockMixin] = [
            m for m in model.modules() if isinstance(m, _Q8BlockMixin)
        ]
        for m in self.discrete_modules:
            m.scale_aware_votes = scale_aware_votes
        aux_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "_grad_anchor" not in n
        ]
        self.aux_optimizer = torch.optim.AdamW(aux_params, lr=aux_lr, weight_decay=0.0) if aux_params else None
        self.aux_base_lr = aux_lr
        self._apply_schedule(0)

    def _schedule(self, step: int) -> tuple[int, float, float, float]:
        if step < self.warmup_steps and self.warmup_steps > 0:
            frac = (step + 1) / self.warmup_steps
            return self.threshold_start, self.vote_gain_start * (0.25 + 0.75 * frac), self.decay_start, self.aux_base_lr * frac

        if self.schedule == "infinite":
            return self.threshold_start, self.vote_gain_start, self.decay_start, self.aux_base_lr

        if self.schedule == "wsd":
            cooldown_steps = max(1, int(round(self.total_steps * self.cooldown_frac)))
            decay_start_step = max(self.warmup_steps, self.total_steps - cooldown_steps)
            if step < decay_start_step:
                return self.threshold_start, self.vote_gain_start, self.decay_start, self.aux_base_lr
            prog = min(1.0, (step - decay_start_step) / max(1, self.total_steps - decay_start_step))
        else:
            prog = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))

        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        thresh = int(round(self.threshold_end - (self.threshold_end - self.threshold_start) * cos))
        gain = self.vote_gain_end + (self.vote_gain_start - self.vote_gain_end) * cos
        dec = self.decay_end - (self.decay_end - self.decay_start) * cos
        alr = self.aux_base_lr * (0.1 + 0.9 * cos)
        return max(1, thresh), gain, dec, alr

    def _apply_schedule(self, step: int):
        thresh, gain, dec, alr = self._schedule(step)
        micro_gain = gain / float(self.grad_accum)
        for m in self.discrete_modules:
            m.vote_gain = micro_gain
            m.vote_clip = self.vote_clip
            m.scale_aware_votes = self.scale_aware_votes
        if self.aux_optimizer is not None:
            for pg in self.aux_optimizer.param_groups:
                pg["lr"] = alr
        return thresh, gain, dec

    def zero_grad(self):
        if self.aux_optimizer is not None:
            self.aux_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def rebalance_scales(
        self,
        rail_trigger: int | None = None,
        grow_factor: float | None = None,
        threshold: int | None = None,
    ) -> int:
        rt = self.rail_trigger if rail_trigger is None else rail_trigger
        gf = self.grow_factor if grow_factor is None else grow_factor
        th = self._schedule(self.step_count)[0] if threshold is None else threshold
        return sum(
            m.rebalance_scales(
                rail_trigger=rt,
                grow_factor=gf,
                threshold=th,
                error_feedback=self.error_feedback,
            )
            for m in self.discrete_modules
        )

    @torch.no_grad()
    def step(self, collect_stats: bool = False) -> dict:
        thresh, gain, dec = self._apply_schedule(self.step_count)
        flips = 0
        total = 0
        max_sat = 0.0
        max_rail = 0.0
        for m in self.discrete_modules:
            st = m.apply_discrete_step(
                threshold=thresh,
                decay=dec,
                weight_decay=self.weight_decay,
                max_stride=self.max_stride,
                error_feedback=self.error_feedback,
                collect_stats=collect_stats,
            )
            if st is not None:
                flips += st["flips"]
                total += st["total"]
                max_sat = max(max_sat, st["sat_pct"])
                max_rail = max(max_rail, st["rail_pct"])

        if self.aux_optimizer is not None:
            ps = [p for pg in self.aux_optimizer.param_groups for p in pg["params"]]
            torch.nn.utils.clip_grad_norm_(ps, 1.0)
            self.aux_optimizer.step()

        self.step_count += 1
        rebal_blocks = 0
        if self.rebalance_every > 0 and (self.step_count % self.rebalance_every == 0):
            rebal_blocks = self.rebalance_scales(threshold=thresh)

        self._apply_schedule(self.step_count)
        return {
            "flips": flips,
            "flip_pct": 100.0 * flips / max(1, total) if collect_stats else 0.0,
            "threshold": thresh,
            "vote_gain": round(gain, 3),
            "decay": round(dec, 4),
            "max_sat_pct": max_sat,
            "max_rail_pct": max_rail,
            "rebal_blocks": rebal_blocks,
        }

    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "aux_optimizer": self.aux_optimizer.state_dict() if self.aux_optimizer is not None else None,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.step_count = int(sd.get("step_count", 0))
        if self.aux_optimizer is not None and sd.get("aux_optimizer") is not None:
            self.aux_optimizer.load_state_dict(sd["aux_optimizer"])
        self._apply_schedule(self.step_count)

    def memory_report(self) -> dict:
        total_p, disc_p = self.model.count_params()
        aux_p = total_p - disc_p
        q_bytes = sum(m.q.numel() * m.q.element_size() for m in self.discrete_modules)
        d_bytes = sum(m.d.numel() * m.d.element_size() for m in self.discrete_modules)
        v_bytes = sum(m.vote_buffer.numel() * m.vote_buffer.element_size() for m in self.discrete_modules)
        aux_bytes = aux_p * 16
        tot_bytes = q_bytes + d_bytes + v_bytes + aux_bytes
        adamw_bytes = total_p * 16
        # Apples-to-apples layer-hooked BF16 weights (2B) + 8-bit AdamW m,v (2B + 2 FP16 block scales = 0.125B -> 4.125B) + 1D BF16+FP32 m,v (10B)
        fused_adamw8bit_bytes = int(disc_p * 4.125) + aux_p * 10
        return {
            "total_params": total_p,
            "discrete_params": disc_p,
            "discrete_pct": 100.0 * disc_p / max(1, total_p),
            "aux_1d_params": aux_p,
            "total_state_mb": tot_bytes / (1024 ** 2),
            "bytes_per_param": tot_bytes / max(1, total_p),
            "fused_adamw8bit_equiv_mb": fused_adamw8bit_bytes / (1024 ** 2),
            "compression_vs_adamw8bit": fused_adamw8bit_bytes / max(1, tot_bytes),
            "adamw_equiv_mb": adamw_bytes / (1024 ** 2),
            "compression_ratio": adamw_bytes / max(1, tot_bytes),
        }


# ==============================================================================
# 3. Compact Qwen2-Regex TinyStories BPE Tokenizer (100% VTOK Compatible)
# ==============================================================================

def train_or_load_bpe_tokenizer(
    tok_dir: str,
    vocab_size: int = 4096,
    max_train_docs: int = 100_000,
):
    """
    Train (or load cached) a compact byte-level BPE tokenizer matching vault3.c's
    exact Qwen2 pre-tokenizer regex + UAX #15 NFC + ChatML special tokens.
    """
    from tokenizers import Tokenizer, Regex, decoders, models, normalizers, pre_tokenizers, trainers

    tok_json = os.path.join(tok_dir, "tokenizer.json")
    if os.path.isfile(tok_json):
        return Tokenizer.from_file(tok_json)

    os.makedirs(tok_dir, exist_ok=True)
    from datasets import load_dataset

    t0 = time.time()
    print(f"[TOK] Training {vocab_size}-vocab Qwen2-compatible ByteLevel BPE on TinyStories ({max_train_docs:,} docs)...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    n_docs = min(len(ds), max_train_docs)
    ds = ds.select(range(n_docs))

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN2_PRETOK_REGEX), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()

    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def _iter_batches():
        bs = 4096
        for s in range(0, n_docs, bs):
            yield ds[s : min(s + bs, n_docs)]["text"]

    tok.train_from_iterator(_iter_batches(), trainer=trainer, length=n_docs)
    tok.save(tok_json)
    print(f"[TOK] Saved {tok_json} (vocab={tok.get_vocab_size()}) in {time.time() - t0:.1f}s")
    return tok


# ==============================================================================
# 4. Zero-PTQ-Loss Direct VLT3 Binary Exporter
# ==============================================================================

def export_vault_primal_vlt3(
    model: VAULTPrimalForCausalLM,
    tok_dir: str,
    out_path: str,
    model_name: str = "VAULT-PRIMAL-10M-q8_0",
    vault_tools_dir: str = "./tools",
) -> dict:
    """
    Write the EXACT trained V3_Q8_0 (d, q) tensors directly into a VLT3 file
    without any float round-trip or re-quantization, and validate with vlt3.read_vlt3().
    """
    if vault_tools_dir not in sys.path:
        sys.path.insert(0, vault_tools_dir)
    import vlt3

    cfg = model.cfg
    # Collect exact raw byte payloads keyed by canonical VLT3 tensor name
    raw_tensors: dict[str, tuple[int, tuple[int, ...], bytes]] = {}

    # 1. Global tensors
    raw_tensors["embed_tokens.weight"] = (
        vlt3.Q8_0,
        (cfg.vocab_size, cfg.d_model),
        model.embed_tokens.pack_v3_q8_0_bytes(),
    )
    raw_tensors["norm.weight"] = (
        vlt3.F32,
        (cfg.d_model,),
        vlt3.encode_tensor(model.norm.weight, vlt3.F32),
    )
    if model.lm_head is not None:
        raw_tensors["lm_head.weight"] = (
            vlt3.Q8_0,
            (cfg.vocab_size, cfg.d_model),
            model.lm_head.pack_v3_q8_0_bytes(),
        )

    # 2. Per-layer tensors (12 per layer matching L_KINDS exactly)
    for li, layer in enumerate(model.layers):
        p = f"layers.{li}"
        raw_tensors[f"{p}.input_layernorm.weight"] = (
            vlt3.F32, (cfg.d_model,), vlt3.encode_tensor(layer.input_layernorm.weight, vlt3.F32)
        )
        for proj_name, mod in [
            ("self_attn.q_proj", layer.self_attn.q_proj),
            ("self_attn.k_proj", layer.self_attn.k_proj),
            ("self_attn.v_proj", layer.self_attn.v_proj),
        ]:
            raw_tensors[f"{p}.{proj_name}.weight"] = (
                vlt3.Q8_0, (mod.out_features, mod.in_features), mod.pack_v3_q8_0_bytes()
            )
            raw_tensors[f"{p}.{proj_name}.bias"] = (
                vlt3.F32, (mod.out_features,), vlt3.encode_tensor(mod.bias, vlt3.F32)
            )
        for proj_name, mod in [
            ("self_attn.o_proj", layer.self_attn.o_proj),
            ("mlp.gate_proj", layer.mlp.gate_proj),
            ("mlp.up_proj", layer.mlp.up_proj),
            ("mlp.down_proj", layer.mlp.down_proj),
        ]:
            raw_tensors[f"{p}.{proj_name}.weight"] = (
                vlt3.Q8_0, (mod.out_features, mod.in_features), mod.pack_v3_q8_0_bytes()
            )
        raw_tensors[f"{p}.post_attention_layernorm.weight"] = (
            vlt3.F32, (cfg.d_model,), vlt3.encode_tensor(layer.post_attention_layernorm.weight, vlt3.F32)
        )

    names = sorted(raw_tensors.keys(), key=vlt3.tensor_id)
    entries = []
    blobs = []
    for nm in names:
        dt, shape, data = raw_tensors[nm]
        n_elem = int(np.prod(shape))
        assert len(data) == vlt3.nbytes_for(dt, n_elem), f"Size mismatch for {nm}"
        dims = list(shape) + [1] * (4 - len(shape))
        entries.append(dict(
            id=vlt3.tensor_id(nm),
            dtype=dt,
            ndim=len(shape),
            dims=dims,
            nbytes=len(data),
            crc=zlib.crc32(data),
            name=nm,
        ))
        blobs.append(data)

    tok_blob, ids = vlt3.build_tokenizer_blob(tok_dir)
    dir_off = vlt3.HEADER_SIZE
    dir_size = vlt3.ENTRY_SIZE * len(entries)
    tok_off = (dir_off + dir_size + vlt3.ALIGN - 1) // vlt3.ALIGN * vlt3.ALIGN
    data_off = (tok_off + len(tok_blob) + vlt3.ALIGN - 1) // vlt3.ALIGN * vlt3.ALIGN
    cur = data_off
    for e in entries:
        cur = (cur + vlt3.ALIGN - 1) // vlt3.ALIGN * vlt3.ALIGN
        e["offset"] = cur
        cur += e["nbytes"]
    file_size = cur

    flags = 1 if cfg.tie_embeddings else 0
    hdr_vals = dict(
        magic=vlt3.MAGIC, version=vlt3.VERSION, header_size=vlt3.HEADER_SIZE, endian_tag=vlt3.ENDIAN_TAG,
        file_size=file_size, alignment=vlt3.ALIGN, n_tensors=len(entries),
        tensor_dir_offset=dir_off, tensor_dir_size=dir_size,
        tokenizer_offset=tok_off, tokenizer_size=len(tok_blob),
        data_offset=data_off, data_size=file_size - data_off,
        vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers,
        n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        d_ff=cfg.d_ff, window=cfg.window,
        n_banks=0, primes=[0] * 8, n_cells=0,
        rms_eps=cfg.rms_eps, rope_theta=cfg.rope_theta, vault_eps=0.0,
        bos_id=ids["bos"], eos_id=ids["eos"], im_start_id=ids["im_start"], im_end_id=ids["im_end"], pad_id=ids["pad"],
        weight_format=vlt3.Q8_0, flags=flags,
        model_name=model_name.encode("utf-8")[:63],
        n_sink=cfg.n_sink,
        seg_budget=0, seg_hidden=0, seg_min=0, seg_max=0, seg_n_feat=0, seg_layers=[0, 0, 0, 0],
    )
    flat = []
    for f in vlt3._HDR_FIELDS:
        v = hdr_vals[f]
        flat.extend(v if f in ("primes", "seg_layers") else [v])
    hdr = struct.pack(vlt3._HDR_FMT, *flat).ljust(vlt3.HEADER_SIZE - 4, b"\0")
    hdr += struct.pack("<I", zlib.crc32(hdr))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(hdr)
        for e in entries:
            f.write(struct.pack(
                vlt3._ENTRY_FMT, e["id"], e["dtype"], e["ndim"], *e["dims"], 0,
                e["offset"], e["nbytes"], e["crc"], e["name"].encode()[:63],
            ))
        f.write(b"\0" * (tok_off - f.tell()))
        f.write(tok_blob)
        for e, b in zip(entries, blobs):
            f.write(b"\0" * (e["offset"] - f.tell()))
            f.write(b)
        assert f.tell() == file_size

    # Full read-back validation (header CRC, all tensor CRCs, NaN/Inf scan, shape checks)
    vf = vlt3.read_vlt3(tmp_path, check_crc=True)
    os.replace(tmp_path, out_path)
    return {
        "path": out_path,
        "file_size_mb": file_size / (1024 ** 2),
        "n_tensors": len(entries),
        "tokenizer_kb": len(tok_blob) / 1024,
        "header": vf.header,
    }
