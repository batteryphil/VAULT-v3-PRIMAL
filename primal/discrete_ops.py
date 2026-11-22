# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
Shadowless Discrete Neural Layers & Vote-Gate Optimization Engine (PRIMAL v7).

Design principles:
1. ZERO shadow weights: 2D weight matrices exist ONLY as discrete LUT indices
   (uint8 for <=8-bit, int16 for >8-bit). Float weights are materialized on the
   fly during forward/backward and immediately discarded.
2. ZERO persistent .grad buffers: backward computes grad_w, immediately folds
   RMS-normalized stochastically-rounded votes into the int8 vote_buffer in-place,
   and frees grad_w before continuing to earlier layers.
3. ZERO AdamW optimizer states (m, v) on any discrete layer: total state+weight
   memory is 2 bytes/param at 8-bit (1B uint8 index + 1B int8 vote) — an 8x
   reduction vs standard BF16 weights + FP32 master + BF16 grad + FP32 AdamW (16B/param).
4. Exact descent sign: negative gradient casts positive votes (+index), positive
   gradient casts negative votes (-index). Verified by unit tests.
5. Unbiased stochastic rounding on BOTH vote injection AND multiplicative vote
   decay, eliminating the int8 truncation deadzone / starvation bug from v6.15.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .manifolds import quantize_to_lut


@torch.no_grad()
def _stochastic_round_int(x: torch.Tensor) -> torch.Tensor:
    """Unbiased stochastic rounding of float tensor to nearest int16."""
    fl = torch.floor(x)
    prob = x - fl
    return (fl + (torch.rand_like(prob) < prob).to(x.dtype)).to(torch.int16)


@torch.no_grad()
def _accumulate_votes(
    grad_w: torch.Tensor,
    vote_buffer: torch.Tensor,
    vote_gain: float,
    vote_clip: float,
    row_indices: torch.Tensor | None = None,
) -> None:
    """
    Convert a weight gradient tensor into RMS-normalized, stochastically-rounded
    int8 descent votes (-grad_w) and accumulate into vote_buffer in-place.
    """
    g = grad_w.float()
    rms = g.pow(2).mean().sqrt().clamp(min=1e-12)
    # Descent direction is -g: if dL/dw > 0 we want w (and LUT index) to DECREASE.
    pressure = (-g / rms) * vote_gain
    pressure = pressure.clamp(-vote_clip, vote_clip)
    delta = _stochastic_round_int(pressure)

    if row_indices is None:
        updated = (vote_buffer.to(torch.int16) + delta).clamp(-127, 127).to(torch.int8)
        vote_buffer.copy_(updated)
    else:
        cur = vote_buffer.index_select(0, row_indices).to(torch.int16)
        updated = (cur + delta).clamp(-127, 127).to(torch.int8)
        vote_buffer.index_copy_(0, row_indices, updated)


class _PrimeLinearFn(torch.autograd.Function):
    """
    Memory-optimal shadowless linear op:
    - Forward materializes w = lut[indices] * scale, runs F.linear, discards w.
    - Backward rematerializes w, computes grad_x and grad_w, immediately folds
      grad_w into int8 vote_buffer, and discards grad_w.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        indices: torch.Tensor,
        scale: torch.Tensor,
        lut: torch.Tensor,
        vote_buffer: torch.Tensor,
        vote_gain: float,
        vote_clip: float,
    ) -> torch.Tensor:
        w = (lut[indices.long()] * scale).to(x.dtype)
        out = F.linear(x, w)
        ctx.save_for_backward(x)
        ctx.indices = indices
        ctx.scale = scale
        ctx.lut = lut
        ctx.vote_buffer = vote_buffer
        ctx.vote_gain = vote_gain
        ctx.vote_clip = vote_clip
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        indices = ctx.indices
        scale = ctx.scale
        lut = ctx.lut
        vote_buffer = ctx.vote_buffer

        # 1. Input gradient (rematerialize w on the fly, never stored across layers)
        w = (lut[indices.long()] * scale).to(grad_output.dtype)
        grad_x = grad_output @ w
        del w

        # 2. Weight gradient -> immediate int8 vote accumulation -> free grad_w
        if ctx.vote_gain > 0.0:
            x_2d = x.reshape(-1, x.shape[-1])
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_w = go_2d.t() @ x_2d
            # Account for scale sign (scale is positive, keep general for safety)
            if scale.numel() == 1:
                if scale.item() < 0:
                    grad_w.neg_()
            else:
                grad_w.mul_(scale.sign().to(grad_w.dtype))
            _accumulate_votes(grad_w, vote_buffer, ctx.vote_gain, ctx.vote_clip)
            del grad_w

        return grad_x, None, None, None, None, None, None


class _PrimeEmbedFn(torch.autograd.Function):
    """
    Shadowless discrete embedding lookup with sparse unique-row backward voting.
    Takes a 1-element dummy grad_anchor (requires_grad=True) so PyTorch records
    this node in the autograd graph during training without any shadow weights.
    """

    @staticmethod
    def forward(
        ctx,
        grad_anchor: torch.Tensor,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
        scale: torch.Tensor,
        lut: torch.Tensor,
        vote_buffer: torch.Tensor,
        vote_gain: float,
        vote_clip: float,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        flat_ids = input_ids.reshape(-1)
        row_idx = indices.index_select(0, flat_ids).long()
        row_vals = lut[row_idx]
        if scale.numel() == 1:
            row_vals = row_vals * scale
        else:
            row_vals = row_vals * scale.index_select(0, flat_ids)
        out = row_vals.to(out_dtype).reshape(*input_ids.shape, indices.shape[1])

        ctx.save_for_backward(input_ids)
        ctx.scale = scale
        ctx.vote_buffer = vote_buffer
        ctx.vote_gain = vote_gain
        ctx.vote_clip = vote_clip
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input_ids,) = ctx.saved_tensors
        if ctx.vote_gain > 0.0:
            flat_ids = input_ids.reshape(-1)
            uniq_ids, inv = torch.unique(flat_ids, return_inverse=True)
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).float()
            uniq_grad = torch.zeros(
                uniq_ids.numel(), grad_flat.shape[1],
                device=grad_flat.device, dtype=torch.float32,
            )
            uniq_grad.index_add_(0, inv, grad_flat)
            if ctx.scale.numel() == 1:
                if ctx.scale.item() < 0:
                    uniq_grad.neg_()
            else:
                uniq_grad.mul_(ctx.scale.index_select(0, uniq_ids).sign().float())
            _accumulate_votes(
                uniq_grad,
                ctx.vote_buffer,
                ctx.vote_gain,
                ctx.vote_clip,
                row_indices=uniq_ids,
            )
        return None, None, None, None, None, None, None, None, None


class _DiscreteLayerMixin:
    """Shared vote-step logic for PrimeLinear and PrimeTiedEmbedding."""

    lut: torch.Tensor
    indices: torch.Tensor
    vote_buffer: torch.Tensor
    scale: torch.Tensor
    vote_gain: float
    vote_clip: float

    @property
    def num_discrete_params(self) -> int:
        return self.indices.numel()

    @property
    def max_lut_idx(self) -> int:
        return self.lut.numel() - 1

    @property
    def zero_lut_idx(self) -> int:
        return self.lut.numel() // 2

    @torch.no_grad()
    def apply_discrete_step(
        self,
        threshold: int = 12,
        decay: float = 0.85,
        weight_decay: float = 0.0,
        max_stride: int = 1,
        error_feedback: bool = False,
    ) -> dict:
        """
        Apply consensus vote-gated index stepping on GPU.

        - Parameters with |vote_buffer| >= threshold step in sign(vote_buffer)
          by up to max_stride LUT indices.
        - Optional discrete weight decay gently biases nonzero weights back
          toward the exact-zero center index before decay.
        - Remaining votes undergo unbiased stochastic multiplicative decay so
          weak consistent gradients never die in an int8 truncation deadzone.
        """
        v = self.vote_buffer
        abs_v = v.abs()
        fired = abs_v >= threshold
        n_flips = int(fired.sum().item())
        n_total = v.numel()

        if n_flips > 0:
            v_fired = v[fired].to(torch.int16)
            step_dir = torch.sign(v_fired)
            if max_stride > 1:
                stride = (v_fired.abs() // threshold).clamp(1, max_stride)
                delta_idx = step_dir * stride
            else:
                delta_idx = step_dir

            cur_idx = self.indices[fired].to(torch.int32)
            new_idx = (cur_idx + delta_idx.to(torch.int32)).clamp(0, self.max_lut_idx)
            self.indices[fired] = new_idx.to(self.indices.dtype)

            if error_feedback:
                residual = v_fired - delta_idx * threshold
                v[fired] = residual.clamp(-127, 127).to(torch.int8)
            else:
                v[fired] = 0

        # Discrete weight decay: cast tiny probabilistic pull toward zero_lut_idx
        if weight_decay > 0.0:
            w_val = self.lut[self.indices.long()]
            # Pull toward 0: opposite sign of current weight, scaled by weight_decay
            wd_vote = _stochastic_round_int(-w_val * weight_decay)
            v.copy_((v.to(torch.int16) + wd_vote).clamp(-127, 127).to(torch.int8))

        # Unbiased stochastic multiplicative decay on surviving votes
        if decay < 1.0:
            v_f = v.float() * decay
            v_dec = _stochastic_round_int(v_f).clamp(-127, 127).to(torch.int8)
            v.copy_(v_dec)

        # Lightweight saturation / zero-sparsity stats
        sat_frac = float((v.abs() >= 120).float().mean().item())
        return {
            "flips": n_flips,
            "total": n_total,
            "flip_pct": 100.0 * n_flips / max(1, n_total),
            "sat_pct": 100.0 * sat_frac,
        }


class PrimeLinear(_DiscreteLayerMixin, nn.Module):
    """
    100% shadowless discrete linear layer (zero FP32 master weights, zero bias).
    Stores only:
      - indices:     uint8 (or int16 for >8-bit) [out_features, in_features]
      - vote_buffer: int8                        [out_features, in_features]
      - scale:       fixed scalar float32 buffer (sets physical dynamic range)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lut: torch.Tensor,
        init_std: float | None = None,
        scale_headroom: float = 5.0,
        vote_gain: float = 4.0,
        vote_clip: float = 16.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.vote_gain = vote_gain
        self.vote_clip = vote_clip

        idx_dtype = torch.uint8 if lut.numel() <= 256 else torch.int16
        self.register_buffer("lut", lut, persistent=False)

        if init_std is None:
            init_std = 1.0 / math.sqrt(in_features)
        # Fixed physical scale giving ample headroom above init_std while utilizing
        # the dense bulk of the Prime-Harmonic LUT immediately at step 0.
        scale_val = max(scale_headroom * init_std, 0.08)
        self.register_buffer("scale", torch.tensor(scale_val, dtype=torch.float32))

        with torch.no_grad():
            raw_w = torch.randn(out_features, in_features, dtype=torch.float32) * init_std
            norm_w = (raw_w / scale_val).clamp(-1.0, 1.0)
            init_idx = quantize_to_lut(norm_w, lut.cpu()).to(idx_dtype)

        self.register_buffer("indices", init_idx)
        self.register_buffer("vote_buffer", torch.zeros(out_features, in_features, dtype=torch.int8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            w = (self.lut[self.indices.long()] * self.scale).to(x.dtype)
            return F.linear(x, w)
        return _PrimeLinearFn.apply(
            x,
            self.indices,
            self.scale,
            self.lut,
            self.vote_buffer,
            self.vote_gain,
            self.vote_clip,
        )


class PrimeTiedEmbedding(_DiscreteLayerMixin, nn.Module):
    """
    Weight-tied shadowless discrete Token Embedding + LM Head.
    Both input embedding lookup and output logit projection share the exact same
    uint8 LUT indices and int8 vote_buffer.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        lut: torch.Tensor,
        init_std: float = 0.02,
        scale_headroom: float = 8.0,
        vote_gain: float = 4.0,
        vote_clip: float = 16.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.vote_gain = vote_gain
        self.vote_clip = vote_clip

        idx_dtype = torch.uint8 if lut.numel() <= 256 else torch.int16
        self.register_buffer("lut", lut, persistent=False)

        # Embeddings grow larger than hidden weights over training; give ample logit headroom
        scale_val = max(scale_headroom * init_std, 0.35)
        self.register_buffer("scale", torch.tensor(scale_val, dtype=torch.float32))

        # Single 4-byte dummy parameter strictly to keep PyTorch autograd connected
        # from the embedding forward pass without storing any shadow weights.
        self._grad_anchor = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        with torch.no_grad():
            raw_w = torch.randn(vocab_size, dim, dtype=torch.float32) * init_std
            norm_w = (raw_w / scale_val).clamp(-1.0, 1.0)
            init_idx = quantize_to_lut(norm_w, lut.cpu()).to(idx_dtype)

        self.register_buffer("indices", init_idx)
        self.register_buffer("vote_buffer", torch.zeros(vocab_size, dim, dtype=torch.int8))

    def embed(self, input_ids: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if not torch.is_grad_enabled():
            flat = input_ids.reshape(-1)
            vals = (self.lut[self.indices.index_select(0, flat).long()] * self.scale).to(out_dtype)
            return vals.reshape(*input_ids.shape, self.dim)
        return _PrimeEmbedFn.apply(
            self._grad_anchor,
            input_ids,
            self.indices,
            self.scale,
            self.lut,
            self.vote_buffer,
            self.vote_gain,
            self.vote_clip,
            out_dtype,
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            w = (self.lut[self.indices.long()] * self.scale).to(x.dtype)
            return F.linear(x, w)
        return _PrimeLinearFn.apply(
            x,
            self.indices,
            self.scale,
            self.lut,
            self.vote_buffer,
            self.vote_gain,
            self.vote_clip,
        )


class PrimeStepEngine:
    """
    Clean, zero-hack discrete optimizer controller for PRIMAL v7.

    Cosine-anneals vote_gain downward and couples vote decay upward (SNR-matched
    consensus window), acting as an exact integer learning-rate schedule: rapid
    exploration early, crisp low-noise crystallization late.
    Also updates the tiny 1D RMSNorm gain vectors (<0.05% of total params).
    """

    def __init__(
        self,
        model: nn.Module,
        total_steps: int,
        warmup_steps: int = 100,
        threshold_start: int = 20,
        threshold_end: int = 24,
        vote_gain_start: float = 4.0,
        vote_gain_end: float = 1.0,
        vote_clip: float = 16.0,
        decay: float = 0.88,
        decay_end: float | None = 0.95,
        weight_decay: float = 0.02,
        max_stride: int = 1,
        error_feedback: bool = True,
        norm_lr: float = 1e-3,
    ):
        self.model = model
        self.total_steps = max(1, total_steps)
        self.warmup_steps = warmup_steps
        self.threshold_start = threshold_start
        self.threshold_end = threshold_end
        self.vote_gain_start = vote_gain_start
        self.vote_gain_end = vote_gain_end
        self.vote_clip = vote_clip
        self.decay_start = decay
        self.decay_end = decay if decay_end is None else decay_end
        self.weight_decay = weight_decay
        self.max_stride = max_stride
        self.error_feedback = error_feedback
        self.step_count = 0

        self.discrete_modules: list[_DiscreteLayerMixin] = [
            m for m in model.modules() if isinstance(m, _DiscreteLayerMixin)
        ]

        # Collect the tiny 1D norm parameters (excluding the 1-element dummy grad_anchor)
        norm_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "_grad_anchor" not in n
        ]
        self.norm_optimizer = (
            torch.optim.AdamW(norm_params, lr=norm_lr, weight_decay=0.0)
            if norm_params else None
        )
        self.norm_base_lr = norm_lr
        self._apply_schedule(0)

    def _schedule_values(self, step: int) -> tuple[int, float, float, float]:
        """Compute (threshold, vote_gain, decay, norm_lr) at current step."""
        if step < self.warmup_steps and self.warmup_steps > 0:
            frac = (step + 1) / self.warmup_steps
            gain = self.vote_gain_start * (0.25 + 0.75 * frac)
            thresh = self.threshold_start
            dec = self.decay_start
            nlr = self.norm_base_lr * frac
            return thresh, gain, dec, nlr

        progress = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1.0 -> 0.0
        thresh = int(round(self.threshold_end - (self.threshold_end - self.threshold_start) * cos))
        gain = self.vote_gain_end + (self.vote_gain_start - self.vote_gain_end) * cos
        dec = self.decay_end - (self.decay_end - self.decay_start) * cos
        nlr = self.norm_base_lr * (0.1 + 0.9 * cos)
        return max(1, thresh), gain, dec, nlr

    def _apply_schedule(self, step: int) -> tuple[int, float, float]:
        thresh, gain, dec, nlr = self._schedule_values(step)
        for m in self.discrete_modules:
            m.vote_gain = gain
            m.vote_clip = self.vote_clip
        if self.norm_optimizer is not None:
            for pg in self.norm_optimizer.param_groups:
                pg["lr"] = nlr
        return thresh, gain, dec

    def zero_grad(self) -> None:
        if self.norm_optimizer is not None:
            self.norm_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self) -> dict:
        thresh, gain, dec = self._apply_schedule(self.step_count)

        total_flips = 0
        total_params = 0
        max_sat = 0.0
        for m in self.discrete_modules:
            st = m.apply_discrete_step(
                threshold=thresh,
                decay=dec,
                weight_decay=self.weight_decay,
                max_stride=self.max_stride,
                error_feedback=self.error_feedback,
            )
            total_flips += st["flips"]
            total_params += st["total"]
            max_sat = max(max_sat, st["sat_pct"])

        if self.norm_optimizer is not None:
            norm_params = [p for pg in self.norm_optimizer.param_groups for p in pg["params"]]
            torch.nn.utils.clip_grad_norm_(norm_params, 1.0)
            self.norm_optimizer.step()

        self.step_count += 1
        next_thresh, _, _ = self._apply_schedule(self.step_count)
        return {
            "flips": total_flips,
            "total_discrete": total_params,
            "flip_pct": 100.0 * total_flips / max(1, total_params),
            "max_sat_pct": max_sat,
            "threshold": thresh,
            "vote_gain": round(gain, 3),
            "decay": round(dec, 4),
            "next_threshold": next_thresh,
        }

    def memory_report(self) -> dict:
        """Exact byte-level accounting of model weights + optimizer state."""
        discrete_idx_bytes = sum(m.indices.numel() * m.indices.element_size() for m in self.discrete_modules)
        discrete_vote_bytes = sum(m.vote_buffer.numel() * m.vote_buffer.element_size() for m in self.discrete_modules)
        discrete_params = sum(m.indices.numel() for m in self.discrete_modules)

        norm_params = sum(
            p.numel() for n, p in self.model.named_parameters() if "_grad_anchor" not in n
        )
        # Norm params in fp32 + AdamW (m, v, grad) = 16 bytes each
        norm_total_bytes = norm_params * 16
        lut_bytes = self.discrete_modules[0].lut.numel() * 4 if self.discrete_modules else 0

        total_params = discrete_params + norm_params
        total_bytes = discrete_idx_bytes + discrete_vote_bytes + norm_total_bytes + lut_bytes
        adamw_equiv_bytes = total_params * 16  # 2B bf16 + 4B fp32 master + 2B grad + 8B AdamW (m,v)

        return {
            "total_params": total_params,
            "discrete_params": discrete_params,
            "discrete_pct": 100.0 * discrete_params / max(1, total_params),
            "norm_params": norm_params,
            "index_mb": discrete_idx_bytes / (1024 ** 2),
            "vote_mb": discrete_vote_bytes / (1024 ** 2),
            "norm_and_lut_mb": (norm_total_bytes + lut_bytes) / (1024 ** 2),
            "total_state_mb": total_bytes / (1024 ** 2),
            "bytes_per_param": total_bytes / max(1, total_params),
            "adamw_equiv_mb": adamw_equiv_bytes / (1024 ** 2),
            "compression_ratio": adamw_equiv_bytes / max(1, total_bytes),
        }
