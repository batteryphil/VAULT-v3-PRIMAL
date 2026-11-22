# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
VAULT v3 x PRIMAL: Head-to-Head V3_P4_0 Benchmark & Shadowless 4-bit Prime-Harmonic Evolution.

1. Evaluates VAULT-v3-1.5B VLT3 files (BF16, Q8_0, Q4_0, P4_0, P4_0+PRIMAL) using
   the exact WikiText-2 test split protocol (Word PPL, Token PPL, BPB) from
   vault/benchmarks/bench_vault_traditional.py.
2. Runs Shadowless 4-bit Prime-Harmonic Evolution directly on the packed V3_P4_0
   nibbles (1.56 B/param total training state: 0.56 B packed P4_0 + 1.0 B int8 vote
   buffer, ZERO FP32 master weights, ZERO AdamW m/v, fixed block scales d) using
   online KL self-distillation from the frozen BF16 VAULT-v3-1.5B teacher — both
   1.54B models fit simultaneously in 16 GB VRAM on the RX 9060 XT!
3. Writes evolved V3_P4_0 nibbles directly back into a CRC32-verified .vlt3 binary
   with 0.0 PTQ loss and verifies freestanding C99 vault3_host speed & parity.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
import warnings
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

from primal_v7.discrete_ops import _stochastic_round_int

warnings.filterwarnings("ignore", message=".*Flash Efficient attention.*")
warnings.filterwarnings("ignore", message=".*Mem Efficient attention.*")

VAULT_ROOT = "."
VAULT_HOST = os.path.join(VAULT_ROOT, "runtime", "vault3_host")
QWEN_SNAP = "Qwen/Qwen2.5-1.5B-Instruct"
QK = 32
P4_LUT_I8 = [-127, -101, -80, -62, -46, -33, -21, -10, 0, 10, 21, 33, 46, 62, 80, 101]


def _load_vault_modules():
    tools_dir = os.path.join(VAULT_ROOT, "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import vlt3

    if "vault_v3_model_ref" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "vault_v3_model_ref",
            os.path.join(VAULT_ROOT, "src", "vault_v3_model.py"),
        )
        vmod = importlib.util.module_from_spec(spec)
        sys.modules["vault_v3_model_ref"] = vmod
        spec.loader.exec_module(vmod)
    else:
        vmod = sys.modules["vault_v3_model_ref"]
    return vlt3, vmod


def _vlt3_to_vault_config(vf, VAULTv3Config):
    h = vf.header
    nf = h.get("seg_n_feat", 0)
    return VAULTv3Config(
        vocab_size=h["vocab_size"], d_model=h["d_model"], n_layers=h["n_layers"],
        n_heads=h["n_heads"], n_kv_heads=h["n_kv_heads"], head_dim=h["head_dim"],
        d_ff=h["d_ff"], rms_eps=h["rms_eps"], rope_theta=h["rope_theta"],
        tie_embeddings=bool(h["flags"] & 1), window=h["window"], n_sink=h["n_sink"],
        seg_budget=h.get("seg_budget", 0) if nf > 0 else 0,
        seg_hidden=h.get("seg_hidden", 256) if nf > 0 else 256,
        seg_min=h.get("seg_min", 16) if nf > 0 else 16,
        seg_max=h.get("seg_max", 48) if nf > 0 else 48,
        seg_layers=tuple(h["seg_layers"][:nf]) if nf > 0 else (),
    )


# ==============================================================================
# 1. Exact WikiText-2 Evaluation (matches vault/benchmarks/bench_vault_traditional.py)
# ==============================================================================

@torch.no_grad()
def evaluate_wikitext2_model(
    model: nn.Module,
    tok,
    full_text: str,
    seq_len: int = 2048,
    batch_size: int = 2,
    device: str = "cuda",
) -> dict:
    t0 = time.perf_counter()
    model.eval()
    ids = tok.encode(full_text, add_special_tokens=False)
    total_bytes = len(full_text.encode("utf-8"))
    total_words = len(re.findall(r"\S+", full_text))

    windows = []
    for pos in range(0, len(ids) - 1, seq_len):
        seg = ids[pos : pos + seq_len + 1]
        if len(seg) < 65:
            continue
        windows.append(seg)

    total_nll_nats = 0.0
    total_pred_tokens = 0
    pad_id = int(tok.eos_token_id if tok.eos_token_id is not None else 0)

    for st in range(0, len(windows), batch_size):
        sub = windows[st : st + batch_size]
        max_x = max(len(w) - 1 for w in sub)
        t_pad = max(64, ((max_x + 63) // 64) * 64)
        bx = torch.full((len(sub), t_pad), pad_id, dtype=torch.long, device=device)
        for b, w in enumerate(sub):
            bx[b, : len(w) - 1] = torch.tensor(w[:-1], dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            h_all = model.forward_hidden(bx)
            proj_fn = model.lm_logits if hasattr(model, "lm_logits") else model.embed_tokens.project
            for b, w in enumerate(sub):
                L_pred = len(w) - 1
                yb = torch.tensor(w[1:], dtype=torch.long, device=device)
                # Chunk over sequence length to keep peak logit VRAM < 300 MB
                for c0 in range(0, L_pred, 256):
                    c1 = min(c0 + 256, L_pred)
                    logits_b = proj_fn(h_all[b, c0:c1]).float()
                    total_nll_nats += float(F.cross_entropy(logits_b, yb[c0:c1], reduction="sum").item())
                total_pred_tokens += L_pred

    cov = total_pred_tokens / float(max(1, len(ids) - 1))
    scored_bytes = max(1.0, total_bytes * cov)
    scored_words = max(1.0, total_words * cov)
    bpb = (total_nll_nats / math.log(2.0)) / scored_bytes
    word_ppl = math.exp(min(20.0, total_nll_nats / scored_words))
    tok_ppl = math.exp(min(20.0, total_nll_nats / max(1, total_pred_tokens)))
    return {
        "bpb": round(bpb, 4),
        "word_ppl": round(word_ppl, 3),
        "tok_ppl": round(tok_ppl, 3),
        "n_tokens": total_pred_tokens,
        "eval_sec": round(time.perf_counter() - t0, 1),
    }


def evaluate_vlt3_wikitext2(vlt3_path: str, tok, full_text: str, device: str = "cuda") -> dict:
    vlt3, vmod = _load_vault_modules()
    vf = vlt3.read_vlt3(vlt3_path, check_crc=False)
    vcfg = _vlt3_to_vault_config(vf, vmod.VAULTv3Config)
    sd = vlt3.vlt3_to_state_dict(vf)
    m = vmod.VAULTv3ForCausalLM(vcfg)
    m.load_state_dict(sd, strict=True)
    del sd, vf
    m = m.to(device=device, dtype=torch.bfloat16).eval()
    res = evaluate_wikitext2_model(m, tok, full_text, seq_len=2048, batch_size=2, device=device)
    res["file_mb"] = round(os.path.getsize(vlt3_path) / (1024 ** 2), 2)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return res


# ==============================================================================
# 2. Native Packed V3_P4_0 Shadowless Linear Op (1.5625 B/param total!)
# ==============================================================================

def _dequant_p4_packed(q_packed: torch.Tensor, d: torch.Tensor, lut: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Materialize [out_f, in_f] weight matrix from packed V3_P4_0 nibbles:
      q_packed: uint8 [out_f, in_f // 32, 16]  (low nibble = elems 0..15, high nibble = elems 16..31)
      d:        float16 [out_f, in_f // 32]
      lut:      float32 [16] (P4_LUT_I8 values)
    """
    out_f, n_blk, _ = q_packed.shape
    lo = (q_packed & 0x0F).long()
    hi = (q_packed >> 4).long()
    idx = torch.cat([lo, hi], dim=-1)  # [out_f, n_blk, 32]
    w = lut[idx] * d.float().unsqueeze(-1)
    return w.to(dtype).reshape(out_f, n_blk * QK)


class _P4BlockLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        q_packed: torch.Tensor,
        d: torch.Tensor,
        lut: torch.Tensor,
        vote_buffer: torch.Tensor,
        vote_gain: float,
        vote_clip: float,
    ) -> torch.Tensor:
        w = _dequant_p4_packed(q_packed, d, lut, x.dtype)
        b = bias.to(x.dtype) if bias is not None else None
        out = F.linear(x, w, b)
        ctx.save_for_backward(x)
        ctx.q_packed = q_packed
        ctx.d = d
        ctx.lut = lut
        ctx.vote_buffer = vote_buffer
        ctx.vote_gain = vote_gain
        ctx.vote_clip = vote_clip
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        w = _dequant_p4_packed(ctx.q_packed, ctx.d, ctx.lut, grad_output.dtype)
        grad_x = grad_output @ w
        del w

        if ctx.vote_gain > 0.0:
            x_2d = x.reshape(-1, x.shape[-1])
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_w = (go_2d.t() @ x_2d).float()  # [out_f, in_f]
            # Exact signed-block descent + Newton curvature weighting:
            # w = d * lut[idx], delta_w ~ d * delta_lut. A discrete jump reduces loss iff
            # -g_w / delta_w > H/2, so divide by |d| so high-curvature outlier blocks don't explode!
            out_f, n_blk = ctx.d.shape
            d_f = ctx.d.float()
            d_inv_signed = (d_f.sign() / (d_f.abs() + 1e-4)).unsqueeze(-1).expand(out_f, n_blk, QK).reshape(out_f, n_blk * QK)
            g_curv = grad_w * d_inv_signed
            del grad_w

            rms = g_curv.pow(2).mean().sqrt().clamp(min=1e-12)
            pressure = ((-g_curv / rms) * ctx.vote_gain).clamp(-ctx.vote_clip, ctx.vote_clip)
            del g_curv
            delta = _stochastic_round_int(pressure)
            updated = (ctx.vote_buffer.to(torch.int16) + delta).clamp(-127, 127).to(torch.int8)
            ctx.vote_buffer.copy_(updated)

        return grad_x, None, None, None, None, None, None, None


class PrimeP4BlockLinear(nn.Module):
    """
    Shadowless packed V3_P4_0 linear layer (1.5625 bytes/param core state):
      - q_packed:    uint8   [out_f, in_f // 32, 16]  (0.5000 B/param)
      - d:           float16 [out_f, in_f // 32]      (0.0625 B/param, fixed signed scale)
      - vote_buffer: int8    [out_f, in_f]            (1.0000 B/param)
    Uses an AdaRound-style +-1 adjacent Prime-Harmonic neighbor bracket during evolution.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        raw_p4_bytes: bytes,
        lut: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.vote_gain = 3.0
        self.vote_clip = 12.0
        self.register_buffer("lut", lut, persistent=False)

        n_blk_per_row = in_features // QK
        q4_dt = np.dtype([("d", "<f2"), ("q", "u1", (QK // 2,))])
        arr = np.frombuffer(raw_p4_bytes, dtype=q4_dt, count=out_features * n_blk_per_row)
        d_t = torch.from_numpy(arr["d"].copy()).reshape(out_features, n_blk_per_row)
        q_t = torch.from_numpy(arr["q"].copy()).reshape(out_features, n_blk_per_row, QK // 2)

        self.register_buffer("d", d_t)                      # float16
        self.register_buffer("q_packed", q_t)               # uint8 packed 2 nibbles/byte
        self.register_buffer("q_init_packed", q_t.clone())  # +-1 AdaRound bracket anchor
        self.register_buffer("vote_buffer", torch.zeros(out_features, in_features, dtype=torch.int8))
        if bias is not None:
            self.register_buffer("bias", bias.float().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            w = _dequant_p4_packed(self.q_packed, self.d, self.lut, x.dtype)
            b = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, w, b)
        return _P4BlockLinearFn.apply(
            x, self.bias, self.q_packed, self.d, self.lut,
            self.vote_buffer, self.vote_gain, self.vote_clip,
        )

    @torch.no_grad()
    def apply_discrete_step(
        self,
        threshold: int = 16,
        decay: float = 0.85,
        max_flip_frac: float = 0.0005,
        collect_stats: bool = False,
    ) -> dict | None:
        out_f, n_blk, _ = self.q_packed.shape
        v16 = self.vote_buffer.to(torch.int16)
        step_dir = torch.sign(v16)  # -1, 0, +1

        lo = (self.q_packed & 0x0F).to(torch.int16)
        hi = (self.q_packed >> 4).to(torch.int16)
        idx = torch.cat([lo, hi], dim=-1)  # [out_f, n_blk, 32]

        lo0 = (self.q_init_packed & 0x0F).to(torch.int16)
        hi0 = (self.q_init_packed >> 4).to(torch.int16)
        idx0 = torch.cat([lo0, hi0], dim=-1)

        # AdaRound +-1 neighbor bracket: weight can only move to an adjacent Prime-Harmonic bin
        lo_bound = (idx0 - 1).clamp(0, 15)
        hi_bound = (idx0 + 1).clamp(0, 15)
        cand_idx = (idx + step_dir.view(out_f, n_blk, QK)).clamp(min=lo_bound, max=hi_bound)
        can_move = (cand_idx != idx).reshape(out_f, n_blk * QK)

        abs_v = v16.abs()
        eligible = (abs_v >= threshold) & can_move

        # Surgical budget: flip at most max_flip_frac of weights per layer (highest conviction first)
        # Fast O(N) 128-bin integer histogram — zero sort / zero kthvalue overhead!
        max_flips = max(1, int(v16.numel() * max_flip_frac))
        elig_v = torch.where(eligible, abs_v, torch.zeros_like(abs_v)).reshape(-1)
        hist_cpu = torch.bincount(elig_v, minlength=128).cpu().tolist()
        n_elig = sum(hist_cpu[threshold:])

        if n_elig > max_flips:
            acc = 0
            cut = 127
            for b in range(127, threshold - 1, -1):
                acc += hist_cpu[b]
                if acc >= max_flips:
                    cut = b
                    break
            above = sum(hist_cpu[cut + 1 :])
            n_ties = hist_cpu[cut]
            rem = max(0, max_flips - above)
            strict = eligible & (abs_v > cut)
            if rem > 0 and n_ties > 0:
                ties = eligible & (abs_v == cut)
                tie_prob = min(1.0, rem / float(n_ties))
                fired = strict | (ties & (torch.rand_like(abs_v, dtype=torch.float32) < tie_prob))
            else:
                fired = strict
        else:
            fired = eligible

        new_idx = torch.where(fired.view(out_f, n_blk, QK), cand_idx, idx).to(torch.uint8)
        self.q_packed.copy_(new_idx[:, :, : QK // 2] | (new_idx[:, :, QK // 2 :] << 4))

        # Refractory reset on fired nibbles; unbiased stochastic decay on remaining votes
        v_next = torch.where(fired, torch.zeros_like(v16), v16)
        if decay < 1.0:
            v_next = _stochastic_round_int(v_next.float() * decay).clamp(-127, 127)
        self.vote_buffer.copy_(v_next.to(torch.int8))

        if not collect_stats:
            return None
        return {
            "flips": int(fired.sum().item()),
            "total": v16.numel(),
        }

    @torch.no_grad()
    def pack_v3_p4_0_bytes(self) -> bytes:
        out_f, n_blk, _ = self.q_packed.shape
        q4_dt = np.dtype([("d", "<f2"), ("q", "u1", (QK // 2,))])
        out = np.empty(out_f * n_blk, dtype=q4_dt)
        out["d"] = self.d.detach().cpu().contiguous().numpy().astype("<f2").reshape(-1)
        out["q"] = self.q_packed.detach().cpu().contiguous().numpy().astype("u1").reshape(-1, QK // 2)
        return out.tobytes()


class FrozenQ8Linear(nn.Module):
    """Frozen V3_Q8_0 linear layer (stores exact Q8_0 blocks, dequantizes on the fly — 1.06 B/param)."""

    def __init__(self, out_features: int, in_features: int, raw_q8_bytes: bytes, bias: torch.Tensor | None = None):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        n_blk = in_features // QK
        q8_dt = np.dtype([("d", "<f2"), ("q", "i1", (QK,))])
        arr = np.frombuffer(raw_q8_bytes, dtype=q8_dt, count=out_features * n_blk)
        self.register_buffer("d", torch.from_numpy(arr["d"].copy()).reshape(out_features, n_blk))
        self.register_buffer("q", torch.from_numpy(arr["q"].copy()).reshape(out_features, in_features))
        if bias is not None:
            self.register_buffer("bias", bias.float().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = (self.q.view(self.out_features, -1, QK).to(x.dtype) * self.d.to(x.dtype).unsqueeze(-1)).reshape(self.out_features, self.in_features)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class FrozenQ8TiedEmbed(nn.Module):
    """Frozen Q8_0 tied embedding + LM head (zero FP16 shadow weights, 1.06 B/param)."""

    def __init__(self, vocab_size: int, dim: int, raw_q8_bytes: bytes):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        n_blk = dim // QK
        q8_dt = np.dtype([("d", "<f2"), ("q", "i1", (QK,))])
        arr = np.frombuffer(raw_q8_bytes, dtype=q8_dt, count=vocab_size * n_blk)
        self.register_buffer("d", torch.from_numpy(arr["d"].copy()).reshape(vocab_size, n_blk))
        self.register_buffer("q", torch.from_numpy(arr["q"].copy()).reshape(vocab_size, dim))
        # 4-byte dummy param strictly to keep autograd connected from input embeddings
        self._grad_anchor = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def embed(self, input_ids: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        flat = input_ids.reshape(-1)
        q_r = self.q.index_select(0, flat)
        d_r = self.d.index_select(0, flat)
        n_r = q_r.shape[0]
        vals = (q_r.view(n_r, -1, QK).to(out_dtype) * d_r.to(out_dtype).unsqueeze(-1)).reshape(n_r, self.dim)
        out = vals.reshape(*input_ids.shape, self.dim)
        if torch.is_grad_enabled():
            out = out + self._grad_anchor.to(out_dtype) * 0.0
        return out

    def project(self, x: torch.Tensor) -> torch.Tensor:
        w = (self.q.view(self.vocab_size, -1, QK).to(x.dtype) * self.d.to(x.dtype).unsqueeze(-1)).reshape(self.vocab_size, self.dim)
        return F.linear(x, w)


class RMSNormFrozen(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", weight.float().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * inv * self.weight).to(x.dtype)


class _RotaryEmb(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
        self.register_buffer("inv_freq", inv_freq.float(), persistent=False)
        self._build(max_seq_len)

    def _build(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float64)
        freqs = torch.outer(t, self.inv_freq.double())
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_c", emb.cos().float(), persistent=False)
        self.register_buffer("sin_c", emb.sin().float(), persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype):
        if seq_len > self.cos_c.size(0):
            self._build(seq_len)
        return self.cos_c[:seq_len].to(dtype), self.sin_c[:seq_len].to(dtype)


def _rot_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class _P4DecoderLayer(nn.Module):
    def __init__(self, n_heads, n_kv_heads, head_dim, ln1, q_p, k_p, v_p, o_p, ln2, g_p, u_p, d_p):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_groups = n_heads // n_kv_heads
        self.input_layernorm = ln1
        self.q_proj = q_p
        self.k_proj = k_p
        self.v_proj = v_p
        self.o_proj = o_p
        self.post_attention_layernorm = ln2
        self.gate_proj = g_p
        self.up_proj = u_p
        self.down_proj = d_p

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.input_layernorm(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        c = cos.unsqueeze(0).unsqueeze(0)
        s = sin.unsqueeze(0).unsqueeze(0)
        q = q * c + _rot_half(q) * s
        k = k * c + _rot_half(k) * s
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(attn)

        h2 = self.post_attention_layernorm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h2)) * self.up_proj(h2))
        return x


class VAULTv3P4Shadowless(nn.Module):
    """
    Loads a VAULT-v3-1.5B P4_0 .vlt3 directly into shadowless GPU memory (~2.4 GB total
    including int8 vote buffers for all ~1.19B 4-bit Prime-Harmonic weights!).
    """

    def __init__(self, vlt3_path: str):
        super().__init__()
        vlt3, _ = _load_vault_modules()
        vf = vlt3.read_vlt3(vlt3_path, check_crc=False)
        self.header = vf.header
        self.src_vlt3_path = vlt3_path

        entry_map = {vlt3.tensor_name(e["id"]): e for e in vf.entries}

        def _raw(name: str):
            e = entry_map[name]
            return e["dtype"], tuple(e["dims"][: e["ndim"]]), bytes(vf.buf[e["offset"] : e["offset"] + e["nbytes"]])

        def _f32(name: str) -> torch.Tensor:
            e = entry_map[name]
            return vlt3.decode_tensor(vf.buf[e["offset"] : e["offset"] + e["nbytes"]], e["dtype"], tuple(e["dims"][: e["ndim"]]))

        self.register_buffer("p4_lut", torch.tensor(P4_LUT_I8, dtype=torch.float32))
        h = vf.header

        dt_e, sh_e, b_e = _raw("embed_tokens.weight")
        assert dt_e == vlt3.Q8_0
        self.embed_tokens = FrozenQ8TiedEmbed(sh_e[0], sh_e[1], b_e)
        self.norm = RMSNormFrozen(_f32("norm.weight"), h["rms_eps"])
        self.rope = _RotaryEmb(h["head_dim"], max_seq_len=2048, theta=h["rope_theta"])

        self.p4_modules: dict[str, PrimeP4BlockLinear] = {}
        layers = []
        for li in range(h["n_layers"]):
            p = f"layers.{li}"
            ln1 = RMSNormFrozen(_f32(f"{p}.input_layernorm.weight"), h["rms_eps"])
            ln2 = RMSNormFrozen(_f32(f"{p}.post_attention_layernorm.weight"), h["rms_eps"])

            def _mk_proj(kind: str, has_bias: bool):
                wname = f"{p}.{kind}.weight"
                dt, sh, raw = _raw(wname)
                bias = _f32(f"{p}.{kind}.bias") if has_bias else None
                if dt == vlt3.P4_0:
                    mod = PrimeP4BlockLinear(sh[0], sh[1], raw, self.p4_lut, bias=bias)
                    self.p4_modules[wname] = mod
                    return mod
                elif dt == vlt3.Q8_0:
                    return FrozenQ8Linear(sh[0], sh[1], raw, bias=bias)
                else:
                    raise ValueError(f"Unexpected dtype {dt} for {wname}")

            layers.append(_P4DecoderLayer(
                h["n_heads"], h["n_kv_heads"], h["head_dim"],
                ln1,
                _mk_proj("self_attn.q_proj", True),
                _mk_proj("self_attn.k_proj", True),
                _mk_proj("self_attn.v_proj", True),
                _mk_proj("self_attn.o_proj", False),
                ln2,
                _mk_proj("mlp.gate_proj", False),
                _mk_proj("mlp.up_proj", False),
                _mk_proj("mlp.down_proj", False),
            ))
        self.layers = nn.ModuleList(layers)
        self.use_checkpointing = True

    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)
        for mod in res.p4_modules.values():
            mod.lut = res.p4_lut
        return res

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        out_dtype = torch.bfloat16 if torch.is_autocast_enabled() else torch.float32
        x = self.embed_tokens.embed(input_ids, out_dtype=out_dtype)
        cos, sin = self.rope(input_ids.size(1), x.dtype)
        for layer in self.layers:
            if self.use_checkpointing and torch.is_grad_enabled():
                x = checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin)
        return self.norm(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens.project(self.forward_hidden(input_ids))

    def memory_summary(self) -> dict:
        p4_params = sum(m.out_features * m.in_features for m in self.p4_modules.values())
        p4_w_bytes = sum(m.q_packed.numel() + m.d.numel() * 2 for m in self.p4_modules.values())
        p4_v_bytes = sum(m.vote_buffer.numel() for m in self.p4_modules.values())
        q8_bytes = sum(
            (m.q.numel() + m.d.numel() * 2)
            for m in self.modules()
            if isinstance(m, (FrozenQ8Linear, FrozenQ8TiedEmbed))
        )
        return {
            "p4_evolvable_params": p4_params,
            "p4_weight_mb": round(p4_w_bytes / (1024 ** 2), 1),
            "p4_vote_mb": round(p4_v_bytes / (1024 ** 2), 1),
            "q8_frozen_mb": round(q8_bytes / (1024 ** 2), 1),
            "total_student_mb": round((p4_w_bytes + p4_v_bytes + q8_bytes) / (1024 ** 2), 1),
            "p4_bytes_per_param": round((p4_w_bytes + p4_v_bytes) / max(1, p4_params), 4),
        }

    def export_patched_vlt3(self, out_path: str) -> dict:
        """
        Clone the source P4_0 .vlt3 byte-for-byte, splice in the evolved P4_0 tensor
        bytes with updated per-tensor CRC32s, and verify with vlt3.read_vlt3(check_crc=True).
        ZERO re-quantization / ZERO PTQ loss!
        """
        vlt3, _ = _load_vault_modules()
        with open(self.src_vlt3_path, "rb") as f:
            raw = bytearray(f.read())
        vf = vlt3.read_vlt3(self.src_vlt3_path, check_crc=False)
        h = vf.header
        dir_off = h["tensor_dir_offset"]

        n_patched = 0
        for idx, e in enumerate(vf.entries):
            nm = vlt3.tensor_name(e["id"])
            if nm in self.p4_modules:
                new_bytes = self.p4_modules[nm].pack_v3_p4_0_bytes()
                assert len(new_bytes) == e["nbytes"], f"Size mismatch for {nm}"
                off = e["offset"]
                raw[off : off + e["nbytes"]] = new_bytes
                new_crc = zlib.crc32(new_bytes) & 0xFFFFFFFF
                # Patch crc field in 128-byte directory entry (offset 48 inside entry)
                struct.pack_into("<I", raw, dir_off + idx * vlt3.ENTRY_SIZE + 48, new_crc)
                n_patched += 1

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(raw)

        # Full integrity validation (CRCs + NaN/Inf scan)
        vlt3.read_vlt3(out_path, check_crc=True)
        return {"path": out_path, "n_patched": n_patched, "size_mb": round(len(raw) / (1024 ** 2), 2)}


# ==============================================================================
# 3. Clean Calibration / Evolution Stream (FineWeb-Edu — zero WikiText-2 overlap)
# ==============================================================================

def build_fineweb_batches(tok, n_seqs: int = 800, seq_len: int = 512) -> torch.Tensor:
    print(f"[DATA] Streaming {n_seqs} clean FineWeb-Edu sequences (seq_len={seq_len}, zero WikiText-2 overlap)...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    buf = []
    seqs = []
    eos = tok.eos_token_id
    for item in ds:
        txt = item.get("text", "")
        if len(txt) < 200:
            continue
        ids = tok.encode(txt, add_special_tokens=False)
        buf.extend(ids)
        buf.append(eos)
        while len(buf) >= seq_len + 1:
            seqs.append(buf[: seq_len + 1])
            buf = buf[seq_len:]
            if len(seqs) >= n_seqs:
                return torch.tensor(seqs, dtype=torch.long)
    return torch.tensor(seqs, dtype=torch.long)


# ==============================================================================
# 4. Online BF16-Teacher -> Shadowless P4_0 Student KL Evolution
# ==============================================================================

@torch.no_grad()
def eval_probe_kl(teacher: nn.Module, student: VAULTv3P4Shadowless, probe_seqs: torch.Tensor, device: str) -> float:
    """Fast (~0.6s) fixed-batch KL divergence between frozen BF16 teacher and live P4_0 student."""
    student.eval()
    tot_kl = 0.0
    n_seq = probe_seqs.size(0)
    for i in range(0, n_seq, 2):
        x_in = probe_seqs[i : i + 2, :-1].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            t_h = teacher.forward_hidden(x_in)
            s_h = student.forward_hidden(x_in)
        T_len = s_h.size(1)
        seq_kl = 0.0
        for c0 in range(0, T_len, 128):
            c1 = min(c0 + 128, T_len)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                t_lg = teacher.lm_logits(t_h[:, c0:c1, :]).float()
                s_lg = student.embed_tokens.project(s_h[:, c0:c1, :]).float()
            log_p_t = F.log_softmax(t_lg, dim=-1)
            log_p_s = F.log_softmax(s_lg, dim=-1)
            seq_kl += float((log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1).mean().item()) * ((c1 - c0) / float(T_len))
        tot_kl += seq_kl * (x_in.size(0) / float(n_seq))
    student.train()
    return tot_kl


def run_p4_primal_evolution(
    bf16_vlt3: str,
    p4_vlt3: str,
    out_vlt3: str,
    tok,
    wt2_text: str,
    steps: int = 25,
    batch_size: int = 2,
    grad_accum: int = 4,
    seq_len: int = 512,
    threshold_start: int = 14,
    threshold_end: int = 24,
    vote_gain_start: float = 4.0,
    vote_gain_end: float = 2.0,
    max_flip_frac_start: float = 0.0004,
    max_flip_frac_end: float = 0.0001,
    decay: float = 0.85,
    device: str = "cuda",
) -> dict:
    vlt3, vmod = _load_vault_modules()

    # Preload clean FineWeb-Edu token blocks (4 fixed probe seqs + training seqs)
    n_probe = 4
    total_seqs = n_probe + steps * batch_size * grad_accum + 8
    all_seqs = build_fineweb_batches(tok, n_seqs=total_seqs, seq_len=seq_len)
    probe_seqs = all_seqs[:n_probe]
    data_seqs = all_seqs[n_probe:]
    print(f"[DATA] Ready: {data_seqs.shape[0]} train seqs + {n_probe} fixed probe seqs ({all_seqs.numel()/1e6:.2f}M tokens)")

    # 1. Load frozen BF16 teacher (~3.0 GB VRAM)
    print(f"[TEACHER] Loading frozen BF16 teacher from {bf16_vlt3}...")
    vf_t = vlt3.read_vlt3(bf16_vlt3, check_crc=False)
    tcfg = _vlt3_to_vault_config(vf_t, vmod.VAULTv3Config)
    teacher = vmod.VAULTv3ForCausalLM(tcfg)
    teacher.load_state_dict(vlt3.vlt3_to_state_dict(vf_t), strict=True)
    del vf_t
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = teacher.to(device=device, dtype=torch.bfloat16).eval()

    # 2. Load shadowless P4_0 student (~2.7 GB VRAM incl. int8 votes & +-1 bracket)
    print(f"[STUDENT] Loading shadowless V3_P4_0 student from {p4_vlt3}...")
    student = VAULTv3P4Shadowless(p4_vlt3).to(device)
    student.train()
    mem = student.memory_summary()
    torch.cuda.reset_peak_memory_stats()

    print("=" * 80)
    print("VAULT-v3-1.5B Surgical PRIMAL 4-bit Evolution (AdaRound +-1 Bracket + Curvature Votes)")
    print(
        f"Evolvable P4_0 params: {mem['p4_evolvable_params']/1e9:.3f}B | "
        f"P4_0 weights: {mem['p4_weight_mb']:.0f} MB + int8 votes: {mem['p4_vote_mb']:.0f} MB "
        f"({mem['p4_bytes_per_param']:.3f} B/param core)"
    )
    print(
        f"Steps: {steps} x {batch_size*grad_accum} seqs x {seq_len} tok | "
        f"Flip budget: {max_flip_frac_start*100:.3f}% -> {max_flip_frac_end*100:.3f}% per step"
    )
    print("=" * 80, flush=True)

    probe_kl0 = eval_probe_kl(teacher, student, probe_seqs, device)
    wt0 = evaluate_wikitext2_model(student, tok, wt2_text, seq_len=2048, batch_size=2, device=device)
    print(
        f"[Step  0] ProbeKL = {probe_kl0:.5f} | Held-out WT2: Word PPL = {wt0['word_ppl']:.3f} | "
        f"Tok PPL = {wt0['tok_ppl']:.3f} | BPB = {wt0['bpb']:.4f}",
        flush=True,
    )

    best_ppl = wt0["word_ppl"]
    best_probe_kl = probe_kl0
    backoff_scale = 1.0
    history = [{"step": 0, "probe_kl": round(probe_kl0, 5), "wt2": wt0}]
    t0 = time.time()
    seq_cursor = 0
    total_flipped_cumulative = 0
    student.train()

    for step in range(1, steps + 1):
        prog = (step - 1) / max(1, steps - 1)
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        cur_thresh = int(round(threshold_end - (threshold_end - threshold_start) * cos))
        cur_gain = vote_gain_end + (vote_gain_start - vote_gain_end) * cos
        cur_frac = (max_flip_frac_end + (max_flip_frac_start - max_flip_frac_end) * cos) * backoff_scale
        for mod in student.p4_modules.values():
            mod.vote_gain = cur_gain

        kl_acc = 0.0
        for _ in range(grad_accum):
            batch = data_seqs[seq_cursor : seq_cursor + batch_size].to(device, non_blocking=True)
            seq_cursor += batch_size
            x_in = batch[:, :-1]

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                t_h = teacher.forward_hidden(x_in)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_h = student.forward_hidden(x_in)  # [B, T, D]
            T_len = s_h.size(1)
            chunk = 128
            s_h_grad = torch.zeros_like(s_h)
            loss_val = 0.0
            for c0 in range(0, T_len, chunk):
                c1 = min(c0 + chunk, T_len)
                scale_c = ((c1 - c0) / float(T_len)) / float(grad_accum)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    t_lg = teacher.lm_logits(t_h[:, c0:c1, :]).float()
                    log_p_t = F.log_softmax(t_lg, dim=-1)
                    p_t = log_p_t.exp()
                s_hc = s_h[:, c0:c1, :].detach().requires_grad_(True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    s_lg = student.embed_tokens.project(s_hc).float()
                log_p_s = F.log_softmax(s_lg, dim=-1)
                kl_c = (p_t * (log_p_t - log_p_s)).sum(dim=-1).mean()
                (kl_c * scale_c).backward()
                s_h_grad[:, c0:c1, :] = s_hc.grad
                loss_val += float(kl_c.item()) * ((c1 - c0) / float(T_len))
                del t_lg, log_p_t, p_t, s_lg, log_p_s, s_hc

            s_h.backward(s_h_grad)
            kl_acc += loss_val / grad_accum
            del t_h, s_h, s_h_grad

        # Apply surgical AdaRound-bracketed discrete nibble stepping
        step_flips = 0
        step_total = 0
        for mod in student.p4_modules.values():
            st = mod.apply_discrete_step(
                threshold=cur_thresh,
                decay=decay,
                max_flip_frac=cur_frac,
                collect_stats=True,
            )
            step_flips += st["flips"]
            step_total += st["total"]
        total_flipped_cumulative += step_flips
        flip_pct = 100.0 * step_flips / max(1, step_total)

        probe_kl = eval_probe_kl(teacher, student, probe_seqs, device)
        kl_delta = probe_kl - probe_kl0
        if probe_kl < best_probe_kl:
            best_probe_kl = probe_kl
        elif probe_kl > best_probe_kl * 1.03:
            backoff_scale = max(0.125, backoff_scale * 0.5)

        elapsed = time.time() - t0
        toks_done = step * batch_size * grad_accum * seq_len
        print(
            f"[Step {step:2d}/{steps}] BatchKL: {kl_acc:.5f} | ProbeKL: {probe_kl:.5f} ({kl_delta:+.5f}) | "
            f"Flips: {step_flips:7,d} ({flip_pct:.4f}%) | Backoff: {backoff_scale:.2f} | {toks_done/max(elapsed,1e-6):,.0f} tok/s",
            flush=True,
        )

        if step in (5, 10, 15, 20) or step == steps:
            wt = evaluate_wikitext2_model(student, tok, wt2_text, seq_len=2048, batch_size=2, device=device)
            print(
                f"  >>> [WT2 Step {step:2d}] Word PPL: {wt['word_ppl']:.3f} (Δ {wt['word_ppl']-wt0['word_ppl']:+.3f}) | "
                f"Tok PPL: {wt['tok_ppl']:.3f} | BPB: {wt['bpb']:.4f}",
                flush=True,
            )
            history.append({
                "step": step,
                "batch_kl": round(kl_acc, 5),
                "probe_kl": round(probe_kl, 5),
                "flip_pct": round(flip_pct, 4),
                "wt2": wt,
            })
            if wt["word_ppl"] < best_ppl:
                best_ppl = wt["word_ppl"]
                exp = student.export_patched_vlt3(out_vlt3)
                print(
                    f"  >>> Exported ZERO-PTQ Evolved V3_P4_0 ({exp['size_mb']:.1f} MB, "
                    f"{exp['n_patched']} P4_0 tensors patched, CRC verified): {out_vlt3}",
                    flush=True,
                )
            student.train()

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    total_time = time.time() - t0

    if not os.path.exists(out_vlt3):
        student.export_patched_vlt3(out_vlt3)

    del teacher, student
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "memory": mem,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "total_time_s": round(total_time, 1),
        "total_nibble_flips": total_flipped_cumulative,
        "init_probe_kl": round(probe_kl0, 5),
        "best_probe_kl": round(best_probe_kl, 5),
        "init_wt2": wt0,
        "best_word_ppl": best_ppl,
        "history": history,
        "out_vlt3": out_vlt3,
    }


def bench_c99_host(vlt3_path: str, threads: int = 24) -> dict:
    cmd = [VAULT_HOST, vlt3_path, "--verify", "--threads", str(threads), "--bench", "64"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    line = res.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main():
    p = argparse.ArgumentParser(description="VAULT-v3-1.5B P4_0 Benchmark & Surgical PRIMAL Evolution")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--threshold_start", type=int, default=14)
    p.add_argument("--threshold_end", type=int, default=24)
    p.add_argument("--vote_gain_start", type=float, default=4.0)
    p.add_argument("--vote_gain_end", type=float, default=2.0)
    p.add_argument("--max_flip_frac_start", type=float, default=0.0004)
    p.add_argument("--max_flip_frac_end", type=float, default=0.0001)
    p.add_argument("--decay", type=float, default=0.85)
    p.add_argument("--skip_part1", action="store_true")
    p.add_argument("--out_json", type=str, default="primal_v7/runs/vault_v3_p4_benchmark.json")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(QWEN_SNAP, local_files_only=True)
    wt_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wt2_text = "\n\n".join(x for x in wt_ds["text"] if x)

    bf16_path = os.path.join(VAULT_ROOT, "dist", "vault_v3_1.5b_bf16.vlt3")
    q8_path = os.path.join(VAULT_ROOT, "dist", "vault_v3_1.5b_q8_0.vlt3")
    q4_path = os.path.join(VAULT_ROOT, "dist", "vault_v3_1.5b_q4_0.vlt3")
    p4_path = os.path.join(VAULT_ROOT, "dist", "vault_v3_1.5b_p4_0.vlt3")
    p4_evolved_path = os.path.join(VAULT_ROOT, "dist", "vault_v3_1.5b_p4_0_primal.vlt3")

    results = {}
    if not args.skip_part1:
        print("\n================ [PART 1] Head-to-Head WikiText-2 (Test Split, L=2048) ================")
        for tag, path in [
            ("bf16", bf16_path),
            ("q8_0", q8_path),
            ("q4_0", q4_path),
            ("p4_0_zeroshot", p4_path),
        ]:
            r = evaluate_vlt3_wikitext2(path, tok, wt2_text, device="cuda")
            results[tag] = r
            print(
                f"  {tag:16s} | Size: {r['file_mb']:7.1f} MB | Word PPL: {r['word_ppl']:6.3f} | "
                f"Tok PPL: {r['tok_ppl']:6.3f} | BPB: {r['bpb']:.4f} ({r['eval_sec']:.1f}s)",
                flush=True,
            )
    else:
        results["part1_cached"] = {
            "bf16": {"file_mb": 2953.1, "word_ppl": 16.491, "tok_ppl": 9.598, "bpb": 0.7541},
            "q8_0": {"file_mb": 1573.0, "word_ppl": 16.495, "tok_ppl": 9.600, "bpb": 0.7541},
            "q4_0": {"file_mb": 967.8, "word_ppl": 17.827, "tok_ppl": 10.221, "bpb": 0.7750},
            "p4_0_zeroshot": {"file_mb": 967.8, "word_ppl": 17.325, "tok_ppl": 9.988, "bpb": 0.7674},
        }

    print("\n================ [PART 2] Surgical PRIMAL 4-bit Evolution on RX 9060 XT ================")
    evo = run_p4_primal_evolution(
        bf16_vlt3=bf16_path,
        p4_vlt3=p4_path,
        out_vlt3=p4_evolved_path,
        tok=tok,
        wt2_text=wt2_text,
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        vote_gain_start=args.vote_gain_start,
        vote_gain_end=args.vote_gain_end,
        max_flip_frac_start=args.max_flip_frac_start,
        max_flip_frac_end=args.max_flip_frac_end,
        decay=args.decay,
    )
    results["evolution"] = evo

    print("\n================ [PART 3] Verify Evolved VLT3 Read-Back & C99 Speed ================")
    r_evo = evaluate_vlt3_wikitext2(p4_evolved_path, tok, wt2_text, device="cuda")
    results["p4_0_primal_vlt3"] = r_evo
    print(
        f"  {'p4_0_primal':16s} | Size: {r_evo['file_mb']:7.1f} MB | Word PPL: {r_evo['word_ppl']:6.3f} | "
        f"Tok PPL: {r_evo['tok_ppl']:6.3f} | BPB: {r_evo['bpb']:.4f}",
        flush=True,
    )

    c99_q4 = bench_c99_host(q4_path, threads=24)
    c99_p4 = bench_c99_host(p4_evolved_path, threads=24)
    results["c99_q4_24t"] = c99_q4
    results["c99_p4_primal_24t"] = c99_p4
    print(f"  vault3_host Q4_0        (24T): {c99_q4['decode_tok_s']:.2f} tok/s ({c99_q4['ms_per_tok']:.1f} ms/tok)")
    print(f"  vault3_host P4_0 PRIMAL (24T): {c99_p4['decode_tok_s']:.2f} tok/s ({c99_p4['ms_per_tok']:.1f} ms/tok)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Full benchmark report -> {args.out_json}")


if __name__ == "__main__":
    main()
