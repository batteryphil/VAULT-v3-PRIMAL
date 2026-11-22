# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VLT3: flat binary checkpoint format for VAULT v3 (see docs/VLT3_FORMAT.md).
#
#   [ header : 4096 bytes, fixed struct, CRC32 ]
#   [ tensor directory : n_tensors x 128-byte fixed entries ]
#   [ tokenizer blob : fixed-layout tables, no JSON ]
#   [ tensor data : each tensor 64-byte aligned ]
#
# No JSON, no strings that must be parsed. Every offset/size is validated before use.

import struct
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

MAGIC = b"VLT3"
VERSION = 1
HEADER_SIZE = 4096
ENTRY_SIZE = 128
ALIGN = 64
ENDIAN_TAG = 0x01020304

# dtypes
F32, F16, BF16, Q8_0, Q4_0, P4_0, T1_58_G32, T1_58_G64 = 0, 1, 2, 8, 9, 10, 11, 12
DTYPE_NAMES = {
    F32: "F32", F16: "F16", BF16: "BF16", Q8_0: "Q8_0", Q4_0: "Q4_0", P4_0: "P4_0",
    T1_58_G32: "T1_58_G32", T1_58_G64: "T1_58_G64",
}
QK = 32  # block size for Q8_0 / Q4_0 / P4_0 / T1_58_G32
QK64 = 64  # block size for T1_58_G64 (1.875 bpw including FP16 scale)

# 16-level Prime-Harmonic int8 LUT (p_k^-0.40 cumulative prime reciprocals for first 8 primes,
# exact 0 at index 8; decoded in 1 cycle via AVX2 VPSHUFB in runtime/vault3.c).
P4_LUT_I8 = np.array([-127, -101, -80, -62, -46, -33, -21, -10, 0, 10, 21, 33, 46, 62, 80, 101], dtype=np.int8)
P4_LUT_F32 = P4_LUT_I8.astype(np.float32)

# Base-3 5-trits-per-byte lookup table (3^5 = 243 <= 256): u_k = q_k + 1 in {0, 1, 2}
_TRIT5_DEC = np.zeros((256, 5), dtype=np.int8)
for _bi in range(243):
    _v = _bi
    for _k in range(5):
        _TRIT5_DEC[_bi, _k] = (_v % 3) - 1
        _v //= 3

# tensor ids: (layer + 1) << 8 | kind for per-layer tensors, kind only for global tensors
G_EMBED, G_NORM, G_LM_HEAD = 1, 2, 3
G_SEG_LN_W, G_SEG_LN_B, G_SEG_W1, G_SEG_B1, G_SEG_W2, G_SEG_B2 = 4, 5, 6, 7, 8, 9
GLOBAL_NAMES = {
    G_EMBED: "embed_tokens.weight",
    G_NORM: "norm.weight",
    G_LM_HEAD: "lm_head.weight",
    G_SEG_LN_W: "seg_scorer.ln.weight",
    G_SEG_LN_B: "seg_scorer.ln.bias",
    G_SEG_W1: "seg_scorer.fc1.weight",
    G_SEG_B1: "seg_scorer.fc1.bias",
    G_SEG_W2: "seg_scorer.fc2.weight",
    G_SEG_B2: "seg_scorer.fc2.bias",
}
GLOBAL_IDS = {v: k for k, v in GLOBAL_NAMES.items()}
L_KINDS = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.q_proj.bias",
    "self_attn.k_proj.weight", "self_attn.k_proj.bias",
    "self_attn.v_proj.weight", "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
]
L_KIND_ID = {n: i + 1 for i, n in enumerate(L_KINDS)}
MATMUL_KINDS = {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}

# header layout (little-endian). Offsets are part of the spec.
# Bytes 112..163 hold legacy Kenyon bank fields (all zero in Kenyon-free VLT3 files).
# Bytes 260..295 hold Segment Memory config (0 in SWA+sinks-only files).
_HDR_FMT = "<4sIII Q II QQQQQQ IIIIIIII I 8I I fff IIIII II 64s I IIIII 4I"
_HDR_FIELDS = [
    "magic", "version", "header_size", "endian_tag",
    "file_size", "alignment", "n_tensors",
    "tensor_dir_offset", "tensor_dir_size", "tokenizer_offset", "tokenizer_size", "data_offset", "data_size",
    "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim", "d_ff", "window",
    "n_banks", "primes", "n_cells", "rms_eps", "rope_theta", "vault_eps",
    "bos_id", "eos_id", "im_start_id", "im_end_id", "pad_id", "weight_format", "flags", "model_name",
    "n_sink",
    "seg_budget", "seg_hidden", "seg_min", "seg_max", "seg_n_feat", "seg_layers",
]
_ENTRY_FMT = "<III4II QQI12x64s"
assert struct.calcsize(_ENTRY_FMT) == ENTRY_SIZE, struct.calcsize(_ENTRY_FMT)
assert struct.calcsize(_HDR_FMT) <= HEADER_SIZE - 4


def tensor_id(name: str) -> int:
    if name in GLOBAL_IDS:
        return GLOBAL_IDS[name]
    assert name.startswith("layers."), f"unknown tensor name: {name}"
    _, li, rest = name.split(".", 2)
    return ((int(li) + 1) << 8) | L_KIND_ID[rest]


def tensor_name(tid: int) -> str:
    if tid < 256:
        return GLOBAL_NAMES[tid]
    return f"layers.{(tid >> 8) - 1}.{L_KINDS[(tid & 0xFF) - 1]}"


# ==============================================================================
# Quantization (GGML Q8_0 / Q4_0 + Prime-Harmonic P4_0 + Base-3 Ternary 1.58-Bit)
# ==============================================================================

Q8_DT = np.dtype([("d", "<f2"), ("q", "i1", (QK,))])          # 34 bytes / 32 weights
Q4_DT = np.dtype([("d", "<f2"), ("q", "u1", (QK // 2,))])     # 18 bytes / 32 weights
T1_58_G32_DT = np.dtype([("d", "<f2"), ("q", "u1", (7,))])    #  9 bytes / 32 weights (2.250 bpw w/ scale)
T1_58_G64_DT = np.dtype([("d", "<f2"), ("q", "u1", (13,))])   # 15 bytes / 64 weights (1.875 bpw w/ scale)


def quantize_q8_0(w: np.ndarray) -> bytes:
    x = w.astype(np.float32).reshape(-1, QK)
    amax = np.abs(x).max(axis=1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    q = np.clip(np.rint(x * inv[:, None]), -127, 127).astype(np.int8)
    out = np.empty(x.shape[0], dtype=Q8_DT)
    out["d"] = d.astype(np.float16)
    out["q"] = q
    return out.tobytes()


def dequantize_q8_0(buf: bytes, n: int) -> np.ndarray:
    b = np.frombuffer(buf, dtype=Q8_DT, count=n // QK)
    return (b["q"].astype(np.float32) * b["d"].astype(np.float32)[:, None]).reshape(-1)


def quantize_q4_0(w: np.ndarray) -> bytes:
    x = w.astype(np.float32).reshape(-1, QK)
    n_blk = x.shape[0]
    out = np.empty(n_blk, dtype=Q4_DT)
    ratios = np.array([-1.0 / 8.0, -0.98 / 8.0, -0.96 / 8.0, -0.94 / 8.0, -0.92 / 8.0, -0.89 / 8.0, 1.0 / 7.5, 0.96 / 7.5], dtype=np.float32)
    step = 262144
    for b0 in range(0, n_blk, step):
        xb = x[b0 : b0 + step]
        m = xb.shape[0]
        idx = np.abs(xb).argmax(axis=1)
        mx = xb[np.arange(m), idx]
        best_err = np.full(m, np.inf, dtype=np.float32)
        best_d = np.zeros(m, dtype=np.float16)
        best_q = np.full((m, QK), 8, dtype=np.uint8)
        nz = mx != 0.0
        if np.any(nz):
            for r in ratios:
                d_f16 = (mx * r).astype(np.float16)
                d_f32 = d_f16.astype(np.float32)
                inv = np.where(d_f32 != 0.0, 1.0 / np.where(d_f32 != 0.0, d_f32, 1.0), 0.0)
                qi = np.clip(np.rint(xb * inv[:, None] + 8.0), 0.0, 15.0).astype(np.uint8)
                rec = (qi.astype(np.float32) - 8.0) * d_f32[:, None]
                err = np.square(rec - xb).sum(axis=1)
                imp = nz & (err < best_err)
                if np.any(imp):
                    best_err[imp] = err[imp]
                    best_d[imp] = d_f16[imp]
                    best_q[imp] = qi[imp]
        out["d"][b0 : b0 + step] = best_d
        out["q"][b0 : b0 + step] = best_q[:, : QK // 2] | (best_q[:, QK // 2 :] << 4)
    return out.tobytes()


def dequantize_q4_0(buf: bytes, n: int) -> np.ndarray:
    b = np.frombuffer(buf, dtype=Q4_DT, count=n // QK)
    lo = (b["q"] & 0x0F).astype(np.int32) - 8
    hi = (b["q"] >> 4).astype(np.int32) - 8
    q = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (q * b["d"].astype(np.float32)[:, None]).reshape(-1)


def quantize_p4_0(w: np.ndarray) -> bytes:
    """
    Clipped-MSE 4-bit Prime-Harmonic block quantization (18 bytes / 32 weights).
    Uses GPU acceleration automatically when available for fast 1.5B+ export.
    """
    x_np = w.astype(np.float32).reshape(-1, QK)
    n_blk = x_np.shape[0]
    out = np.empty(n_blk, dtype=Q4_DT)
    max_neg = float(-P4_LUT_F32[0])   # 127.0
    max_pos = float(P4_LUT_F32[-1])   # 101.0
    ratios = [
        -1.00 / max_neg, -0.96 / max_neg, -0.92 / max_neg,
        -0.88 / max_neg, -0.84 / max_neg, -0.80 / max_neg,
        1.00 / max_pos, 0.94 / max_pos,
    ]
    use_gpu = torch.cuda.is_available() and n_blk >= 16384
    dev = "cuda" if use_gpu else "cpu"
    lut_t = torch.tensor(P4_LUT_F32, device=dev, dtype=torch.float32)
    mids_t = 0.5 * (lut_t[:-1] + lut_t[1:])
    step = 262144
    with torch.no_grad():
        for b0 in range(0, n_blk, step):
            xb = torch.from_numpy(x_np[b0 : b0 + step]).to(dev)
            m = xb.shape[0]
            mx = xb.gather(1, xb.abs().argmax(dim=1, keepdim=True)).squeeze(1)
            best_err = torch.full((m,), float("inf"), device=dev, dtype=torch.float32)
            best_d = torch.zeros(m, device=dev, dtype=torch.float16)
            best_q = torch.full((m, QK), 8, device=dev, dtype=torch.uint8)
            nz = mx != 0.0
            if nz.any():
                for r in ratios:
                    d_f16 = (mx * r).to(torch.float16)
                    d_f32 = d_f16.float()
                    inv = torch.where(d_f32 != 0.0, 1.0 / torch.where(d_f32 != 0.0, d_f32, torch.ones_like(d_f32)), torch.zeros_like(d_f32))
                    qi = torch.searchsorted(mids_t, xb * inv.unsqueeze(1)).to(torch.uint8)
                    rec = lut_t[qi.long()] * d_f32.unsqueeze(1)
                    err = (rec - xb).pow(2).sum(dim=1)
                    imp = nz & (err < best_err)
                    best_err = torch.where(imp, err, best_err)
                    best_d = torch.where(imp, d_f16, best_d)
                    best_q = torch.where(imp.unsqueeze(1), qi, best_q)
            packed = best_q[:, : QK // 2] | (best_q[:, QK // 2 :] << 4)
            out["d"][b0 : b0 + step] = best_d.cpu().numpy()
            out["q"][b0 : b0 + step] = packed.cpu().numpy()
    return out.tobytes()


def dequantize_p4_0(buf: bytes, n: int) -> np.ndarray:
    b = np.frombuffer(buf, dtype=Q4_DT, count=n // QK)
    lo = P4_LUT_F32[(b["q"] & 0x0F).astype(np.int32)]
    hi = P4_LUT_F32[(b["q"] >> 4).astype(np.int32)]
    q = np.concatenate([lo, hi], axis=1)
    return (q * b["d"].astype(np.float32)[:, None]).reshape(-1)


def pack_t1_58(q_i8: np.ndarray, d_f16: np.ndarray, group_size: int) -> bytes:
    """
    Losslessly pack exact ternary trits q_i8 in {-1, 0, +1} and FP16 block scales d_f16
    into V3_T1_58_G32 (group_size=32, 9B/blk) or V3_T1_58_G64 (group_size=64, 15B/blk).
    """
    assert group_size in (32, 64), f"unsupported ternary group_size={group_size}"
    q = np.clip(q_i8.astype(np.int8), -1, 1).reshape(-1, group_size)
    n_blk = q.shape[0]
    d = d_f16.astype("<f2").reshape(n_blk)
    n_bytes = 7 if group_size == 32 else 13
    pad_len = n_bytes * 5 - group_size  # 3 for G=32 (35-32), 1 for G=64 (65-64)
    if pad_len > 0:
        q_pad = np.zeros((n_blk, n_bytes * 5), dtype=np.uint8)
        q_pad[:, :group_size] = (q + 1).astype(np.uint8)
        q_pad[:, group_size:] = 1  # u = 1 => trit 0 in unused tail slots
    else:
        q_pad = (q + 1).astype(np.uint8)
    u5 = q_pad.reshape(n_blk, n_bytes, 5)
    pk = (
        u5[:, :, 0]
        + 3 * u5[:, :, 1]
        + 9 * u5[:, :, 2]
        + 27 * u5[:, :, 3]
        + 81 * u5[:, :, 4]
    ).astype(np.uint8)
    dt = T1_58_G32_DT if group_size == 32 else T1_58_G64_DT
    out = np.empty(n_blk, dtype=dt)
    out["d"] = d
    out["q"] = pk
    return out.tobytes()


def _lloyd_ternary_quantize(w: np.ndarray, group_size: int) -> Tuple[np.ndarray, np.ndarray]:
    x = w.astype(np.float32).reshape(-1, group_size)
    d = np.maximum(x.abs().mean(axis=1) if hasattr(x, "abs") else np.abs(x).mean(axis=1), 1e-8)
    # Initialize at 0.75 * max(|x|) so pre-quantized ternary tensors (w = q * d) recover exact (q, d) in 1 step!
    amax = np.abs(x).max(axis=1)
    d = np.where(amax > 0.0, amax, 0.0).astype(np.float32)
    for _ in range(4):
        inv = np.where(d > 0.0, 1.0 / np.where(d > 0.0, d, 1.0), 0.0)
        q = np.clip(np.rint(x * inv[:, None]), -1.0, 1.0)
        nz_cnt = (q != 0.0).sum(axis=1).astype(np.float32)
        num = (x * q).sum(axis=1)
        d = np.where(nz_cnt > 0.0, np.maximum(num / np.maximum(nz_cnt, 1.0), 0.0), 0.0)
    d_f16 = d.astype(np.float16)
    d_f32 = d_f16.astype(np.float32)
    inv = np.where(d_f32 > 0.0, 1.0 / np.where(d_f32 > 0.0, d_f32, 1.0), 0.0)
    q_i8 = np.clip(np.rint(x * inv[:, None]), -1.0, 1.0).astype(np.int8)
    return q_i8, d_f16


def quantize_t1_58_g32(w: np.ndarray) -> bytes:
    q_i8, d_f16 = _lloyd_ternary_quantize(w, 32)
    return pack_t1_58(q_i8, d_f16, 32)


def quantize_t1_58_g64(w: np.ndarray) -> bytes:
    q_i8, d_f16 = _lloyd_ternary_quantize(w, 64)
    return pack_t1_58(q_i8, d_f16, 64)


def dequantize_t1_58_g32(buf: bytes, n: int) -> np.ndarray:
    b = np.frombuffer(buf, dtype=T1_58_G32_DT, count=n // QK)
    trits = _TRIT5_DEC[b["q"]].reshape(-1, 35)[:, :32].astype(np.float32)
    return (trits * b["d"].astype(np.float32)[:, None]).reshape(-1)


def dequantize_t1_58_g64(buf: bytes, n: int) -> np.ndarray:
    b = np.frombuffer(buf, dtype=T1_58_G64_DT, count=n // QK64)
    trits = _TRIT5_DEC[b["q"]].reshape(-1, 65)[:, :64].astype(np.float32)
    return (trits * b["d"].astype(np.float32)[:, None]).reshape(-1)


def nbytes_for(dtype: int, n: int) -> int:
    if dtype == F32:
        return 4 * n
    if dtype in (F16, BF16):
        return 2 * n
    if dtype == Q8_0:
        assert n % QK == 0
        return (n // QK) * Q8_DT.itemsize
    if dtype in (Q4_0, P4_0):
        assert n % QK == 0
        return (n // QK) * Q4_DT.itemsize
    if dtype == T1_58_G32:
        assert n % QK == 0
        return (n // QK) * T1_58_G32_DT.itemsize
    if dtype == T1_58_G64:
        assert n % QK64 == 0
        return (n // QK64) * T1_58_G64_DT.itemsize
    raise ValueError(dtype)


def encode_tensor(t: torch.Tensor, dtype: int) -> bytes:
    a = t.detach().float().cpu().contiguous()
    if dtype == F32:
        return a.numpy().astype("<f4").tobytes()
    if dtype == F16:
        return a.numpy().astype("<f2").tobytes()
    if dtype == BF16:
        return a.to(torch.bfloat16).view(torch.int16).numpy().astype("<i2").tobytes()
    if dtype == Q8_0:
        return quantize_q8_0(a.numpy())
    if dtype == Q4_0:
        return quantize_q4_0(a.numpy())
    if dtype == P4_0:
        return quantize_p4_0(a.numpy())
    if dtype == T1_58_G32:
        return quantize_t1_58_g32(a.numpy())
    if dtype == T1_58_G64:
        return quantize_t1_58_g64(a.numpy())
    raise ValueError(dtype)


def decode_tensor(buf: bytes, dtype: int, shape: Tuple[int, ...]) -> torch.Tensor:
    n = int(np.prod(shape))
    if dtype == F32:
        a = np.frombuffer(buf, dtype="<f4", count=n).copy()
    elif dtype == F16:
        a = np.frombuffer(buf, dtype="<f2", count=n).astype(np.float32)
    elif dtype == BF16:
        raw = np.frombuffer(buf, dtype="<u2", count=n).astype(np.uint32) << 16
        a = raw.view(np.float32).copy()
    elif dtype == Q8_0:
        a = dequantize_q8_0(buf, n)
    elif dtype == Q4_0:
        a = dequantize_q4_0(buf, n)
    elif dtype == P4_0:
        a = dequantize_p4_0(buf, n)
    elif dtype == T1_58_G32:
        a = dequantize_t1_58_g32(buf, n)
    elif dtype == T1_58_G64:
        a = dequantize_t1_58_g64(buf, n)
    else:
        raise ValueError(dtype)
    return torch.from_numpy(a.reshape(shape))


# ==============================================================================
# Tokenizer blob (byte-level BPE, raw-byte token strings, merges as id triples)
# ==============================================================================

_TOK_HDR_FMT = "<4sIIIII QQQ QQQ"  # magic, version, n_vocab, n_merges, n_special, n_ranges, then 6 offsets/sizes
TOK_CAT_L, TOK_CAT_N = 1, 2


def _bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _unicode_ranges() -> List[Tuple[int, int, int]]:
    """Compact (lo, hi, cat) ranges for Unicode letters (L*) and numbers (N*), used by the pre-tokenizer."""
    import unicodedata
    ranges = []
    cur = None
    for cp in range(0x110000):
        c = unicodedata.category(chr(cp))
        cat = TOK_CAT_L if c[0] == "L" else (TOK_CAT_N if c[0] == "N" else 0)
        if cur and cur[2] == cat and cur[1] == cp - 1:
            cur[1] = cp
        else:
            if cur and cur[2]:
                ranges.append(tuple(cur))
            cur = [cp, cp, cat]
    if cur and cur[2]:
        ranges.append(tuple(cur))
    return ranges


def build_tokenizer_blob(hf_tokenizer_dir: str) -> Tuple[bytes, Dict[str, int]]:
    import json
    import os

    with open(os.path.join(hf_tokenizer_dir, "tokenizer.json"), encoding="utf-8") as f:
        tj = json.load(f)
    b2u = _bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}
    vocab: Dict[str, int] = tj["model"]["vocab"]
    added = {a["content"]: a["id"] for a in tj.get("added_tokens", [])}
    n_vocab = max(max(vocab.values()), max(added.values()) if added else 0) + 1
    tok_bytes: List[bytes] = [b""] * n_vocab
    for s, i in vocab.items():
        tok_bytes[i] = bytes(u2b[ch] for ch in s)
    for s, i in added.items():
        tok_bytes[i] = s.encode("utf-8")
    merges_raw = tj["model"]["merges"]
    merges = []
    for m in merges_raw:
        a, b = (m.split(" ", 1) if isinstance(m, str) else m)
        merges.append((vocab[a], vocab[b], vocab[a + b]))
    byte_tok = [vocab[b2u[b]] for b in range(256)]
    special = sorted(added.values())
    ranges = _unicode_ranges()

    pool = bytearray()
    table = np.zeros((n_vocab, 2), dtype="<u4")
    for i, tb in enumerate(tok_bytes):
        table[i] = (len(pool), len(tb))
        pool += tb
    sections = [
        table.tobytes(),
        bytes(pool),
        np.asarray(merges, dtype="<u4").tobytes(),
        np.asarray(special, dtype="<u4").tobytes(),
        np.asarray(ranges, dtype="<u4").tobytes(),
        np.asarray(byte_tok, dtype="<u4").tobytes(),
    ]
    head_size = 128
    offs = []
    cur = head_size
    for s in sections:
        cur = (cur + 7) & ~7
        offs.append(cur)
        cur += len(s)
    hdr = struct.pack("<4sIIIIII", b"VTOK", 1, n_vocab, len(merges), len(special), len(ranges), 0)
    hdr += struct.pack("<6Q", *offs)
    hdr = hdr.ljust(head_size, b"\0")
    blob = bytearray(hdr)
    for o, s in zip(offs, sections):
        blob += b"\0" * (o - len(blob))
        blob += s
    ids = {
        "bos": added.get("<|endoftext|>", added.get("<｜begin▁of▁sentence｜>", 0)),
        "eos": added.get("<|im_end|>", added.get("<｜end▁of▁sentence｜>", 0)),
        "im_start": added.get("<|im_start|>", added.get("<｜User｜>", 0)),
        "im_end": added.get("<|im_end|>", added.get("<｜Assistant｜>", 0)),
        "pad": added.get("<|endoftext|>", added.get("<｜end▁of▁sentence｜>", 0)),
    }
    return bytes(blob), ids


# ==============================================================================
# Writer
# ==============================================================================

def choose_dtype(name: str, fmt: str, n_layers: int = 0) -> int:
    if fmt == "f32":
        return F32
    if name in ("embed_tokens.weight", "lm_head.weight"):
        if fmt in ("t1_58_hybrid", "t1_58_g64"):
            return P4_0
        return {"bf16": BF16, "q8_0": Q8_0, "q4_0": Q8_0, "p4_0": Q8_0}[fmt]   # embed & output head stay >= 8-bit in 4/8-bit modes
    if name.startswith("layers."):
        _, li_s, kind = name.split(".", 2)
        li = int(li_s)
    else:
        li, kind = -1, name
    if kind in MATMUL_KINDS:
        if fmt == "t1_58_hybrid":
            if kind in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):
                return Q8_0
            if kind in ("self_attn.q_proj.weight", "self_attn.o_proj.weight"):
                return Q8_0 if (n_layers >= 8 and li in (0, 1, n_layers - 2, n_layers - 1)) else P4_0
            return T1_58_G32
        if fmt == "t1_58_g64":
            if kind in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):
                return Q8_0
            return T1_58_G64
        if fmt in ("q4_0", "p4_0"):
            # Keep 6:1 GQA KV projections (+11.6 MB) and boundary Q/O projections (+9.4 MB) in Q8_0
            if kind in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):
                return Q8_0
            if n_layers >= 8 and li in (0, 1, n_layers - 2, n_layers - 1) and kind in ("self_attn.q_proj.weight", "self_attn.o_proj.weight"):
                return Q8_0
        return {"bf16": BF16, "q8_0": Q8_0, "q4_0": Q4_0, "p4_0": P4_0}[fmt]
    return F32  # norms, biases, segment scorer LayerNorm/biases (small)


def write_vlt3(
    path: str,
    state: Dict[str, torch.Tensor],
    cfg,
    tokenizer_dir: str,
    fmt: str,
    model_name: str = "VAULT-v3",
    prepacked: Optional[Dict[str, Tuple[int, bytes]]] = None,
) -> Dict:
    names = sorted(state.keys(), key=tensor_id)
    entries = []
    blobs = []
    n_layers_cfg = int(getattr(cfg, "n_layers", 0))
    for nm in names:
        t = state[nm]
        if prepacked is not None and nm in prepacked:
            dt, data = prepacked[nm]
        else:
            dt = choose_dtype(nm, fmt, n_layers=n_layers_cfg)
            if dt in (Q8_0, Q4_0, P4_0, T1_58_G32) and t.shape[-1] % QK != 0:
                dt = BF16
            elif dt == T1_58_G64 and t.shape[-1] % QK64 != 0:
                dt = BF16
            data = encode_tensor(t, dt)
        assert len(data) == nbytes_for(dt, t.numel())
        dims = list(t.shape) + [1] * (4 - t.dim())
        entries.append(dict(id=tensor_id(nm), dtype=dt, ndim=t.dim(), dims=dims, nbytes=len(data), crc=zlib.crc32(data), name=nm))
        blobs.append(data)

    tok_blob, ids = build_tokenizer_blob(tokenizer_dir)
    dir_off = HEADER_SIZE
    dir_size = ENTRY_SIZE * len(entries)
    tok_off = (dir_off + dir_size + ALIGN - 1) // ALIGN * ALIGN
    data_off = (tok_off + len(tok_blob) + ALIGN - 1) // ALIGN * ALIGN
    cur = data_off
    for e in entries:
        cur = (cur + ALIGN - 1) // ALIGN * ALIGN
        e["offset"] = cur
        cur += e["nbytes"]
    file_size = cur
    primes = [0] * 8
    seg_layers_raw = list(getattr(cfg, "seg_layers", ()) or ())
    seg_budget = int(getattr(cfg, "seg_budget", 0))
    has_sc = bool(seg_layers_raw) and ("seg_scorer.ln.weight" in state)
    seg_n_feat = len(seg_layers_raw) if has_sc else 0
    seg_layers = seg_layers_raw[:4] + [0] * (4 - len(seg_layers_raw[:4]))
    seg_hidden = int(getattr(cfg, "seg_hidden", 256)) if seg_n_feat > 0 else 0
    seg_min = int(getattr(cfg, "seg_min", 16)) if seg_n_feat > 0 else 0
    seg_max = int(getattr(cfg, "seg_max", 48)) if seg_n_feat > 0 else 0
    flags = (1 if cfg.tie_embeddings else 0) | (4 if seg_n_feat > 0 else 0)
    fmt_map = {
        "f32": F32, "bf16": BF16, "q8_0": Q8_0, "q4_0": Q4_0, "p4_0": P4_0,
        "t1_58_hybrid": T1_58_G32, "t1_58_g64": T1_58_G64,
    }
    hdr_vals = dict(
        magic=MAGIC, version=VERSION, header_size=HEADER_SIZE, endian_tag=ENDIAN_TAG,
        file_size=file_size, alignment=ALIGN, n_tensors=len(entries),
        tensor_dir_offset=dir_off, tensor_dir_size=dir_size, tokenizer_offset=tok_off, tokenizer_size=len(tok_blob),
        data_offset=data_off, data_size=file_size - data_off,
        vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim, d_ff=cfg.d_ff, window=cfg.window,
        n_banks=0, primes=primes, n_cells=0,
        rms_eps=cfg.rms_eps, rope_theta=cfg.rope_theta, vault_eps=0.0,
        bos_id=ids["bos"], eos_id=ids["eos"], im_start_id=ids["im_start"], im_end_id=ids["im_end"], pad_id=ids["pad"],
        weight_format=fmt_map[fmt], flags=flags,
        model_name=model_name.encode("utf-8")[:63],
        n_sink=int(getattr(cfg, "n_sink", 0)),
        seg_budget=seg_budget if seg_n_feat > 0 else 0,
        seg_hidden=seg_hidden, seg_min=seg_min, seg_max=seg_max, seg_n_feat=seg_n_feat, seg_layers=seg_layers,
    )
    flat = []
    for f in _HDR_FIELDS:
        v = hdr_vals[f]
        flat.extend(v if f in ("primes", "seg_layers") else [v])
    hdr = struct.pack(_HDR_FMT, *flat).ljust(HEADER_SIZE - 4, b"\0")
    hdr += struct.pack("<I", zlib.crc32(hdr))

    with open(path, "wb") as f:
        f.write(hdr)
        for e in entries:
            f.write(struct.pack(_ENTRY_FMT, e["id"], e["dtype"], e["ndim"], *e["dims"], 0, e["offset"], e["nbytes"], e["crc"],
                                e["name"].encode()[:63]))
        f.write(b"\0" * (tok_off - f.tell()))
        f.write(tok_blob)
        for e, b in zip(entries, blobs):
            f.write(b"\0" * (e["offset"] - f.tell()))
            f.write(b)
        assert f.tell() == file_size
    return {"file_size": file_size, "n_tensors": len(entries), "tokenizer_bytes": len(tok_blob)}


# ==============================================================================
# Reader / validator
# ==============================================================================

class VLT3Error(ValueError):
    pass


@dataclass
class VLT3File:
    header: Dict
    entries: List[Dict]
    buf: memoryview

    def tensor(self, name: str) -> torch.Tensor:
        tid = tensor_id(name)
        for e in self.entries:
            if e["id"] == tid:
                shape = tuple(e["dims"][: e["ndim"]])
                return decode_tensor(self.buf[e["offset"]: e["offset"] + e["nbytes"]], e["dtype"], shape)
        raise KeyError(name)


def _check_finite_payload(buf: memoryview, dt: int, n: int, tid: int) -> None:
    if dt == F32:
        a = np.frombuffer(buf, dtype="<f4", count=n)
        if not np.isfinite(a).all():
            raise VLT3Error(f"tensor {tid}: non-finite F32 values")
    elif dt == F16:
        a = np.frombuffer(buf, dtype="<f2", count=n)
        if not np.isfinite(a).all():
            raise VLT3Error(f"tensor {tid}: non-finite F16 values")
    elif dt == BF16:
        u = np.frombuffer(buf, dtype="<u2", count=n)
        if ((u & 0x7F80) == 0x7F80).any():
            raise VLT3Error(f"tensor {tid}: non-finite BF16 values")
    elif dt == Q8_0:
        b = np.frombuffer(buf, dtype=Q8_DT, count=n // QK)
        if not np.isfinite(b["d"]).all():
            raise VLT3Error(f"tensor {tid}: non-finite Q8_0 scale")
    elif dt in (Q4_0, P4_0):
        b = np.frombuffer(buf, dtype=Q4_DT, count=n // QK)
        if not np.isfinite(b["d"]).all():
            raise VLT3Error(f"tensor {tid}: non-finite {DTYPE_NAMES[dt]} scale")
    elif dt == T1_58_G32:
        b = np.frombuffer(buf, dtype=T1_58_G32_DT, count=n // QK)
        if not np.isfinite(b["d"]).all():
            raise VLT3Error(f"tensor {tid}: non-finite T1_58_G32 scale")
    elif dt == T1_58_G64:
        b = np.frombuffer(buf, dtype=T1_58_G64_DT, count=n // QK64)
        if not np.isfinite(b["d"]).all():
            raise VLT3Error(f"tensor {tid}: non-finite T1_58_G64 scale")


def read_vlt3(path: str, check_crc: bool = True) -> VLT3File:
    with open(path, "rb") as f:
        data = f.read()
    buf = memoryview(data)
    size = len(data)
    if size < HEADER_SIZE:
        raise VLT3Error("file smaller than header")
    if zlib.crc32(data[: HEADER_SIZE - 4]) != struct.unpack_from("<I", data, HEADER_SIZE - 4)[0]:
        raise VLT3Error("header CRC mismatch")
    vals = struct.unpack_from(_HDR_FMT, data, 0)
    h: Dict = {}
    i = 0
    for f in _HDR_FIELDS:
        if f == "primes":
            h[f] = list(vals[i: i + 8])
            i += 8
        elif f == "seg_layers":
            h[f] = list(vals[i: i + 4])
            i += 4
        else:
            h[f] = vals[i]
            i += 1
    if h["magic"] != MAGIC or h["version"] != VERSION or h["header_size"] != HEADER_SIZE or h["endian_tag"] != ENDIAN_TAG:
        raise VLT3Error("bad magic/version/header_size/endianness")
    if h["file_size"] != size:
        raise VLT3Error(f"file_size {h['file_size']} != actual {size}")
    if h["alignment"] != ALIGN:
        raise VLT3Error("unsupported alignment")

    def _region(off, n, what):
        if off < HEADER_SIZE or off > size or n > size - off:
            raise VLT3Error(f"{what} out of bounds")

    _region(h["tensor_dir_offset"], h["tensor_dir_size"], "tensor directory")
    _region(h["tokenizer_offset"], h["tokenizer_size"], "tokenizer")
    _region(h["data_offset"], h["data_size"], "data")
    if h["tensor_dir_size"] != h["n_tensors"] * ENTRY_SIZE:
        raise VLT3Error("tensor_dir_size != n_tensors * 128")
    if not (0 < h["n_layers"] <= 64) or not (0 < h["n_heads"] <= 256) or not (0 < h["head_dim"] <= 256):
        raise VLT3Error("bad layer/head dimensions")
    if h["n_kv_heads"] == 0 or h["n_heads"] % h["n_kv_heads"] != 0 or h["n_heads"] * h["head_dim"] != h["d_model"]:
        raise VLT3Error("inconsistent GQA/d_model dimensions")
    if h["n_sink"] > 512 or h["n_sink"] > h["window"]:
        raise VLT3Error("bad n_sink")
    if h["n_banks"] != 0 or h["n_cells"] != 0 or any(p != 0 for p in h["primes"]):
        raise VLT3Error("bad vault bank description (expected 0 in Kenyon-free VLT3)")
    if not (np.isfinite(h["rms_eps"]) and h["rms_eps"] > 0 and np.isfinite(h["rope_theta"]) and h["rope_theta"] > 1.0
            and np.isfinite(h["vault_eps"])):
        raise VLT3Error("non-finite or invalid eps/theta")
    if h["seg_n_feat"] > 0:
        if h["seg_n_feat"] > 4 or not (0 < h["seg_hidden"] <= 1024) or not (0 < h["seg_min"] <= h["seg_max"] <= h["window"]):
            raise VLT3Error("invalid segment scorer hyperparameters")
        if h["seg_budget"] > 16384 or any(l >= h["n_layers"] for l in h["seg_layers"][: h["seg_n_feat"]]):
            raise VLT3Error("invalid segment budget or feature layers")

    entries = []
    seen = set()
    spans = []
    for k in range(h["n_tensors"]):
        o = h["tensor_dir_offset"] + k * ENTRY_SIZE
        tid, dt, ndim, d0, d1, d2, d3, rsv, off, nb, crc, nm = struct.unpack_from(_ENTRY_FMT, data, o)
        dims = [d0, d1, d2, d3]
        if rsv != 0:
            raise VLT3Error(f"tensor entry {k}: reserved field not zero")
        if tid in seen:
            raise VLT3Error(f"duplicate tensor id {tid}")
        seen.add(tid)
        if dt not in DTYPE_NAMES or not (1 <= ndim <= 4) or any(x == 0 for x in dims[:ndim]):
            raise VLT3Error(f"bad dtype/ndim/zero-dim for tensor {tid}")
        n = 1
        for x in dims[:ndim]:
            n *= int(x)
        if nb != nbytes_for(dt, n):
            raise VLT3Error(f"tensor {tid}: nbytes {nb} inconsistent with dtype/dims")
        if off % ALIGN or off < h["data_offset"] or nb > h["data_offset"] + h["data_size"] - off:
            raise VLT3Error(f"tensor {tid}: offset out of data region or misaligned")
        if check_crc:
            if zlib.crc32(data[off: off + nb]) != crc:
                raise VLT3Error(f"tensor {tid}: CRC mismatch")
        _check_finite_payload(buf[off: off + nb], dt, n, tid)
        spans.append((off, off + nb))
        entries.append(dict(id=tid, dtype=dt, ndim=ndim, dims=dims, offset=off, nbytes=nb, crc=crc,
                            name=nm.rstrip(b"\0").decode("utf-8", "replace")))
    spans.sort()
    for (a0, a1), (b0, _b1) in zip(spans, spans[1:]):
        if b0 < a1:
            raise VLT3Error("overlapping tensors")
    need = {G_EMBED, G_NORM} | {((l + 1) << 8) | (j + 1) for l in range(h["n_layers"]) for j in range(len(L_KINDS))}
    if h["seg_n_feat"] > 0:
        need |= {G_SEG_LN_W, G_SEG_LN_B, G_SEG_W1, G_SEG_B1, G_SEG_W2, G_SEG_B2}
    if need - seen:
        raise VLT3Error(f"missing tensors: {sorted(need - seen)[:5]}")
    return VLT3File(h, entries, buf)


def vlt3_to_state_dict(vf: VLT3File) -> Dict[str, torch.Tensor]:
    return {tensor_name(e["id"]): vf.tensor(tensor_name(e["id"])) for e in vf.entries}


def vlt3_config(vf: VLT3File):
    from src.vault_v3_model import VAULTv3Config
    h = vf.header
    nf = h.get("seg_n_feat", 0)
    return VAULTv3Config(
        vocab_size=h["vocab_size"], d_model=h["d_model"], n_layers=h["n_layers"], n_heads=h["n_heads"],
        n_kv_heads=h["n_kv_heads"], head_dim=h["head_dim"], d_ff=h["d_ff"], rms_eps=h["rms_eps"],
        rope_theta=h["rope_theta"], tie_embeddings=bool(h["flags"] & 1), window=h["window"],
        n_sink=h["n_sink"],
        seg_budget=h.get("seg_budget", 0) if nf > 0 else 0,
        seg_hidden=h.get("seg_hidden", 256) if nf > 0 else 256,
        seg_min=h.get("seg_min", 16) if nf > 0 else 16,
        seg_max=h.get("seg_max", 48) if nf > 0 else 48,
        seg_layers=tuple(h["seg_layers"][:nf]) if nf > 0 else (),
    )
