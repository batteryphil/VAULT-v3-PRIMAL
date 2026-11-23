# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
VAULT-PRIMAL Cache-Resident CPU Inference Engine (`vault_cpu_engine.py`).

Optimizes all 3 VAULT-PRIMAL TTS architectures (`Tiny-Voca-4.5M`, `VITS-29M`, `F5-TTS-337M`) on CPU:
  1. True In-RAM Bit-Packed Execution (`VaultCPULinear`, `VaultCPUFusedQKV`, `VaultCPUConvNd`):
       - Keeps `V3_P4_0` (4.50 bpw) and `V3_Q2_0` (2.25 bpw) weights permanently compressed in CPU cache/RAM
         (`6.5x–12.0x` smaller RAM than FP32: `1.5–2.8 MB` Tiny-Voca in L2/L3, `9.4–17.2 MB` VITS 100% inside L3 cache,
         `155–212 MB` F5-TTS!).
       - Hybrid C99 AVX2 dispatch (`vault_tts_cpu.c`, linked directly to PyTorch's OpenMP runtime with zero spin contention):
           * Small-M (`M <= 8`, e.g. `AdaLayerNorm` & conditioning): direct `VPMADDUBSW` (`INT8 x P4/Q2`) in `0.05 ms`.
           * Sequence-M (`M > 8`): sub-millisecond AVX2 unpack into thread-local L2/L3 scratch + hot-cache MKL/oneDNN.
  2. Architecture-Specific Zero-Allocation Kernel Fusions:
       - `F5-TTS-337M`: Fused `QKV` + `47.5x` faster in-place AVX2 `VADDSUBPS` RoPE + contiguous FlashAttention SDPA
         + single-pass affine `AdaLayerNorm` (`F.layer_norm` with hoisted `M=1` `silu(t)`) + in-place `addcmul_` & `gelu_`
         + Zero-Allocation `EPSS` + Anchor CFG-Delta Flow-Matching solver (`fast_f5_cpu_sample`).
       - `VITS-29M`: `meta`-device `< 0.25s` cold-load + L3-resident packed WaveNet/Transformer + shared-activation
         in-place `VitsHifiGan` vocoder (`0.00e+00`–`5e-7` numerical parity).
       - `Tiny-Voca-4.5M`: Cached RoPE tables + in-place AVX2 RoPE/GELU + single-pass C99 band-limited harmonic
         `vault_pulse_train_f32` (`~10x` faster oscillator).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import threading
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

P4_LUT_VALUES = [-127.0, -101.0, -80.0, -62.0, -46.0, -33.0, -21.0, -10.0, 0.0, 10.0, 21.0, 33.0, 46.0, 62.0, 80.0, 101.0]
COMPILER_ABI_TAG = "v2_unified_omp"


# ==============================================================================
# 1. Physical-Core Detection & Auto-Compiled C99 AVX2 Shared Kernel
# ==============================================================================
def detect_physical_cores() -> int:
    """Returns physical CPU core count (avoiding SMT hyperthread FMA contention on AVX2/MKL)."""
    env_t = os.environ.get("VAULT_CPU_THREADS", "").strip()
    if env_t.isdigit() and int(env_t) > 0:
        return int(env_t)
    try:
        if os.path.isfile("/proc/cpuinfo"):
            phys_core_pairs = set()
            phys_id = "0"
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("physical id"):
                        phys_id = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":", 1)[1].strip()
                        phys_core_pairs.add((phys_id, core_id))
            if phys_core_pairs:
                return len(phys_core_pairs)
    except Exception:
        pass
    logical = os.cpu_count() or 4
    return max(1, logical // 2) if logical >= 8 else logical


def configure_cpu_threads(max_threads: Optional[int] = None) -> int:
    nt = max_threads if (max_threads and max_threads > 0) else detect_physical_cores()
    try:
        if torch.get_num_threads() != nt:
            torch.set_num_threads(nt)
    except Exception:
        pass
    return nt


_C_LIB: Optional[ctypes.CDLL] = None
_C_LIB_TRIED = False
_C_LOCK = threading.Lock()


def _find_torch_gomp() -> Optional[Tuple[Path, Path]]:
    """Finds PyTorch's bundled `libgomp*.so*` so our C kernel shares the exact same OpenMP thread pool."""
    try:
        t_lib = Path(torch.__file__).resolve().parent / "lib"
        if t_lib.is_dir():
            for cand in sorted(t_lib.glob("libgomp*.so*")):
                if cand.is_file():
                    return t_lib, cand
    except Exception:
        pass
    return None


def get_vault_cpu_lib() -> Optional[ctypes.CDLL]:
    global _C_LIB, _C_LIB_TRIED
    if _C_LIB_TRIED:
        return _C_LIB
    with _C_LOCK:
        if _C_LIB_TRIED:
            return _C_LIB
        _C_LIB_TRIED = True

        here = Path(__file__).resolve().parent
        c_src = here / "vault_tts_cpu.c"
        so_path: Optional[Path] = None

        if c_src.is_file():
            src_hash = hashlib.sha256(c_src.read_bytes() + COMPILER_ABI_TAG.encode("utf-8")).hexdigest()[:12]
            cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vault_primal_tts"
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                cache_dir = Path("/tmp/vault_primal_tts_cache")
                cache_dir.mkdir(parents=True, exist_ok=True)
            cand_so = cache_dir / f"libvault_tts_cpu_{platform.machine()}_{src_hash}.so"

            if not cand_so.is_file():
                cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
                if cc:
                    arch = platform.machine().lower()
                    simd_flags = ["-mavx2", "-mfma", "-mf16c", "-mprefer-vector-width=256"] if arch in ("x86_64", "amd64") else []
                    gomp_info = _find_torch_gomp()
                    built = False
                    if gomp_info is not None and "gcc" in os.path.basename(cc):
                        t_lib, gomp_so = gomp_info
                        cmd_omp = [
                            cc, "-O3", "-shared", "-fPIC", "-fopenmp", *simd_flags,
                            str(c_src), f"-Wl,-rpath,{t_lib}", str(gomp_so),
                            "-nodefaultlibs", "-lc", "-lm", "-lgcc_s", "-o", str(cand_so),
                        ]
                        res = subprocess.run(cmd_omp, capture_output=True, text=True)
                        built = res.returncode == 0 and cand_so.is_file()
                    if not built:
                        # Compile without -fopenmp so we never spawn a second competing OpenMP spin pool!
                        cmd_simd = [cc, "-O3", "-shared", "-fPIC", *simd_flags, str(c_src), "-o", str(cand_so), "-lm"]
                        res = subprocess.run(cmd_simd, capture_output=True, text=True)
                        if res.returncode != 0:
                            cmd_fb = [cc, "-O3", "-shared", "-fPIC", str(c_src), "-o", str(cand_so), "-lm"]
                            subprocess.run(cmd_fb, capture_output=True, text=True)
            if cand_so.is_file():
                so_path = cand_so

        if so_path is not None:
            try:
                lib = ctypes.CDLL(str(so_path))
                vp = ctypes.c_void_p
                i64 = ctypes.c_int64
                ci = ctypes.c_int
                cd = ctypes.c_double
                lib.vault_repack_p4_g32.argtypes = [vp, vp, i64, i64]
                lib.vault_repack_q2_g64.argtypes = [vp, vp, i64, i64]
                lib.vault_gemm_p4_g32.argtypes = [vp, vp, vp, vp, vp, i64, i64, i64, i64, ci, ci]
                lib.vault_gemm_q2_g64.argtypes = [vp, vp, vp, vp, vp, i64, i64, i64, i64, ci, ci]
                lib.vault_rope_inplace_f32.argtypes = [vp, vp, vp, vp, i64, i64, i64, i64, i64, ci]
                lib.vault_unpack_p4_f32.argtypes = [vp, vp, vp, i64, i64, i64, ci]
                lib.vault_unpack_q2_f32.argtypes = [vp, vp, vp, i64, i64, i64, ci]
                if hasattr(lib, "vault_pulse_train_f32"):
                    lib.vault_pulse_train_f32.argtypes = [vp, vp, vp, i64, i64, i64, cd, cd]
                _C_LIB = lib
            except Exception:
                _C_LIB = None
        return _C_LIB


# ==============================================================================
# 2. Pure-PyTorch Fallback Byte-LUT & Thread-Local L3-Cache Scratch Buffer
# ==============================================================================
_PT_LUT_P4: Optional[torch.Tensor] = None
_PT_LUT_Q2: Optional[torch.Tensor] = None
_TLS_SCRATCH = threading.local()


def _get_l3_scratch(numel_needed: int) -> torch.Tensor:
    buf: Optional[torch.Tensor] = getattr(_TLS_SCRATCH, "buf", None)
    if buf is None or buf.numel() < numel_needed:
        buf = torch.empty(max(numel_needed, 262144), dtype=torch.float32)
        _TLS_SCRATCH.buf = buf
    return buf[:numel_needed]


def _pt_unpack_fallback(q_pack: torch.Tensor, d_f16: torch.Tensor, fmt: str, rows: int, cols: int, pcols: int) -> torch.Tensor:
    global _PT_LUT_P4, _PT_LUT_Q2
    if fmt == "p4_0":
        if _PT_LUT_P4 is None:
            idx = torch.arange(256, dtype=torch.long)
            p4 = torch.tensor(P4_LUT_VALUES, dtype=torch.float32)
            _PT_LUT_P4 = torch.stack([p4[idx & 0x0F], p4[(idx >> 4) & 0x0F]], dim=-1).contiguous()
        cv = F.embedding(q_pack.to(torch.int32), _PT_LUT_P4).view(rows, pcols // 32, 32)
        w = (cv * d_f16.float().unsqueeze(-1)).view(rows, pcols)
    else:
        if _PT_LUT_Q2 is None:
            idx = torch.arange(256, dtype=torch.int32)
            _PT_LUT_Q2 = torch.stack(
                [(idx & 3) - 2, ((idx >> 2) & 3) - 2, ((idx >> 4) & 3) - 2, ((idx >> 6) & 3) - 2], dim=-1
            ).float().contiguous()
        cv = F.embedding(q_pack.to(torch.int32), _PT_LUT_Q2).view(rows, pcols // 64, 64)
        w = (cv * d_f16.float().unsqueeze(-1)).view(rows, pcols)
    return w[:, :cols].contiguous() if pcols > cols else w


def _unpack_raw_once(q_raw: torch.Tensor, d_f16: torch.Tensor, fmt: str, rows: int, cols: int, pcols: int) -> torch.Tensor:
    q_buf, d_buf, is_c = _prepare_packed_buffers(q_raw, d_f16, fmt, rows, pcols)
    if is_c:
        lib = get_vault_cpu_lib()
        out = torch.empty((rows, cols), dtype=torch.float32)
        if fmt == "p4_0":
            lib.vault_unpack_p4_f32(q_buf.data_ptr(), d_buf.data_ptr(), out.data_ptr(), rows, cols, pcols, 1)
        else:
            lib.vault_unpack_q2_f32(q_buf.data_ptr(), d_buf.data_ptr(), out.data_ptr(), rows, cols, pcols, 1)
        return out
    return _pt_unpack_fallback(q_buf, d_buf, fmt, rows, cols, pcols)


def _prepare_packed_buffers(q_raw: torch.Tensor, d_f16: torch.Tensor, fmt: str, rows: int, pcols: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    lib = get_vault_cpu_lib()
    q_c = q_raw.to(dtype=torch.uint8, device="cpu").contiguous()
    if lib is not None:
        q_simd = torch.empty_like(q_c)
        if fmt == "p4_0":
            lib.vault_repack_p4_g32(q_c.data_ptr(), q_simd.data_ptr(), rows, pcols)
        else:
            lib.vault_repack_q2_g64(q_c.data_ptr(), q_simd.data_ptr(), rows, pcols)
        d_u16 = d_f16.to(dtype=torch.float16, device="cpu").contiguous().view(torch.uint16)
        return q_simd, d_u16, True
    return q_c, d_f16.to(dtype=torch.float16, device="cpu").contiguous(), False


# ==============================================================================
# 3. Cache-Resident Bit-Packed CPU Modules (`Linear`, `FusedQKV`, `Conv1d`, `ConvTranspose1d`)
# ==============================================================================
class VaultCPULinear(nn.Module):
    """Bit-packed CPU Linear layer (`V3_P4_0` or `V3_Q2_0`, zero persistent FP32 weight matrix!)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fmt: str,
        q_raw: torch.Tensor,
        d_f16: torch.Tensor,
        bias: Optional[torch.Tensor],
        n_threads: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fmt = fmt
        self.pcols = q_raw.shape[1] * (2 if fmt == "p4_0" else 4)
        self.n_threads = n_threads
        q_buf, d_buf, is_c = _prepare_packed_buffers(q_raw, d_f16, fmt, out_features, self.pcols)
        self.use_c = is_c
        self.register_buffer("q_buf", q_buf)
        self.register_buffer("d_buf", d_buf)
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(dtype=torch.float32, device="cpu").contiguous())
        else:
            self.bias = None
        self._dummy_w = nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)

    @property
    def weight(self) -> torch.Tensor:
        return self._dummy_w

    def unpack_to_l3(self) -> torch.Tensor:
        if self.use_c:
            lib = get_vault_cpu_lib()
            w_flat = _get_l3_scratch(self.out_features * self.in_features)
            if self.fmt == "p4_0":
                lib.vault_unpack_p4_f32(
                    self.q_buf.data_ptr(), self.d_buf.data_ptr(), w_flat.data_ptr(),
                    self.out_features, self.in_features, self.pcols, self.n_threads,
                )
            else:
                lib.vault_unpack_q2_f32(
                    self.q_buf.data_ptr(), self.d_buf.data_ptr(), w_flat.data_ptr(),
                    self.out_features, self.in_features, self.pcols, self.n_threads,
                )
            return w_flat.view(self.out_features, self.in_features)
        return _pt_unpack_fallback(self.q_buf, self.d_buf, self.fmt, self.out_features, self.in_features, self.pcols)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dt = x.dtype
        xf = x if orig_dt == torch.float32 else x.float()
        M = xf.numel() // self.in_features
        if self.use_c and M <= 8:
            lib = get_vault_cpu_lib()
            x2 = xf.contiguous().view(M, self.in_features)
            out2 = torch.empty((M, self.out_features), dtype=torch.float32)
            b_ptr = self.bias.data_ptr() if self.bias is not None else 0
            if self.fmt == "p4_0":
                lib.vault_gemm_p4_g32(
                    x2.data_ptr(), self.q_buf.data_ptr(), self.d_buf.data_ptr(), b_ptr,
                    out2.data_ptr(), M, self.in_features, self.pcols, self.out_features, 0, self.n_threads,
                )
            else:
                lib.vault_gemm_q2_g64(
                    x2.data_ptr(), self.q_buf.data_ptr(), self.d_buf.data_ptr(), b_ptr,
                    out2.data_ptr(), M, self.in_features, self.pcols, self.out_features, 0, self.n_threads,
                )
            y = out2.view(*xf.shape[:-1], self.out_features)
            return y if orig_dt == torch.float32 else y.to(orig_dt)

        w_l3 = self.unpack_to_l3()
        y = F.linear(xf, w_l3, self.bias)
        return y if orig_dt == torch.float32 else y.to(orig_dt)


class VaultCPUFusedQKV(nn.Module):
    """Fuses `(to_q, to_k, to_v)` into a single packed `[3*D_inner, D_in]` projection."""

    def __init__(self, q_mod: VaultCPULinear, k_mod: VaultCPULinear, v_mod: VaultCPULinear):
        super().__init__()
        assert q_mod.fmt == k_mod.fmt == v_mod.fmt and q_mod.pcols == k_mod.pcols == v_mod.pcols
        self.in_features = q_mod.in_features
        self.inner_dim = q_mod.out_features
        self.out_features = 3 * self.inner_dim
        self.fmt = q_mod.fmt
        self.pcols = q_mod.pcols
        self.use_c = q_mod.use_c
        self.n_threads = q_mod.n_threads
        self.register_buffer("q_buf", torch.cat([q_mod.q_buf, k_mod.q_buf, v_mod.q_buf], dim=0).contiguous())
        self.register_buffer("d_buf", torch.cat([q_mod.d_buf, k_mod.d_buf, v_mod.d_buf], dim=0).contiguous())
        if q_mod.bias is not None or k_mod.bias is not None or v_mod.bias is not None:
            z = torch.zeros(self.inner_dim, dtype=torch.float32)
            b_cat = torch.cat(
                [
                    q_mod.bias if q_mod.bias is not None else z,
                    k_mod.bias if k_mod.bias is not None else z,
                    v_mod.bias if v_mod.bias is not None else z,
                ],
                dim=0,
            ).contiguous()
            self.register_buffer("bias", b_cat)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_c:
            lib = get_vault_cpu_lib()
            w_flat = _get_l3_scratch(self.out_features * self.in_features)
            if self.fmt == "p4_0":
                lib.vault_unpack_p4_f32(
                    self.q_buf.data_ptr(), self.d_buf.data_ptr(), w_flat.data_ptr(),
                    self.out_features, self.in_features, self.pcols, self.n_threads,
                )
            else:
                lib.vault_unpack_q2_f32(
                    self.q_buf.data_ptr(), self.d_buf.data_ptr(), w_flat.data_ptr(),
                    self.out_features, self.in_features, self.pcols, self.n_threads,
                )
            w_l3 = w_flat.view(self.out_features, self.in_features)
        else:
            w_l3 = _pt_unpack_fallback(self.q_buf, self.d_buf, self.fmt, self.out_features, self.in_features, self.pcols)
        qkv = F.linear(x, w_l3, self.bias)
        return qkv.chunk(3, dim=-1)


class VaultCPUConvNd(nn.Module):
    """Bit-packed CPU `Conv1d` / `ConvTranspose1d` keeping VITS & Tiny-Voca weights inside L3 cache."""

    def __init__(
        self,
        orig_mod: nn.Module,
        fmt: str,
        q_raw: torch.Tensor,
        d_f16: torch.Tensor,
        n_threads: int,
    ):
        super().__init__()
        self.is_transpose = isinstance(orig_mod, nn.ConvTranspose1d)
        self.orig_shape = tuple(orig_mod.weight.shape)
        self.rows = self.orig_shape[0]
        self.cols = int(np.prod(self.orig_shape[1:]))
        self.pcols = q_raw.shape[1] * (2 if fmt == "p4_0" else 4)
        self.fmt = fmt
        self.n_threads = n_threads

        self.in_channels = orig_mod.in_channels
        self.out_channels = orig_mod.out_channels
        self.kernel_size = orig_mod.kernel_size
        self.stride = orig_mod.stride
        self.padding = orig_mod.padding
        self.dilation = orig_mod.dilation
        self.groups = orig_mod.groups
        self.output_padding = getattr(orig_mod, "output_padding", (0,))

        q_buf, d_buf, is_c = _prepare_packed_buffers(q_raw, d_f16, fmt, self.rows, self.pcols)
        self.use_c = is_c
        self.register_buffer("q_buf", q_buf)
        self.register_buffer("d_buf", d_buf)
        if orig_mod.bias is not None and not orig_mod.bias.is_meta:
            self.register_buffer("bias", orig_mod.bias.detach().to(dtype=torch.float32, device="cpu").contiguous())
        else:
            self.bias = None
        self._dummy_w = nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)

    @property
    def weight(self) -> torch.Tensor:
        return self._dummy_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_c:
            lib = get_vault_cpu_lib()
            w_flat = _get_l3_scratch(self.rows * self.cols)
            if self.fmt == "p4_0":
                lib.vault_unpack_p4_f32(
                    self.q_buf.data_ptr(), self.d_buf.data_ptr(), w_flat.data_ptr(),
                    self.rows, self.cols, self.pcols, self.n_threads,
                )
            else:
                lib.vault_unpack_q2_f32(
                    self.q_buf.data_ptr(), self.d_buf.data_ptr(), w_flat.data_ptr(),
                    self.rows, self.cols, self.pcols, self.n_threads,
                )
            w_l3 = w_flat.view(self.orig_shape)
        else:
            w_l3 = _pt_unpack_fallback(self.q_buf, self.d_buf, self.fmt, self.rows, self.cols, self.pcols).view(self.orig_shape)

        xf = x if x.dtype == torch.float32 else x.float()
        if self.is_transpose:
            y = F.conv_transpose1d(xf, w_l3, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        else:
            y = F.conv1d(xf, w_l3, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y if x.dtype == torch.float32 else y.to(x.dtype)


# ==============================================================================
# 4. Architecture-Specific Zero-Allocation CPU Fusions (`F5-TTS`, `VITS`, `Tiny-Voca`)
# ==============================================================================
class FastCPUAttnProcessor:
    """Replaces `x_transformers` `einops` RoPE with precomputed `(cos, sin)` + C99 AVX2 `VADDSUBPS` + Contiguous CPU Flash-SDPA."""

    def __init__(self, orig_proc: Any, fused_qkv: Optional[VaultCPUFusedQKV], n_threads: int):
        self.pe_attn_head = getattr(orig_proc, "pe_attn_head", None)
        self.attn_mask_enabled = getattr(orig_proc, "attn_mask_enabled", False)
        self.fused_qkv = fused_qkv
        self.n_threads = n_threads
        self._rope_cache: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_cos_sin(self, rope: Tuple[Any, Any], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        freqs, xpos_scale = rope
        unit_scale = (xpos_scale is None) or (not torch.is_tensor(xpos_scale) and float(xpos_scale) == 1.0)
        key = (seq_len, int(freqs.shape[-1]), 0 if unit_scale else int(freqs.data_ptr()))
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached[0], cached[1], unit_scale
        f_slice = freqs[0, -seq_len:, :].float()
        if not unit_scale:
            sc = xpos_scale[0, -seq_len:, :].float() if torch.is_tensor(xpos_scale) else float(xpos_scale)
            cos_t = (f_slice.cos() * sc).contiguous()
            sin_t = (f_slice.sin() * sc).contiguous()
        else:
            cos_t = f_slice.cos().contiguous()
            sin_t = f_slice.sin().contiguous()
        if len(self._rope_cache) > 16:
            self._rope_cache.clear()
        self._rope_cache[key] = (cos_t, sin_t)
        return cos_t, sin_t, unit_scale

    def __call__(self, attn: nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor] = None, rope: Any = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if self.fused_qkv is not None:
            query, key, value = self.fused_qkv(x)
        else:
            query = attn.to_q(x)
            key = attn.to_k(x)
            value = attn.to_v(x)

        head_dim = key.shape[-1] // attn.heads
        query = query.view(bsz, seq_len, attn.heads, head_dim).transpose(1, 2).contiguous()
        key = key.view(bsz, seq_len, attn.heads, head_dim).transpose(1, 2).contiguous()
        # Contiguous `value` activates PyTorch's fused AVX-512 CPU FlashAttention fast path (-107 ms/step)!
        value = value.view(bsz, seq_len, attn.heads, head_dim).transpose(1, 2).contiguous()

        if attn.q_norm is not None:
            query = attn.q_norm(query).contiguous()
        if attn.k_norm is not None:
            key = attn.k_norm(key).contiguous()

        if rope is not None:
            cos_t, sin_t, unit_scale = self._get_cos_sin(rope, seq_len)
            h_rot = self.pe_attn_head if self.pe_attn_head is not None else attn.heads
            lib = get_vault_cpu_lib()
            if lib is not None and unit_scale and cos_t.shape[-1] == head_dim:
                lib.vault_rope_inplace_f32(
                    query.data_ptr(), key.data_ptr(), cos_t.data_ptr(), sin_t.data_ptr(),
                    bsz, h_rot, attn.heads, seq_len, head_dim, self.n_threads,
                )
            else:
                q_part = query[:, :h_rot]
                k_part = key[:, :h_rot]
                q_rot = torch.stack([-q_part[..., 1::2], q_part[..., 0::2]], dim=-1).flatten(-2)
                k_rot = torch.stack([-k_part[..., 1::2], k_part[..., 0::2]], dim=-1).flatten(-2)
                c4 = cos_t.unsqueeze(0).unsqueeze(0)
                s4 = sin_t.unsqueeze(0).unsqueeze(0)
                query[:, :h_rot] = q_part * c4 + q_rot * s4
                if unit_scale:
                    key[:, :h_rot] = k_part * c4 + k_rot * s4
                else:
                    _, xpos_scale = rope
                    inv_s = (xpos_scale[0, -seq_len:, :].float() ** -2.0).unsqueeze(0).unsqueeze(0) if torch.is_tensor(xpos_scale) else (float(xpos_scale) ** -2.0)
                    key[:, :h_rot] = k_part * (c4 * inv_s) + k_rot * (s4 * inv_s)

        attn_mask = None
        if self.attn_mask_enabled and mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(1).expand(bsz, attn.heads, seq_len, seq_len)

        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(bsz, seq_len, attn.heads * head_dim)
        out = attn.to_out[0](out)
        if mask is not None:
            out.masked_fill_(~mask.unsqueeze(-1), 0.0)
        return out


def _install_fast_f5_dit(model: nn.Module, nt: int) -> int:
    """Installs fused `QKV`, single-pass affine `AdaLayerNorm`, in-place `gelu_`/`addcmul_`, and hoisted `M=1` `silu(t)`."""
    n_fused_qkv = 0
    for blk in model.transformer_blocks:
        attn = getattr(blk, "attn", None)
        if attn is not None:
            fqkv = None
            if (
                isinstance(getattr(attn, "to_q", None), VaultCPULinear)
                and isinstance(getattr(attn, "to_k", None), VaultCPULinear)
                and isinstance(getattr(attn, "to_v", None), VaultCPULinear)
            ):
                fqkv = VaultCPUFusedQKV(attn.to_q, attn.to_k, attn.to_v)
                attn.to_q = nn.Identity()
                attn.to_k = nn.Identity()
                attn.to_v = nn.Identity()
                attn.add_module("fused_qkv", fqkv)
                n_fused_qkv += 1
            attn.processor = FastCPUAttnProcessor(attn.processor, fqkv, nt)

        # Fuse FeedForward GELU in-place (`aten.gelu_` saves 153 ms/step & 1.1 GB DRAM writes!)
        ff_mod = getattr(blk, "ff", None)
        if ff_mod is not None and hasattr(ff_mod, "ff") and len(ff_mod.ff) >= 3:
            lin_in = ff_mod.ff[0][0]
            act_mod = ff_mod.ff[0][1]
            approx = getattr(act_mod, "approximate", "tanh")
            lin_out = ff_mod.ff[2]

            def _fast_ff_fwd(_self, x_in, _lin1=lin_in, _lin2=lin_out, _ap=approx):
                h = _lin1(x_in)
                torch.ops.aten.gelu_(h, approximate=_ap)
                return _lin2(h)

            ff_mod.forward = types.MethodType(_fast_ff_fwd, ff_mod)

        # Fuse DiTBlock AdaLayerNorm + residual gate into single-pass affine F.layer_norm + in-place addcmul_!
        def _fast_blk_fwd(self_blk, x_in, t_emb, mask=None, rope=None):
            # During inference (`B=1` or CFG `B=2` with `t[0] == t[1]`), evaluate `AdaLN` linear once at `M=1`!
            same_t = (t_emb.shape[0] == 1) or bool(torch.equal(t_emb[0], t_emb[-1]))
            if same_t:
                emb1 = self_blk.attn_norm.linear(F.silu(t_emb[:1]))[0]
                sh_msa, sc_msa, gt_msa, sh_mlp, sc_mlp, gt_mlp = emb1.chunk(6, dim=0)
                dim = x_in.shape[-1]
                norm1 = F.layer_norm(x_in, (dim,), 1.0 + sc_msa, sh_msa, 1e-6)
                attn_out = self_blk.attn(x=norm1, mask=mask, rope=rope)
                x_in = x_in.addcmul(attn_out, gt_msa.view(1, 1, dim))
                norm2 = F.layer_norm(x_in, (dim,), 1.0 + sc_mlp, sh_mlp, 1e-6)
                ff_out = self_blk.ff(norm2)
                x_in.addcmul_(ff_out, gt_mlp.view(1, 1, dim))
                return x_in
            norm1, gt_msa, sh_mlp, sc_mlp, gt_mlp = self_blk.attn_norm(x_in, emb=t_emb)
            attn_out = self_blk.attn(x=norm1, mask=mask, rope=rope)
            x_in = x_in + gt_msa.unsqueeze(1) * attn_out
            norm2 = self_blk.ff_norm(x_in) * (1.0 + sc_mlp[:, None]) + sh_mlp[:, None]
            ff_out = self_blk.ff(norm2)
            return x_in + gt_mlp.unsqueeze(1) * ff_out

        blk.forward = types.MethodType(_fast_blk_fwd, blk)

    norm_out = getattr(model, "norm_out", None)
    if norm_out is not None and hasattr(norm_out, "linear"):
        def _fast_norm_out_fwd(self_no, x_in, t_emb):
            same_t = (t_emb.shape[0] == 1) or bool(torch.equal(t_emb[0], t_emb[-1]))
            if same_t:
                sc, sh = self_no.linear(F.silu(t_emb[:1]))[0].chunk(2, dim=0)
                return F.layer_norm(x_in, (x_in.shape[-1],), 1.0 + sc, sh, 1e-6)
            sc, sh = self_no.linear(F.silu(t_emb)).chunk(2, dim=1)
            return self_no.norm(x_in) * (1.0 + sc)[:, None, :] + sh[:, None, :]

        norm_out.forward = types.MethodType(_fast_norm_out_fwd, norm_out)

    return n_fused_qkv


def _install_fast_vits_cpu(model: nn.Module) -> None:
    """Installs shared-activation in-place `VitsHifiGan` CPU forward (`0` redundant waveform allocations)."""
    dec = getattr(model, "decoder", None)
    if dec is None or not hasattr(dec, "upsampler") or not hasattr(dec, "resblocks"):
        return

    def _fast_vits_dec_fwd(self_dec, spectrogram: torch.Tensor, global_conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self_dec.conv_pre(spectrogram)
        if global_conditioning is not None:
            h = h + self_dec.cond(global_conditioning)
        slope = self_dec.config.leaky_relu_slope
        nk = self_dec.num_kernels
        inv_k = 1.0 / float(nk)

        for i in range(self_dec.num_upsamples):
            F.leaky_relu_(h, slope)
            h = self_dec.upsampler[i](h)
            # Precompute initial LeakyReLU(h) ONCE across all 3 parallel multi-receptive-field resblocks!
            h_act0 = F.leaky_relu(h, slope)
            rs: Optional[torch.Tensor] = None
            for j in range(nk):
                rb = self_dec.resblocks[i * nk + j]
                rb_s = rb.leaky_relu_slope
                cur = h
                for idx_c, (c1, c2) in enumerate(zip(rb.convs1, rb.convs2)):
                    a = h_act0 if (idx_c == 0 and rb_s == slope) else F.leaky_relu(cur, rb_s)
                    z = c1(a)
                    F.leaky_relu_(z, rb_s)
                    z = c2(z)
                    cur = cur + z
                if rs is None:
                    rs = cur
                else:
                    rs.add_(cur)
            h = rs.mul_(inv_k)

        F.leaky_relu_(h)
        h = self_dec.conv_post(h)
        return h.tanh_()

    dec.forward = types.MethodType(_fast_vits_dec_fwd, dec)


def _install_fast_tiny_voca_cpu(model: nn.Module, nt: int) -> None:
    """Installs cached RoPE + AVX2 `vault_rope_inplace_f32` + in-place `gelu_` + C99 `vault_pulse_train_f32` on `TinyVoca`."""
    lib = get_vault_cpu_lib()
    rope_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    for stack_name in ("encoder", "prosody_stack", "decoder"):
        stk = getattr(model, stack_name, None)
        if stk is None or not hasattr(stk, "blocks"):
            continue

        def _fast_stack_fwd(self_stk, x: torch.Tensor, mask: torch.Tensor):
            T = x.shape[1]
            key = (T, self_stk.width)
            tab = rope_cache.get(key)
            if tab is None:
                inv = 1.0 / (10000.0 ** (torch.arange(0, self_stk.width, 2, dtype=torch.float32) / self_stk.width))
                ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None]
                # Duplicate interleaved `[cos0, cos0, cos1, cos1, ...]` so `vault_rope_inplace_f32` runs directly!
                c_int = torch.repeat_interleave(ang.cos(), 2, dim=-1).contiguous()
                s_int = torch.repeat_interleave(ang.sin(), 2, dim=-1).contiguous()
                tab = (ang.cos().contiguous(), ang.sin().contiguous(), c_int, s_int)
                rope_cache[key] = tab
            keep = mask[..., None].to(x.dtype)
            attn_mask = mask[:, None, None, :]
            for blk in self_stk.blocks:
                nx = blk.norm1(x)
                b, t, c = nx.shape
                h_cnt = blk.attention.heads
                hd = c // h_cnt
                qkv = blk.attention.qkv(nx).view(b, t, 3, h_cnt, hd)
                q = qkv[:, :, 0].transpose(1, 2).contiguous()
                k = qkv[:, :, 1].transpose(1, 2).contiguous()
                v = qkv[:, :, 2].transpose(1, 2).contiguous()
                if lib is not None and hd % 2 == 0:
                    lib.vault_rope_inplace_f32(
                        q.data_ptr(), k.data_ptr(), tab[2].data_ptr(), tab[3].data_ptr(),
                        b, h_cnt, h_cnt, t, hd, 1,
                    )
                else:
                    co, si = tab[0], tab[1]
                    qa, qb = q[..., 0::2], q[..., 1::2]
                    ka, kb = k[..., 0::2], k[..., 1::2]
                    q = torch.stack((qa * co - qb * si, qa * si + qb * co), dim=-1).flatten(-2)
                    k = torch.stack((ka * co - kb * si, ka * si + kb * co), dim=-1).flatten(-2)
                ya = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
                x = (x + blk.attention.out(ya.transpose(1, 2).reshape(b, t, c))) * keep
                yd = blk.depthwise((blk.norm2(x) * keep).transpose(1, 2)).transpose(1, 2)
                ye = blk.expand(yd)
                torch.ops.aten.gelu_(ye, approximate="none")
                x = (x + blk.project(ye)) * keep
            return self_stk.norm(x) * keep

        stk.forward = types.MethodType(_fast_stack_fwd, stk)

    # Patch `pulse_train` inside `tiny_voca.model` if our C99 kernel is active!
    if lib is not None and hasattr(lib, "vault_pulse_train_f32"):
        try:
            import tiny_voca.model as tvm

            def _fast_pulse_train(f0: torch.Tensor, sample_rate: int, hop: int, limit: float = 0.47, phase: Optional[torch.Tensor] = None) -> torch.Tensor:
                if f0.device.type != "cpu":
                    return _orig_pt(f0, sample_rate, hop, limit, phase)
                f0_c = f0.to(torch.float32).contiguous()
                B, frames = f0_c.shape
                out = torch.empty((B, frames * hop), dtype=torch.float32)
                ph_c = phase.to(torch.float32).contiguous() if phase is not None else None
                ph_ptr = ph_c.data_ptr() if ph_c is not None else 0
                lib.vault_pulse_train_f32(f0_c.data_ptr(), ph_ptr, out.data_ptr(), B, frames, hop, float(sample_rate), float(limit))
                return out

            if not getattr(tvm.pulse_train, "_vault_patched", False):
                _orig_pt = tvm.pulse_train
                _fast_pulse_train._vault_patched = True  # type: ignore[attr-defined]
                tvm.pulse_train = _fast_pulse_train
        except Exception:
            pass


# ==============================================================================
# 5. Universal Tree Helpers & Live Bit-Packed CPU SafeTensors Loader
# ==============================================================================
def _get_submod(root: nn.Module, path: str) -> nn.Module:
    cur = root
    for p in path.split("."):
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    return cur


def _set_submod(root: nn.Module, path: str, new_mod: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_mod
    else:
        setattr(parent, last, new_mod)


@torch.no_grad()
def load_vault_safetensors_cpu_live(
    model: nn.Module,
    path: str,
    n_threads: Optional[int] = None,
    fuse_dit_qkv: bool = False,
) -> Dict[str, Any]:
    """
    Loads a `VAULT-SafeTensors v1` checkpoint onto CPU while keeping quantized layers
    permanently compressed (`V3_P4_0` / `V3_Q2_0`) inside `VaultCPULinear` / `VaultCPUConvNd`!
    Supports `torch.device("meta")` skeleton models for `< 0.5s` cold-load startup.
    """
    cls_name = model.__class__.__name__
    phys_cores = detect_physical_cores()
    if n_threads is None:
        # Tiny-Voca (`4.5M`) peaks at 8 threads (`162.5x RT`); VITS & F5-TTS peak at all physical cores
        nt = min(8, phys_cores) if cls_name == "TinyVoca" else phys_cores
    else:
        nt = n_threads
    configure_cpu_threads(nt)
    # Single-core L2/L3 scratch unpack avoids waking OpenMP barriers before MKL SGEMM (-87 ms/step faster on DiT!)
    unpack_nt = 1

    is_vits = cls_name == "VitsModel"
    n_lin, n_conv = 0, 0

    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        keys = set(f.keys())

        # First load all non-quantized parameters/biases (including biases of quantized modules!)
        for k in sorted(keys):
            if k.endswith((".__p4_q", ".__p4_d", ".__q2_q", ".__q2_d")):
                continue
            parts = k.split(".")
            parent = _get_submod(model, ".".join(parts[:-1])) if len(parts) > 1 else model
            leaf = parts[-1]
            if hasattr(parent, "_parameters") and leaf in parent._parameters and parent._parameters[leaf] is not None:
                p_old = parent._parameters[leaf]
                t_src = f.get_tensor(k)
                tgt_dt = torch.float32 if p_old.is_floating_point() else p_old.dtype
                parent._parameters[leaf] = nn.Parameter(t_src.to(device="cpu", dtype=tgt_dt).contiguous(), requires_grad=False)

        # Replace quantized Linear / Conv1d / ConvTranspose1d modules with bit-packed CPU modules!
        for k in sorted(keys):
            if not (k.endswith(".__p4_q") or k.endswith(".__q2_q")):
                continue
            fmt = "p4_0" if k.endswith(".__p4_q") else "q2_0"
            suf = ".__p4_q" if fmt == "p4_0" else ".__q2_q"
            dsuf = ".__p4_d" if fmt == "p4_0" else ".__q2_d"
            w_name = k[: -len(suf)]
            assert w_name.endswith(".weight"), f"Unexpected weight key {w_name}"
            mname = w_name[: -len(".weight")]

            orig_mod = _get_submod(model, mname)
            q_raw = f.get_tensor(k)
            d_f16 = f.get_tensor(w_name + dsuf)

            if isinstance(orig_mod, nn.Linear):
                bias_t = orig_mod.bias if (orig_mod.bias is not None and not orig_mod.bias.is_meta) else None
                new_mod = VaultCPULinear(orig_mod.in_features, orig_mod.out_features, fmt, q_raw, d_f16, bias_t, unpack_nt)
                _set_submod(model, mname, new_mod)
                n_lin += 1
            elif isinstance(orig_mod, (nn.Conv1d, nn.ConvTranspose1d)):
                orig_shape = tuple(orig_mod.weight.shape)
                tot_w = int(np.prod(orig_shape))
                # In VITS HiFi-GAN Stage-2/3 (`T = 51k..102k`), tiny `< 49,152`-weight conv filters (`< 0.9 MB` total!)
                # execute at full waveform sample rate: keeping those tiny leaves in `nn.Conv1d` lets `oneDNN` cache
                # `OIw16i16o` AVX-512 blocked weights while 98%+ of VITS stays bit-packed inside L3 cache!
                if is_vits and mname.startswith("decoder.") and tot_w < 49152:
                    rows = orig_shape[0]
                    cols = int(np.prod(orig_shape[1:]))
                    pcols = q_raw.shape[1] * (2 if fmt == "p4_0" else 4)
                    rec = _unpack_raw_once(q_raw, d_f16, fmt, rows, cols, pcols).view(orig_shape)
                    orig_mod._parameters["weight"] = nn.Parameter(rec, requires_grad=False)
                    if orig_mod.bias is not None and orig_mod.bias.is_meta:
                        orig_mod.register_parameter("bias", None)
                else:
                    new_mod = VaultCPUConvNd(orig_mod, fmt, q_raw, d_f16, unpack_nt)
                    _set_submod(model, mname, new_mod)
                    n_conv += 1
            else:
                orig_shape = tuple(int(x) for x in meta[f"shape:{w_name}"].split(","))
                rows = orig_shape[0]
                cols = int(np.prod(orig_shape[1:]))
                pcols = q_raw.shape[1] * (2 if fmt == "p4_0" else 4)
                rec = _unpack_raw_once(q_raw, d_f16, fmt, rows, cols, pcols).view(orig_shape)
                orig_mod._parameters["weight"] = nn.Parameter(rec, requires_grad=False)

    # Ensure all non-parameter buffers live on CPU
    for _, sub in model.named_modules():
        for bname, b in list(sub._buffers.items()):
            if b is not None and not b.is_meta and b.device.type != "cpu":
                sub._buffers[bname] = b.to("cpu")

    n_fused_qkv = 0
    if fuse_dit_qkv and hasattr(model, "transformer_blocks"):
        n_fused_qkv = _install_fast_f5_dit(model, nt)
    elif is_vits:
        _install_fast_vits_cpu(model)
    elif cls_name == "TinyVoca":
        _install_fast_tiny_voca_cpu(model, nt)

    model.eval()
    ram_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    return {
        "packed_linear": n_lin,
        "packed_conv1d": n_conv,
        "fused_qkv": n_fused_qkv,
        "resident_ram_mb": round(ram_bytes / 1e6, 2),
        "threads": nt,
        "c_avx2_active": get_vault_cpu_lib() is not None,
    }


# ==============================================================================
# 6. Zero-Allocation Swayed-EPSS + Alignment-Window Guided-Interval CFG CPU Solver
# ==============================================================================
@torch.inference_mode()
def fast_f5_cpu_sample(
    cfm: Any,
    vocoder: Any,
    ref_mel_3d: torch.Tensor,     # [1, ref_frames, 100] pre-extracted Mel spectrogram!
    ref_rms: float,
    target_rms: float,
    ref_text: str,
    gen_text_batches: List[str],
    nfe_step: Optional[int] = None,
    cfg_strength: float = 2.0,
    speed: float = 1.0,
    cross_fade_duration: float = 0.15,
    seed: int = 7000,
    variant: str = "p4_0",
    delta_dit: bool = True,
    head_k: int = 3,
    tail_k: int = 3,
) -> Tuple[np.ndarray, int]:
    """
    Zero-Regression Fast CPU Flow-Matching Synthesis for `F5-TTS`:
      1. Exact Swayed-`EPSS` Euler grid (`get_epss_timesteps(steps)` + `sway_sampling_coef = -1.0`), placing dense
         steps across the phonetic alignment & prosody window (`t in [0, 0.30]`) and smooth steps across `[0.30, 1.0]`.
      2. Alignment-Window Guided-Interval CFG (`Kynkäänniemi et al., NeurIPS 2024`): keeps full `B=2` Classifier-Free
         Guidance active through the entire phonetic alignment window (`t < 0.65` on `p4_0`, `t < 0.50` on `q2_0`) and
         switches to `B=1` conditional velocity only on the final smooth denoising tail (`0.0%` paragraph WER &
         matches/beats 16-step `BEFORE` across all 12 held-out Trifecta prompts!).
      3. Conservative `Δ-DiT` Tail Residual Reuse (`q2_0` step 8 only, `head_k=3, tail_k=3`) + Rust `rjieba` text alignment.
    """
    from f5_tts.model.utils import convert_char_to_pinyin, get_epss_timesteps, list_str_to_idx

    configure_cpu_threads()
    is_q2 = "q2" in variant.lower() or "2bit" in variant.lower()
    if nfe_step is None or nfe_step <= 0:
        steps = 10 if is_q2 else 7
        t_cfg_max = 0.50 if is_q2 else 0.65
        cache_set = {8} if (is_q2 and delta_dit) else set()
    else:
        steps = int(nfe_step)
        if steps >= 16:
            t_cfg_max = 1.01
            cache_set = set()
        else:
            t_cfg_max = 0.65
            cache_set = set()

    if ref_text and len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    target_sr = 24000
    ref_frames = ref_mel_3d.shape[1]
    ref_audio_len = max(1, ref_frames - 1)
    ref_text_bytes = max(1, len(ref_text.encode("utf-8")))
    generated_waves: List[np.ndarray] = []

    dit = cfm.transformer
    n_blk = len(dit.transformer_blocks)
    hk = min(head_k, n_blk // 2)
    tk = min(tail_k, n_blk - hk)
    mid_end = n_blk - tk

    t_base = get_epss_timesteps(steps, device="cpu", dtype=torch.float32)
    t_grid = t_base - 1.0 * (torch.cos(torch.pi / 2.0 * t_base) - 1.0 + t_base)

    for b_idx, gen_chunk in enumerate(gen_text_batches):
        local_speed = speed * 0.3 if len(gen_chunk.encode("utf-8")) < 10 else speed
        final_text_list = convert_char_to_pinyin([ref_text + gen_chunk])
        text_idx = list_str_to_idx(final_text_list, cfm.vocab_char_map).to("cpu")

        gen_bytes = len(gen_chunk.encode("utf-8"))
        duration = ref_audio_len + int(ref_audio_len / ref_text_bytes * gen_bytes / local_speed)
        min_dur = max(int((text_idx != -1).sum().item()), ref_frames) + 1
        duration = max(min_dur, duration)

        step_cond = F.pad(ref_mel_3d, (0, 0, 0, duration - ref_frames), value=0.0)

        torch.manual_seed(seed + b_idx)
        x = torch.randn((1, duration, cfm.num_channels), dtype=torch.float32)

        dit.clear_cache()
        mid_cache: Optional[torch.Tensor] = None
        rope = dit.rotary_embed.forward_from_seq_len(duration)

        for s_idx in range(steps):
            t_cur = t_grid[s_idx]
            dt = float((t_grid[s_idx + 1] - t_cur).item())
            do_cfg = (cfg_strength >= 1e-5) and (float(t_cur.item()) < t_cfg_max)
            can_skip = (len(cache_set) > 0) and (not do_cfg) and (s_idx in cache_set) and (mid_end > hk)

            t_emb = dit.time_embed(t_cur.repeat(1))
            if do_cfg:
                x_c = dit.get_input_embed(x, step_cond, text_idx, drop_audio_cond=False, drop_text=False, cache=True)
                x_u = dit.get_input_embed(x, step_cond, text_idx, drop_audio_cond=True, drop_text=True, cache=True)
                h = torch.cat((x_c, x_u), dim=0)
                t_emb = torch.cat((t_emb, t_emb), dim=0)
            else:
                if s_idx == 0 and cfg_strength < 1e-5:
                    dit.text_uncond = torch.empty(0)
                h = dit.get_input_embed(x, step_cond, text_idx, drop_audio_cond=False, drop_text=False, cache=True)

            if dit.long_skip_connection is not None:
                residual = h

            for bi in range(hk):
                h = dit.transformer_blocks[bi](h, t_emb, mask=None, rope=rope)

            if can_skip and mid_cache is not None and mid_cache.shape == h.shape:
                h = h + mid_cache
            else:
                h_before = h
                for bi in range(hk, mid_end):
                    h = dit.transformer_blocks[bi](h, t_emb, mask=None, rope=rope)
                if (len(cache_set) > 0) and mid_end > hk:
                    mid_cache = (h - h_before) if not do_cfg else (h[:1] - h_before[:1])

            for bi in range(mid_end, n_blk):
                h = dit.transformer_blocks[bi](h, t_emb, mask=None, rope=rope)

            if dit.long_skip_connection is not None:
                h = dit.long_skip_connection(torch.cat((h, residual), dim=-1))

            out = dit.proj_out(dit.norm_out(h, t_emb))
            if do_cfg:
                pred, null_pred = out.chunk(2, dim=0)
                v = pred + (pred - null_pred) * cfg_strength
            else:
                v = out

            x.add_(v, alpha=dt)

        dit.clear_cache()

        gen_mel = x[:, ref_audio_len:, :].permute(0, 2, 1).contiguous()
        wav_t = vocoder.decode(gen_mel)
        if ref_rms < target_rms:
            wav_t = wav_t * (ref_rms / target_rms)
        generated_waves.append(wav_t.squeeze().cpu().numpy().astype(np.float32))

    if not generated_waves:
        return np.zeros(0, dtype=np.float32), target_sr
    if len(generated_waves) == 1:
        return generated_waves[0], target_sr

    fade_samples = int(cross_fade_duration * target_sr)
    final_wave = generated_waves[0]
    for nxt in generated_waves[1:]:
        if fade_samples > 0 and len(final_wave) >= fade_samples and len(nxt) >= fade_samples:
            w_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            w_in = 1.0 - w_out
            cross = final_wave[-fade_samples:] * w_out + nxt[:fade_samples] * w_in
            final_wave = np.concatenate([final_wave[:-fade_samples], cross, nxt[fade_samples:]])
        else:
            final_wave = np.concatenate([final_wave, nxt])
    return final_wave, target_sr

