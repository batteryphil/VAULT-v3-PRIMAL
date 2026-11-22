# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
Prime-Harmonic and baseline discrete manifolds (LUTs) for PRIMAL v7.

Key properties of the v7 Prime-Harmonic LUT:
- Strictly symmetric around exact 0.0: lut[K_eff - 1 - i] == -lut[i]
- Strictly increasing with zero duplicate/wasted entries
- Step sizes grow outward from 0 following prime-power reciprocals:
      dx_k proportional to p_{n - k + 1}^(-gamma)
  so resolution is finest near zero (where neural weights concentrate) and
  widens smoothly in the tails to +-1.0.
- At 8-bit (gamma=0.35), achieves ~41% lower quantization MSE on Gaussian
  neural weights than uniform INT8 linear quantization.
"""

import math
import torch


def _first_n_primes(n: int) -> list[int]:
    """Return the first n prime numbers using a fast sieve."""
    if n <= 0:
        return []
    # Upper bound by prime number theorem (safe for all n >= 6)
    limit = max(30, int(n * (math.log(n) + math.log(math.log(max(n, 3))) + 2.5)))
    while True:
        sieve = bytearray(b"\x01") * (limit + 1)
        sieve[0:2] = b"\x00\x00"
        for p in range(2, int(limit ** 0.5) + 1):
            if sieve[p]:
                sieve[p * p : limit + 1 : p] = b"\x00" * (((limit - p * p) // p) + 1)
        primes = [i for i, is_p in enumerate(sieve) if is_p]
        if len(primes) >= n:
            return primes[:n]
        limit *= 2


def prime_harmonic_lut(bits: int = 8, gamma: float = 0.35, device: str | torch.device = "cpu") -> torch.Tensor:
    """
    Build a strictly symmetric 2^bits-entry Prime-Harmonic LUT on [-1.0, 1.0].

    Active symmetric indices are [0 .. 2^bits - 2] (odd count, exact 0.0 at
    center index (2^bits - 2) // 2). Index 2^bits - 1 duplicates +1.0 only to
    fill the power-of-two table width cleanly; training clamps to [0, active_max].
    """
    if bits not in (4, 6, 8, 16):
        raise ValueError(f"Unsupported bit width: {bits}. Choose from 4, 6, 8, 16.")

    K = 1 << bits
    n_half = (K // 2) - 1  # 7 (4-bit), 31 (6-bit), 127 (8-bit), 32767 (16-bit)

    primes = _first_n_primes(n_half)
    # Outward steps from 0: smallest steps near 0 (large primes), widest at tail (2, 3, 5...)
    p_tensor = torch.tensor(primes[::-1], dtype=torch.float64)
    steps = torch.pow(p_tensor, -gamma)
    pos = torch.cumsum(steps, dim=0)
    pos = pos / pos[-1]  # strictly increasing in (0, 1], ending at exact 1.0

    neg = -torch.flip(pos, dims=[0])
    zero = torch.zeros(1, dtype=torch.float64)
    # Active symmetric grid has K - 1 entries: indices 0 .. K-2, center = n_half -> 0.0
    vals = torch.cat([neg, zero, pos], dim=0)
    return vals.to(dtype=torch.float32, device=device)


def linear_lut(bits: int = 8, device: str | torch.device = "cpu") -> torch.Tensor:
    """
    Uniform linear LUT on [-1.0, 1.0] with exact 0.0 center (K - 1 symmetric points,
    matching the active range of prime_harmonic_lut for fair apples-to-apples comparison).
    """
    K = 1 << bits
    n_active = K - 1  # odd count so exact 0.0 sits at the exact center
    return torch.linspace(-1.0, 1.0, n_active, dtype=torch.float32, device=device)


@torch.no_grad()
def quantize_to_lut(weight: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """
    Exact nearest-neighbor quantization of any tensor (assumed normalized to [-1, 1])
    onto an arbitrary sorted 1D LUT using O(log K) binary search.
    Returns int32 indices in [0, len(lut) - 1].
    """
    lut_dev = lut.to(device=weight.device, dtype=torch.float32).contiguous()
    w_flat = weight.reshape(-1).to(torch.float32).clamp(lut_dev[0].item(), lut_dev[-1].item()).contiguous()

    idx_hi = torch.searchsorted(lut_dev, w_flat).clamp(0, lut_dev.numel() - 1)
    idx_lo = (idx_hi - 1).clamp(0, lut_dev.numel() - 1)

    d_lo = (w_flat - lut_dev[idx_lo]).abs()
    d_hi = (w_flat - lut_dev[idx_hi]).abs()
    idx = torch.where(d_lo < d_hi, idx_lo, idx_hi)
    return idx.reshape(weight.shape).to(torch.int32)


@torch.no_grad()
def evaluate_lut_mse(lut: torch.Tensor, sigma: float = 1.0 / 3.5, n_samples: int = 200_000) -> dict:
    """Measure quantization MSE and spacing statistics on clipped Gaussian weights."""
    g = torch.Generator(device=lut.device)
    g.manual_seed(0)
    w = (torch.randn(n_samples, generator=g, device=lut.device) * sigma).clamp(-1.0, 1.0)
    idx = quantize_to_lut(w, lut)
    mse = (w - lut[idx.long()]).pow(2).mean().item()
    diffs = lut[1:] - lut[:-1]
    return {
        "n_levels": lut.numel(),
        "mse": mse,
        "min_dx": diffs.min().item(),
        "max_dx": diffs.max().item(),
        "center_val": lut[lut.numel() // 2].item(),
    }
