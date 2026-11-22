"""
PRIMAL v7: Clean-Room Shadowless Discrete LLM Training on Prime-Harmonic Manifolds.
"""

from .manifolds import (
    prime_harmonic_lut,
    linear_lut,
    quantize_to_lut,
    evaluate_lut_mse,
)
from .discrete_ops import (
    PrimeLinear,
    PrimeTiedEmbedding,
    PrimeStepEngine,
)
from .model import PrimeConfig, PrimeTransformer

__all__ = [
    "prime_harmonic_lut",
    "linear_lut",
    "quantize_to_lut",
    "evaluate_lut_mse",
    "PrimeLinear",
    "PrimeTiedEmbedding",
    "PrimeStepEngine",
    "PrimeConfig",
    "PrimeTransformer",
]
