# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
PRIMAL v3.1: Shadowless Full-Parameter Discrete LLM Training & Prime-Harmonic Manifolds.
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
from .vault_primal import (
    VAULTPrimalConfig,
    VAULTPrimalForCausalLM,
    Q8StepEngine,
    load_pretrained_qwen2_into_primal_q8,
    export_vault_primal_vlt3,
)

__all__ = [
    "prime_harmonic_lut",
    "linear_lut",
    "quantize_to_lut",
    "evaluate_lut_mse",
    "PrimeLinear",
    "PrimeTiedEmbedding",
    "PrimeStepEngine",
    "VAULTPrimalConfig",
    "VAULTPrimalForCausalLM",
    "Q8StepEngine",
    "load_pretrained_qwen2_into_primal_q8",
    "export_vault_primal_vlt3",
]
