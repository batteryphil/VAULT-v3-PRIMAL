#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
Exact 82-probe O(1) 8K -> 100.4K VAULT v3 Long-Context & Adversarial Suite
on VAULT-v3-3B PRIMAL (V3_Q8_0 and Prime-Harmonic Hessian V3_P4_0).

Uses the exact `eval_long_context_retrieval(..., run_100k=True)` harness from
`benchmarks/bench_vault_v3_p4_eval.py` for 100% apples-to-apples comparison
with VAULT-v3-1.5B, measuring flat O(1) peak VRAM all the way to 100,400 tokens!
"""

import gc
import json
import os
import sys
import time

if "expandable_segments" in os.environ.get("PYTORCH_ALLOC_CONF", ""):
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import torch
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from bench_vault_v3_p4_eval import eval_long_context_retrieval, load_vlt3_model  # noqa: E402

QWEN_3B = (
    "Qwen/Qwen2.5-3B-Instruct"
    "snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
)
MODELS = [
    ("VAULT-v3-3B-PRIMAL-Q8_0", os.path.join(ROOT, "dist", "vault_v3_3b_q8_0_primal.vlt3")),
    ("VAULT-v3-3B-PRIMAL-P4_0", os.path.join(ROOT, "dist", "vault_v3_3b_p4_0_primal.vlt3")),
]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(QWEN_3B)
    results = {}
    for label, path in MODELS:
        print(f"\n==================== [O(1) 8K->100.4K SUITE: {label}] ====================", flush=True)
        t0 = time.perf_counter()
        model = load_vlt3_model(path, dev)
        print(
            f"  Loaded {os.path.basename(path)} | window={model.cfg.window} seg_budget={model.cfg.seg_budget} "
            f"seg_layers={model.cfg.seg_layers} | base_vram={torch.cuda.memory_allocated()/1024**3:.3f} GiB",
            flush=True,
        )
        res = eval_long_context_retrieval(model, tok, run_100k=True)
        res["file_mb"] = round(os.path.getsize(path) / (1024 ** 2), 2)
        res["eval_sec"] = round(time.perf_counter() - t0, 1)
        results[label] = res
        del model
        gc.collect()
        torch.cuda.empty_cache()

    out_path = os.path.join(ROOT, "benchmarks", "vault_v3_3b_primal_o1_100k.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved O(1) 100K report -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
