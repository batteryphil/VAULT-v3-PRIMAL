#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
Master 1-Command Pipeline Orchestrator for VAULT-v3-PRIMAL:

  1. Instant Prebuilt C99 CPU Inference (Zero Python runtime needed after download):
       python3 run_pipeline.py --mode quickstart-c99 --model deepseek-1.5b --prompt "Solve: 3x + 7 = 22"
       python3 run_pipeline.py --mode quickstart-c99 --model deepseek-1.5b --nothink --prompt "Is Taiwan a de facto independent country?"
       python3 run_pipeline.py --mode quickstart-c99 --model qwen-3b --bench 128

  2. Instant Prebuilt Python `transformers` Inference (True O(1) 100K+ Segment KV Memory):
       python3 run_pipeline.py --mode quickstart-python --model deepseek-1.5b --prompt "Find x^2 - y^2 = 15 in integers."
       python3 run_pipeline.py --mode quickstart-python --model qwen-3b --prompt "Explain RoPE re-rotation."

  3. Full End-to-End Reproducible Training, Quantization, SegmentScorer & Packaging Pipeline from Upstream Base Weights:
       python3 run_pipeline.py --mode full-build --model deepseek-1.5b
       python3 run_pipeline.py --mode full-build --model qwen-3b
       (Add `--dry-run` to preview every stage and exact command without executing.)

  4. Freestanding C99 & Unit-Test Verification Gate:
       python3 run_pipeline.py --mode verify
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "deepseek-1.5b": {
        "upstream_base": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "hf_main_repo": "batteryphil/VAULT-v3-DeepSeek-R1-1.5B-PRIMAL",
        "hf_vlt3_repo": "batteryphil/VAULT-v3-DeepSeek-R1-1.5B-PRIMAL-VLT3",
        "p4_filename": "vault_v3_deepseek_r1_1_5b_p4_0_primal.vlt3",
        "q8_filename": "vault_v3_deepseek_r1_1_5b_q8_0_primal.vlt3",
    },
    "qwen-3b": {
        "upstream_base": "Qwen/Qwen2.5-3B-Instruct",
        "hf_main_repo": "batteryphil/VAULT-v3-3B-PRIMAL",
        "hf_vlt3_repo": "batteryphil/VAULT-v3-3B-PRIMAL-VLT3",
        "p4_filename": "vault_v3_3b_p4_0_primal.vlt3",
        "q8_filename": "vault_v3_3b_q8_0_primal.vlt3",
    },
}


def _run_step(title: str, cmd: List[str], env: Dict[str, str] | None = None, dry_run: bool = False) -> None:
    banner = f"\n{'=' * 84}\n[PIPELINE STAGE] {title}\n  $ {' '.join(shlex.quote(c) for c in cmd)}\n{'=' * 84}"
    print(banner, flush=True)
    if dry_run:
        return
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=REPO_ROOT, env=merged_env, check=True)


def run_quickstart_c99(args: argparse.Namespace) -> None:
    spec = MODEL_SPECS[args.model]
    fname = spec["p4_filename"] if args.quant == "p4_0" else spec["q8_filename"]
    local_vlt3 = os.path.join(REPO_ROOT, "dist", fname)

    # 1. Build freestanding C99 host binary
    _run_step("Compile Freestanding Zero-Dependency C99 Engine (runtime/vault3_host)", ["make", "-C", "runtime"], dry_run=args.dry_run)

    # 2. Download .vlt3 binary from Hugging Face if not already present locally
    if not os.path.isfile(local_vlt3):
        dl_cmd = [
            sys.executable,
            "-c",
            (
                "from huggingface_hub import hf_hub_download; "
                f"p = hf_hub_download(repo_id={spec['hf_vlt3_repo']!r}, filename={fname!r}, local_dir='dist'); "
                "print('[DOWNLOADED]', p)"
            ),
        ]
        _run_step(f"Download Prebuilt {args.quant.upper()} .vlt3 Binary from Hugging Face ({spec['hf_vlt3_repo']})", dl_cmd, dry_run=args.dry_run)

    # 3. Execute vault3_host
    host_bin = os.path.join(REPO_ROOT, "runtime", "vault3_host")
    if args.bench > 0:
        cmd = [host_bin, local_vlt3, "--bench", str(args.bench), "--threads", str(args.threads)]
        _run_step(f"Run Freestanding C99 Speed & RSS Benchmark ({args.model}, {args.quant})", cmd, dry_run=args.dry_run)
    else:
        prompt = args.prompt or "Solve step by step: Find all integer solutions to x^2 - y^2 = 15."
        cmd = [host_bin, local_vlt3, "--chat", prompt, "--n", str(args.max_new_tokens), "--threads", str(args.threads)]
        if args.nothink:
            cmd.append("--nothink")
        _run_step(f"Run Freestanding C99 Inference ({args.model}, {args.quant})", cmd, dry_run=args.dry_run)


def run_quickstart_python(args: argparse.Namespace) -> None:
    spec = MODEL_SPECS[args.model]
    prompt = args.prompt or "Explain why O(1) Segment-Evicted KV Memory prevents VRAM exhaustion at 100K tokens."
    py_code = (
        "import torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        f"repo = {spec['hf_main_repo']!r}\n"
        "dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32\n"
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)\n"
        "model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=dtype).to(device).eval()\n"
        f"ans = model.chat(tok, {prompt!r}, max_new_tokens={args.max_new_tokens}, nothink={bool(args.nothink)!r})\n"
        "print('\\n=== RESPONSE ===\\n' + ans)\n"
    )
    _run_step(f"Run Python `transformers` O(1) 100K+ Inference ({spec['hf_main_repo']})", [sys.executable, "-c", py_code], dry_run=args.dry_run)


def run_full_build(args: argparse.Namespace) -> None:
    py = sys.executable
    _run_step("Stage 0: Compile & Freestanding-Audit C99 Runtime (scalar C99 + AVX2/NEON)", ["make", "-C", "runtime"], dry_run=args.dry_run)

    if args.model == "deepseek-1.5b":
        wsp_dir = args.wsp_dir or os.path.join(REPO_ROOT, "models", "deepseek-r1-1.5b-wsp")
        _run_step(
            "Stage 1 (WSP): Direct Weight-Space Uncensoring & 18-Probe Verification Gate",
            [py, "-u", "wsp/wsp_deepseek_1_5b.py", "--base-model", MODEL_SPECS["deepseek-1.5b"]["upstream_base"], "--save-dir", wsp_dir],
            dry_run=args.dry_run,
        )
        _run_step(
            "Stages 2-4: On-Policy Stream -> Shadowless PRIMAL v3.1 (V3_Q8_0) -> Hessian V3_P4_0 -> O(1) SegmentScorer -> Post-PRIMAL Q-ROME -> HF Packaging & Full 100K Eval",
            [py, "-u", "primal/run_deepseek_1_5b_primal_pipeline.py"],
            env={"WSP_MODEL_DIR": wsp_dir},
            dry_run=args.dry_run,
        )
        _run_step(
            "Stage 5: Render Empirical Hugging Face Model Cards (README.md)",
            [py, "-u", "tools/upload_hf_deepseek_1_5b.py", "--cards-only"],
            dry_run=args.dry_run,
        )
    else:
        _run_step(
            "Stage 1: Train 3B O(1) SegmentScorer MLP (seg_layers=10,20,30) through APL Causal EMA Habituation",
            [py, "-u", "primal/train_vault_v3_3b_seg_scorer.py", "--out_dir", "runs/vault_v3_3b_seg_scorer"],
            dry_run=args.dry_run,
        )
        _run_step(
            "Stage 2: On-Policy CoT Stream + Full-Parameter Shadowless PRIMAL v3.1 (3.09B V3_Q8_0) + Activation-Hessian V3_P4_0 Export",
            [py, "-u", "primal/run_primal_v3_1_onpolicy_and_controls.py"],
            dry_run=args.dry_run,
        )
        _run_step(
            "Stage 3: Build Self-Contained Hugging Face Release Repositories (dist/hf_vault_v3_3b & dist/hf_vault_v3_3b_vlt3)",
            [py, "-u", "tools/build_hf_vault_v3_3b.py"],
            dry_run=args.dry_run,
        )
        _run_step(
            "Stage 4: Run True O(1) 8K -> 100.5K Long-Context Verification Suite",
            [py, "-u", "benchmarks/bench_vault_v3_3b_o1_100k.py"],
            dry_run=args.dry_run,
        )


def run_verify(args: argparse.Namespace) -> None:
    _run_step("Verify Freestanding Zero-Libc C99 Symbols (Scalar + AVX2 + AArch64 NEON)", ["make", "-C", "runtime"], dry_run=args.dry_run)
    _run_step("Run VLT3 Binary Encoding, BPE Tokenizer & C99 Parity Unit Suite", [sys.executable, "-u", "tests/test_vlt3_runtime.py"], dry_run=args.dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="VAULT-v3-PRIMAL Unified Pipeline Runner (Quickstart Inference, Full From-Scratch Build, & Verification)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--mode",
        choices=["quickstart-c99", "quickstart-python", "full-build", "verify"],
        default="quickstart-c99",
        help="Pipeline stage/workflow to execute (default: quickstart-c99)",
    )
    ap.add_argument(
        "--model",
        choices=["deepseek-1.5b", "qwen-3b"],
        default="deepseek-1.5b",
        help="Target model family: uncensored DeepSeek-R1-1.5B-PRIMAL or Qwen2.5-3B-PRIMAL",
    )
    ap.add_argument("--quant", choices=["p4_0", "q8_0"], default="p4_0", help=".vlt3 quantization format for quickstart-c99 (default: p4_0)")
    ap.add_argument("--prompt", default="", help="Prompt for quickstart-c99 or quickstart-python")
    ap.add_argument("--nothink", action="store_true", help="Enable instant zero-latency --nothink mode (DeepSeek-1.5B)")
    ap.add_argument("--bench", type=int, default=0, help="Run N-token C99 decode speed/RSS benchmark instead of chat")
    ap.add_argument("--max-new-tokens", type=int, default=384, help="Maximum new tokens to generate")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8)), help="CPU thread count for C99 runtime")
    ap.add_argument("--wsp-dir", default="", help="Optional custom output/input directory for WSP-uncensored checkpoint")
    ap.add_argument("--dry-run", action="store_true", help="Print every pipeline stage and exact command without executing")
    args = ap.parse_args()

    if args.mode == "quickstart-c99":
        run_quickstart_c99(args)
    elif args.mode == "quickstart-python":
        run_quickstart_python(args)
    elif args.mode == "full-build":
        run_full_build(args)
    elif args.mode == "verify":
        run_verify(args)


if __name__ == "__main__":
    main()
