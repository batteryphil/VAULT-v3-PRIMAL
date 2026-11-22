#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)
"""
Losslessly packs GPTQ + PRIMAL 1.58-Bit Ternary checkpoints (`mlp_1_58_g32.pt` and
`full_body_1_58_g64.pt`) into CRC32-verified `.vlt3` binaries, benchmarks the
zero-multiply freestanding C99 runtime (`runtime/vault3_host`), and verifies
`O(1)` long-context NIAH passkey retrieval at 16K, 32K, and 100K tokens.
"""

import gc
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.vault_v3_model import VAULTv3ForCausalLM
from tools import vlt3

Q8_TEACHER_VLT3 = os.path.join(REPO_ROOT, "dist", "vault_v3_deepseek_r1_1_5b_q8_0_primal.vlt3")
P4_TEACHER_VLT3 = os.path.join(REPO_ROOT, "dist", "vault_v3_deepseek_r1_1_5b_p4_0_primal.vlt3")
HF_TOK_DIR = os.path.join(REPO_ROOT, "dist", "hf_vault_v3_deepseek_r1_1_5b")
TERNARY_DIR = os.path.join(REPO_ROOT, "runs", "ternary_1_58")
REPORT_JSON = os.path.join(REPO_ROOT, "benchmarks", "results", "vault_v3_deepseek_1_5b_ternary_1_58_report.json")
HOST_BIN = os.path.join(REPO_ROOT, "runtime", "vault3_host")


def export_ternary_vlt3(
    base_sd: Dict[str, torch.Tensor],
    cfg,
    pt_path: str,
    out_vlt3: str,
    fmt: str,
    group_size: int,
    model_name: str,
) -> Dict[str, Any]:
    t0 = time.time()
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    td: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = ckpt["ternary_data"]
    norm_w: Dict[str, torch.Tensor] = ckpt.get("norm_weights", {})
    bias_w: Dict[str, torch.Tensor] = ckpt.get("bias_weights", {})

    sd = {k: v.clone() for k, v in base_sd.items()}
    for extra_dict in (norm_w, bias_w):
        for nk, nv in extra_dict.items():
            v_key = nk[len("model."):] if nk.startswith("model.") else nk
            if v_key in sd:
                sd[v_key] = nv.detach().float().cpu().clone()

    dt_code = vlt3.T1_58_G32 if group_size == 32 else vlt3.T1_58_G64
    prepacked: Dict[str, Tuple[int, bytes]] = {}
    ternary_params = 0
    zero_trits = 0
    for mod_key, (q_cpu, d_cpu) in td.items():
        w_key = f"{mod_key}.weight"
        q_np = q_cpu.detach().cpu().numpy().astype(np.int8)
        d_np = d_cpu.detach().cpu().numpy().astype(np.float16)
        raw_bytes = vlt3.pack_t1_58(q_np, d_np, group_size)
        prepacked[w_key] = (dt_code, raw_bytes)
        ternary_params += int(q_np.size)
        zero_trits += int((q_np == 0).sum())

    meta_w = vlt3.write_vlt3(
        out_vlt3,
        sd,
        cfg,
        HF_TOK_DIR,
        fmt=fmt,
        model_name=model_name,
        prepacked=prepacked,
    )
    # Full CRC32 + NaN/Inf header/payload verification
    vf = vlt3.read_vlt3(out_vlt3, check_crc=True)
    total_params = sum(int(np.prod(e["dims"][: e["ndim"]])) for e in vf.entries)
    weight_bytes = sum(int(e["nbytes"]) for e in vf.entries)
    fsize = int(meta_w["file_size"])
    res = {
        "vlt3_file": os.path.basename(out_vlt3),
        "file_bytes": fsize,
        "file_mb": round(fsize / 1e6, 2),
        "file_gib": round(fsize / (1024 ** 3), 4),
        "total_params": total_params,
        "ternary_params": ternary_params,
        "ternary_sparsity_zero_pct": round(100.0 * zero_trits / max(1, ternary_params), 2),
        "overall_effective_bpw": round(8.0 * weight_bytes / total_params, 3),
        "crc32_verified": True,
        "export_seconds": round(time.time() - t0, 2),
    }
    print(
        f"[EXPORT {res['vlt3_file']}] {res['file_mb']:.2f} MB ({res['file_gib']:.3f} GiB) | "
        f"eff_bpw={res['overall_effective_bpw']} | zero_trits={res['ternary_sparsity_zero_pct']:.1f}% | CRC32=OK",
        flush=True,
    )
    return res


def bench_c99_vlt3(vlt3_path: str, run_chat: bool = True) -> Dict[str, Any]:
    best_tok_s = 0.0
    best_rec: Dict[str, Any] = {}
    for threads in (12, 24):
        cpu_mask = "0-11" if threads == 12 else "0-23"
        for _ in range(3):
            cmd = [
                "taskset", "-c", cpu_mask,
                HOST_BIN, vlt3_path,
                "--bench-prefill", "256",
                "--bench", "128",
                "--threads", str(threads),
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
            for line in (out.stdout or "").splitlines():
                line = line.strip()
                if line.startswith("{") and '"decode_tok_s"' in line:
                    jb = json.loads(line)
                    tok_s = float(jb["decode_tok_s"])
                    if tok_s > best_tok_s:
                        best_tok_s = tok_s
                        best_rec = {
                            "best_threads": threads,
                            "decode_tok_s": tok_s,
                            "decode_ms_tok": float(jb["ms_per_tok"]),
                            "prefill_tok_s": float(jb["prefill_tok_s"]),
                            "peak_vmhwm_mib": round(float(jb["peak_rss_mb"]) / 1.048576, 2),
                            "peak_vmhwm_mb_dec": float(jb["peak_rss_mb"]),
                        }
    if run_chat:
        c_math = subprocess.run(
            [HOST_BIN, vlt3_path, "--verify", "--chat", "Compute 17 * 23 step by step concisely.", "--nothink", "--n", "80", "--temp", "0", "--threads", "24"],
            capture_output=True, text=True, check=True,
        )
        c_tw = subprocess.run(
            [HOST_BIN, vlt3_path, "--chat", "Is Taiwan a de facto independent country? Answer directly in two sentences.", "--nothink", "--n", "80", "--temp", "0", "--threads", "24"],
            capture_output=True, text=True, check=True,
        )
        best_rec["c99_math_sample"] = c_math.stdout.strip()[:220]
        best_rec["c99_taiwan_sample"] = c_tw.stdout.strip()[:220]
    print(
        f"[C99 BENCH {os.path.basename(vlt3_path)}] {best_rec['decode_tok_s']:.2f} tok/s "
        f"({best_rec['decode_ms_tok']:.2f} ms/tok, threads={best_rec['best_threads']}) | "
        f"VmHWM={best_rec['peak_vmhwm_mib']:.2f} MiB",
        flush=True,
    )
    return best_rec


@torch.no_grad()
def eval_niah_passkey_vlt3(vlt3_path: str, tok) -> Dict[str, Any]:
    vf = vlt3.read_vlt3(vlt3_path, check_crc=False)
    cfg = vlt3.vlt3_config(vf)
    m = VAULTv3ForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    sd = {k: v.to(device="cuda", dtype=(torch.float32 if v.ndim == 1 or k.startswith("seg_scorer.") else torch.bfloat16))
          for k, v in vlt3.vlt3_to_state_dict(vf).items()}
    m.load_state_dict(sd, strict=True)

    hay_unit = (
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
        "Water boils at one hundred degrees Celsius at standard atmospheric pressure. "
    )
    unit_ids = tok.encode(hay_unit, add_special_tokens=False)
    tests = [
        (16384, 0.25, "739184"),
        (32768, 0.25, "482915"),
        (100000, 0.25, "916427"),
    ]
    hits = 0
    cases = []
    for ctx_target, depth, pkey in tests:
        needle = f" Remember this secret passkey number: {pkey}. The secret passkey is {pkey}. "
        prompt_q = (
            f"<｜User｜>What is the 6-digit secret passkey number mentioned earlier in the text? "
            f"Reply with the exact digits.<｜Assistant｜><think>\n</think>\n\nThe secret passkey number is "
        )
        n_units = max(10, ctx_target // len(unit_ids))
        pre_u = int(n_units * depth)
        post_u = n_units - pre_u
        ids = (
            [151646]
            + unit_ids * pre_u
            + tok.encode(needle, add_special_tokens=False)
            + unit_ids * post_u
            + tok.encode(prompt_q, add_special_tokens=False)
        )
        inp = torch.tensor([ids], dtype=torch.long, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        lg, st = m.prefill(inp, vault_on=True)
        gen = []
        nxt = int(lg[0].argmax().item())
        gen.append(nxt)
        for _ in range(15):
            lg2 = m.step(torch.tensor([nxt], dtype=torch.long, device="cuda"), st)
            nxt = int(lg2[0].argmax().item())
            if nxt == 151643:
                break
            gen.append(nxt)
        gtxt = tok.decode(gen, skip_special_tokens=True).strip()
        ok = pkey in gtxt
        if ok:
            hits += 1
        vram_mib = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
        cases.append({
            "total_tokens": inp.shape[1],
            "depth": depth,
            "passkey": pkey,
            "output": gtxt[:60],
            "exact_hit": ok,
            "peak_vram_mib": vram_mib,
        })
        print(f"  [NIAH {os.path.basename(vlt3_path)} @ {inp.shape[1]:,} tok] passkey={pkey} -> '{gtxt[:40]}' | hit={ok} | VRAM={vram_mib} MiB", flush=True)

    del m, sd, vf
    gc.collect()
    torch.cuda.empty_cache()
    return {"hits": hits, "total": len(tests), "accuracy_pct": round(100.0 * hits / len(tests), 2), "cases": cases}


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_TOK_DIR, trust_remote_code=True)
    print("=== Loading Q8_0 Teacher VLT3 Base State (including trained SegmentScorer) ===", flush=True)
    vf_base = vlt3.read_vlt3(Q8_TEACHER_VLT3, check_crc=False)
    cfg = vlt3.vlt3_config(vf_base)
    base_sd = vlt3.vlt3_to_state_dict(vf_base)
    del vf_base

    hyb_pt = os.path.join(TERNARY_DIR, "mlp_1_58_g32.pt")
    pure_pt = os.path.join(TERNARY_DIR, "full_body_1_58_g64.pt")
    hyb_vlt3 = os.path.join(REPO_ROOT, "dist", "vault_v3_deepseek_r1_1_5b_t1_58_hybrid.vlt3")
    pure_vlt3 = os.path.join(REPO_ROOT, "dist", "vault_v3_deepseek_r1_1_5b_t1_58_pure.vlt3")

    exp_hyb = export_ternary_vlt3(
        base_sd, cfg, hyb_pt, hyb_vlt3, fmt="t1_58_hybrid", group_size=32,
        model_name="VAULT-v3-DeepSeek-1.5B-T1.58-Hybrid",
    )
    exp_pure = export_ternary_vlt3(
        base_sd, cfg, pure_pt, pure_vlt3, fmt="t1_58_g64", group_size=64,
        model_name="VAULT-v3-DeepSeek-1.5B-T1.58-Pure",
    )
    del base_sd
    gc.collect()

    print("\n=== Benchmarking Zero-Multiply Freestanding C99 Speed Curve (Q8_0 -> P4_0 -> T1.58-Hybrid -> T1.58-Pure) ===", flush=True)
    c99_q8 = bench_c99_vlt3(Q8_TEACHER_VLT3, run_chat=False)
    c99_p4 = bench_c99_vlt3(P4_TEACHER_VLT3, run_chat=False)
    c99_hyb = bench_c99_vlt3(hyb_vlt3, run_chat=True)
    c99_pure = bench_c99_vlt3(pure_vlt3, run_chat=True)

    print("\n=== Testing O(1) Long-Context NIAH Passkey Rescue (16K / 32K / 100K) on 1.58-Bit VLT3 Binaries ===", flush=True)
    niah_hyb = eval_niah_passkey_vlt3(hyb_vlt3, tok)
    niah_pure = eval_niah_passkey_vlt3(pure_vlt3, tok)

    rep: Dict[str, Any] = {}
    if os.path.exists(REPORT_JSON):
        with open(REPORT_JSON) as f:
            rep = json.load(f)

    rep["vlt3_c99_and_niah"] = {
        "q8_0_reference": c99_q8,
        "p4_0_reference": c99_p4,
        "mlp_1_58_hybrid_g32": {
            "export": exp_hyb,
            "c99_bench": c99_hyb,
            "niah_o1_passkey": niah_hyb,
        },
        "full_body_1_58_g64": {
            "export": exp_pure,
            "c99_bench": c99_pure,
            "niah_o1_passkey": niah_pure,
        },
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\n[COMPLETE] Updated full 1.58-bit report -> {REPORT_JSON}", flush=True)


if __name__ == "__main__":
    main()
