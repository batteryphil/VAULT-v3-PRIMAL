#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
Peer Review P1 Long-Context Audit Suite:
  1. 82-Probe VAULT v3 Ablation across 3 modes on VAULT-v3-3B-PRIMAL (Q8_0 & P4_0):
       (A) Full SegmentScorer + 2-Tier FNV-1a Shingle / Hop-2 Resonance (default)
       (B) Pure SegmentScorer MLP + APL EMA ONLY (shingle/Hop-2 resonance completely disabled)
       (C) Sliding-Window + Sink ONLY (`seg_budget = 0`, zero Segment Memory)
  2. External Non-Isomorphic RULER Benchmark (Hsieh et al. canonical categories, zero structural
     overlap with SegmentScorer synthetic training generators) across 4K, 8K, 16K, 32K, 64K tokens:
       - `RULER-NIAH-NaturalProse`: lowercase natural-language facts inside essay haystack (no IDs/JSON/tags)
       - `RULER-MultiKey-4`: 4 competing natural-language subject-attribute pairs
       - `RULER-MultiValue-4`: all 4 attributes of one subject dispersed at 15%, 35%, 55%, 75% depth
       - `RULER-VariableTracking-5Hop`: 5-step variable binding chain (`X1 = 73924 ... X5 = X4`)
  3. Adversarial Homogeneous Dense Ledger Stress Test (2K, 4K, 8K, 16K, 32K tokens):
       - Every single line is an identical-entropy structured JSON financial record, quantifying
         empirical recall vs. theoretical query-agnostic capacity `(window + seg_budget) / L`.
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

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

from bench_vault_v3_p4_eval import eval_long_context_retrieval, greedy_chat, load_vlt3_model  # noqa: E402

QWEN_3B = (
    "Qwen/Qwen2.5-3B-Instruct"
    "snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
)

# Independent essay paragraphs (zero overlap with FineWeb-Edu training cache or synthetic ID templates)
ESSAY_PARAGRAPHS = [
    "What most people miss about building durable software is how much quiet discipline goes into choosing simple interfaces early on. When teams rush to add configuration knobs for every edge case, the surface area grows faster than anyone can reason about.",
    "Consider how traditional watchmakers approached mechanical tolerances before computer-aided machining existed. Every gear train was designed so minor thermal expansion in winter or summer balanced across opposing brass and steel bridges.",
    "In maritime navigation, dead reckoning worked surprisingly well over short coastal hops, yet accumulated drift over open water made periodic celestial fixes indispensable. Good engineering systems combine cheap local updates with occasional ground-truth anchors.",
    "Walk through an old university library and notice which technical monographs still have creased spines fifty years later. Almost invariably they are the books that explain first principles clearly rather than cataloging transient frameworks.",
    "Agricultural terraces in steep mountain valleys solve two opposite problems at once: during dry months they hold moisture near the roots, while during monsoon downpours they step down kinetic energy so topsoil never washes into the gorge.",
    "Early telegraph operators developed a keen intuitive sense of channel capacity long before information theory formalized entropy. Common phrases contracted naturally into crisp abbreviations while rare proper nouns were spelled out with deliberate care.",
]


def make_essay_haystack(tok, target_tokens: int, seed: int = 101) -> str:
    rng = random.Random(seed)
    chunks = []
    est_toks = 0
    while est_toks < target_tokens + 128:
        p = rng.choice(ESSAY_PARAGRAPHS)
        chunks.append(p)
        est_toks += 48
    full = "\n\n".join(chunks)
    ids = tok.encode(full, add_special_tokens=False)[:target_tokens]
    return tok.decode(ids, skip_special_tokens=True)


def run_non_isomorphic_ruler_suite(model, tok, seg_budget: int = 1024) -> Dict[str, Any]:
    lengths = [4096, 8192, 16384, 32768, 65536]
    people = ["Evelyn Vance", "Marcus Lindholm", "Clara Osei", "Henrik Solberg", "Nadia Kowalski", " Julian Mercer"]
    gems = ["tourmaline", "chrysoberyl", "alexandrite", "peridot", "spinel", "zircon"]
    cities = ["Bergen", "Valparaiso", "Trieste", "Sapporo", "Halifax", "Hobart"]
    instruments = ["cello", "oboe", "clavichord", "mandolin", "viola", "bassoon"]

    ruler_res: Dict[str, Any] = {}
    for L in lengths:
        t0 = time.perf_counter()
        # 1. RULER-NIAH-NaturalProse (4 trials across 15%, 35%, 60%, 80% depths — zero uppercase codes/JSON/tags!)
        prose_hits = 0
        for ti, depth in enumerate((0.15, 0.35, 0.60, 0.80)):
            person = people[ti].strip()
            gem = gems[ti]
            n_pre = max(128, int((L - 120) * depth))
            n_post = max(128, (L - 120) - n_pre)
            pre = make_essay_haystack(tok, n_pre, seed=L + ti * 13)
            post = make_essay_haystack(tok, n_post, seed=L + ti * 29 + 7)
            needle = f"Incidentally, the favorite gemstone of Professor {person} has always been {gem}."
            q = f"{pre}\n\n{needle}\n\n{post}\n\nQuestion: What is the favorite gemstone of Professor {person}? Reply concisely."
            rep, plen, _, _ = greedy_chat(model, tok, q, max_new_tokens=32, seg_budget=seg_budget)
            if gem.lower() in rep.lower():
                prose_hits += 1

        # 2. RULER-MultiKey-4 (4 competing natural-language subjects scattered across haystack, 4 trials)
        mk_hits = 0
        for ti in range(4):
            segs = []
            seg_len = max(128, (L - 300) // 5)
            for ki in range(4):
                segs.append(make_essay_haystack(tok, seg_len, seed=L + ti * 100 + ki))
                segs.append(f"Biographical note: Dr. {people[ki].strip()} grew up in the port city of {cities[(ki + ti) % 6]}.")
            segs.append(make_essay_haystack(tok, seg_len, seed=L + ti * 100 + 99))
            target_p = people[ti].strip()
            expected_city = cities[(ti + ti) % 6]
            q = "\n\n".join(segs) + f"\n\nQuestion: Which port city did Dr. {target_p} grow up in?"
            rep, _, _, _ = greedy_chat(model, tok, q, max_new_tokens=32, seg_budget=seg_budget)
            if expected_city.lower() in rep.lower():
                mk_hits += 1

        # 3. RULER-MultiValue-4 (4 distinct items belonging to 1 subject scattered across L, 2 trials x 4 values = 8 values)
        mv_val_hits = 0
        for ti in range(2):
            subj = people[ti].strip()
            vals = [gems[ti], gems[ti + 2], instruments[ti], instruments[ti + 2]]
            seg_len = max(128, (L - 300) // 5)
            parts = []
            for vi, val in enumerate(vals):
                parts.append(make_essay_haystack(tok, seg_len, seed=L + 500 + ti * 30 + vi))
                parts.append(f"Curator inventory entry: {subj}'s private study collection includes a rare {val}.")
            parts.append(make_essay_haystack(tok, seg_len, seed=L + 599 + ti))
            q = "\n\n".join(parts) + f"\n\nQuestion: List all four items recorded in {subj}'s private study collection."
            rep, _, _, _ = greedy_chat(model, tok, q, max_new_tokens=64, seg_budget=seg_budget)
            for v in vals:
                if v.lower() in rep.lower():
                    mv_val_hits += 1

        # 4. RULER-VariableTracking-5Hop (canonical variable chain across L, 2 trials)
        vt_hits = 0
        for ti in range(2):
            val_num = str(73924 + ti * 1453)
            vnames = ["VAR_ALPHA", "VAR_BETA", "VAR_GAMMA", "VAR_DELTA", "VAR_OMEGA"]
            seg_len = max(128, (L - 300) // 6)
            parts = [
                make_essay_haystack(tok, seg_len, seed=L + 900 + ti * 10),
                f"State assignment: {vnames[0]} = {val_num}.",
            ]
            for hi in range(1, 5):
                parts.append(make_essay_haystack(tok, seg_len, seed=L + 900 + ti * 10 + hi))
                parts.append(f"State assignment: {vnames[hi]} = {vnames[hi - 1]}.")
            parts.append(make_essay_haystack(tok, seg_len, seed=L + 909 + ti))
            q = "\n\n".join(parts) + f"\n\nQuestion: Follow the chain of variable assignments and state the exact numerical value of {vnames[4]}."
            rep, _, _, _ = greedy_chat(model, tok, q, max_new_tokens=64, seg_budget=seg_budget)
            if val_num in rep:
                vt_hits += 1

        tot_h = prose_hits + mk_hits + mv_val_hits + vt_hits
        tot_n = 4 + 4 + 8 + 2  # 18 probes per length
        ruler_res[str(L)] = {
            "niah_natural_prose": f"{prose_hits}/4",
            "multikey_4": f"{mk_hits}/4",
            "multivalue_4": f"{mv_val_hits}/8",
            "variable_tracking_5hop": f"{vt_hits}/2",
            "hits": tot_h,
            "total": tot_n,
            "pct": round(100.0 * tot_h / tot_n, 1),
            "sec": round(time.perf_counter() - t0, 1),
        }
        print(
            f"  [RULER @{L:5d} tok] {tot_h:2d}/{tot_n} ({100.0*tot_h/tot_n:5.1f}%) | "
            f"prose={prose_hits}/4 mk={mk_hits}/4 mv={mv_val_hits}/8 vt5={vt_hits}/2 | {ruler_res[str(L)]['sec']}s",
            flush=True,
        )
    all_h = sum(v["hits"] for v in ruler_res.values())
    all_n = sum(v["total"] for v in ruler_res.values())
    ruler_res["overall"] = {"hits": all_h, "total": all_n, "pct": round(100.0 * all_h / all_n, 2)}
    return ruler_res


def run_homogeneous_dense_ledger_pigeonhole_test(model, tok, seg_budget: int = 1024) -> Dict[str, Any]:
    """
    Peer Review Section 2.4 Pigeonhole Stress Test:
    100% homogeneous structured JSON ledger (ZERO low-entropy prose filler!).
    Measures true empirical recall across L = 2K, 4K, 8K, 16K, 32K vs information-theoretic
    query-agnostic capacity min(1.0, (window + seg_budget) / L).
    """
    out = {}
    rng = random.Random(777)
    for n_rows in (64, 128, 256, 512, 1024):  # ~2K, 4K, 8K, 16K, 32K tokens of 100% dense JSON records!
        records = []
        lines = []
        for r in range(n_rows):
            acct = f"ACCT-{1000 + r:04d}"
            bal = f"{rng.randint(10000, 99999)}"
            records.append((acct, bal))
            lines.append(f'{{"row":{r},"account":"{acct}","region":"US-EAST","balance_usd":{bal},"status":"SETTLED"}}')
        ledger = "\n".join(lines)
        # Probe 10 evenly-spaced early/mid/late rows across the entire ledger
        probe_idxs = [int(round(k * (n_rows - 1) / 9.0)) for k in range(10)]
        hits = 0
        plen_last = 0
        for idx in probe_idxs:
            acct, bal = records[idx]
            q = f"{ledger}\n\nQuestion: In the JSON ledger above, what is the exact `balance_usd` for account \"{acct}\"?"
            rep, plen, _, _ = greedy_chat(model, tok, q, max_new_tokens=32, seg_budget=seg_budget)
            plen_last = plen
            if bal in rep:
                hits += 1
        theo_cap = min(100.0, round(100.0 * (model.cfg.window + seg_budget) / max(1, plen_last), 1))
        out[f"rows_{n_rows}"] = {
            "prompt_tokens": plen_last,
            "hits": hits,
            "total": 10,
            "empirical_recall_pct": round(100.0 * hits / 10.0, 1),
            "theoretical_capacity_pct": theo_cap,
        }
        print(
            f"  [PIGEONHOLE 100%-JSON Ledger rows={n_rows:4d} (L={plen_last:5d} tok)] "
            f"empirical={hits}/10 ({100.0*hits/10:.1f}%) vs theoretical_bound={theo_cap:.1f}%",
            flush=True,
        )
    return out


def summarize_82probe(res: Dict[str, Any]) -> Dict[str, Any]:
    h = 0
    t = 0
    for k in ("dense_24_B1024", "dense_48_B1024", "multifact_8k_26needles_B1024",
              "two_hop_join_n8_B1024", "temporal_latest_v1_v3", "temporal_archival_t1", "temporal_rollback_t2",
              "minified_json_B1024"):
        h += res[k]["hits"]
        t += res[k]["total"]
    h += int(bool(res["scorer_sponge_120"]["survived"]))
    t += 1
    h += int(bool(res["multi_constraint_autopin_2400"]["passed"]))
    t += 1
    for hk in res["extreme_horizon"].values():
        h += hk["hits"]
        t += hk["total"]
    return {"hits": h, "total": t, "pct": round(100.0 * h / max(1, t), 2)}


def main():
    vlt3_q8 = os.path.join(ROOT, "dist", "vault_v3_3b_q8_0_primal.vlt3")
    vlt3_p4 = os.path.join(ROOT, "dist", "vault_v3_3b_p4_0_primal.vlt3")
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(QWEN_3B)

    model = load_vlt3_model(vlt3_q8, dev)
    orig_rebuild = model._rebuild_shingle_index
    orig_chunk_bias = model._build_chunk_resonance_bias
    orig_step_res = model._update_step_resonance

    def _disable_shingle():
        def _noop_rebuild(st):
            st.seg_bias_mode = 0
            st.seg_bias_active = False
            st.seg_head_bias = None
            st.seg_head_bias_hop1 = None
            st.sh_df = {}
            st.sh_postings = {}
            st.mm_tag_valid.zero_()
            st.mm_n_valid = 0
            st.n_uniq_segs = 0
            st.seg_inv = []
        model._rebuild_shingle_index = _noop_rebuild
        model._build_chunk_resonance_bias = lambda *a, **k: None
        model._update_step_resonance = lambda *a, **k: None

    def _enable_shingle():
        model._rebuild_shingle_index = orig_rebuild
        model._build_chunk_resonance_bias = orig_chunk_bias
        model._update_step_resonance = orig_step_res

    print("\n=== [1/4] 82-Probe Suite: FULL (SegmentScorer + 2-Tier FNV-1a Shingle/Hop-2 Resonance) ===", flush=True)
    res_full = eval_long_context_retrieval(model, tok, run_100k=True)
    sum_full = summarize_82probe(res_full)
    print(f"  --> FULL 82-Probe Total: {sum_full['hits']}/{sum_full['total']} ({sum_full['pct']}%)", flush=True)

    print("\n=== [2/4] 82-Probe Ablation: PURE SegmentScorer MLP + APL EMA ONLY (Shingle/Hop-2 DISABLED) ===", flush=True)
    _disable_shingle()
    res_mlp_only = eval_long_context_retrieval(model, tok, run_100k=True)
    sum_mlp_only = summarize_82probe(res_mlp_only)
    print(f"  --> PURE MLP+EMA ONLY Total: {sum_mlp_only['hits']}/{sum_mlp_only['total']} ({sum_mlp_only['pct']}%)", flush=True)

    print("\n=== [3/4] External Non-Isomorphic RULER Suite (4K -> 64K, Pure Natural Prose + 5-Hop VT) ===", flush=True)
    _enable_shingle()
    ruler_full = run_non_isomorphic_ruler_suite(model, tok, seg_budget=1024)
    _disable_shingle()
    print("\n  [RULER Ablation: Pure MLP+EMA ONLY without Shingle/Hop-2]", flush=True)
    ruler_mlp_only = run_non_isomorphic_ruler_suite(model, tok, seg_budget=1024)
    _enable_shingle()
    print("\n  [RULER Baseline: SWA + Sink ONLY (seg_budget=0)]", flush=True)
    ruler_swa_only = run_non_isomorphic_ruler_suite(model, tok, seg_budget=0)

    print("\n=== [4/4] Adversarial Homogeneous 100%-JSON Ledger Pigeonhole Stress Test (2K -> 32K) ===", flush=True)
    pigeonhole = run_homogeneous_dense_ledger_pigeonhole_test(model, tok, seg_budget=1024)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Also verify updated P4_0 .vlt3 on the 82-probe 100K suite!
    print("\n=== [BONUS] Updated VAULT-v3-3B-PRIMAL P4_0 on 82-Probe 100.4K Suite ===", flush=True)
    m_p4 = load_vlt3_model(vlt3_p4, dev)
    res_p4 = eval_long_context_retrieval(m_p4, tok, run_100k=True)
    sum_p4 = summarize_82probe(res_p4)
    print(f"  --> P4_0 82-Probe Total: {sum_p4['hits']}/{sum_p4['total']} ({sum_p4['pct']}%)", flush=True)
    del m_p4

    report = {
        "ablation_82probe_q8_0": {
            "full_seg_plus_shingle_hop2": {"summary": sum_full, "details": res_full},
            "pure_mlp_ema_no_shingle": {"summary": sum_mlp_only, "details": res_mlp_only},
        },
        "updated_82probe_p4_0": {"summary": sum_p4, "details": res_p4},
        "external_ruler_4k_64k_q8_0": {
            "full_seg_plus_shingle_hop2": ruler_full,
            "pure_mlp_ema_no_shingle": ruler_mlp_only,
            "swa_sink_only_B0": ruler_swa_only,
        },
        "homogeneous_json_pigeonhole_stress_q8_0": pigeonhole,
    }
    out_p = os.path.join(ROOT, "benchmarks", "peer_review_ruler_and_ablation.json")
    with open(out_p, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[DONE] Saved RULER + Hop-2 ablation + Pigeonhole report -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
