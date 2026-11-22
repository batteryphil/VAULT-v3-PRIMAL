#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
Unconfounded Peer-Review Baseline Suite for VAULT-v3-3B-PRIMAL:
  1. Untouched `Qwen/Qwen2.5-3B` (Base, Raw BF16) on WikiText-2 + exact 9-task EleutherAI
     zero-shot harness (both n=1000 historical match AND full un-subsampled splits on
     LAMBADA-5153, BoolQ-3270, HellaSwag-10042) to isolate Base-vs-Instruct chat calibration.
  2. Untouched `Qwen/Qwen2.5-3B-Instruct` (Raw BF16) on FULL GSM8K-1319 at max_new_tokens=1024
     (plus first-200 slice at 1024 tok) + full un-subsampled 9-task EleutherAI splits.
  3. Saves item-level binary correctness vectors for paired bootstrap 95% CIs & McNemar tests.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

if "expandable_segments" in os.environ.get("PYTORCH_ALLOC_CONF", ""):
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from primal.train_vault_v3_3b_primal import (
    _resolve_qwen_3b_dir,
    GSM8K_SYSTEM_PROMPT,
    GSM8K_USER_SUFFIX,
    _encode_pair_lm_eval,
    build_wikitext2_val,
    extract_gsm8k_answer,
)


def bootstrap_ci_95(binary_hits: List[int], n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    arr = np.asarray(binary_hits, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1) * 100.0
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(float(lo), 2), round(float(hi), 2)


@torch.no_grad()
def eval_wt2_hf_bf16(model: AutoModelForCausalLM, wt_bundle: Tuple[torch.Tensor, int, int], device: str = "cuda") -> Dict[str, float]:
    wt_ids, total_bytes, total_words = wt_bundle
    seq_len = 2048
    n_seq = (wt_ids.numel() - 1) // seq_len
    xs = wt_ids[: n_seq * seq_len].view(n_seq, seq_len)
    ys = wt_ids[1 : n_seq * seq_len + 1].view(n_seq, seq_len)
    total_nll = 0.0
    total_toks = 0
    for i in range(0, n_seq, 2):
        xb = xs[i : i + 2].to(device)
        yb = ys[i : i + 2].to(device).reshape(-1)
        hid = model.model(input_ids=xb, use_cache=False).last_hidden_state.reshape(-1, model.config.hidden_size)
        for s0 in range(0, yb.numel(), 512):
            s1 = min(yb.numel(), s0 + 512)
            lg = model.lm_head(hid[s0:s1]).float()
            total_nll += float(F.cross_entropy(lg, yb[s0:s1], reduction="sum").item())
            del lg
        total_toks += int(yb.numel())
        del hid, xb, yb
    bpb = (total_nll / math.log(2.0)) / max(1, total_bytes)
    word_ppl = math.exp(min(20.0, total_nll / max(1, total_words)))
    tok_ppl = math.exp(min(20.0, total_nll / max(1, total_toks)))
    return {
        "wt2_bpb": round(bpb, 4),
        "wt2_word_ppl": round(word_ppl, 2),
        "wt2_tok_ppl": round(tok_ppl, 2),
    }


@torch.no_grad()
def score_pairs_hf_bf16(
    model: AutoModelForCausalLM,
    tok: Tokenizer,
    pairs: List[Tuple[str, str]],
    batch_size: int = 32,
    max_seq_len: int = 2048,
    device: str = "cuda",
) -> List[Tuple[float, float, bool, int]]:
    n = len(pairs)
    encoded = [(i, *_encode_pair_lm_eval(tok, ctx, cont, max_seq_len)) for i, (ctx, cont) in enumerate(pairs)]
    encoded.sort(key=lambda x: len(x[1]))
    results: List[Optional[Tuple[float, float, bool, int]]] = [None] * n
    pad_id = tok.token_to_id("<|endoftext|>") or 0

    for st in range(0, n, batch_size):
        chunk = encoded[st : st + batch_size]
        max_len = max(len(item[1]) for item in chunk)
        sub_bs = min(batch_size, 8) if max_len >= 512 else (min(batch_size, 16) if max_len >= 256 else batch_size)
        for sub_st in range(0, len(chunk), sub_bs):
            sub = chunk[sub_st : sub_st + sub_bs]
            t_pad = max(32, ((max(len(item[1]) for item in sub) + 31) // 32) * 32)
            inp = torch.full((len(sub), t_pad), pad_id, dtype=torch.long, device=device)
            for b, (_, w_ids, _, _) in enumerate(sub):
                inp[b, : len(w_ids)] = torch.tensor(w_ids, dtype=torch.long, device=device)
            # Use model.model to get hidden states BEFORE lm_head so we only project continuation tokens!
            hidden = model.model(input_ids=inp, use_cache=False).last_hidden_state
            h_slices, targets, spans = [], [], []
            cursor = 0
            for b, (_, w_ids, c_len, _) in enumerate(sub):
                L = len(w_ids)
                c0 = L - c_len
                h_slices.append(hidden[b, c0 - 1 : L - 1])
                targets.extend(w_ids[c0:L])
                spans.append((cursor, cursor + c_len))
                cursor += c_len
            logits_cat = model.lm_head(torch.cat(h_slices, dim=0)).float()
            lp_cat = F.log_softmax(logits_cat, dim=-1)
            tgt = torch.tensor(targets, dtype=torch.long, device=device)
            tok_lp = lp_cat.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            greedy = lp_cat.argmax(dim=-1)
            for b, (orig_idx, _, c_len, b_len) in enumerate(sub):
                s0, s1 = spans[b]
                sum_lp = float(tok_lp[s0:s1].sum().item())
                exact = bool((greedy[s0:s1] == tgt[s0:s1]).all().item())
                results[orig_idx] = (sum_lp, sum_lp / float(b_len), exact, c_len)
    return results  # type: ignore[return-value]


def load_all_9_tasks() -> Dict[str, Any]:
    lam_texts = list(load_dataset("EleutherAI/lambada_openai", "en", split="test")["text"])

    def _arc(cfg_name):
        items = []
        for r in load_dataset("allenai/ai2_arc", cfg_name, split="test"):
            labs, txts = list(r["choices"]["label"]), list(r["choices"]["text"])
            if r["answerKey"] in labs and len(txts) >= 2:
                ctx = f"Question: {r['question'].strip()}\nAnswer:"
                items.append(([(ctx, " " + t.strip()) for t in txts], labs.index(r["answerKey"])))
        return items

    sciq_rc, sciq_cb = [], []
    for r in load_dataset("allenai/sciq", split="test"):
        q, sup = r["question"].strip(), (r.get("support") or "").strip()
        cands = [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]]
        ctx_cb = f"Question: {q}\nAnswer:"
        ctx_rc = f"{sup}\nQuestion: {q}\nAnswer:" if sup else ctx_cb
        sciq_rc.append(([(ctx_rc, " " + c.strip()) for c in cands], 3))
        sciq_cb.append(([(ctx_cb, " " + c.strip()) for c in cands], 3))

    piqa_items = [
        ([(f"Question: {r['goal'].strip()}\nAnswer:", " " + r["sol1"].strip()),
          (f"Question: {r['goal'].strip()}\nAnswer:", " " + r["sol2"].strip())], int(r["label"]))
        for r in load_dataset("baber/piqa", split="validation") if int(r["label"]) in (0, 1)
    ]

    def _hs_clean(t: str) -> str:
        return re.sub(r"\[.*?\]", "", t.strip().replace(" [title]", ". ")).replace("  ", " ")

    hs_ds = load_dataset("Rowan/hellaswag", split="validation")
    hs_items = [
        ([(_hs_clean(hs_ds[i]["activity_label"] + ": " + hs_ds[i]["ctx_a"] + " " + hs_ds[i]["ctx_b"].capitalize()),
           " " + _hs_clean(e)) for e in hs_ds[i]["endings"]], int(hs_ds[i]["label"]))
        for i in range(len(hs_ds))
    ]

    wg_items = []
    for r in load_dataset("allenai/winogrande", "winogrande_xl", split="validation"):
        if "_" in r["sentence"]:
            i = r["sentence"].index("_")
            p, s = r["sentence"][:i], r["sentence"][i + 1 :]
            wg_items.append(([(p + r["option1"], s), (p + r["option2"], s)], int(r["answer"]) - 1))

    bq_ds = load_dataset("google/boolq", split="validation")
    bq_items = []
    for i in range(len(bq_ds)):
        r = bq_ds[i]
        q = r["question"].strip()
        if not q.endswith("?"):
            q += "?"
        ctx = f"{r['passage'].strip()}\nQuestion: {q}\nAnswer:"
        bq_items.append(([(ctx, " no"), (ctx, " yes")], 1 if bool(r["answer"]) else 0))

    return {
        "lambada_texts": lam_texts,
        "mc": [
            ("arc_easy", "ARC-Easy", _arc("ARC-Easy"), "acc_norm"),
            ("arc_challenge", "ARC-Challenge", _arc("ARC-Challenge"), "acc_norm"),
            ("sciq_rc", "SciQ-RC", sciq_rc, "acc_norm"),
            ("sciq_closedbook", "SciQ-ClosedBook", sciq_cb, "acc_norm"),
            ("piqa", "PIQA", piqa_items, "acc_norm"),
            ("hellaswag", "HellaSwag", hs_items, "acc_norm"),
            ("winogrande", "Winogrande", wg_items, "acc"),
            ("boolq", "BoolQ", bq_items, "acc"),
        ],
    }


@torch.no_grad()
def eval_9task_full_and_1k(
    model: AutoModelForCausalLM,
    tok: Tokenizer,
    tasks: Dict[str, Any],
    wt_bundle: Tuple[torch.Tensor, int, int],
    label: str,
    device: str = "cuda",
) -> Tuple[Dict[str, Any], Dict[str, List[int]]]:
    print(f"\n==================== [FULL + 1K BENCHMARK SUITE: {label}] ====================", flush=True)
    wt2 = eval_wt2_hf_bf16(model, wt_bundle, device=device)
    print(f"  [WikiText-2      ] BPB={wt2['wt2_bpb']:.4f} | Word_PPL={wt2['wt2_word_ppl']:.2f} | Tok_PPL={wt2['wt2_tok_ppl']:.2f}", flush=True)

    # LAMBADA full (5,153) + 1k slice
    t0 = time.perf_counter()
    lam_pairs = []
    for txt in tasks["lambada_texts"]:
        t = txt.strip()
        if " " in t:
            ctx, last = t.rsplit(" ", 1)
            lam_pairs.append((ctx, " " + last))
    lam_scored = score_pairs_hf_bf16(model, tok, lam_pairs, batch_size=32, device=device)
    lam_hits_full = [1 if ex else 0 for _, _, ex, _ in lam_scored]
    lam_hits_1k = lam_hits_full[:1000]
    lam_full = {
        "n_samples": len(lam_scored),
        "acc": round(100.0 * sum(lam_hits_full) / len(lam_scored), 2),
        "ci95": bootstrap_ci_95(lam_hits_full),
        "target_word_ppl": round(math.exp(min(20.0, -sum(lp for lp, _, _, _ in lam_scored) / len(lam_scored))), 2),
        "eval_sec": round(time.perf_counter() - t0, 1),
    }
    lam_1k = {
        "n_samples": 1000,
        "acc": round(100.0 * sum(lam_hits_1k) / 1000.0, 2),
        "target_word_ppl": round(math.exp(min(20.0, -sum(lp for lp, _, _, _ in lam_scored[:1000]) / 1000.0)), 2),
    }
    print(
        f"  [LAMBADA         ] FULL(n={lam_full['n_samples']}): {lam_full['acc']:.2f}% {lam_full['ci95']} (ppl={lam_full['target_word_ppl']:.2f}) | "
        f"1K: {lam_1k['acc']:.2f}% (ppl={lam_1k['target_word_ppl']:.2f}) | {lam_full['eval_sec']:.1f}s",
        flush=True,
    )

    item_vectors: Dict[str, List[int]] = {"lambada": lam_hits_full}
    mc_full_res = {}
    mc_1k_res = {}
    prim_full = [lam_full["acc"]]
    prim_1k = [lam_1k["acc"]]

    from transformers.cache_utils import DynamicCache
    pad_id = tok.token_to_id("<|endoftext|>") or 0

    def _score_mc_fast(items_mc: List) -> List[List[Tuple[float, float]]]:
        """Exact shared-prefix KV-cache multi-choice scorer (9.54e-7 parity, 4x faster on HellaSwag/BoolQ/ARC/SciQ)."""
        n_it = len(items_mc)
        out_it: List[Optional[List[Tuple[float, float]]]] = [None] * n_it
        fast_bucket = []
        slow_pairs = []
        slow_map = []
        for idx_i, (cands, _gold) in enumerate(items_mc):
            enc_c = [_encode_pair_lm_eval(tok, ctx, cont, 2048) for ctx, cont in cands]
            pl = min(len(w) - cl - 1 for w, cl, _ in enc_c)
            p0 = enc_c[0][0][:pl] if pl >= 8 else []
            if pl >= 8 and all(w[:pl] == p0 for w, _, _ in enc_c):
                fast_bucket.append((idx_i, pl, p0, enc_c))
            else:
                s0 = len(slow_pairs)
                slow_pairs.extend(cands)
                slow_map.append((idx_i, s0, len(slow_pairs)))

        if slow_pairs:
            sc_slow = score_pairs_hf_bf16(model, tok, slow_pairs, batch_size=24, device=device)
            for idx_i, s0, s1 in slow_map:
                out_it[idx_i] = [(x[0], x[1]) for x in sc_slow[s0:s1]]

        fast_bucket.sort(key=lambda x: x[1])
        q_bs = 64
        for st_i in range(0, len(fast_bucket), q_bs):
            ch = fast_bucket[st_i : st_i + q_bs]
            max_p = max(x[1] for x in ch)
            sub_qbs = 8 if max_p >= 384 else (24 if max_p >= 192 else q_bs)
            for sst in range(0, len(ch), sub_qbs):
                sub = ch[sst : sst + sub_qbs]
                B_q = len(sub)
                mp = max(x[1] for x in sub)
                pinp = torch.full((B_q, mp), pad_id, dtype=torch.long, device=device)
                pam = torch.zeros((B_q, mp), dtype=torch.long, device=device)
                for b, (_, pl, p0, _) in enumerate(sub):
                    pad = mp - pl
                    pinp[b, pad:] = torch.tensor(p0, dtype=torch.long, device=device)
                    pam[b, pad:] = 1
                pkv = DynamicCache()
                ppos = (pam.cumsum(-1) - 1).clamp(min=0)
                model.model(input_ids=pinp, attention_mask=pam, position_ids=ppos, past_key_values=pkv, use_cache=True)

                parent_idx, suffs, clens, blens, plens = [], [], [], [], []
                spans_q = []
                for b, (idx_i, pl, _, enc_c) in enumerate(sub):
                    q_s0 = len(suffs)
                    for w, cl, bl in enc_c:
                        parent_idx.append(b)
                        suffs.append(w[pl:])  # length = (L - pl) >= cl + 1
                        clens.append(cl)
                        blens.append(bl)
                        plens.append(pl)
                    spans_q.append((idx_i, q_s0, len(suffs)))

                pt = torch.tensor(parent_idx, dtype=torch.long, device=device)
                pkv.batch_select_indices(pt)
                pam_exp = pam[pt]
                N_s = len(suffs)
                max_s = max(len(s) for s in suffs)
                sinp = torch.full((N_s, max_s), pad_id, dtype=torch.long, device=device)
                sam = torch.ones((N_s, max_s), dtype=torch.long, device=device)
                spos = torch.zeros((N_s, max_s), dtype=torch.long, device=device)
                for si, (s_ids, pl) in enumerate(zip(suffs, plens)):
                    sinp[si, : len(s_ids)] = torch.tensor(s_ids, dtype=torch.long, device=device)
                    spos[si] = pl + torch.arange(max_s, device=device)
                full_am = torch.cat([pam_exp, sam], dim=1)
                hid = model.model(input_ids=sinp, attention_mask=full_am, position_ids=spos, past_key_values=pkv, use_cache=True).last_hidden_state
                h_slices, targets, c_spans = [], [], []
                cur = 0
                for si, (s_ids, cl) in enumerate(zip(suffs, clens)):
                    Ls = len(s_ids)
                    c0 = Ls - cl
                    h_slices.append(hid[si, c0 - 1 : Ls - 1])
                    targets.extend(s_ids[c0:Ls])
                    c_spans.append((cur, cur + cl))
                    cur += cl
                h_cat = torch.cat(h_slices, dim=0)
                tgt_t = torch.tensor(targets, dtype=torch.long, device=device)
                tok_lp = torch.empty((h_cat.size(0),), dtype=torch.float32, device=device)
                for hs0 in range(0, h_cat.size(0), 512):
                    hs1 = min(h_cat.size(0), hs0 + 512)
                    lg_c = model.lm_head(h_cat[hs0:hs1]).float()
                    lp_c = F.log_softmax(lg_c, dim=-1)
                    tok_lp[hs0:hs1] = lp_c.gather(-1, tgt_t[hs0:hs1].unsqueeze(-1)).squeeze(-1)
                    del lg_c, lp_c
                cand_scores = []
                for (c0s, c1s), bl in zip(c_spans, blens):
                    slp = float(tok_lp[c0s:c1s].sum().item())
                    cand_scores.append((slp, slp / float(bl)))
                for idx_i, qs0, qs1 in spans_q:
                    out_it[idx_i] = cand_scores[qs0:qs1]
                del pkv, hid, h_cat, tok_lp
        return out_it  # type: ignore[return-value]

    for key, name, items, metric in tasks["mc"]:
        t_m = time.perf_counter()
        scored_items = _score_mc_fast(items)
        acc_vec, norm_vec = [], []
        for cs, (_cands, gold) in zip(scored_items, items):
            acc_vec.append(1 if max(range(len(cs)), key=lambda i: cs[i][0]) == gold else 0)
            norm_vec.append(1 if max(range(len(cs)), key=lambda i: cs[i][1]) == gold else 0)
        p_vec = norm_vec if metric == "acc_norm" else acc_vec
        item_vectors[key] = p_vec
        n_f = len(items)
        r_f = {
            "n_samples": n_f,
            "acc": round(100.0 * sum(acc_vec) / n_f, 2),
            "acc_norm": round(100.0 * sum(norm_vec) / n_f, 2),
            "ci95": bootstrap_ci_95(p_vec),
            "eval_sec": round(time.perf_counter() - t_m, 1),
        }
        mc_full_res[key] = r_f
        prim_full.append(r_f[metric])

        n_1k = min(1000, n_f) if key in ("hellaswag", "boolq") else n_f
        r_1k = {
            "n_samples": n_1k,
            "acc": round(100.0 * sum(acc_vec[:n_1k]) / n_1k, 2),
            "acc_norm": round(100.0 * sum(norm_vec[:n_1k]) / n_1k, 2),
        }
        mc_1k_res[key] = r_1k
        prim_1k.append(r_1k[metric])
        print(
            f"  [{name:<16s}] FULL(n={n_f:<5d}) {metric}={r_f[metric]:5.2f}% {r_f['ci95']} | "
            f"1K-match={r_1k[metric]:5.2f}% | {r_f['eval_sec']:.1f}s",
            flush=True,
        )

    mc8_full = round(sum(prim_full[1:]) / 8.0, 2)
    avg9_full = round(sum(prim_full) / 9.0, 2)
    mc8_1k = round(sum(prim_1k[1:]) / 8.0, 2)
    avg9_1k = round(sum(prim_1k) / 9.0, 2)
    print(
        f"  --> FULL Splits: 8-Task MC={mc8_full:.2f}% | 9-Task={avg9_full:.2f}% || "
        f"1K-Match: 8-Task MC={mc8_1k:.2f}% | 9-Task={avg9_1k:.2f}%",
        flush=True,
    )
    return {
        "label": label,
        "wikitext2": wt2,
        "full_splits": {
            "lambada": lam_full,
            "multiple_choice": mc_full_res,
            "mc_8task_avg_pct": mc8_full,
            "zero_shot_9task_avg_pct": avg9_full,
        },
        "historical_1k_match": {
            "lambada": lam_1k,
            "multiple_choice": mc_1k_res,
            "mc_8task_avg_pct": mc8_1k,
            "zero_shot_9task_avg_pct": avg9_1k,
        },
    }, item_vectors


@torch.no_grad()
def eval_gsm8k_full_hf_bf16(
    model: AutoModelForCausalLM,
    tok: Tokenizer,
    max_new_tokens: int = 1024,
    batch_size: int = 64,
    device: str = "cuda",
) -> Tuple[Dict[str, Any], List[int]]:
    from transformers.cache_utils import DynamicCache

    t0 = time.perf_counter()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = len(ds)
    im_start = tok.token_to_id("<|im_start|>")
    im_end = tok.token_to_id("<|im_end|>")
    eos_id = tok.token_to_id("<|endoftext|>")
    nl = tok.encode("\n", add_special_tokens=False).ids

    encoded_prompts = []
    golds = []
    for i in range(n):
        q = ds[i]["question"].strip() + GSM8K_USER_SUFFIX
        golds.append(extract_gsm8k_answer(ds[i]["answer"]))
        p_ids = (
            [im_start] + tok.encode(f"system\n{GSM8K_SYSTEM_PROMPT}", add_special_tokens=False).ids + [im_end] + nl
            + [im_start] + tok.encode(f"user\n{q}", add_special_tokens=False).ids + [im_end] + nl
            + [im_start] + tok.encode("assistant\n", add_special_tokens=False).ids
        )
        encoded_prompts.append((i, p_ids))

    encoded_prompts.sort(key=lambda x: len(x[1]))
    outputs = [""] * n
    gen_lens = [0] * n
    truncated = [1] * n

    for st in range(0, n, batch_size):
        chunk = encoded_prompts[st : st + batch_size]
        B = len(chunk)
        max_p = max(len(c[1]) for c in chunk)
        inp = torch.full((B, max_p), eos_id, dtype=torch.long, device=device)
        cur_am = torch.zeros((B, max_p), dtype=torch.long, device=device)
        for b, (_, p_ids) in enumerate(chunk):
            pl = max_p - len(p_ids)
            inp[b, pl:] = torch.tensor(p_ids, dtype=torch.long, device=device)
            cur_am[b, pl:] = 1

        pkv = DynamicCache()
        pos0 = (cur_am.cumsum(-1) - 1).clamp(min=0)
        hid0 = model.model(input_ids=inp, attention_mask=cur_am, position_ids=pos0, past_key_values=pkv, use_cache=True).last_hidden_state
        nt = model.lm_head(hid0[:, -1, :]).argmax(-1)
        del hid0
        active = list(range(B))
        cur_pos = torch.tensor([len(c[1]) for c in chunk], dtype=torch.long, device=device)
        gen_toks_b: List[List[int]] = [[] for _ in range(B)]

        for step in range(max_new_tokens):
            nt_list = nt.tolist()
            keep = []
            for idx_act, b_orig in enumerate(active):
                tid = nt_list[idx_act]
                if tid in (im_end, eos_id):
                    truncated[chunk[b_orig][0]] = 0
                else:
                    gen_toks_b[b_orig].append(tid)
                    keep.append(idx_act)
            if not keep or step == max_new_tokens - 1:
                break
            if len(keep) < len(active):
                kt = torch.tensor(keep, dtype=torch.long, device=device)
                active = [active[i] for i in keep]
                nt = nt[kt]
                cur_pos = cur_pos[kt]
                cur_am = cur_am[kt]
                pkv.batch_select_indices(kt)
            cur_am = torch.cat([cur_am, torch.ones((len(active), 1), dtype=torch.long, device=device)], dim=1)
            hid_s = model.model(input_ids=nt.unsqueeze(1), attention_mask=cur_am, position_ids=cur_pos.unsqueeze(1), past_key_values=pkv, use_cache=True).last_hidden_state
            nt = model.lm_head(hid_s[:, -1, :]).argmax(-1)
            cur_pos = cur_pos + 1

        del pkv
        for b, (orig_idx, _) in enumerate(chunk):
            gen_lens[orig_idx] = len(gen_toks_b[b])
            outputs[orig_idx] = tok.decode(gen_toks_b[b], skip_special_tokens=True).strip()

        if ((st // batch_size) + 1) % 5 == 0 or st + B >= n:
            print(f"  [GSM8K-1319] {min(n, st + B)}/{n} completed ({time.perf_counter() - t0:.1f}s)...", flush=True)

    hits_vec = []
    boxed_vec = []
    for raw, g in zip(outputs, golds):
        boxed_vec.append(1 if "\\boxed{" in raw else 0)
        p = extract_gsm8k_answer(raw)
        hits_vec.append(1 if (p is not None and g is not None and abs(p - g) < 1e-4) else 0)

    dt = time.perf_counter() - t0
    res = {
        "n_samples": n,
        "max_new_tokens": max_new_tokens,
        "hits": sum(hits_vec),
        "acc": round(100.0 * sum(hits_vec) / n, 2),
        "ci95": bootstrap_ci_95(hits_vec),
        "boxed_pct": round(100.0 * sum(boxed_vec) / n, 2),
        "truncated_pct": round(100.0 * sum(truncated) / n, 2),
        "mean_gen_tokens": round(float(np.mean(gen_lens)), 1),
        "slice_first_200": {
            "n_samples": 200,
            "hits": sum(hits_vec[:200]),
            "acc": round(100.0 * sum(hits_vec[:200]) / 200.0, 2),
            "boxed_pct": round(100.0 * sum(boxed_vec[:200]) / 200.0, 2),
            "truncated_pct": round(100.0 * sum(truncated[:200]) / 200.0, 2),
        },
        "eval_sec": round(dt, 1),
    }
    print(
        f"  [GSM8K-1319 FULL ] Acc={res['acc']:.2f}% {res['ci95']} ({res['hits']}/{n}) | "
        f"boxed={res['boxed_pct']:.1f}% | trunc={res['truncated_pct']:.2f}% | mean_tok={res['mean_gen_tokens']} | "
        f"first200@1024={res['slice_first_200']['acc']:.2f}% (boxed={res['slice_first_200']['boxed_pct']:.1f}%) | {dt:.1f}s",
        flush=True,
    )
    return res, hits_vec


def main():
    DEFAULT_QWEN_3B = _resolve_qwen_3b_dir()
    out_dir = "runs/peer_review_v3"
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda"

    print("[1/3] Downloading / resolving Qwen/Qwen2.5-3B (Base)...", flush=True)
    base_dir = snapshot_download("Qwen/Qwen2.5-3B")
    print(f"      Resolved Qwen2.5-3B Base -> {base_dir}", flush=True)

    tok = Tokenizer.from_file(os.path.join(_resolve_qwen_3b_dir, "tokenizer.json"))
    wt_bundle = build_wikitext2_val(tok, seq_len=2048)
    tasks = load_all_9_tasks()

    # 1. Evaluate Qwen2.5-3B (Base)
    print("\n[2/3] Loading Qwen/Qwen2.5-3B (Base, Raw BF16)...", flush=True)
    m_base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16).to(device).eval()
    base_rep, base_vecs = eval_9task_full_and_1k(
        m_base, tok, tasks, wt_bundle, label="Qwen/Qwen2.5-3B (Base, Untouched BF16)", device=device
    )
    del m_base
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Evaluate Qwen2.5-3B-Instruct (Raw BF16) on Full Splits + Full GSM8K-1319 (1024 tok)
    print("\n[3/3] Loading Qwen/Qwen2.5-3B-Instruct (Untouched BF16)...", flush=True)
    m_inst = AutoModelForCausalLM.from_pretrained(_resolve_qwen_3b_dir, dtype=torch.bfloat16).to(device).eval()
    inst_rep, inst_vecs = eval_9task_full_and_1k(
        m_inst, tok, tasks, wt_bundle, label="Qwen/Qwen2.5-3B-Instruct (Untouched BF16)", device=device
    )
    inst_gsm, inst_gsm_vec = eval_gsm8k_full_hf_bf16(m_inst, tok, max_new_tokens=1024, batch_size=64, device=device)
    inst_rep["gsm8k_1319"] = inst_gsm
    inst_vecs["gsm8k_1319"] = inst_gsm_vec
    del m_inst
    gc.collect()
    torch.cuda.empty_cache()

    out_json = os.path.join(out_dir, "unconfounded_raw_baselines.json")
    with open(out_json, "w") as f:
        json.dump({"qwen25_3b_base_bf16": base_rep, "qwen25_3b_instruct_bf16": inst_rep}, f, indent=2)
    torch.save(
        {"base_vecs": base_vecs, "instruct_vecs": inst_vecs},
        os.path.join(out_dir, "raw_baseline_item_vectors.pt"),
    )
    print(f"\n[DONE] Saved unconfounded baselines -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
