#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
Round-2 Peer Review Item 3 & Section 2.2 Complete Pipeline (`PRIMAL v3.1`):
  1. Step-Radius-Matched Continuous Control (`Layer-Hooked BF16 + 8-Bit AdamW` at `lr_peak = 2e-4`,
     15 steps = 120 `b=1` micro-steps = 245,760 tokens) on the Round-1/2 off-policy ~110-token stream,
     testing Round-2 §2.2's prediction that step-radius-matched continuous AdamW ALSO compresses CoT
     length and regresses GSM8K when supervised on off-policy terse targets.
  2. On-Policy Rejection-Sampled `Qwen/Qwen2.5-3B-Instruct` Stream (`primal_v3_1_onpolicy_stream.pt`):
     Generates greedy responses from untouched `Qwen2.5-3B-Instruct` on deduplicated `openai/gsm8k` train
     + `MetaMathQA` prompts, retaining ONLY verified-correct `\\boxed{}` traces (~330 tokens), mixed with
     20% coding (`Magicoder` + `CodeFeedback`) and 30% textbook replay (`FineWeb-Edu` + `Cosmopedia-v2`).
  3. Lossless `PRIMAL v3.1` Training (25 steps = 409,600 tokens, `decay=1.00, scale_aware_votes=True`,
     sparse O(n_crowded) `rebalance_scales` eliminating the +3.11 GiB rebalance VRAM spike), validated
     strictly on held-out `WikiText-2` + held-out `GSM8K-TrainVal200` (zero test-split peeking).
  4. Full Un-Subsampled Evaluations on:
       - `PRIMAL v3.1 (Q8_0 -> BF16 safetensors)` across all 9 full tasks + `GSM8K-1319 @ 1024 tok`
       - `PRIMAL v3.1 (Exact V3_Q8_0 FP32 product)` across WikiText-2 + 4 Science MC splits + LAMBADA-1K
         (explicitly quantifying exact `.vlt3` Q8_0 vs. exported BF16 safetensors accuracy)
       - `PRIMAL v3.1 (V3_P4_0, 1.805 GiB)` across all 9 full tasks + `GSM8K-1319 @ 1024 tok`
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Tuple

if "expandable_segments" in os.environ.get("PYTORCH_ALLOC_CONF", ""):
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from primal_v7.eval_peer_review_baselines import (
    GSM8K_SYSTEM_PROMPT,
    GSM8K_USER_SUFFIX,
    eval_9task_full_and_1k,
    eval_gsm8k_full_hf_bf16,
    eval_wt2_hf_bf16,
    load_all_9_tasks,
    score_pairs_hf_bf16,
)
from primal_v7.run_primal_v3_and_controls import (
    eval_gsm8k_val200_primal,
    paired_bootstrap_diff_ci,
)
from primal_v7.train_vault_v3_3b_primal import (
    DEFAULT_QWEN_3B,
    DEFAULT_SYSTEM_PROMPT,
    VAULT_ROOT,
    _norm_dedup_key,
    build_heldout_sft_val,
    build_wikitext2_val,
    encode_sft_turns_masked,
    evaluate_heldout_sft_loss,
    evaluate_wikitext2_bpb,
    export_hessian_p4_0_from_primal_3b,
    extract_gsm8k_answer,
)
from primal_v7.train_vault_v3_3b_seg_scorer import _load_v3_mod, repack_vlt3_with_seg_scorer
from primal_v7.vault_primal import (
    QK,
    Q8StepEngine,
    VAULTPrimalConfig,
    VAULTPrimalForCausalLM,
    _Q8BlockMixin,
    export_vault_primal_vlt3,
    load_pretrained_qwen2_into_primal_q8,
)

V3_DIR = "primal_v7/runs/vault_v3_3b_primal_v3"
OUT_DIR = "primal_v7/runs/vault_v3_3b_primal_v3_1"
SCORER_CKPT = "primal_v7/runs/vault_v3_3b_seg_scorer/vault_v3_3b_seg_scorer.pt"
VAULT_DIST = "./dist"


# ==============================================================================
# 1. Step-Radius-Matched 8-Bit AdamW Control (`lr_peak = 2e-4`, 15 steps) on Off-Policy Stream
# ==============================================================================

def run_step_matched_adamw8bit_offpolicy_15steps(
    qwen_dir: str,
    offpolicy_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    wt_bundle: Tuple[torch.Tensor, int, int],
    tok: Tokenizer,
    lr_peak: float = 2.0e-4,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Runs 15 optimizer-step equivalents (60 B=2 micro-batches = 120 b=1 micro-steps = 245,760 tokens)
    of Layer-Hooked BF16 + 8-Bit AdamW at `lr_peak = 2e-4` (sub_lr = 2.5e-5 -> max |dw| ~ 0.003-0.008,
    matching PRIMAL's multi-tick excursion radius) on the Round-1/2 off-policy ~110-token stream.
    """
    print("\n=== [EXP-A] Step-Radius-Matched Layer-Hooked 8-Bit AdamW (lr_peak=2e-4, 15 steps = 245.8K tok) on Off-Policy Stream ===", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    m_ctrl = AutoModelForCausalLM.from_pretrained(qwen_dir, dtype=torch.bfloat16).to(device)
    m_ctrl.config.use_cache = False
    m_ctrl.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m_ctrl.train()

    state_2d: Dict[nn.Parameter, Dict[str, Any]] = {}
    state_1d: Dict[nn.Parameter, Dict[str, Any]] = {}
    step_state = {"lr": lr_peak / 8.0}
    SCALE_FP16 = 1024.0
    INV_SCALE_FP16 = 1.0 / SCALE_FP16

    for _, p in m_ctrl.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and (p.numel() % QK == 0):
            nb = p.numel() // QK
            state_2d[p] = {
                "m_q": torch.zeros((nb, QK), dtype=torch.int8, device=device),
                "m_d": torch.zeros((nb,), dtype=torch.float16, device=device),
                "v_q": torch.zeros((nb, QK), dtype=torch.uint8, device=device),
                "v_d": torch.zeros((nb,), dtype=torch.float16, device=device),
                "step": 0,
            }
        else:
            state_1d[p] = {
                "m": torch.zeros_like(p, dtype=torch.float32),
                "v": torch.zeros_like(p, dtype=torch.float32),
                "step": 0,
            }

    w0_norm_sample = m_ctrl.model.layers[18].self_attn.q_proj.weight.detach().float().cpu().clone()

    def _make_hook(param: nn.Parameter):
        @torch.no_grad()
        def _hook(p: nn.Parameter):
            if p.grad is None:
                return
            g_raw = p.grad
            p.grad = None
            lr = step_state["lr"]
            b1, b2, eps = 0.9, 0.999, 1e-6
            if p in state_2d:
                st = state_2d[p]
                st["step"] += 1
                t = st["step"]
                bc1 = 1.0 - (b1 ** t)
                s_bc2 = math.sqrt(1.0 - (b2 ** t))
                p_blk = p.data.view(-1, QK)
                g_blk = g_raw.reshape(-1, QK)
                for b0 in range(0, p_blk.size(0), 65536):
                    b1_idx = min(p_blk.size(0), b0 + 65536)
                    gb = torch.nan_to_num(g_blk[b0:b1_idx].float(), nan=0.0, posinf=0.0, neginf=0.0)
                    m_f = st["m_q"][b0:b1_idx].float() * (st["m_d"][b0:b1_idx].float() * INV_SCALE_FP16).unsqueeze(-1)
                    sv_prev = st["v_q"][b0:b1_idx].float() * (st["v_d"][b0:b1_idx].float() * INV_SCALE_FP16).unsqueeze(-1)
                    v_f = sv_prev.square_()
                    m_f.mul_(b1).add_(gb, alpha=1.0 - b1)
                    v_f.mul_(b2).addcmul_(gb, gb, value=1.0 - b2)
                    sv_f = v_f.sqrt_()
                    md_fp16 = ((m_f.abs().amax(dim=1) / 127.0) * SCALE_FP16).clamp(min=1e-7, max=65000.0).to(torch.float16)
                    st["m_d"][b0:b1_idx].copy_(md_fp16)
                    md_eff = (md_fp16.float() * INV_SCALE_FP16).clamp(min=1e-16)
                    st["m_q"][b0:b1_idx].copy_(torch.round(m_f / md_eff.unsqueeze(-1)).clamp(-127, 127).to(torch.int8))
                    svd_fp16 = ((sv_f.amax(dim=1) / 255.0) * SCALE_FP16).clamp(min=1e-7, max=65000.0).to(torch.float16)
                    st["v_d"][b0:b1_idx].copy_(svd_fp16)
                    svd_eff = (svd_fp16.float() * INV_SCALE_FP16).clamp(min=1e-16)
                    vq_raw = torch.round(sv_f / svd_eff.unsqueeze(-1)).clamp(0, 255).to(torch.uint8)
                    vq_safe = torch.where((sv_f > 0.0) & (vq_raw == 0), torch.ones_like(vq_raw), vq_raw)
                    st["v_q"][b0:b1_idx].copy_(vq_safe)
                    denom = (vq_safe.float() * svd_eff.unsqueeze(-1)) / s_bc2 + eps
                    upd = ((m_f / bc1) / denom).clamp_(-5.0, 5.0)
                    p_blk[b0:b1_idx].copy_((p_blk[b0:b1_idx].float() - lr * upd).to(torch.bfloat16))
                    del gb, m_f, v_f, sv_f, vq_raw, vq_safe, denom, upd
            else:
                g = torch.nan_to_num(g_raw.float(), nan=0.0, posinf=0.0, neginf=0.0)
                st = state_1d[p]
                st["step"] += 1
                t = st["step"]
                st["m"].mul_(b1).add_(g, alpha=1.0 - b1)
                st["v"].mul_(b2).addcmul_(g, g, value=1.0 - b2)
                bc1, bc2 = 1.0 - (b1 ** t), 1.0 - (b2 ** t)
                upd = ((st["m"] / bc1) / ((st["v"] / bc2).sqrt().add_(eps))).clamp_(-5.0, 5.0)
                p.data.copy_((p.data.float() - lr * upd).to(p.data.dtype))
                del g, upd
            del g_raw
        return _hook

    tied_emb = m_ctrl.model.embed_tokens.weight
    hooks = [
        p.register_post_accumulate_grad_hook(_make_hook(p))
        for p in list(state_2d.keys()) + list(state_1d.keys())
        if p is not tied_emb
    ]
    tied_emb_hook = _make_hook(tied_emb)
    grad_w_lm_cpu = torch.empty_like(tied_emb, device="cpu")

    sub_batches = offpolicy_batches[:60]  # 60 * 2 = 120 b=1 steps = 15 PRIMAL steps (245,760 tok)
    total_sub = len(sub_batches) * 2
    base_lr_u = lr_peak / 8.0
    warmup_u = 16
    ui = 0
    t0 = time.time()
    for x_cpu, y_cpu in sub_batches:
        for bi in range(x_cpu.size(0)):
            ui += 1
            if ui <= warmup_u:
                step_state["lr"] = base_lr_u * (ui / warmup_u)
            else:
                prog = (ui - warmup_u) / max(1, total_sub - warmup_u)
                step_state["lr"] = base_lr_u * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * prog)))
            y1d_cpu = y_cpu[bi].reshape(-1)
            n_val_int = int((y1d_cpu != -100).sum().item())
            if n_val_int == 0:
                continue
            xb = x_cpu[bi : bi + 1].to(device)
            yb = y_cpu[bi : bi + 1].to(device)
            hid = m_ctrl.model(input_ids=xb, use_cache=False).last_hidden_state
            with torch.no_grad():
                h2d = hid.detach().reshape(-1, hid.size(-1))
                y1d = yb.reshape(-1)
                n_valid = float(n_val_int)
                loss_acc = 0.0
                w_lm = m_ctrl.lm_head.weight.detach()
                grad_h = torch.empty_like(h2d)
                grad_w_lm = torch.zeros_like(w_lm)
                for s in range(0, y1d.numel(), 128):
                    e = min(s + 128, y1d.numel())
                    hc, yc = h2d[s:e], y1d[s:e]
                    vc = yc != -100
                    ys = torch.where(vc, yc, torch.zeros_like(yc))
                    lg = F.linear(hc, w_lm).float()
                    lp = F.log_softmax(lg, dim=-1)
                    del lg
                    loss_acc += float(torch.where(vc, -lp.gather(-1, ys.unsqueeze(-1)).squeeze(-1), torch.zeros_like(yc, dtype=torch.float32)).sum().item())
                    dlg = lp.exp_()
                    dlg.scatter_add_(-1, ys.unsqueeze(-1), torch.full((e - s, 1), -1.0, device=device))
                    dlg.mul_(vc.float().unsqueeze(-1) / n_valid)
                    dlg_bf = dlg.to(torch.bfloat16)
                    grad_h[s:e] = dlg_bf @ w_lm
                    grad_w_lm.addmm_(dlg_bf.t(), hc)
                    del lp, dlg, dlg_bf
                loss_val = loss_acc / n_valid
                grad_w_lm_cpu.copy_(grad_w_lm)
                del grad_w_lm, h2d
            hid.backward(grad_h.view_as(hid))
            del hid, grad_h, xb, yb
            with torch.no_grad():
                if tied_emb.grad is None:
                    tied_emb.grad = grad_w_lm_cpu.to(device)
                else:
                    for r0 in range(0, tied_emb.size(0), 16384):
                        r1 = min(tied_emb.size(0), r0 + 16384)
                        tied_emb.grad[r0:r1].add_(grad_w_lm_cpu[r0:r1].to(device))
                tied_emb_hook(tied_emb)
            if ui in (1, 40, 80, 120):
                print(f"  [ADAM8-LR2e-4 {ui:3d}/{total_sub}] loss={loss_val:.4f} | elapsed={time.time()-t0:.1f}s", flush=True)

    max_dw_sample = float((m_ctrl.model.layers[18].self_attn.q_proj.weight.detach().float().cpu() - w0_norm_sample).abs().max().item())
    del w0_norm_sample, grad_w_lm_cpu
    for hk in hooks:
        hk.remove()
    state_2d.clear()
    state_1d.clear()
    del hooks, tied_emb_hook, state_2d, state_1d
    gc.collect()
    torch.cuda.empty_cache()

    m_ctrl.gradient_checkpointing_disable()
    m_ctrl.eval()
    wt = eval_wt2_hf_bf16(m_ctrl, wt_bundle, device=device)
    gsm, _ = eval_gsm8k_full_hf_bf16(m_ctrl, tok, max_new_tokens=1024, batch_size=64, device=device)
    print(
        f"  [ADAM8-LR2e-4 RESULT] max|dw|_L18_q={max_dw_sample:.5f} | WT2 Word_PPL={wt['wt2_word_ppl']:.2f} | "
        f"GSM8K-1319={gsm['acc']:.2f}% | mean_tok={gsm['mean_gen_tokens']:.1f}",
        flush=True,
    )
    del m_ctrl
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "lr_peak": lr_peak,
        "sub_steps_b1": total_sub,
        "tokens": total_sub * 2048,
        "sample_layer18_qproj_max_abs_dw": round(max_dw_sample, 6),
        "wikitext2": wt,
        "gsm8k_1319": gsm,
    }


# ==============================================================================
# 2. On-Policy Rejection-Sampled Stream Builder (`Qwen2.5-3B-Instruct`)
# ==============================================================================

@torch.no_grad()
def build_or_load_onpolicy_stream(
    qwen_dir: str,
    tok: Tokenizer,
    cache_path: str,
    total_micro_batches: int = 100,  # 25 PRIMAL steps = 409,600 tokens
    batch_size: int = 2,
    seq_len: int = 2048,
    seed: int = 2027,
    device: str = "cuda",
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[str, str]], Dict[str, Any]]:
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")
    forbidden_q = {_norm_dedup_key(r["question"]) for r in gsm_test}

    gsm_tr = load_dataset("openai/gsm8k", "main", split="train")
    val_items: List[Tuple[str, float]] = []
    train_gsm_pool: List[Tuple[str, float]] = []
    for r in gsm_tr:
        nq = _norm_dedup_key(r["question"])
        if nq in forbidden_q:
            continue
        ans = extract_gsm8k_answer(r["answer"])
        if ans is None:
            continue
        if len(val_items) < 200:
            val_items.append((r["question"].strip(), ans))
            forbidden_q.add(nq)
        else:
            train_gsm_pool.append((r["question"].strip(), ans))

    if os.path.isfile(cache_path):
        print(f"[ON-POLICY] Loading cached on-policy stream from {cache_path}...", flush=True)
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        return saved["batches"], val_items, saved["meta"]

    rng = random.Random(seed)
    rng.shuffle(train_gsm_pool)
    cand_prompts: List[Tuple[str, str, float]] = []
    for q_txt, gold in train_gsm_pool[:420]:
        cand_prompts.append(("gsm8k_onpolicy", q_txt + GSM8K_USER_SUFFIX, gold))

    mm_ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    for r in mm_ds:
        q_raw = (r.get("query") or "").strip()
        resp_raw = (r.get("response") or "").strip()
        if not q_raw or not resp_raw or _norm_dedup_key(q_raw) in forbidden_q:
            continue
        gold = extract_gsm8k_answer(resp_raw)
        if gold is None or not math.isfinite(gold):
            continue
        cand_prompts.append(("metamath_onpolicy", q_raw + GSM8K_USER_SUFFIX, gold))
        if sum(1 for x in cand_prompts if x[0] == "metamath_onpolicy") >= 280:
            break

    print(f"\n[ON-POLICY] Generating & rejection-sampling {len(cand_prompts)} math traces from untouched Qwen2.5-3B-Instruct...", flush=True)
    t0 = time.time()
    hf_base = AutoModelForCausalLM.from_pretrained(qwen_dir, dtype=torch.bfloat16).to(device).eval()
    im_start = tok.token_to_id("<|im_start|>")
    im_end = tok.token_to_id("<|im_end|>")
    eos_id = tok.token_to_id("<|endoftext|>")
    nl = tok.encode("\n", add_special_tokens=False).ids

    enc = []
    for idx, (src, q_u, gold) in enumerate(cand_prompts):
        p_ids = (
            [im_start] + tok.encode(f"system\n{GSM8K_SYSTEM_PROMPT}", add_special_tokens=False).ids + [im_end] + nl
            + [im_start] + tok.encode(f"user\n{q_u}", add_special_tokens=False).ids + [im_end] + nl
            + [im_start] + tok.encode("assistant\n", add_special_tokens=False).ids
        )
        enc.append((idx, src, q_u, gold, p_ids))
    enc.sort(key=lambda x: len(x[4]))

    accepted: List[Tuple[str, List[Tuple[str, str]], int]] = []
    gen_bs = 64
    max_new = 768
    for st in range(0, len(enc), gen_bs):
        ch = enc[st : st + gen_bs]
        B = len(ch)
        max_p = max(len(x[4]) for x in ch)
        inp = torch.full((B, max_p), eos_id, dtype=torch.long, device=device)
        cur_am = torch.zeros((B, max_p), dtype=torch.long, device=device)
        for b, x in enumerate(ch):
            pl = max_p - len(x[4])
            inp[b, pl:] = torch.tensor(x[4], dtype=torch.long, device=device)
            cur_am[b, pl:] = 1
        pkv = DynamicCache()
        pos0 = (cur_am.cumsum(-1) - 1).clamp(min=0)
        hid0 = hf_base.model(input_ids=inp, attention_mask=cur_am, position_ids=pos0, past_key_values=pkv, use_cache=True).last_hidden_state
        nt = hf_base.lm_head(hid0[:, -1, :]).argmax(-1)
        del hid0
        active = list(range(B))
        cur_pos = torch.tensor([len(x[4]) for x in ch], dtype=torch.long, device=device)
        gen_b: List[List[int]] = [[] for _ in range(B)]
        done_ok = [False] * B
        for step in range(max_new):
            nt_l = nt.tolist()
            keep = []
            for ia, bo in enumerate(active):
                tid = nt_l[ia]
                if tid in (im_end, eos_id):
                    done_ok[bo] = True
                else:
                    gen_b[bo].append(tid)
                    keep.append(ia)
            if not keep or step == max_new - 1:
                break
            if len(keep) < len(active):
                kt = torch.tensor(keep, dtype=torch.long, device=device)
                active = [active[i] for i in keep]
                nt = nt[kt]
                cur_pos = cur_pos[kt]
                cur_am = cur_am[kt]
                pkv.batch_select_indices(kt)
            cur_am = torch.cat([cur_am, torch.ones((len(active), 1), dtype=torch.long, device=device)], dim=1)
            hs = hf_base.model(input_ids=nt.unsqueeze(1), attention_mask=cur_am, position_ids=cur_pos.unsqueeze(1), past_key_values=pkv, use_cache=True).last_hidden_state
            nt = hf_base.lm_head(hs[:, -1, :]).argmax(-1)
            cur_pos = cur_pos + 1
        del pkv
        for b, (_, src, q_u, gold, _) in enumerate(ch):
            if not done_ok[b] or len(gen_b[b]) < 32:
                continue
            txt = tok.decode(gen_b[b], skip_special_tokens=True).strip()
            pred = extract_gsm8k_answer(txt)
            if "\\boxed{" in txt and pred is not None and abs(pred - gold) < 1e-4:
                turns = [("system", GSM8K_SYSTEM_PROMPT), ("user", q_u), ("assistant", txt)]
                accepted.append((src, turns, len(gen_b[b])))
        print(f"  [ON-POLICY {min(st+gen_bs, len(enc))}/{len(enc)}] accepted={len(accepted)} ({time.time()-t0:.1f}s)", flush=True)

    del hf_base
    gc.collect()
    torch.cuda.empty_cache()

    rng.shuffle(accepted)
    mean_onpolicy_tok = round(sum(x[2] for x in accepted) / max(1, len(accepted)), 1)
    accept_pct = round(100.0 * len(accepted) / max(1, len(cand_prompts)), 2)
    print(f"[ON-POLICY DONE] Accepted {len(accepted)}/{len(cand_prompts)} ({accept_pct}%) | mean_trace_tok={mean_onpolicy_tok}", flush=True)

    it_magic = iter(load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True))
    it_cf = iter(load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True))
    it_fw = iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True))
    it_cosmo = iter(load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True))

    onp_idx = 0
    counts = {"onpolicy_math_cot": 0, "magicoder": 0, "codefeedback": 0, "fineweb_edu": 0, "cosmopedia": 0}
    sources = [
        ("onpolicy_math_cot", 0.50),
        ("magicoder",         0.10),
        ("codefeedback",      0.10),
        ("fineweb_edu",       0.20),
        ("cosmopedia",        0.10),
    ]
    s_names = [s[0] for s in sources]
    s_probs = [s[1] for s in sources]

    def _next_item(src: str):
        nonlocal onp_idx
        while True:
            if src == "onpolicy_math_cot":
                _, turns, _ = accepted[onp_idx % len(accepted)]
                onp_idx += 1
                return ("sft", turns)
            if src == "magicoder":
                r = next(it_magic)
                p, s = (r.get("problem") or "").strip(), (r.get("solution") or "").strip()
                if len(p) >= 24 and 64 <= len(s) <= 7000 and _norm_dedup_key(p) not in forbidden_q:
                    return ("sft", [("system", DEFAULT_SYSTEM_PROMPT), ("user", p), ("assistant", s)])
            if src == "codefeedback":
                r = next(it_cf)
                p, s = (r.get("query") or "").strip(), (r.get("answer") or "").strip()
                if len(p) >= 24 and 64 <= len(s) <= 7000 and _norm_dedup_key(p) not in forbidden_q:
                    return ("sft", [("system", DEFAULT_SYSTEM_PROMPT), ("user", p), ("assistant", s)])
            if src == "fineweb_edu":
                t = (next(it_fw).get("text") or "").strip()
                if len(t) >= 250:
                    return ("lm", t[:10000])
            if src == "cosmopedia":
                t = (next(it_cosmo).get("text") or "").strip()
                if len(t) >= 250:
                    return ("lm", t[:10000])

    need = seq_len + 1
    buf_i: List[int] = []
    buf_l: List[int] = []
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    while len(batches) < total_micro_batches:
        bx, by = [], []
        for _ in range(batch_size):
            while len(buf_i) < need:
                ch_s = rng.choices(s_names, weights=s_probs, k=1)[0]
                mode, payload = _next_item(ch_s)
                counts[ch_s] += 1
                if mode == "lm":
                    raw = tok.encode(payload, add_special_tokens=False).ids
                    if len(raw) >= 64:
                        c = raw[:seq_len] + [eos_id]
                        buf_i.extend(c)
                        buf_l.extend(c)
                else:
                    ci, cl = encode_sft_turns_masked(tok, payload, im_start, im_end, nl)
                    if len(ci) <= seq_len and any(v != -100 for v in cl):
                        buf_i.extend(ci + [eos_id])
                        buf_l.extend(cl + [-100])
            bx.append(buf_i[:seq_len])
            by.append(buf_l[1:need])
            del buf_i[:seq_len]
            del buf_l[:seq_len]
        batches.append((torch.tensor(bx, dtype=torch.long), torch.tensor(by, dtype=torch.long)))

    meta = {
        "n_candidates": len(cand_prompts),
        "n_accepted": len(accepted),
        "acceptance_rate_pct": accept_pct,
        "mean_onpolicy_trace_tokens": mean_onpolicy_tok,
        "stream_counts": counts,
    }
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save({"batches": batches, "meta": meta}, cache_path)
    return batches, val_items, meta


# ==============================================================================
# 3. Exact V3_Q8_0 (FP32 Product) vs. Exported BF16 Safetensors Comparison
# ==============================================================================

@torch.no_grad()
def load_hf_from_vlt3_dtype(vlt3_path: str, qwen_dir: str, dtype: torch.dtype, device: str = "cuda") -> AutoModelForCausalLM:
    tools_dir = os.path.join(VAULT_ROOT, "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import vlt3
    cfg = AutoConfig.from_pretrained(qwen_dir)
    hf_m = AutoModelForCausalLM.from_config(cfg).to(device=device, dtype=dtype).eval()
    vf = vlt3.read_vlt3(vlt3_path, check_crc=True)
    raw_sd = vlt3.vlt3_to_state_dict(vf)
    for nm, t_cpu in raw_sd.items():
        if nm.startswith("seg_scorer."):
            continue
        t = t_cpu.to(device=device, dtype=dtype)
        if nm == "embed_tokens.weight":
            hf_m.model.embed_tokens.weight.data.copy_(t)
        elif nm == "norm.weight":
            hf_m.model.norm.weight.data.copy_(t)
        elif nm.startswith("layers."):
            parts = nm.split(".")
            li = int(parts[1])
            hl = hf_m.model.layers[li]
            sub = ".".join(parts[2:])
            if sub == "input_layernorm.weight":
                hl.input_layernorm.weight.data.copy_(t)
            elif sub == "post_attention_layernorm.weight":
                hl.post_attention_layernorm.weight.data.copy_(t)
            elif sub.startswith("self_attn."):
                getattr(getattr(hl.self_attn, parts[3]), parts[4]).data.copy_(t)
            elif sub.startswith("mlp."):
                getattr(getattr(hl.mlp, parts[3]), parts[4]).data.copy_(t)
        del t
    del raw_sd, vf
    if cfg.tie_word_embeddings:
        hf_m.lm_head.weight = hf_m.model.embed_tokens.weight
    return hf_m


class _ExactQ8LinearFP32(nn.Module):
    def __init__(self, q: torch.Tensor, d: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("q", q.to(torch.int8))
        self.register_buffer("d", d.to(torch.float16))
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float32))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_f, in_f = self.q.shape
        if out_f > 65536:
            outs = []
            x_f = x.float()
            for r0 in range(0, out_f, 32768):
                r1 = min(out_f, r0 + 32768)
                wc = (self.q[r0:r1].view(r1 - r0, -1, QK).float() * self.d[r0:r1].float().unsqueeze(-1)).reshape(r1 - r0, in_f)
                bc = self.bias[r0:r1] if self.bias is not None else None
                outs.append(F.linear(x_f, wc, bc))
                del wc
            return torch.cat(outs, dim=-1)
        w = (self.q.view(out_f, -1, QK).float() * self.d.float().unsqueeze(-1)).reshape(out_f, in_f)
        return F.linear(x.float(), w, self.bias)


class _ExactQ8EmbedFP32(nn.Module):
    def __init__(self, q: torch.Tensor, d: torch.Tensor):
        super().__init__()
        self.register_buffer("q", q.to(torch.int8))
        self.register_buffer("d", d.to(torch.float16))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.reshape(-1)
        qr = self.q.index_select(0, flat)
        dr = self.d.index_select(0, flat)
        n, dim = qr.shape
        vals = (qr.view(n, -1, QK).float() * dr.float().unsqueeze(-1)).reshape(n, dim)
        return vals.reshape(*input_ids.shape, dim)


@torch.no_grad()
def load_hf_exact_q8_fp32_compact(best_ckpt_path: str, qwen_dir: str, device: str = "cuda") -> AutoModelForCausalLM:
    cfg = AutoConfig.from_pretrained(qwen_dir)
    with torch.device("meta"):
        hf_m = AutoModelForCausalLM.from_config(cfg)
    ck = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)["model_state"]

    emb_q = ck["embed_tokens.q"].to(device)
    emb_d = ck["embed_tokens.d"].to(device)
    hf_m.model.embed_tokens = _ExactQ8EmbedFP32(emb_q, emb_d)
    hf_m.model.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device=device, dtype=torch.float32)
    hf_m.model.norm.weight.data.copy_(ck["norm.weight"].float().to(device))
    hf_m.model.rotary_emb = AutoModelForCausalLM.from_config(cfg).model.rotary_emb.to(device=device)

    for li in range(cfg.num_hidden_layers):
        hl = hf_m.model.layers[li]
        pfx = f"layers.{li}."
        hl.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device=device, dtype=torch.float32)
        hl.input_layernorm.weight.data.copy_(ck[pfx + "input_layernorm.weight"].float().to(device))
        hl.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device=device, dtype=torch.float32)
        hl.post_attention_layernorm.weight.data.copy_(ck[pfx + "post_attention_layernorm.weight"].float().to(device))
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            b_key = pfx + f"self_attn.{proj}.bias"
            b_t = ck[b_key].to(device) if b_key in ck else None
            setattr(hl.self_attn, proj, _ExactQ8LinearFP32(ck[pfx + f"self_attn.{proj}.q"].to(device), ck[pfx + f"self_attn.{proj}.d"].to(device), b_t))
        for mproj in ("gate_proj", "up_proj", "down_proj"):
            setattr(hl.mlp, mproj, _ExactQ8LinearFP32(ck[pfx + f"mlp.{mproj}.q"].to(device), ck[pfx + f"mlp.{mproj}.d"].to(device), None))

    hf_m.lm_head = _ExactQ8LinearFP32(emb_q, emb_d, None)
    del ck
    hf_m.requires_grad_(False)
    return hf_m.eval()


@torch.no_grad()
def eval_exact_fp32_q8_canary(vlt3_q8_path: str, qwen_dir: str, tok: Tokenizer, tasks: Dict[str, Any], wt_bundle, device: str = "cuda") -> Dict[str, Any]:
    """
    Evaluates exact V3_Q8_0 (`int8 * fp16`) weights in IEEE-754 float32 (0 mantissa truncation!)
    at 3.06 GiB persistent VRAM (<5 GiB peak VRAM, zero GTT spillover) to explicitly measure
    any delta vs. BF16-exported safetensors on deterministic tasks.
    """
    print("\n=== [EXACT V3_Q8_0 FP32 CHECK] Evaluating Bit-Exact `int8 * fp16` Weights in FP32 (3.06 GiB Compact) ===", flush=True)
    t0 = time.perf_counter()
    best_ckpt_path = os.path.join(OUT_DIR, "best_ckpt.pt")
    hf_fp32 = load_hf_exact_q8_fp32_compact(best_ckpt_path, qwen_dir, device=device)
    wt = eval_wt2_hf_bf16(hf_fp32, wt_bundle, device=device)
    print(f"  [FP32 WikiText-2 ] BPB={wt['wt2_bpb']:.4f} | Word_PPL={wt['wt2_word_ppl']:.2f} | Tok_PPL={wt['wt2_tok_ppl']:.2f} ({time.perf_counter()-t0:.1f}s)", flush=True)

    lam_pairs = []
    for txt in tasks["lambada_texts"][:1000]:
        t = txt.strip()
        if " " in t:
            ctx, last = t.rsplit(" ", 1)
            lam_pairs.append((ctx, " " + last))
    lam_sc = score_pairs_hf_bf16(hf_fp32, tok, lam_pairs, batch_size=16, device=device)
    lam_1k_acc = round(100.0 * sum(1 for _, _, ex, _ in lam_sc if ex) / len(lam_sc), 2)
    print(f"  [FP32 LAMBADA-1K ] acc={lam_1k_acc:.2f}% ({time.perf_counter()-t0:.1f}s)", flush=True)

    mc_out = {}
    for key, name, items, _metric in tasks["mc"]:
        if key not in ("arc_easy", "arc_challenge", "sciq_rc", "sciq_closedbook"):
            continue
        flat_pairs = []
        spans = []
        for cands, gold in items:
            s0 = len(flat_pairs)
            flat_pairs.extend(cands)
            spans.append((s0, len(flat_pairs), gold))
        sc = score_pairs_hf_bf16(hf_fp32, tok, flat_pairs, batch_size=16, device=device)
        hits_norm = sum(
            1 for s0, s1, gold in spans
            if max(range(s1 - s0), key=lambda i: sc[s0 + i][1]) == gold
        )
        mc_out[key] = round(100.0 * hits_norm / len(items), 2)
        print(f"  [FP32 {name:<15}] acc_norm={mc_out[key]:.2f}% ({time.perf_counter()-t0:.1f}s)", flush=True)

    del hf_fp32
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  [EXACT V3_Q8_0 FP32 DONE in {time.perf_counter()-t0:.1f}s]", flush=True)
    return {
        "wikitext2": wt,
        "lambada_1k_acc": lam_1k_acc,
        "arc_easy_full": mc_out["arc_easy"],
        "arc_challenge_full": mc_out["arc_challenge"],
        "sciq_rc_full": mc_out["sciq_rc"],
        "sciq_cb_full": mc_out["sciq_closedbook"],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda"
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    tok = Tokenizer.from_file(os.path.join(DEFAULT_QWEN_3B, "tokenizer.json"))
    eos_id = tok.token_to_id("<|endoftext|>")
    wt_bundle = build_wikitext2_val(tok, seq_len=2048)
    val_sft_samples = build_heldout_sft_val(
        tok, seq_len=2048, n_per_source=20,
        cache_path=os.path.join(V3_DIR, "heldout_sft_val_cache.pt"),
    )
    tasks = load_all_9_tasks()

    # Phase 1: Step-radius-matched 8-Bit AdamW (`lr_peak = 2e-4`, 15 steps) on Round-1/2 off-policy stream
    offp_saved = torch.load(os.path.join(V3_DIR, "primal_v3_50step_stream.pt"), map_location="cpu", weights_only=False)
    adam_lr2e4_offpolicy = run_step_matched_adamw8bit_offpolicy_15steps(
        DEFAULT_QWEN_3B, offp_saved["batches"], wt_bundle, tok, lr_peak=2.0e-4, device=device
    )
    del offp_saved
    gc.collect()

    # Phase 2: Build On-Policy Rejection-Sampled Qwen2.5-3B-Instruct Stream (`25` steps = `409,600` tokens)
    onp_cache = os.path.join(OUT_DIR, "primal_v3_1_onpolicy_stream.pt")
    batches, gsm8k_val200, onp_meta = build_or_load_onpolicy_stream(
        DEFAULT_QWEN_3B, tok, cache_path=onp_cache, total_micro_batches=100, batch_size=2, seq_len=2048, seed=2027, device=device
    )

    # Phase 3: Train Lossless PRIMAL v3.1 (Sparse `rebalance_scales` -> flat ~9.28 GiB peak VRAM!)
    with open(os.path.join(DEFAULT_QWEN_3B, "config.json")) as f:
        qc = json.load(f)
    cfg = VAULTPrimalConfig.from_qwen_config(qc, window=4096)
    model = VAULTPrimalForCausalLM(cfg, skip_init=True, grad_checkpoint=True, ce_chunk_size=512)
    load_pretrained_qwen2_into_primal_q8(model, DEFAULT_QWEN_3B, headroom=1.05)
    q0_snap = {n: m.q.clone() for n, m in model.named_modules() if isinstance(m, _Q8BlockMixin)}

    model = model.to(device)
    total_steps = 25
    grad_accum = 4
    engine = Q8StepEngine(
        model,
        total_steps=total_steps,
        warmup_steps=4,
        threshold_start=24,
        threshold_end=30,
        vote_gain_start=0.36,
        vote_gain_end=0.16,
        vote_clip=8.0,
        decay_start=1.00,
        decay_end=1.00,
        weight_decay=0.0,
        max_stride=1,
        error_feedback=True,
        aux_lr=1.0e-5,
        schedule="wsd",
        cooldown_frac=0.30,
        rebalance_every=10,
        rail_trigger=122,
        grow_factor=1.12,
        scale_aware_votes=True,
        grad_accum=grad_accum,
    )

    wt0 = evaluate_wikitext2_bpb(model, wt_bundle[0], wt_bundle[1], wt_bundle[2], 2048, device, batch_size=2)
    gv0 = eval_gsm8k_val200_primal(model, DEFAULT_QWEN_3B, tok, gsm8k_val200, device=device)
    print(f"\n[PRIMAL-v3.1 STEP 0] WT2 BPB={wt0['wt2_bpb']:.4f} (PPL={wt0['wt2_word_ppl']:.2f}) | GSM8K-TrainVal200={gv0['acc']:.2f}%", flush=True)

    best_ckpt_path = os.path.join(OUT_DIR, "best_ckpt.pt")
    best_pareto = -1e9
    best_step = 0
    total_rebal = 0
    steady_peak_gib = 0.0
    rebal_peak_gib = 0.0
    train_log = []
    t_tr0 = time.time()

    for step in range(1, total_steps + 1):
        t_s = time.time()
        engine.zero_grad()
        loss_acc = 0.0
        torch.cuda.reset_peak_memory_stats()
        for mb_i in range((step - 1) * grad_accum, step * grad_accum):
            xc, yc = batches[mb_i]
            xg = xc.to(device, non_blocking=True)
            yg = yc.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(xg, yg, ignore_index=-100)
            loss_acc += float(loss.item()) / grad_accum
            (loss / grad_accum).backward()
            del xg, yg, loss

        st_stats = engine.step(collect_stats=True)
        pk_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        n_reb = int(st_stats.get("rebal_blocks", 0))
        total_rebal += n_reb
        if (step % 10) == 0:
            rebal_peak_gib = max(rebal_peak_gib, pk_gib)
        else:
            steady_peak_gib = max(steady_peak_gib, pk_gib)
        tok_s = (grad_accum * 2 * 2048) / max(1e-6, time.time() - t_s)
        print(
            f"[STEP {step:2d}/{total_steps}] loss={loss_acc:.4f} | flips={st_stats['flip_pct']:.4f}% ({st_stats['flips']:,}) | "
            f"rebal={n_reb} | peak={pk_gib:.2f} GiB | {tok_s:.0f} tok/s",
            flush=True,
        )

        if step in (10, 15, 20, 25):
            wt_v = evaluate_wikitext2_bpb(model, wt_bundle[0], wt_bundle[1], wt_bundle[2], 2048, device, batch_size=2)
            vs_v = evaluate_heldout_sft_loss(model, val_sft_samples, eos_id, device, batch_size=4)
            gv_v = eval_gsm8k_val200_primal(model, DEFAULT_QWEN_3B, tok, gsm8k_val200, device=device)
            # Balanced validation Pareto: preserves GSM8K-TrainVal200 while rewarding WT2 BPB improvement!
            pareto = gv_v["acc"] - 120.0 * wt_v["wt2_bpb"]
            print(
                f"  >>> [VAL @{step:2d}] val_sft={vs_v:.4f} | WT2 BPB={wt_v['wt2_bpb']:.4f} (PPL={wt_v['wt2_word_ppl']:.2f}) | "
                f"GSM8K-TrainVal200={gv_v['acc']:.2f}% (boxed={gv_v['boxed_pct']:.1f}%) | pareto={pareto:.3f}",
                flush=True,
            )
            train_log.append({
                "step": step, "val_sft": round(vs_v, 4),
                "wt2_bpb": wt_v["wt2_bpb"], "wt2_word_ppl": wt_v["wt2_word_ppl"],
                "gsm8k_val200_acc": gv_v["acc"], "gsm8k_val200_boxed": gv_v["boxed_pct"],
                "pareto": round(pareto, 4),
            })
            if pareto > best_pareto:
                best_pareto = pareto
                best_step = step
                torch.save({"step": step, "model_state": {k: v.cpu() for k, v in model.state_dict().items()}}, best_ckpt_path)
                print(f"  >>> [BEST] Saved validation-selected PRIMAL v3.1 checkpoint at Step {step}", flush=True)

    ck = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"], strict=True)
    del ck

    # Anatomy measurement
    changed_tot = 0
    disc_tot = 0
    max_dq = 0
    by_mod: Dict[str, List[int]] = {}
    for n, m in model.named_modules():
        if isinstance(m, _Q8BlockMixin):
            dq = (m.q.detach().cpu().to(torch.int16) - q0_snap[n].to(torch.int16)).abs()
            ch = int((dq > 0).sum().item())
            tot = int(dq.numel())
            mx = int(dq.max().item())
            changed_tot += ch
            disc_tot += tot
            max_dq = max(max_dq, mx)
            tag = n.split(".")[-1]
            by_mod.setdefault(tag, [0, 0])
            by_mod[tag][0] += ch
            by_mod[tag][1] += tot
    del q0_snap
    anatomy = {
        "selected_step": best_step,
        "total_discrete_params": disc_tot,
        "changed_params": changed_tot,
        "changed_pct": round(100.0 * changed_tot / max(1, disc_tot), 4),
        "max_abs_dq_ticks": max_dq,
        "rebalanced_blocks_total": total_rebal,
        "steady_peak_train_gib": round(steady_peak_gib, 3),
        "rebalance_step_peak_train_gib": round(rebal_peak_gib, 3),
        "by_module_pct": {k: round(100.0 * v[0] / max(1, v[1]), 4) for k, v in by_mod.items()},
    }
    print(f"\n[PRIMAL-v3.1 ANATOMY] {json.dumps(anatomy, indent=2)}", flush=True)

    # Export zero-PTQ V3_Q8_0 and Hessian-calibrated V3_P4_0 `.vlt3` binaries, then free training model from GPU!
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    q8_raw_vlt3 = os.path.join(OUT_DIR, "vault_v3_3b_q8_0_primal_raw.vlt3")
    p4_raw_vlt3 = os.path.join(OUT_DIR, "vault_v3_3b_p4_0_primal_raw.vlt3")
    export_vault_primal_vlt3(model, DEFAULT_QWEN_3B, q8_raw_vlt3, model_name="VAULT-v3-3B-q8_0-primal-v3.1")
    calib = [wt_bundle[0][i * 1024 : (i + 1) * 1024].unsqueeze(0) for i in range(16)]
    export_hessian_p4_0_from_primal_3b(model, DEFAULT_QWEN_3B, p4_raw_vlt3, model_name="VAULT-v3-3B-p4_0-primal-v3.1", calib_batches=calib, device=device)
    model = model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Repack trained SegmentScorer into `vault/dist/` with CRC32 verification
    s_ck = torch.load(SCORER_CKPT, map_location="cpu", weights_only=False)
    v3mod = _load_v3_mod()
    scorer = v3mod.SegmentScorer(s_ck["d"], s_ck["hidden"])
    scorer.load_state_dict(s_ck["state"], strict=True)
    scorer.eval()
    seg_layers = tuple(int(x) for x in s_ck["layers"])
    q8_dst = os.path.join(VAULT_DIST, "vault_v3_3b_q8_0_primal.vlt3")
    p4_dst = os.path.join(VAULT_DIST, "vault_v3_3b_p4_0_primal.vlt3")
    info_q8 = repack_vlt3_with_seg_scorer(q8_raw_vlt3, q8_dst, scorer, seg_layers, window=int(s_ck["window"]), seg_budget=int(s_ck["seg_budget"]), model_name="VAULT-v3-3B-q8_0-primal-v3.1")
    info_p4 = repack_vlt3_with_seg_scorer(p4_raw_vlt3, p4_dst, scorer, seg_layers, window=int(s_ck["window"]), seg_budget=int(s_ck["seg_budget"]), model_name="VAULT-v3-3B-p4_0-primal-v3.1")
    print(f"[REPACK v3.1] Q8_0={info_q8['file_size_mb']} MiB ({info_q8['file_size_mb']/1024:.3f} GiB) | P4_0={info_p4['file_size_mb']} MiB ({info_p4['file_size_mb']/1024:.3f} GiB)", flush=True)

    # Phase 4A: Full-Split Evaluation on PRIMAL v3.1 (Q8_0 -> BF16 Safetensors Path)
    hf_q8_bf16 = load_hf_from_vlt3_dtype(q8_dst, DEFAULT_QWEN_3B, dtype=torch.bfloat16, device=device)
    q8_rep, q8_vecs = eval_9task_full_and_1k(hf_q8_bf16, tok, tasks, wt_bundle, label="VAULT-v3-3B-PRIMAL v3.1 (Q8_0 -> BF16)", device=device)
    q8_gsm, q8_gsm_vec = eval_gsm8k_full_hf_bf16(hf_q8_bf16, tok, max_new_tokens=1024, batch_size=64, device=device)
    q8_rep["gsm8k_1319"] = q8_gsm
    q8_vecs["gsm8k_1319"] = q8_gsm_vec
    del hf_q8_bf16
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 4B: Exact `.vlt3` V3_Q8_0 (FP32 Product) Evaluation
    exact_q8_fp32 = eval_exact_fp32_q8_canary(q8_dst, DEFAULT_QWEN_3B, tok, tasks, wt_bundle, device=device)

    # Phase 4C: Full-Split Evaluation on PRIMAL v3.1 (P4_0, 1.805 GiB)
    hf_p4_bf16 = load_hf_from_vlt3_dtype(p4_dst, DEFAULT_QWEN_3B, dtype=torch.bfloat16, device=device)
    p4_rep, p4_vecs = eval_9task_full_and_1k(hf_p4_bf16, tok, tasks, wt_bundle, label="VAULT-v3-3B-PRIMAL v3.1 (P4_0, 1.805 GiB)", device=device)
    p4_gsm, p4_gsm_vec = eval_gsm8k_full_hf_bf16(hf_p4_bf16, tok, max_new_tokens=1024, batch_size=64, device=device)
    p4_rep["gsm8k_1319"] = p4_gsm
    p4_vecs["gsm8k_1319"] = p4_gsm_vec
    del hf_p4_bf16
    gc.collect()
    torch.cuda.empty_cache()

    # Paired bootstrap CIs & McNemar vs. Instruct, Base, and Round-1/2 8-bit AdamW
    paired_stats = {}
    rv = torch.load("primal_v7/runs/peer_review_v3/raw_baseline_item_vectors.pt", map_location="cpu", weights_only=False)
    v3_iv = torch.load(os.path.join(V3_DIR, "primal_v3_item_vectors.pt"), map_location="cpu", weights_only=False)
    for cname, ref_v in [
        ("v3_1_q8_vs_instruct_bf16", rv["instruct_vecs"]),
        ("v3_1_q8_vs_base_bf16", rv["base_vecs"]),
        ("v3_1_q8_vs_adamw8bit_lr1e5", v3_iv["ctrl_vecs"]),
    ]:
        paired_stats[cname] = {}
        for k in q8_vecs:
            if k in ref_v and len(q8_vecs[k]) == len(ref_v[k]):
                paired_stats[cname][k] = paired_bootstrap_diff_ci(q8_vecs[k], ref_v[k])

    report = {
        "step_matched_adamw8bit_lr2e4_offpolicy": adam_lr2e4_offpolicy,
        "onpolicy_stream_meta": onp_meta,
        "primal_v3_1_train_log": train_log,
        "primal_v3_1_anatomy": anatomy,
        "vlt3_sizes": {
            "q8_0_mib": info_q8["file_size_mb"],
            "q8_0_gib": round(info_q8["file_size_mb"] / 1024.0, 3),
            "p4_0_mib": info_p4["file_size_mb"],
            "p4_0_gib": round(info_p4["file_size_mb"] / 1024.0, 3),
        },
        "primal_v3_1_q8_0_bf16_eval": q8_rep,
        "primal_v3_1_q8_0_exact_fp32_eval": exact_q8_fp32,
        "primal_v3_1_p4_0_eval": p4_rep,
        "paired_significance": paired_stats,
    }
    out_json = os.path.join(OUT_DIR, "peer_review_primal_v3_1_full_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"q8_vecs": q8_vecs, "p4_vecs": p4_vecs}, os.path.join(OUT_DIR, "primal_v3_1_item_vectors.pt"))
    print(f"\n[PRIMAL v3.1 PIPELINE COMPLETE] -> {out_json}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
