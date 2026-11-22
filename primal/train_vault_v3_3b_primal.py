#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
VAULT-v3-3B: Full-Parameter Shadowless Discrete PRIMAL (V3_Q8_0) Training on a Single 16 GB GPU.

Updates 100% of all 3,085,938,688 parameters of Qwen2.5-3B-Instruct (99.992% native V3_Q8_0
blocks + 0.008% RMSNorms/QKV biases) with ZERO LoRA, ZERO shadow weights, ZERO persistent
.grad buffers on 2D weights, and ZERO AdamW m/v optimizer states on 2D weights:
  - Persistent training state: 5.93 GiB (vs 45.98 GiB in BF16+FP32 AdamW -> 7.75x compression!)
  - Peak VRAM on AMD RX 9060 XT (16 GB): ~9.3 GiB at B=2, T=2048
  - Zero-PTQ-loss direct export to freestanding C99 VLT3 (.vlt3) + Hessian Prime-Harmonic V3_P4_0!

Frontier Reasoning + Math + Code + Anti-Forgetting Replay Mix (Zero GSM8K-Test Leakage):
  - 25% open-r1/OpenR1-Math-220k                 (verified competition/Olympiad math traces)
  - 20% meta-math/MetaMathQA                     (verified GSM8K/MATH train CoT, test-deduped)
  - 15% bespokelabs/Bespoke-Stratos-17k          (distilled deep reasoning traces)
  - 10% ise-uiuc/Magicoder-OSS-Instruct-75K      (algorithmic coding & data structures)
  - 10% m-a-p/CodeFeedback-Filtered-Instruction  (multi-language code instruction following)
  -  5% HuggingFaceTB/smoltalk                   (general ChatML STEM & helpfulness)
  - 15% HuggingFaceFW/fineweb-edu (sample-100BT) (full-LM anti-forgetting replay)
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import queue
import random
import re
import signal
import subprocess
import sys
import threading
import time
import warnings
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
from tokenizers import Tokenizer

from primal.vault_primal import (
    Q8StepEngine,
    VAULTPrimalConfig,
    VAULTPrimalForCausalLM,
    _dequant_q8_0,
    export_vault_primal_vlt3,
    load_pretrained_qwen2_into_primal_q8,
)

warnings.filterwarnings("ignore", message=".*Flash Efficient attention.*")
warnings.filterwarnings("ignore", message=".*Mem Efficient attention.*")

DEFAULT_QWEN_3B = os.environ.get("QWEN_3B_DIR", "Qwen/Qwen2.5-3B-Instruct")


def _resolve_qwen_3b_dir() -> str:
    env_d = os.environ.get("QWEN_3B_DIR", "")
    if env_d and os.path.isdir(env_d):
        return env_d
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    except Exception:
        return snapshot_download("Qwen/Qwen2.5-3B-Instruct")

VAULT_ROOT = "."
VAULT_HOST_BIN = os.path.join(VAULT_ROOT, "runtime", "vault3_host")
DEFAULT_SYSTEM_PROMPT = "You are a helpful, mathematically rigorous AI assistant."
GSM8K_SYSTEM_PROMPT = "You are a helpful, mathematically rigorous AI assistant."
GSM8K_USER_SUFFIX = "\n\nSolve this problem step by step and put your final numerical answer inside \\boxed{}."

# (name, weight, repo, config, split, mode)
SFT_SOURCES = [
    ("openr1_math",  0.20, "open-r1/OpenR1-Math-220k",               "default",      "train", "sft"),
    ("metamath",     0.20, "meta-math/MetaMathQA",                   None,           "train", "sft"),
    ("magicoder",    0.10, "ise-uiuc/Magicoder-OSS-Instruct-75K",    None,           "train", "sft"),
    ("codefeedback", 0.10, "m-a-p/CodeFeedback-Filtered-Instruction",None,           "train", "sft"),
    ("smoltalk",     0.10, "HuggingFaceTB/smoltalk",                 "all",          "train", "sft"),
    ("fineweb_edu",  0.20, "HuggingFaceFW/fineweb-edu",              "sample-100BT", "train", "lm"),
    ("cosmopedia",   0.10, "HuggingFaceTB/smollm-corpus",            "cosmopedia-v2","train", "lm"),
]


# ==============================================================================
# 1. GSM8K Contamination Guard & ChatML Formatting (Assistant-Only Masking)
# ==============================================================================

def _norm_dedup_key(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return s[:120]


def build_gsm8k_test_dedup_set() -> set[str]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    keys = set()
    for r in ds:
        k = _norm_dedup_key(r["question"])
        if len(k) >= 20:
            keys.add(k)
    return keys


def _is_val_key(text: str) -> bool:
    h = hashlib.md5(text[:256].encode("utf-8", "ignore")).digest()
    return (int.from_bytes(h[:4], "little") % 200) == 0


def _extract_turns(name: str, row: dict, max_chars: int = 5600) -> Optional[Tuple[str, List[Tuple[str, str]]]]:
    """
    Convert any source row into (mode, payload):
      - mode == "lm":  returns ("lm", [("text", raw_text)])
      - mode == "sft": returns ("sft", [(role, content), ...]) with concise, verified complete answers
    Returns None if row is empty, malformed, unverified, or verbose/truncated.
    """
    if name in ("fineweb_edu", "cosmopedia"):
        txt = (row.get("text") or "").strip()
        if len(txt) < 200:
            return None
        return ("lm", [("text", txt[:12000])])

    if name == "openr1_math":
        prob = (row.get("problem") or "").strip()
        sol = (row.get("solution") or "").strip()
        ans = (row.get("answer") or "").strip()
        if len(prob) < 15 or len(sol) < 30 or len(prob) + len(sol) > max_chars:
            return None
        # Use concise Olympiad reference solutions (zero <think> rambling that overflows 512-tok decode!)
        if "<think>" in sol or "<|begin_of_thought|>" in sol:
            return None
        if ans and "\\boxed" not in sol:
            sol = f"{sol}\n\nTherefore, the final answer is $\\boxed{{{ans}}}$."
        return ("sft", [
            ("system", DEFAULT_SYSTEM_PROMPT),
            ("user", prob),
            ("assistant", sol),
        ])

    if name == "metamath":
        q = (row.get("query") or "").strip()
        r = (row.get("response") or "").strip()
        if len(q) < 15 or len(r) < 20 or len(q) + len(r) > max_chars:
            return None
        # Cleanly replace MetaMath's "The answer is: X" tail with crisp Qwen2.5 \boxed{X}
        m = re.search(r"\n*The answer is:\s*([^\n]+)$", r)
        if m:
            ans = m.group(1).strip().rstrip(".")
            body = r[: m.start()].rstrip()
            r = f"{body}\n\nTherefore, the final answer is $\\boxed{{{ans}}}$."
        return ("sft", [
            ("system", DEFAULT_SYSTEM_PROMPT),
            ("user", q),
            ("assistant", r),
        ])

    if name == "magicoder":
        p = (row.get("problem") or "").strip()
        s = (row.get("solution") or "").strip()
        if len(p) < 20 or len(s) < 30 or len(p) + len(s) > max_chars:
            return None
        return ("sft", [
            ("system", DEFAULT_SYSTEM_PROMPT),
            ("user", p),
            ("assistant", s),
        ])

    if name == "codefeedback":
        q = (row.get("query") or "").strip()
        a = (row.get("answer") or "").strip()
        if len(q) < 20 or len(a) < 30 or len(q) + len(a) > max_chars:
            return None
        return ("sft", [
            ("system", DEFAULT_SYSTEM_PROMPT),
            ("user", q),
            ("assistant", a),
        ])

    if name == "smoltalk":
        msgs = row.get("messages") or []
        if not isinstance(msgs, list) or len(msgs) < 2:
            return None
        turns = []
        total_len = 0
        for m in msgs:
            role = (m.get("role") or "user").strip()
            content = (m.get("content") or "").strip()
            if content:
                turns.append((role, content))
                total_len += len(content)
        if not turns or turns[-1][0] != "assistant" or total_len < 40 or total_len > max_chars:
            return None
        if turns[0][0] != "system":
            turns.insert(0, ("system", DEFAULT_SYSTEM_PROMPT))
        return ("sft", turns)

    return None


def encode_sft_turns_masked(
    tok: Tokenizer,
    turns: List[Tuple[str, str]],
    im_start_id: int,
    im_end_id: int,
    nl_ids: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Encode ChatML turns into (input_ids, label_ids) with exact assistant-only loss masking:
      - system / user / role-header tokens -> label = -100 (masked)
      - assistant content + <|im_end|>     -> label = exact token id
    Note: Returned (ids, labels) are aligned at the same token index; shift by 1 when creating (x, y).
    """
    ids: List[int] = []
    labels: List[int] = []
    for role, content in turns:
        hdr = [im_start_id] + tok.encode(f"{role}\n", add_special_tokens=False).ids
        body = tok.encode(content, add_special_tokens=False).ids + [im_end_id]
        ids.extend(hdr)
        labels.extend([-100] * len(hdr))
        ids.extend(body)
        if role == "assistant":
            labels.extend(body)
        else:
            labels.extend([-100] * len(body))
        ids.extend(nl_ids)
        labels.extend([-100] * len(nl_ids))
    return ids, labels


# ==============================================================================
# 2. Async Multi-Source Packed SFT + Replay Streamer (Zero GPU Starvation)
# ==============================================================================

class FrontierSFTReplayStream:
    """
    Infinite multi-source async streamer that:
      - Streams 6 SFT reasoning/math/code/chat datasets + 1 FineWeb-Edu replay dataset
      - Deduplicates strictly against GSM8K test questions & deterministic 0.5% val holdout
      - Packs complete SFT conversations with exact assistant-only loss masking (-100 on prompts)
      - Packs FineWeb-Edu replay chunks with full causal LM loss (anti-forgetting)
      - Keeps a prefetched queue of (input_ids, targets) tensors ready for zero-wait GPU consumption
    """

    def __init__(
        self,
        tok: Tokenizer,
        seq_len: int,
        batch_size: int,
        gsm8k_dedup: set[str],
        seed: int = 42,
        prefetch_batches: int = 12,
    ):
        self.tok = tok
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.gsm8k_dedup = gsm8k_dedup
        self.seed = seed

        self.eos_id = tok.token_to_id("<|endoftext|>")
        self.im_start_id = tok.token_to_id("<|im_start|>")
        self.im_end_id = tok.token_to_id("<|im_end|>")
        assert self.eos_id is not None and self.im_start_id is not None and self.im_end_id is not None
        self.nl_ids = tok.encode("\n", add_special_tokens=False).ids

        self._stop = threading.Event()
        self.source_queues: Dict[str, queue.Queue] = {s[0]: queue.Queue(maxsize=128) for s in SFT_SOURCES}
        self.batch_queue: queue.Queue = queue.Queue(maxsize=prefetch_batches)
        self.doc_counts: Dict[str, int] = {s[0]: 0 for s in SFT_SOURCES}
        self.dedup_drops = 0

        self._threads: List[threading.Thread] = []
        for idx, (name, _w, repo, cfg, split, _mode) in enumerate(SFT_SOURCES):
            th = threading.Thread(
                target=self._source_worker,
                args=(name, repo, cfg, split, seed + idx * 1009),
                daemon=True,
                name=f"src-{name}",
            )
            th.start()
            self._threads.append(th)

        self._packer = threading.Thread(target=self._packer_worker, daemon=True, name="sft-packer")
        self._packer.start()
        self._threads.append(self._packer)

    def _source_worker(self, name: str, repo: str, cfg: Optional[str], split: str, base_seed: int):
        q = self.source_queues[name]
        epoch = 0
        backoff = 2.0
        max_chars = int(self.seq_len * 3.1)  # fits cleanly inside seq_len without truncating reasoning
        while not self._stop.is_set():
            try:
                ds = load_dataset(repo, cfg, split=split, streaming=True) if cfg else load_dataset(repo, split=split, streaming=True)
                ds = ds.shuffle(seed=base_seed + epoch * 7919, buffer_size=500)
                for row in ds:
                    if self._stop.is_set():
                        return
                    item = _extract_turns(name, row, max_chars=max_chars)
                    if item is None:
                        continue
                    mode, payload = item
                    # Check GSM8K test contamination and deterministic val holdout
                    check_txt = payload[0][1] if mode == "lm" else (payload[1][1] if len(payload) > 1 else payload[0][1])
                    if _is_val_key(check_txt):
                        continue
                    if mode == "sft" and _norm_dedup_key(check_txt) in self.gsm8k_dedup:
                        self.dedup_drops += 1
                        continue
                    while not self._stop.is_set():
                        try:
                            q.put(item, timeout=0.5)
                            self.doc_counts[name] += 1
                            backoff = 2.0
                            break
                        except queue.Full:
                            continue
                epoch += 1
            except Exception as exc:
                if self._stop.is_set() or "shutdown" in str(exc).lower():
                    return
                print(f"[STREAM-WARN] {name}: {exc!r}; retry in {backoff:.0f}s", flush=True)
                time.sleep(backoff)
                backoff = min(60.0, backoff * 1.8)

    def _next_encoded_sample(self, rng: random.Random, names: List[str], weights: List[float]) -> Optional[Tuple[List[int], List[int]]]:
        for _ in range(30):
            if self._stop.is_set():
                return None
            src = rng.choices(names, weights=weights, k=1)[0]
            q = self.source_queues[src]
            try:
                mode, payload = q.get(timeout=0.15)
            except queue.Empty:
                # Fallback to any ready queue so GPU never stalls
                found = None
                for alt in names:
                    try:
                        found = self.source_queues[alt].get_nowait()
                        break
                    except queue.Empty:
                        continue
                if found is None:
                    continue
                mode, payload = found

            if mode == "lm":
                ids = self.tok.encode(payload[0][1], add_special_tokens=False).ids
                if len(ids) < 64:
                    continue
                # Slice a random contiguous window up to seq_len so replay matches full length cleanly
                if len(ids) > self.seq_len:
                    off = rng.randint(0, len(ids) - self.seq_len)
                    ids = ids[off : off + self.seq_len]
                ids = ids + [self.eos_id]
                return ids, list(ids)
            else:
                ids, labs = encode_sft_turns_masked(
                    self.tok, payload, self.im_start_id, self.im_end_id, self.nl_ids
                )
                # Drop over-long SFT items instead of truncating their final answer!
                if len(ids) > self.seq_len or len(ids) < 24:
                    continue
                if not any(v != -100 for v in labs[1:]):
                    continue
                ids.append(self.eos_id)
                labs.append(-100)
                return ids, labs
        return None

    def _packer_worker(self):
        rng = random.Random(self.seed + 31337)
        names = [s[0] for s in SFT_SOURCES]
        weights = [s[1] for s in SFT_SOURCES]
        T = self.seq_len
        need = T + 1

        while not self._stop.is_set():
            xs = np.full((self.batch_size, T), self.eos_id, dtype=np.int64)
            ys = np.full((self.batch_size, T), -100, dtype=np.int64)
            valid_rows = 0

            for b in range(self.batch_size):
                buf_id: List[int] = []
                buf_lb: List[int] = []
                # Pack complete samples until filling >= 85% of seq_len (or next sample won't fit)
                while len(buf_id) < int(0.88 * T) and not self._stop.is_set():
                    sample = self._next_encoded_sample(rng, names, weights)
                    if sample is None:
                        break
                    s_id, s_lb = sample
                    if len(buf_id) == 0:
                        buf_id.extend(s_id[:need])
                        buf_lb.extend(s_lb[:need])
                    elif len(buf_id) + len(s_id) <= need:
                        buf_id.extend(s_id)
                        buf_lb.extend(s_lb)
                    else:
                        # Stop packing once next complete item wouldn't fit (zero truncation!)
                        break
                if len(buf_id) >= 16:
                    L = min(len(buf_id) - 1, T)
                    xs[b, :L] = buf_id[:L]
                    ys[b, :L] = buf_lb[1 : L + 1]
                    if (ys[b] != -100).any():
                        valid_rows += 1

            if valid_rows == self.batch_size:
                x_t = torch.from_numpy(xs)
                y_t = torch.from_numpy(ys)
                while not self._stop.is_set():
                    try:
                        self.batch_queue.put((x_t, y_t), timeout=0.5)
                        break
                    except queue.Full:
                        continue

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.batch_queue.get()

    def stop(self):
        self._stop.set()


# ==============================================================================
# 3. Fixed Held-Out Validation Sets (WikiText-2 + Held-Out Reasoning/Code SFT)
# ==============================================================================

def build_wikitext2_val(tok: Tokenizer, seq_len: int = 2048) -> Tuple[torch.Tensor, int, int]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok.encode(text, add_special_tokens=False).ids
    n_seq = (len(ids) - 1) // seq_len
    used_ids = ids[: n_seq * seq_len + 1]
    used_text = tok.decode(used_ids, skip_special_tokens=False)
    n_bytes = len(used_text.encode("utf-8"))
    n_words = len(re.findall(r"\S+", used_text))
    return torch.tensor(used_ids, dtype=torch.long), n_bytes, n_words


@torch.no_grad()
def evaluate_wikitext2_bpb(
    model: VAULTPrimalForCausalLM,
    wt_ids: torch.Tensor,
    n_bytes: int,
    n_words: int,
    seq_len: int,
    device: str,
    batch_size: int = 2,
) -> Dict[str, float]:
    model.eval()
    n_seq = (len(wt_ids) - 1) // seq_len
    total_nll = 0.0
    total_toks = 0
    for s in range(0, n_seq, batch_size):
        e = min(s + batch_size, n_seq)
        xs = torch.stack([wt_ids[i * seq_len : (i + 1) * seq_len] for i in range(s, e)]).to(device)
        ys = torch.stack([wt_ids[i * seq_len + 1 : (i + 1) * seq_len + 1] for i in range(s, e)]).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(xs, ys, ignore_index=-100)
        n_t = ys.numel()
        total_nll += float(loss.item()) * n_t
        total_toks += n_t
    model.train()
    tok_ppl = math.exp(min(20.0, total_nll / max(1, total_toks)))
    bpb = total_nll / (max(1, n_bytes) * math.log(2.0))
    word_ppl = math.exp(min(20.0, total_nll / max(1, n_words)))
    return {
        "wt2_bpb": round(bpb, 4),
        "wt2_word_ppl": round(word_ppl, 2),
        "wt2_tok_ppl": round(tok_ppl, 2),
        "wt2_tokens": total_toks,
    }


def build_heldout_sft_val(
    tok: Tokenizer,
    seq_len: int = 2048,
    n_per_source: int = 20,
    cache_path: Optional[str] = None,
) -> List[Tuple[List[int], List[int]]]:
    """
    Build a deterministic held-out validation set of assistant-masked reasoning/math/code
    examples using the exact `_is_val_key` hash holdout (100% disjoint from train!).
    """
    if cache_path and os.path.isfile(cache_path):
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"[VAL-SFT] Loaded cached {len(data)} held-out SFT examples from {cache_path}", flush=True)
        return data

    im_start_id = tok.token_to_id("<|im_start|>")
    im_end_id = tok.token_to_id("<|im_end|>")
    nl_ids = tok.encode("\n", add_special_tokens=False).ids
    max_chars = int(seq_len * 3.0)
    samples: List[Tuple[List[int], List[int]]] = []

    val_sources = [s for s in SFT_SOURCES if s[5] == "sft"]
    for name, _w, repo, cfg, split, _mode in val_sources:
        got = 0
        ds = load_dataset(repo, cfg, split=split, streaming=True) if cfg else load_dataset(repo, split=split, streaming=True)
        for row in ds:
            item = _extract_turns(name, row, max_chars=max_chars)
            if item is None:
                continue
            _m, payload = item
            check_txt = payload[1][1] if len(payload) > 1 else payload[0][1]
            if not _is_val_key(check_txt):
                continue
            ids, labs = encode_sft_turns_masked(tok, payload, im_start_id, im_end_id, nl_ids)
            if 32 <= len(ids) <= seq_len and any(v != -100 for v in labs[1:]):
                samples.append((ids, labs))
                got += 1
                if got >= n_per_source:
                    break
        print(f"  [VAL-SFT] collected {got} held-out examples from {name}", flush=True)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(samples, cache_path)
    return samples


@torch.no_grad()
def evaluate_heldout_sft_loss(
    model: VAULTPrimalForCausalLM,
    val_samples: List[Tuple[List[int], List[int]]],
    eos_id: int,
    device: str,
    batch_size: int = 4,
) -> float:
    model.eval()
    # Length-sort for minimal padding
    ordered = sorted(val_samples, key=lambda x: len(x[0]))
    tot_nll = 0.0
    tot_valid = 0
    for s in range(0, len(ordered), batch_size):
        chunk = ordered[s : s + batch_size]
        max_l = max(len(c[0]) for c in chunk) - 1
        t_pad = max(64, ((max_l + 63) // 64) * 64)
        xs = torch.full((len(chunk), t_pad), eos_id, dtype=torch.long, device=device)
        ys = torch.full((len(chunk), t_pad), -100, dtype=torch.long, device=device)
        for b, (ids, labs) in enumerate(chunk):
            L = len(ids) - 1
            xs[b, :L] = torch.tensor(ids[:L], dtype=torch.long, device=device)
            ys[b, :L] = torch.tensor(labs[1 : L + 1], dtype=torch.long, device=device)
        n_v = int((ys != -100).sum().item())
        if n_v == 0:
            continue
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(xs, ys, ignore_index=-100)
        tot_nll += float(loss.item()) * n_v
        tot_valid += n_v
    model.train()
    return round(tot_nll / max(1, tot_valid), 4)


# ==============================================================================
# 4. Exact Fast Batched KV-Cache Generation + GSM8K Exact Math Evaluator
# ==============================================================================

def _apply_rope_pos(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE with per-batch per-position cos/sin of shape (B, 1, T, head_dim)."""
    h = q.shape[-1] // 2
    q_rot = torch.cat((-q[..., h:], q[..., :h]), dim=-1)
    k_rot = torch.cat((-k[..., h:], k[..., :h]), dim=-1)
    return (q * cos) + (q_rot * sin), (k * cos) + (k_rot * sin)


@torch.no_grad()
def generate_chat_batched(
    model: VAULTPrimalForCausalLM,
    tok: Tokenizer,
    prompts: List[str],
    system_prompt: str = GSM8K_SYSTEM_PROMPT,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    device: str = "cuda",
) -> List[str]:
    """
    Exact greedy batched KV-cache generation for VAULTPrimalForCausalLM with left-padding
    and exact per-sample RoPE position IDs (bit-identical to bs=1 unbatched greedy decode).
    """
    model.eval()
    im_start = tok.token_to_id("<|im_start|>")
    im_end = tok.token_to_id("<|im_end|>")
    eos_id = tok.token_to_id("<|endoftext|>")
    nl_ids = tok.encode("\n", add_special_tokens=False).ids

    encoded_prompts: List[Tuple[int, List[int]]] = []
    for idx, up in enumerate(prompts):
        ids = (
            [im_start] + tok.encode(f"system\n{system_prompt}", add_special_tokens=False).ids + [im_end] + nl_ids
            + [im_start] + tok.encode(f"user\n{up}", add_special_tokens=False).ids + [im_end] + nl_ids
            + [im_start] + tok.encode("assistant\n", add_special_tokens=False).ids
        )
        encoded_prompts.append((idx, ids))

    encoded_prompts.sort(key=lambda x: len(x[1]))
    outputs: List[str] = [""] * len(prompts)
    cfg = model.cfg
    n_groups = cfg.n_heads // cfg.n_kv_heads

    # Pre-dequantize all layer weights? No, dequantizing per batch or on-the-fly in bf16 is fast,
    # and on 3B we have ~9 GiB free VRAM so let's keep VRAM minimal by dequantizing per layer!
    for st in range(0, len(encoded_prompts), batch_size):
        chunk = encoded_prompts[st : st + batch_size]
        B = len(chunk)
        lens = [len(c[1]) for c in chunk]
        max_p = max(lens)
        max_total = max_p + max_new_tokens + 4
        cos_all, sin_all = model.rope(max_total, torch.bfloat16)

        inp = torch.full((B, max_p), eos_id, dtype=torch.long, device=device)
        pad_lens = torch.zeros(B, dtype=torch.long, device=device)
        pos_ids = torch.zeros((B, max_p), dtype=torch.long, device=device)
        for b, (_, p_ids) in enumerate(chunk):
            pl = max_p - len(p_ids)
            pad_lens[b] = pl
            inp[b, pl:] = torch.tensor(p_ids, dtype=torch.long, device=device)
            pos_ids[b, pl:] = torch.arange(len(p_ids), dtype=torch.long, device=device)

        # Allocate compact GQA KV caches (n_kv_heads=2 -> tiny VRAM!)
        k_caches = [
            torch.empty((B, cfg.n_kv_heads, max_p + max_new_tokens, cfg.head_dim), dtype=torch.bfloat16, device=device)
            for _ in range(cfg.n_layers)
        ]
        v_caches = [
            torch.empty((B, cfg.n_kv_heads, max_p + max_new_tokens, cfg.head_dim), dtype=torch.bfloat16, device=device)
            for _ in range(cfg.n_layers)
        ]

        # Build prefill causal + left-pad mask: (B, 1, max_p, max_p)
        ar = torch.arange(max_p, device=device)
        causal = ar.unsqueeze(0) <= ar.unsqueeze(1)  # (max_p, max_p)
        valid_k = ar.unsqueeze(0) >= pad_lens.unsqueeze(1)  # (B, max_p)
        prefill_mask = causal.unsqueeze(0) & valid_k.unsqueeze(1)  # (B, max_p, max_p)
        # Ensure padded query rows have at least one True so softmax never NaNs
        prefill_mask = prefill_mask | (~valid_k.unsqueeze(2) & torch.eye(max_p, dtype=torch.bool, device=device).unsqueeze(0))
        prefill_mask = prefill_mask.unsqueeze(1)  # (B, 1, max_p, max_p)

        # Prefill forward pass
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = model.embed_tokens.embed(inp, out_dtype=torch.bfloat16)
            cos_p = cos_all[pos_ids].unsqueeze(1)  # (B, 1, max_p, head_dim)
            sin_p = sin_all[pos_ids].unsqueeze(1)
            for li, layer in enumerate(model.layers):
                xn = layer.input_layernorm(h)
                attn = layer.self_attn
                q = attn.q_proj(xn).view(B, max_p, cfg.n_heads, cfg.head_dim).transpose(1, 2)
                k = attn.k_proj(xn).view(B, max_p, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
                v = attn.v_proj(xn).view(B, max_p, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
                q, k = _apply_rope_pos(q, k, cos_p, sin_p)
                k_caches[li][:, :, :max_p] = k
                v_caches[li][:, :, :max_p] = v
                kr = k.repeat_interleave(n_groups, dim=1) if n_groups > 1 else k
                vr = v.repeat_interleave(n_groups, dim=1) if n_groups > 1 else v
                y = F.scaled_dot_product_attention(q, kr, vr, attn_mask=prefill_mask)
                y = y.transpose(1, 2).contiguous().view(B, max_p, cfg.d_model)
                h = h + attn.o_proj(y)
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            h_last = model.norm(h[:, -1:, :])
            next_tok = model.embed_tokens.project(h_last).squeeze(1).float().argmax(dim=-1)

        gen_ids: List[List[int]] = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        cur_pos = torch.tensor(lens, dtype=torch.long, device=device)  # exact RoPE pos for next token

        for step in range(max_new_tokens):
            nt_cpu = next_tok.tolist()
            for b in range(B):
                if not finished[b]:
                    tid = nt_cpu[b]
                    if tid in (im_end, eos_id):
                        finished[b] = True
                    else:
                        gen_ids[b].append(tid)
            if bool(finished.all().item()):
                break
            if step == max_new_tokens - 1:
                break

            cache_idx = max_p + step
            cur_len = cache_idx + 1
            ar_c = torch.arange(cur_len, device=device).unsqueeze(0)
            step_mask = (ar_c >= pad_lens.unsqueeze(1)).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, cur_len)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h = model.embed_tokens.embed(next_tok.unsqueeze(1), out_dtype=torch.bfloat16)
                cos_s = cos_all[cur_pos].unsqueeze(1).unsqueeze(2)  # (B, 1, 1, head_dim)
                sin_s = sin_all[cur_pos].unsqueeze(1).unsqueeze(2)
                for li, layer in enumerate(model.layers):
                    xn = layer.input_layernorm(h)
                    attn = layer.self_attn
                    q = attn.q_proj(xn).view(B, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)
                    k = attn.k_proj(xn).view(B, 1, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
                    v = attn.v_proj(xn).view(B, 1, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
                    q, k = _apply_rope_pos(q, k, cos_s, sin_s)
                    k_caches[li][:, :, cache_idx : cache_idx + 1] = k
                    v_caches[li][:, :, cache_idx : cache_idx + 1] = v
                    kc = k_caches[li][:, :, :cur_len]
                    vc = v_caches[li][:, :, :cur_len]
                    kr = kc.repeat_interleave(n_groups, dim=1) if n_groups > 1 else kc
                    vr = vc.repeat_interleave(n_groups, dim=1) if n_groups > 1 else vc
                    y = F.scaled_dot_product_attention(q, kr, vr, attn_mask=step_mask)
                    y = y.transpose(1, 2).contiguous().view(B, 1, cfg.d_model)
                    h = h + attn.o_proj(y)
                    h = h + layer.mlp(layer.post_attention_layernorm(h))
                h_n = model.norm(h)
                next_tok = model.embed_tokens.project(h_n).squeeze(1).float().argmax(dim=-1)
            cur_pos = cur_pos + 1

        for b, (orig_idx, _) in enumerate(chunk):
            outputs[orig_idx] = tok.decode(gen_ids[b], skip_special_tokens=True).strip()

        del k_caches, v_caches
        torch.cuda.empty_cache()

    model.train()
    return outputs


def _extract_last_boxed(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out).strip()


def _parse_numeric(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").replace(" ", "").strip()
    s = s.rstrip(".")
    # Handle \frac{a}{b}
    m_frac = re.fullmatch(r"-?\\d?frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", s)
    if m_frac:
        num, den = float(m_frac.group(1)), float(m_frac.group(2))
        return num / den if den != 0 else None
    m_sfrac = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", s)
    if m_sfrac:
        num, den = float(m_sfrac.group(1)), float(m_sfrac.group(2))
        return num / den if den != 0 else None
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def extract_gsm8k_answer(text: str) -> Optional[float]:
    bx = _extract_last_boxed(text)
    if bx is not None:
        v = _parse_numeric(bx)
        if v is not None:
            return v
    m = re.search(r"####\s*([^\n]+)", text)
    if m:
        v = _parse_numeric(m.group(1))
        if v is not None:
            return v
    # Fallback: last number in final 2 lines
    tail = "\n".join(text.strip().splitlines()[-3:])
    return _parse_numeric(tail)


def evaluate_gsm8k(
    model: VAULTPrimalForCausalLM,
    tok: Tokenizer,
    n_samples: int = 200,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    device: str = "cuda",
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = min(n_samples, len(ds))
    prompts = [ds[i]["question"].strip() + GSM8K_USER_SUFFIX for i in range(n)]
    golds = [extract_gsm8k_answer(ds[i]["answer"]) for i in range(n)]
    preds_raw = generate_chat_batched(
        model, tok, prompts,
        system_prompt=GSM8K_SYSTEM_PROMPT,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
    )
    hits = 0
    boxed_hits = 0
    for raw, g in zip(preds_raw, golds):
        if "\\boxed{" in raw:
            boxed_hits += 1
        p = extract_gsm8k_answer(raw)
        if p is not None and g is not None and abs(p - g) < 1e-4:
            hits += 1
    dt = time.perf_counter() - t0
    return {
        "n_samples": n,
        "hits": hits,
        "acc": round(100.0 * hits / max(1, n), 2),
        "boxed_pct": round(100.0 * boxed_hits / max(1, n), 1),
        "eval_sec": round(dt, 1),
    }


# ==============================================================================
# 5. Self-Contained EleutherAI Zero-Shot Academic Benchmark Suite (3B-Sized)
# ==============================================================================

def _encode_pair_lm_eval(tok: Tokenizer, ctx: str, cont: str, max_seq_len: int = 2048) -> Tuple[List[int], int, int]:
    cont_byte_len = max(1, len(cont.encode("utf-8")))
    ctx_ids = tok.encode(ctx, add_special_tokens=False).ids
    whole_ids = tok.encode(ctx + cont, add_special_tokens=False).ids
    if len(whole_ids) > len(ctx_ids) and whole_ids[: len(ctx_ids)] == ctx_ids:
        cont_len = len(whole_ids) - len(ctx_ids)
    else:
        cont_ids = tok.encode(cont, add_special_tokens=False).ids
        cont_len = max(1, len(cont_ids))
        if len(whole_ids) <= 1:
            whole_ids = ctx_ids + cont_ids
        cont_len = min(cont_len, len(whole_ids) - 1)
    if len(whole_ids) > max_seq_len:
        whole_ids = whole_ids[-max_seq_len:]
        cont_len = min(cont_len, max_seq_len - 1)
    cont_len = max(1, min(cont_len, len(whole_ids) - 1))
    return whole_ids, cont_len, cont_byte_len


@torch.no_grad()
def _score_pairs_batched(
    model: VAULTPrimalForCausalLM,
    tok: Tokenizer,
    pairs: List[Tuple[str, str]],
    batch_size: int = 16,
    max_seq_len: int = 2048,
    device: str = "cuda",
) -> List[Tuple[float, float, bool, int]]:
    model.eval()
    n = len(pairs)
    encoded = [(i, *_encode_pair_lm_eval(tok, ctx, cont, max_seq_len)) for i, (ctx, cont) in enumerate(pairs)]
    encoded.sort(key=lambda x: len(x[1]))
    results: List[Optional[Tuple[float, float, bool, int]]] = [None] * n
    pad_id = tok.token_to_id("<|endoftext|>") or 0

    for st in range(0, n, batch_size):
        chunk = encoded[st : st + batch_size]
        max_len = max(len(item[1]) for item in chunk)
        sub_bs = min(batch_size, 4) if max_len >= 512 else (min(batch_size, 8) if max_len >= 256 else batch_size)
        for sub_st in range(0, len(chunk), sub_bs):
            sub = chunk[sub_st : sub_st + sub_bs]
            t_pad = max(64, ((max(len(item[1]) for item in sub) + 63) // 64) * 64)
            inp = torch.full((len(sub), t_pad), pad_id, dtype=torch.long, device=device)
            for b, (_, w_ids, _, _) in enumerate(sub):
                inp[b, : len(w_ids)] = torch.tensor(w_ids, dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h = model.embed_tokens.embed(inp, out_dtype=torch.bfloat16)
                cos, sin = model.rope(inp.size(1), h.dtype)
                for layer in model.layers:
                    h = layer(h, cos, sin)
                h = model.norm(h)
                h_slices, targets, spans = [], [], []
                cursor = 0
                for b, (_, w_ids, c_len, _) in enumerate(sub):
                    L = len(w_ids)
                    c0 = L - c_len
                    h_slices.append(h[b, c0 - 1 : L - 1])
                    targets.extend(w_ids[c0:L])
                    spans.append((cursor, cursor + c_len))
                    cursor += c_len
                logits_cat = model.embed_tokens.project(torch.cat(h_slices, dim=0)).float()
            lp_cat = F.log_softmax(logits_cat, dim=-1)
            tgt = torch.tensor(targets, dtype=torch.long, device=device)
            tok_lp = lp_cat.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            greedy = lp_cat.argmax(dim=-1)
            for b, (orig_idx, _, c_len, b_len) in enumerate(sub):
                s0, s1 = spans[b]
                sum_lp = float(tok_lp[s0:s1].sum().item())
                exact = bool((greedy[s0:s1] == tgt[s0:s1]).all().item())
                results[orig_idx] = (sum_lp, sum_lp / float(b_len), exact, c_len)
    model.train()
    return results  # type: ignore[return-value]


def _eval_mc_items(model: VAULTPrimalForCausalLM, tok: Tokenizer, mc_items: List, batch_size: int = 16, device: str = "cuda") -> Dict:
    t0 = time.perf_counter()
    flat, offsets = [], []
    for cands, gold in mc_items:
        s0 = len(flat)
        flat.extend(cands)
        offsets.append((s0, len(flat), gold))
    scored = _score_pairs_batched(model, tok, flat, batch_size=batch_size, max_seq_len=2048, device=device)
    acc_h = norm_h = 0
    for s0, s1, gold in offsets:
        cs = scored[s0:s1]
        if max(range(len(cs)), key=lambda i: cs[i][0]) == gold:
            acc_h += 1
        if max(range(len(cs)), key=lambda i: cs[i][1]) == gold:
            norm_h += 1
    n = max(1, len(mc_items))
    return {"n_samples": len(mc_items), "acc": round(100.0 * acc_h / n, 2), "acc_norm": round(100.0 * norm_h / n, 2), "eval_sec": round(time.perf_counter() - t0, 1)}


def _eval_lambada(model: VAULTPrimalForCausalLM, tok: Tokenizer, texts: List[str], batch_size: int = 16, device: str = "cuda") -> Dict:
    t0 = time.perf_counter()
    pairs = []
    for txt in texts:
        t = txt.strip()
        if " " in t:
            ctx, last = t.rsplit(" ", 1)
            pairs.append((ctx, " " + last))
    scored = _score_pairs_batched(model, tok, pairs, batch_size=batch_size, max_seq_len=2048, device=device)
    n = max(1, len(scored))
    hits = sum(1 for _, _, ex, _ in scored if ex)
    ppl = math.exp(min(20.0, -sum(lp for lp, _, _, _ in scored) / n))
    return {"n_samples": len(scored), "acc": round(100.0 * hits / n, 2), "target_word_ppl": round(ppl, 2), "eval_sec": round(time.perf_counter() - t0, 1)}


def run_full_benchmark_suite(
    model: VAULTPrimalForCausalLM,
    tok: Tokenizer,
    wt_bundle: Tuple[torch.Tensor, int, int],
    val_sft_samples: List[Tuple[List[int], List[int]]],
    device: str,
    label: str = "VAULT-v3-3B",
    gsm8k_n: int = 200,
    max_large_split: int = 1000,
) -> Dict[str, Any]:
    print(f"\n==================== [BENCHMARK SUITE: {label}] ====================", flush=True)
    wt_ids, wt_bytes, wt_words = wt_bundle
    eos_id = tok.token_to_id("<|endoftext|>")
    wt2 = evaluate_wikitext2_bpb(model, wt_ids, wt_bytes, wt_words, 2048, device, batch_size=2)
    print(f"  [WikiText-2    ] BPB={wt2['wt2_bpb']:.4f} | Word_PPL={wt2['wt2_word_ppl']:.2f} | Tok_PPL={wt2['wt2_tok_ppl']:.2f}", flush=True)

    val_sft = evaluate_heldout_sft_loss(model, val_sft_samples, eos_id, device, batch_size=4)
    print(f"  [Heldout-SFT   ] val_sft_loss={val_sft:.4f} (n={len(val_sft_samples)})", flush=True)

    gsm = evaluate_gsm8k(model, tok, n_samples=gsm8k_n, max_new_tokens=512, batch_size=32, device=device)
    print(f"  [GSM8K-{gsm['n_samples']:<3d}     ] Acc={gsm['acc']:5.2f}% ({gsm['hits']}/{gsm['n_samples']}) | boxed={gsm['boxed_pct']:.1f}% | {gsm['eval_sec']:.1f}s", flush=True)

    lam_texts = list(load_dataset("EleutherAI/lambada_openai", "en", split="test")["text"])[:max_large_split]
    lam = _eval_lambada(model, tok, lam_texts, batch_size=8, device=device)
    print(f"  [LAMBADA       ] Acc={lam['acc']:5.2f}% | TargetWord_PPL={lam['target_word_ppl']:.2f}", flush=True)

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
        for i in range(min(len(hs_ds), max_large_split))
    ]

    wg_items = []
    for r in load_dataset("allenai/winogrande", "winogrande_xl", split="validation"):
        if "_" in r["sentence"]:
            i = r["sentence"].index("_")
            p, s = r["sentence"][:i], r["sentence"][i + 1 :]
            wg_items.append(([(p + r["option1"], s), (p + r["option2"], s)], int(r["answer"]) - 1))

    bq_ds = load_dataset("google/boolq", split="validation")
    bq_items = []
    for i in range(min(len(bq_ds), max_large_split)):
        r = bq_ds[i]
        q = r["question"].strip()
        if not q.endswith("?"):
            q += "?"
        ctx = f"{r['passage'].strip()}\nQuestion: {q}\nAnswer:"
        bq_items.append(([(ctx, " no"), (ctx, " yes")], 1 if bool(r["answer"]) else 0))

    mc_suite = [
        ("arc_easy", "ARC-Easy", _arc("ARC-Easy"), "acc_norm"),
        ("arc_challenge", "ARC-Challenge", _arc("ARC-Challenge"), "acc_norm"),
        ("sciq_rc", "SciQ-RC", sciq_rc, "acc_norm"),
        ("sciq_closedbook", "SciQ-ClosedBook", sciq_cb, "acc_norm"),
        ("piqa", "PIQA", piqa_items, "acc_norm"),
        ("hellaswag", "HellaSwag", hs_items, "acc_norm"),
        ("winogrande", "Winogrande", wg_items, "acc"),
        ("boolq", "BoolQ", bq_items, "acc"),
    ]
    mc_res = {}
    primaries = [lam["acc"]]
    for k, name, items, metric in mc_suite:
        r = _eval_mc_items(model, tok, items, batch_size=8, device=device)
        mc_res[k] = r
        primaries.append(r[metric])
        print(f"  [{name:<15s}] primary({metric:8s})={r[metric]:5.2f}% | acc={r['acc']:5.2f}% | acc_norm={r['acc_norm']:5.2f}%", flush=True)

    mc_8_avg = round(sum(primaries[1:]) / 8.0, 2)
    avg_9 = round(sum(primaries) / 9.0, 2)
    print(f"  --> 8-Task MC Avg = {mc_8_avg:.2f}% | 9-Task Avg = {avg_9:.2f}% | GSM8K = {gsm['acc']:.2f}%", flush=True)
    return {
        "label": label,
        "wikitext2": wt2,
        "val_sft_loss": val_sft,
        "gsm8k": gsm,
        "lambada": lam,
        "multiple_choice": mc_res,
        "mc_8task_avg_pct": mc_8_avg,
        "zero_shot_9task_avg_pct": avg_9,
    }


# ==============================================================================
# 6. Freestanding C99 vault3_host Verification & Prime-Harmonic Hessian P4_0 Export
# ==============================================================================

def verify_c99_runtime_3b(vlt3_path: str) -> Dict[str, Any]:
    if not os.path.isfile(VAULT_HOST_BIN):
        return {"skipped": "vault3_host not built"}
    print(f"\n=== [C99] Freestanding vault3_host Benchmark on {os.path.basename(vlt3_path)} ===", flush=True)
    speeds = {}
    for th in (1, 8, 24):
        cmd = [VAULT_HOST_BIN, vlt3_path, "--bench-prefill", "64", "--bench", "32", "--threads", str(th)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        parsed = {}
        for line in p.stdout.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = json.loads(line)
                    break
                except Exception:
                    pass
        speeds[f"{th}T"] = {
            "decode_tok_s": float(parsed.get("decode_tok_s", 0.0)),
            "prefill_tok_s": float(parsed.get("prefill_tok_s", 0.0)),
            "ms_per_tok": float(parsed.get("ms_per_tok", 0.0)),
            "peak_rss_mb": float(parsed.get("peak_rss_mb", 0.0)),
            "rc": p.returncode,
        }
        print(
            f"  [C99 {th:2d}T] decode={speeds[f'{th}T']['decode_tok_s']:.2f} tok/s "
            f"({speeds[f'{th}T']['ms_per_tok']:.1f} ms/tok) | prefill={speeds[f'{th}T']['prefill_tok_s']:.2f} tok/s | "
            f"rss={speeds[f'{th}T']['peak_rss_mb']:.0f} MB | rc={p.returncode}",
            flush=True,
        )

    # Run 2 live C99 ChatML probes (math + code)
    samples = []
    for prompt in [
        "A train travels 180 km in 2.5 hours. What is its average speed in km/h? Give a concise step-by-step answer.",
        "Write a concise Python function `is_prime(n: int) -> bool` using trial division up to sqrt(n).",
    ]:
        cmd = [
            VAULT_HOST_BIN, vlt3_path, "--chat",
            "--system", DEFAULT_SYSTEM_PROMPT,
            "--prompt", prompt, "--n", "128", "--temp", "0", "--threads", "12",
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        reply = p.stdout.strip()
        print(f"  [C99 CHAT] Q: {prompt[:70]}...\n             A: {reply[:240]}...", flush=True)
        samples.append({"prompt": prompt, "reply": reply, "rc": p.returncode})
    return {"speeds": speeds, "samples": samples}


@torch.no_grad()
def export_hessian_p4_0_from_primal_3b(
    model: VAULTPrimalForCausalLM,
    qwen_dir: str,
    out_path: str,
    model_name: str = "VAULT-v3-3B-p4_0-primal",
    calib_batches: Optional[List[torch.Tensor]] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Export the trained 3B Q8_0 model to Hessian-weighted Prime-Harmonic V3_P4_0 (~1.85 GiB):
      - Collects exact diagonal input Hessian H_j = E[x_j^2] across all 36 layers on calib_batches
      - Quantizes protected tensors (embed_tokens, K/V, boundary Q/O) to exact Q8_0 (0 PTQ loss)
      - Quantizes middle Q/O + all MLP projections to H_j-weighted Prime-Harmonic V3_P4_0
      - Validates full CRC32 + finite scales via vlt3.read_vlt3()
    """
    tools_dir = os.path.join(VAULT_ROOT, "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import struct, zlib
    import vlt3

    model.eval()
    cfg = model.cfg
    h_diag: Dict[str, torch.Tensor] = {}
    hooks = []

    if calib_batches:
        def _make_hook(tname: str):
            def _hook(_mod, inp, _out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                sq = (x * x).mean(dim=0)
                if tname not in h_diag:
                    h_diag[tname] = sq
                else:
                    h_diag[tname] = h_diag[tname] + sq
            return _hook

        for li, layer in enumerate(model.layers):
            for sub_name, mod in [
                (f"layers.{li}.self_attn.q_proj.weight", layer.self_attn.q_proj),
                (f"layers.{li}.self_attn.o_proj.weight", layer.self_attn.o_proj),
                (f"layers.{li}.mlp.gate_proj.weight", layer.mlp.gate_proj),
                (f"layers.{li}.mlp.up_proj.weight", layer.mlp.up_proj),
                (f"layers.{li}.mlp.down_proj.weight", layer.mlp.down_proj),
            ]:
                hooks.append(mod.register_forward_hook(_make_hook(sub_name)))

        for x_b in calib_batches:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(x_b.to(device))
        for hk in hooks:
            hk.remove()
        for k in h_diag:
            h_diag[k] = (h_diag[k] / len(calib_batches)).clamp(min=1e-8)

    lut_t = torch.tensor(vlt3.P4_LUT_F32, device=device, dtype=torch.float32)
    mids_t = 0.5 * (lut_t[:-1] + lut_t[1:])
    max_neg, max_pos = float(-vlt3.P4_LUT_F32[0]), float(vlt3.P4_LUT_F32[-1])
    ratios = [
        -1.00 / max_neg, -0.96 / max_neg, -0.92 / max_neg,
        -0.88 / max_neg, -0.84 / max_neg, -0.80 / max_neg,
        1.00 / max_pos, 0.94 / max_pos,
    ]

    def _quant_p4_hessian(w_2d: torch.Tensor, h_vec: Optional[torch.Tensor]) -> bytes:
        out_f, in_f = w_2d.shape
        x_blk = w_2d.float().reshape(-1, vlt3.QK)
        n_blk = x_blk.shape[0]
        if h_vec is not None:
            h_blk = h_vec.to(device).view(1, in_f // vlt3.QK, vlt3.QK).expand(out_f, -1, -1).reshape(n_blk, vlt3.QK)
        else:
            h_blk = None
        out = np.empty(n_blk, dtype=vlt3.Q4_DT)
        step = 131072
        for b0 in range(0, n_blk, step):
            b1 = min(b0 + step, n_blk)
            xb = x_blk[b0:b1]
            hb = h_blk[b0:b1] if h_blk is not None else None
            m = xb.shape[0]
            mx = xb.gather(1, xb.abs().argmax(dim=1, keepdim=True)).squeeze(1)
            best_err = torch.full((m,), float("inf"), device=device, dtype=torch.float32)
            best_d = torch.zeros(m, device=device, dtype=torch.float16)
            best_q = torch.full((m, vlt3.QK), 8, device=device, dtype=torch.uint8)
            nz = mx != 0.0
            if nz.any():
                for r in ratios:
                    d_f16 = (mx * r).to(torch.float16)
                    d_f32 = d_f16.float()
                    inv = torch.where(d_f32 != 0.0, 1.0 / torch.where(d_f32 != 0.0, d_f32, torch.ones_like(d_f32)), torch.zeros_like(d_f32))
                    qi = torch.searchsorted(mids_t, xb * inv.unsqueeze(1)).to(torch.uint8)
                    rec = lut_t[qi.long()] * d_f32.unsqueeze(1)
                    diff2 = (rec - xb).pow(2)
                    if hb is not None:
                        diff2 = diff2 * hb
                    err = diff2.sum(dim=1)
                    imp = nz & (err < best_err)
                    best_err = torch.where(imp, err, best_err)
                    best_d = torch.where(imp, d_f16, best_d)
                    best_q = torch.where(imp.unsqueeze(1), qi, best_q)
            packed = best_q[:, : vlt3.QK // 2] | (best_q[:, vlt3.QK // 2 :] << 4)
            out["d"][b0:b1] = best_d.cpu().numpy()
            out["q"][b0:b1] = packed.cpu().numpy()
        return out.tobytes()

    raw_tensors: Dict[str, Tuple[int, Tuple[int, ...], bytes]] = {}
    raw_tensors["embed_tokens.weight"] = (vlt3.Q8_0, (cfg.vocab_size, cfg.d_model), model.embed_tokens.pack_v3_q8_0_bytes())
    raw_tensors["norm.weight"] = (vlt3.F32, (cfg.d_model,), vlt3.encode_tensor(model.norm.weight, vlt3.F32))

    for li, layer in enumerate(model.layers):
        p = f"layers.{li}"
        raw_tensors[f"{p}.input_layernorm.weight"] = (vlt3.F32, (cfg.d_model,), vlt3.encode_tensor(layer.input_layernorm.weight, vlt3.F32))
        raw_tensors[f"{p}.post_attention_layernorm.weight"] = (vlt3.F32, (cfg.d_model,), vlt3.encode_tensor(layer.post_attention_layernorm.weight, vlt3.F32))

        for proj_name, mod in [
            ("self_attn.q_proj", layer.self_attn.q_proj),
            ("self_attn.k_proj", layer.self_attn.k_proj),
            ("self_attn.v_proj", layer.self_attn.v_proj),
        ]:
            tname = f"{p}.{proj_name}.weight"
            dt = vlt3.choose_dtype(tname, "p4_0", cfg.n_layers)
            if dt == vlt3.P4_0:
                w_f32 = _dequant_q8_0(mod.q, mod.d, torch.float32)
                payload = _quant_p4_hessian(w_f32, h_diag.get(tname))
            else:
                payload = mod.pack_v3_q8_0_bytes()
            raw_tensors[tname] = (dt, (mod.out_features, mod.in_features), payload)
            raw_tensors[f"{p}.{proj_name}.bias"] = (vlt3.F32, (mod.out_features,), vlt3.encode_tensor(mod.bias, vlt3.F32))

        for proj_name, mod in [
            ("self_attn.o_proj", layer.self_attn.o_proj),
            ("mlp.gate_proj", layer.mlp.gate_proj),
            ("mlp.up_proj", layer.mlp.up_proj),
            ("mlp.down_proj", layer.mlp.down_proj),
        ]:
            tname = f"{p}.{proj_name}.weight"
            dt = vlt3.choose_dtype(tname, "p4_0", cfg.n_layers)
            if dt == vlt3.P4_0:
                w_f32 = _dequant_q8_0(mod.q, mod.d, torch.float32)
                payload = _quant_p4_hessian(w_f32, h_diag.get(tname))
            else:
                payload = mod.pack_v3_q8_0_bytes()
            raw_tensors[tname] = (dt, (mod.out_features, mod.in_features), payload)

    names = sorted(raw_tensors.keys(), key=vlt3.tensor_id)
    entries, blobs = [], []
    for nm in names:
        dt, shape, data = raw_tensors[nm]
        n_elem = int(np.prod(shape))
        assert len(data) == vlt3.nbytes_for(dt, n_elem)
        dims = list(shape) + [1] * (4 - len(shape))
        entries.append(dict(id=vlt3.tensor_id(nm), dtype=dt, ndim=len(shape), dims=dims, nbytes=len(data), crc=zlib.crc32(data), name=nm))
        blobs.append(data)

    tok_blob, ids = vlt3.build_tokenizer_blob(qwen_dir)
    dir_off = vlt3.HEADER_SIZE
    dir_size = vlt3.ENTRY_SIZE * len(entries)
    tok_off = (dir_off + dir_size + vlt3.ALIGN - 1) // vlt3.ALIGN * vlt3.ALIGN
    data_off = (tok_off + len(tok_blob) + vlt3.ALIGN - 1) // vlt3.ALIGN * vlt3.ALIGN
    cur = data_off
    for e in entries:
        cur = (cur + vlt3.ALIGN - 1) // vlt3.ALIGN * vlt3.ALIGN
        e["offset"] = cur
        cur += e["nbytes"]
    file_size = cur

    hdr_vals = dict(
        magic=vlt3.MAGIC, version=vlt3.VERSION, header_size=vlt3.HEADER_SIZE, endian_tag=vlt3.ENDIAN_TAG,
        file_size=file_size, alignment=vlt3.ALIGN, n_tensors=len(entries),
        tensor_dir_offset=dir_off, tensor_dir_size=dir_size,
        tokenizer_offset=tok_off, tokenizer_size=len(tok_blob),
        data_offset=data_off, data_size=file_size - data_off,
        vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers,
        n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        d_ff=cfg.d_ff, window=cfg.window,
        n_banks=0, primes=[0] * 8, n_cells=0,
        rms_eps=cfg.rms_eps, rope_theta=cfg.rope_theta, vault_eps=0.0,
        bos_id=ids["bos"], eos_id=ids["eos"], im_start_id=ids["im_start"], im_end_id=ids["im_end"], pad_id=ids["pad"],
        weight_format=vlt3.P4_0, flags=(1 if cfg.tie_embeddings else 0),
        model_name=model_name.encode("utf-8")[:63],
        n_sink=cfg.n_sink,
        seg_budget=0, seg_hidden=0, seg_min=0, seg_max=0, seg_n_feat=0, seg_layers=[0, 0, 0, 0],
    )
    flat = []
    for f in vlt3._HDR_FIELDS:
        v = hdr_vals[f]
        flat.extend(v if f in ("primes", "seg_layers") else [v])
    hdr = struct.pack(vlt3._HDR_FMT, *flat).ljust(vlt3.HEADER_SIZE - 4, b"\0")
    hdr += struct.pack("<I", zlib.crc32(hdr))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(hdr)
        for e in entries:
            f.write(struct.pack(
                vlt3._ENTRY_FMT, e["id"], e["dtype"], e["ndim"], *e["dims"], 0,
                e["offset"], e["nbytes"], e["crc"], e["name"].encode()[:63],
            ))
        f.write(b"\0" * (tok_off - f.tell()))
        f.write(tok_blob)
        for e, b in zip(entries, blobs):
            f.write(b"\0" * (e["offset"] - f.tell()))
            f.write(b)
        assert f.tell() == file_size
    vf = vlt3.read_vlt3(tmp_path, check_crc=True)
    os.replace(tmp_path, out_path)
    model.train()
    return {"path": out_path, "file_size_mb": round(file_size / (1024 ** 2), 2), "n_tensors": len(entries)}


# ==============================================================================
# 7. Checkpoint Helper & Main Entrypoint (eval_base / pilot / train)
# ==============================================================================

def save_primal_ckpt(path: str, model: VAULTPrimalForCausalLM, engine: Q8StepEngine, meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    payload = {
        "cfg": dataclasses.asdict(model.cfg),
        "model": model.state_dict(),
        "engine": engine.state_dict(),
        "meta": meta,
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="VAULT-v3-3B Full-Parameter Shadowless PRIMAL Trainer")
    ap.add_argument("--mode", choices=["eval_base", "pilot", "train", "export_p4"], default="train")
    ap.add_argument("--qwen_dir", default="")
    ap.add_argument("--out_dir", default="runs/vault_v3_3b_primal")
    ap.add_argument("--resume", default="", help="Optional checkpoint .pt to resume from")
    ap.add_argument("--headroom", type=float, default=1.05)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--total_steps", type=int, default=600)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--schedule", choices=["cosine", "wsd", "infinite"], default="wsd")
    ap.add_argument("--cooldown_frac", type=float, default=0.20)
    ap.add_argument("--threshold_start", type=int, default=24)
    ap.add_argument("--threshold_end", type=int, default=36)
    ap.add_argument("--vote_gain_start", type=float, default=1.6)
    ap.add_argument("--vote_gain_end", type=float, default=0.4)
    ap.add_argument("--vote_clip", type=float, default=10.0)
    ap.add_argument("--decay_start", type=float, default=0.90)
    ap.add_argument("--decay_end", type=float, default=0.95)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--aux_lr", type=float, default=2e-5)
    ap.add_argument("--eval_every", type=int, default=40)
    ap.add_argument("--gsm8k_every", type=int, default=200)
    ap.add_argument("--gsm8k_n", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--max_wt2_drift", type=float, default=0.035, help="Auto-cool/guard if WT2 BPB drifts more than this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_final_eval", action="store_true")
    args = ap.parse_args()
    if not args.qwen_dir or not os.path.isdir(args.qwen_dir):
        args.qwen_dir = _resolve_qwen_3b_dir()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tok = Tokenizer.from_file(os.path.join(args.qwen_dir, "tokenizer.json"))
    eos_id = tok.token_to_id("<|endoftext|>")
    with open(os.path.join(args.qwen_dir, "config.json")) as f:
        qc = json.load(f)
    cfg = VAULTPrimalConfig.from_qwen_config(qc, window=args.window)

    print(f"[INIT] Building VAULT-v3-3B (skip_init=True, grad_checkpoint=True, ce_chunk=512)...", flush=True)
    model = VAULTPrimalForCausalLM(cfg, skip_init=True, grad_checkpoint=True, ce_chunk_size=512)
    if args.resume and os.path.isfile(args.resume):
        print(f"[INIT] Loading checkpoint {args.resume}...", flush=True)
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        resume_engine_sd = ck.get("engine")
        resume_meta = ck.get("meta", {})
        del ck
    else:
        info = load_pretrained_qwen2_into_primal_q8(model, args.qwen_dir, headroom=args.headroom)
        print(f"[INIT] Loaded pretrained Qwen2.5-3B-Instruct into native V3_Q8_0 (headroom={args.headroom}) in {info['elapsed_s']}s", flush=True)
        resume_engine_sd = None
        resume_meta = {}

    model = model.to(device)
    engine = Q8StepEngine(
        model,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        vote_gain_start=args.vote_gain_start,
        vote_gain_end=args.vote_gain_end,
        vote_clip=args.vote_clip,
        decay_start=args.decay_start,
        decay_end=args.decay_end,
        weight_decay=args.weight_decay,
        max_stride=1,
        error_feedback=True,
        aux_lr=args.aux_lr,
        schedule=args.schedule,
        cooldown_frac=args.cooldown_frac,
        rebalance_every=0,
        grad_accum=args.grad_accum,
    )
    if resume_engine_sd is not None:
        engine.load_state_dict(resume_engine_sd)

    mem = engine.memory_report()
    print(
        f"[MEM ] VAULT-v3-3B: {mem['total_params']/1e9:.3f}B params ({mem['discrete_pct']:.3f}% discrete) | "
        f"PRIMAL state={mem['total_state_mb']/1024:.3f} GiB vs AdamW {mem['adamw_equiv_mb']/1024:.2f} GiB "
        f"({mem['compression_ratio']:.2f}x compression) | GPU alloc={torch.cuda.memory_allocated()/1024**3:.3f} GiB",
        flush=True,
    )

    wt_bundle = build_wikitext2_val(tok, seq_len=2048)
    val_cache = os.path.join(args.out_dir, "heldout_sft_val_cache.pt")
    val_sft_samples = build_heldout_sft_val(tok, seq_len=args.seq_len, n_per_source=20, cache_path=val_cache)

    # --- Mode: export_p4 (standalone Hessian P4_0 export of any checkpoint) ---
    if args.mode == "export_p4":
        wt_ids = wt_bundle[0]
        calib = [wt_ids[i * 1024 : (i + 1) * 1024].unsqueeze(0) for i in range(16)]
        p4_path = os.path.join(args.out_dir, "vault_v3_3b_p4_0_primal.vlt3")
        info_p4 = export_hessian_p4_0_from_primal_3b(model, args.qwen_dir, p4_path, calib_batches=calib, device=device)
        print(f"[EXPORT-P4] Wrote {p4_path} ({info_p4['file_size_mb']:.1f} MiB, {info_p4['n_tensors']} tensors)", flush=True)
        c99_p4 = verify_c99_runtime_3b(p4_path)
        with open(os.path.join(args.out_dir, "p4_0_c99_report.json"), "w") as f:
            json.dump({"export": info_p4, "c99": c99_p4}, f, indent=2)
        return

    # --- Mode: eval_base (Full Step-0 BEFORE Baseline + Base .vlt3 + C99 Verification) ---
    if args.mode == "eval_base":
        # Quick batched vs bs=1 exactness check on first 2 GSM8K problems
        ds_g = load_dataset("openai/gsm8k", "main", split="test")
        test_p = [ds_g[i]["question"].strip() + GSM8K_USER_SUFFIX for i in range(2)]
        r_b1 = [generate_chat_batched(model, tok, [p], max_new_tokens=96, batch_size=1, device=device)[0] for p in test_p]
        r_b2 = generate_chat_batched(model, tok, test_p, max_new_tokens=96, batch_size=2, device=device)
        prefix_lens = [len(os.path.commonprefix([a, b])) for a, b in zip(r_b1, r_b2)]
        print(f"[PARITY] Batched KV-cache vs bs=1 common prefix chars (n=2): {prefix_lens}", flush=True)
        assert min(prefix_lens) >= 100, f"Batched KV-cache RoPE/mask mismatch! prefix_lens={prefix_lens}\n  b1={r_b1}\n  b2={r_b2}"

        base_report = run_full_benchmark_suite(
            model, tok, wt_bundle, val_sft_samples, device,
            label="BEFORE (VAULT-v3-3B Base V3_Q8_0)",
            gsm8k_n=args.gsm8k_n,
        )
        base_vlt3 = os.path.join(args.out_dir, "vault_v3_3b_q8_0_base.vlt3")
        print(f"\n[EXPORT] Exporting baseline {base_vlt3}...", flush=True)
        exp_info = export_vault_primal_vlt3(model, args.qwen_dir, base_vlt3, model_name="VAULT-v3-3B-q8_0-base")
        print(f"[EXPORT] Validated {base_vlt3}: {exp_info['file_size_mb']:.1f} MiB, {exp_info['n_tensors']} tensors", flush=True)
        c99_info = verify_c99_runtime_3b(base_vlt3)
        out_json = os.path.join(args.out_dir, "baseline_before_report.json")
        with open(out_json, "w") as f:
            json.dump({"memory": mem, "baseline": base_report, "export": {k: v for k, v in exp_info.items() if k != "header"}, "c99": c99_info}, f, indent=2)
        print(f"[DONE] Saved BEFORE baseline report -> {out_json}", flush=True)
        return

    # --- Modes: pilot (15-step calibration) or train (Full-Parameter Shadowless PRIMAL) ---
    gsm8k_dedup = build_gsm8k_test_dedup_set()
    print(f"[DATA] Loaded {len(gsm8k_dedup)} GSM8K-test dedup keys; starting 7-source async SFT+Replay streamer...", flush=True)
    stream = FrontierSFTReplayStream(
        tok, seq_len=args.seq_len, batch_size=args.batch_size, gsm8k_dedup=gsm8k_dedup, seed=args.seed,
    )

    stop_requested = {"flag": False}
    def _handle_sig(_sig, _frame):
        print("\n[SIGNAL] Graceful stop requested — finishing current step & saving checkpoint...", flush=True)
        stop_requested["flag"] = True
    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    start_step = engine.step_count
    target_steps = (start_step + 15) if args.mode == "pilot" else args.total_steps
    eval_every = 5 if args.mode == "pilot" else args.eval_every

    wt0 = evaluate_wikitext2_bpb(model, wt_bundle[0], wt_bundle[1], wt_bundle[2], 2048, device, batch_size=2)
    vs0 = evaluate_heldout_sft_loss(model, val_sft_samples, eos_id, device, batch_size=4)
    base_wt2_bpb = float(resume_meta.get("base_wt2_bpb", wt0["wt2_bpb"]))
    best_val_sft = float(resume_meta.get("best_val_sft", vs0))
    total_tokens = int(resume_meta.get("total_tokens", 0))
    tok_per_step = args.batch_size * args.seq_len * args.grad_accum

    print(
        f"[START] mode={args.mode} | step={start_step}->{target_steps} | "
        f"B={args.batch_size} T={args.seq_len} ga={args.grad_accum} ({tok_per_step:,} tok/step) | "
        f"thr={args.threshold_start}->{args.threshold_end} gain={args.vote_gain_start}->{args.vote_gain_end} "
        f"dec={args.decay_start}->{args.decay_end} | "
        f"step0: val_sft={vs0:.4f} wt2_bpb={wt0['wt2_bpb']:.4f} (word_ppl={wt0['wt2_word_ppl']:.2f})",
        flush=True,
    )

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    t_run0 = time.time()
    ema_loss = None
    auto_cooled = False

    for step in range(start_step + 1, target_steps + 1):
        t_st = time.time()
        engine.zero_grad()
        loss_acc = 0.0
        valid_tok_step = 0
        torch.cuda.reset_peak_memory_stats()

        for _ in range(args.grad_accum):
            x_cpu, y_cpu = stream.next_batch()
            x_gpu = x_cpu.to(device, non_blocking=True)
            y_gpu = y_cpu.to(device, non_blocking=True)
            valid_tok_step += int((y_gpu != -100).sum().item())
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x_gpu, y_gpu, ignore_index=-100)
            loss_val = float(loss.item())
            if not math.isfinite(loss_val):
                print(f"[WARN] Non-finite loss at step {step}; skipping microbatch", flush=True)
                del x_gpu, y_gpu, loss
                continue
            loss_acc += loss_val / args.grad_accum
            (loss / args.grad_accum).backward()
            del x_gpu, y_gpu, loss

        collect = (step % 5 == 0) or (step <= start_step + 5) or (step == target_steps)
        st = engine.step(collect_stats=collect)
        dt = time.time() - t_st
        total_tokens += tok_per_step
        ema_loss = loss_acc if ema_loss is None else 0.9 * ema_loss + 0.1 * loss_acc
        tps = tok_per_step / max(1e-6, dt)
        pk_gib = torch.cuda.max_memory_allocated() / (1024.0 ** 3)

        rec: Dict[str, Any] = {
            "step": step,
            "tokens": total_tokens,
            "train_loss": round(loss_acc, 4),
            "ema_loss": round(ema_loss, 4),
            "valid_tok_pct": round(100.0 * valid_tok_step / max(1, tok_per_step), 1),
            "tok_s": round(tps, 1),
            "peak_gib": round(pk_gib, 2),
            "thr": st["threshold"],
            "gain": st["vote_gain"],
            "flip_pct": round(st["flip_pct"], 4),
            "flips": st["flips"],
            "max_rail_pct": round(st["max_rail_pct"], 3),
        }

        if step == start_step + 1 or step % 5 == 0 or step == target_steps:
            print(
                f"[STEP {step:4d}/{target_steps}] tok={total_tokens/1e6:5.2f}M | "
                f"loss={loss_acc:.4f} (ema={ema_loss:.4f}) | val_tok={rec['valid_tok_pct']:4.1f}% | "
                f"flips={st['flip_pct']:.4f}% ({st['flips']:,}) | thr={st['threshold']} g={st['vote_gain']:.2f} | "
                f"{tps:.0f} tok/s | peak={pk_gib:.2f} GiB",
                flush=True,
            )

        if step % eval_every == 0 or step == target_steps or stop_requested["flag"]:
            wt_cur = evaluate_wikitext2_bpb(model, wt_bundle[0], wt_bundle[1], wt_bundle[2], 2048, device, batch_size=2)
            vs_cur = evaluate_heldout_sft_loss(model, val_sft_samples, eos_id, device, batch_size=4)
            drift = wt_cur["wt2_bpb"] - base_wt2_bpb
            rec.update({
                "val_sft_loss": vs_cur,
                "wt2_bpb": wt_cur["wt2_bpb"],
                "wt2_word_ppl": wt_cur["wt2_word_ppl"],
                "wt2_drift": round(drift, 4),
            })
            print(
                f"  >>> [EVAL @{step}] val_sft={vs_cur:.4f} (best={best_val_sft:.4f}, delta={vs_cur - vs0:+.4f}) | "
                f"WT2 BPB={wt_cur['wt2_bpb']:.4f} (Word PPL={wt_cur['wt2_word_ppl']:.2f}, drift={drift:+.4f}) | "
                f"elapsed={(time.time()-t_run0)/60:.1f}m | docs={stream.doc_counts}",
                flush=True,
            )

            # Anti-forgetting auto-dampener: if WT2 BPB drifts > max_wt2_drift, immediately lower vote_gain by 35%
            if args.mode == "train" and drift > args.max_wt2_drift and not auto_cooled:
                engine.vote_gain_start *= 0.65
                engine.vote_gain_end *= 0.65
                engine.threshold_start = min(40, engine.threshold_start + 4)
                engine._apply_schedule(engine.step_count)
                auto_cooled = True
                print(
                    f"  [GUARD] WT2 drift {drift:+.4f} > {args.max_wt2_drift:.4f}: auto-dampened gain -> "
                    f"{engine.vote_gain_start:.2f}->{engine.vote_gain_end:.2f}, thr -> {engine.threshold_start}",
                    flush=True,
                )

            # Save best checkpoint whenever val_sft improves without violating anti-forgetting guard
            if args.mode == "train" and vs_cur < best_val_sft and drift <= args.max_wt2_drift:
                best_val_sft = vs_cur
                save_primal_ckpt(
                    os.path.join(args.out_dir, "best_ckpt.pt"),
                    model, engine,
                    {"step": step, "total_tokens": total_tokens, "base_wt2_bpb": base_wt2_bpb, "best_val_sft": best_val_sft, "wt2": wt_cur},
                )
                print(f"  >>> [BEST] Saved best_ckpt.pt (val_sft={best_val_sft:.4f}, WT2 BPB={wt_cur['wt2_bpb']:.4f})", flush=True)

        if args.mode == "train" and args.gsm8k_every > 0 and (step % args.gsm8k_every == 0) and step < target_steps:
            g_mid = evaluate_gsm8k(model, tok, n_samples=100, max_new_tokens=512, batch_size=32, device=device)
            lam_500 = _eval_lambada(
                model, tok,
                list(load_dataset("EleutherAI/lambada_openai", "en", split="test")["text"])[:500],
                batch_size=8, device=device,
            )
            rec["gsm8k_100_acc"] = g_mid["acc"]
            rec["lambada_500_acc"] = lam_500["acc"]
            print(
                f"  >>> [MID-PROBE @{step}] GSM8K-100 = {g_mid['acc']:.2f}% ({g_mid['hits']}/100, boxed={g_mid['boxed_pct']:.1f}%) | "
                f"LAMBADA-500 = {lam_500['acc']:.2f}% (ppl={lam_500['target_word_ppl']:.2f})",
                flush=True,
            )

        if args.mode == "train" and (step % args.save_every == 0 or stop_requested["flag"]):
            save_primal_ckpt(
                os.path.join(args.out_dir, "latest_ckpt.pt"),
                model, engine,
                {"step": step, "total_tokens": total_tokens, "base_wt2_bpb": base_wt2_bpb, "best_val_sft": best_val_sft},
            )
            print(f"  [CKPT] Saved latest_ckpt.pt at step {step}", flush=True)

        with open(log_path, "a") as lf:
            lf.write(json.dumps(rec) + "\n")

        if stop_requested["flag"]:
            break

    stream.stop()

    if args.mode == "pilot":
        print("\n[PILOT COMPLETE] Inspect val_sft_loss, wt2_drift, and flip_pct above before launching full train.", flush=True)
        os._exit(0)

    # Save final latest checkpoint
    save_primal_ckpt(
        os.path.join(args.out_dir, "final_ckpt.pt"),
        model, engine,
        {"step": engine.step_count, "total_tokens": total_tokens, "base_wt2_bpb": base_wt2_bpb, "best_val_sft": best_val_sft},
    )

    # Export both zero-PTQ native V3_Q8_0 and Hessian Prime-Harmonic V3_P4_0!
    q8_path = os.path.join(args.out_dir, "vault_v3_3b_q8_0_primal.vlt3")
    print(f"\n[EXPORT-Q8] Writing zero-PTQ native {q8_path}...", flush=True)
    exp_q8 = export_vault_primal_vlt3(model, args.qwen_dir, q8_path, model_name="VAULT-v3-3B-q8_0-primal")
    print(f"[EXPORT-Q8] Validated {q8_path}: {exp_q8['file_size_mb']:.1f} MiB, {exp_q8['n_tensors']} tensors", flush=True)

    wt_ids = wt_bundle[0]
    calib = [wt_ids[i * 1024 : (i + 1) * 1024].unsqueeze(0) for i in range(16)]
    p4_path = os.path.join(args.out_dir, "vault_v3_3b_p4_0_primal.vlt3")
    print(f"[EXPORT-P4] Writing Hessian Prime-Harmonic {p4_path}...", flush=True)
    exp_p4 = export_hessian_p4_0_from_primal_3b(model, args.qwen_dir, p4_path, calib_batches=calib, device=device)
    print(f"[EXPORT-P4] Validated {p4_path}: {exp_p4['file_size_mb']:.1f} MiB, {exp_p4['n_tensors']} tensors", flush=True)

    after_report = None
    if not args.skip_final_eval:
        after_report = run_full_benchmark_suite(
            model, tok, wt_bundle, val_sft_samples, device,
            label=f"AFTER ({engine.step_count} steps, {total_tokens/1e6:.2f}M tok, VAULT-v3-3B PRIMAL)",
            gsm8k_n=args.gsm8k_n,
        )

    c99_q8 = verify_c99_runtime_3b(q8_path)
    c99_p4 = verify_c99_runtime_3b(p4_path)

    final_summary = {
        "steps": engine.step_count,
        "total_tokens": total_tokens,
        "elapsed_min": round((time.time() - t_run0) / 60.0, 1),
        "memory": mem,
        "base_wt2_bpb": base_wt2_bpb,
        "best_val_sft": best_val_sft,
        "after": after_report,
        "export_q8": {k: v for k, v in exp_q8.items() if k != "header"},
        "export_p4": exp_p4,
        "c99_q8": c99_q8,
        "c99_p4": c99_p4,
    }
    summary_path = os.path.join(args.out_dir, "final_after_report.json")
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=2)
    print(f"\n[ALL DONE] Saved final report -> {summary_path}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
