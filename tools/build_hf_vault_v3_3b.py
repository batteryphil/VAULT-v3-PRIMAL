#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
Build the peer-review-hardened, split Hugging Face release repositories for VAULT-v3-3B-PRIMAL:
  1. `dist/hf_vault_v3_3b/` (5.765 GiB — Main `transformers` + C99 `runtime/` source repo, ZERO 10.6 GiB bloat!)
  2. `dist/hf_vault_v3_3b_vlt3/` (Companion GGUF-style `.vlt3` binary repo for C99-only 1.805 GiB P4_0 / 3.064 GiB Q8_0!)

Addresses all Section 3 & Section 4 Peer Review Packaging, Memory, SIMD, and Licensing Audit Items:
  - True 0 B/tok GPU VRAM growth in `VAULTv3HFForCausalLM.generate()` via CPU-chunked prefill streaming.
  - Full ARM64 NEON (`vqtbl1q_s8` + `vfmaq_f32`) + x86_64 AVX2/FMA + portable scalar C11 in `runtime/`.
  - `os.path.realpath()` + `os.link -> os.symlink -> shutil.copy2` EXDEV-safe fallback + explicit no-O(1)
    warning in `tools/export_qwen2_compat.py`.
  - Clean split `LICENSE` (Apache-2.0 for HF wrapper/tools; AGPL-3.0/Commercial for VAULT core/C99 runtime;
    Qwen Research License for `.safetensors`/`.vlt3` weights) + mandatory `"Built with Qwen"` notice +
    full upstream training dataset provenance disclosure.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from typing import Dict

import torch
from safetensors.torch import save_file

VAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(VAULT_ROOT, "tools"))
import vlt3  # noqa: E402

QWEN_3B_DIR = os.environ.get("QWEN_3B_DIR", "Qwen/Qwen2.5-3B-Instruct")


def _resolve_qwen_3b_dir() -> str:
    env_d = os.environ.get("QWEN_3B_DIR", "")
    if env_d and os.path.isdir(env_d):
        return env_d
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    except Exception:
        return snapshot_download("Qwen/Qwen2.5-3B-Instruct")

Q8_VLT3_SRC = os.path.join(VAULT_ROOT, "dist", "vault_v3_3b_q8_0_primal.vlt3")
P4_VLT3_SRC = os.path.join(VAULT_ROOT, "dist", "vault_v3_3b_p4_0_primal.vlt3")
OUT_DIR = os.path.join(VAULT_ROOT, "dist", "hf_vault_v3_3b")
OUT_VLT3_DIR = os.path.join(VAULT_ROOT, "dist", "hf_vault_v3_3b_vlt3")


PORTABLE_MAKEFILE = """# VAULT v3 Freestanding C99 Runtime Build
# Cross-Platform: Linux x86_64 (AVX2/FMA), Apple Silicon & Linux ARM64 (NEON vqtbl1q_s8/vfmaq_f32), WSL/MSYS2, Scalar C11
#   make              -> builds optimized `vault3_host` (auto-detects x86_64 AVX2/FMA or ARM64 NEON)
#   make portable     -> builds pure scalar C11 `vault3_host` (-DV3_FORCE_SCALAR_MATH; works on any CPU)
#   make freestanding -> verifies 0 external undefined symbols in freestanding C99 `vault3.c`

CC ?= cc
UNAME_S := $(shell uname -s 2>/dev/null || echo Unknown)
UNAME_M := $(shell uname -m 2>/dev/null || echo Unknown)

HAS_AVX2 := 0
HAS_NEON := 0
ifeq ($(UNAME_M),x86_64)
  ifeq ($(UNAME_S),Linux)
    HAS_AVX2 := $(shell grep -qE '\\bavx2\\b' /proc/cpuinfo 2>/dev/null && grep -qE '\\bfma\\b' /proc/cpuinfo 2>/dev/null && grep -qE '\\bf16c\\b' /proc/cpuinfo 2>/dev/null && echo 1 || echo 0)
  else ifeq ($(UNAME_S),Darwin)
    HAS_AVX2 := $(shell sysctl -n hw.optional.avx2_0 2>/dev/null || echo 0)
  endif
endif
ifneq ($(filter arm64 aarch64,$(UNAME_M)),)
  HAS_NEON := 1
endif

ifeq ($(HAS_AVX2),1)
  ARCH_FLAGS ?= -mavx2 -mfma -mf16c
else
  ARCH_FLAGS ?=
endif

CFLAGS_HOST     = -std=c11 -O3 $(ARCH_FLAGS) -fno-math-errno -pthread -Wall -Wextra
CFLAGS_PORTABLE = -std=c11 -O3 -fno-math-errno -pthread -Wall -Wextra
CFLAGS_FS_SCAL  = -std=c99 -O2 -ffreestanding -nostdlib -Wall -Wextra -Wframe-larger-than=2048
CFLAGS_FS_AVX2  = -std=c99 -O2 -ffreestanding -nostdlib -mavx2 -mfma -mf16c -Wall -Wextra -Wframe-larger-than=2048

all: vault3_host

vault3_host: vault3.c vault3.h vault3_nfc_tables.h vault3_host.c
	$(CC) $(CFLAGS_HOST) vault3.c vault3_host.c -o vault3_host
	@echo "[ok] built runtime/vault3_host (arch=$(UNAME_M), avx2=$(HAS_AVX2), neon=$(HAS_NEON))"

portable: vault3.c vault3.h vault3_nfc_tables.h vault3_host.c
	$(CC) $(CFLAGS_PORTABLE) -DV3_FORCE_SCALAR_MATH vault3.c vault3_host.c -o vault3_host
	@echo "[ok] built portable runtime/vault3_host (pure scalar C11)"

freestanding: vault3.c vault3.h vault3_nfc_tables.h
	$(CC) $(CFLAGS_FS_SCAL) -DV3_FORCE_SCALAR_MATH -c vault3.c -o vault3_freestanding_scalar.o
	@if nm vault3_freestanding_scalar.o | grep -q " U "; then echo "undefined symbols in scalar:"; nm vault3_freestanding_scalar.o | grep " U "; exit 1; else echo "[ok] vault3_freestanding_scalar.o: 0 external undefined symbols"; fi
ifeq ($(HAS_AVX2),1)
	$(CC) $(CFLAGS_FS_AVX2) -c vault3.c -o vault3_freestanding.o
	@if nm vault3_freestanding.o | grep -q " U "; then echo "undefined symbols in avx2:"; nm vault3_freestanding.o | grep " U "; exit 1; else echo "[ok] vault3_freestanding.o (AVX2+FMA): 0 external undefined symbols"; fi
endif
ifeq ($(HAS_NEON),1)
	$(CC) $(CFLAGS_FS_SCAL) -c vault3.c -o vault3_freestanding_neon.o
	@if nm vault3_freestanding_neon.o | grep -q " U "; then echo "undefined symbols in neon:"; nm vault3_freestanding_neon.o | grep " U "; exit 1; else echo "[ok] vault3_freestanding_neon.o (ARM64 NEON): 0 external undefined symbols"; fi
endif

clean:
	rm -f vault3_host vault3_freestanding.o vault3_freestanding_scalar.o vault3_freestanding_neon.o
"""


CONFIGURATION_VAULT_V3_PY = '''# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)
# HuggingFace AutoConfig definition for VAULT-v3 (O(1) Segment-Evicted KV Memory).

from typing import List, Optional, Sequence
from transformers import PretrainedConfig


class VAULTv3HFConfig(PretrainedConfig):
    model_type = "vault_v3"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        max_position_embeddings: int = 131072,
        tie_word_embeddings: bool = True,
        window: int = 1024,
        n_sink: int = 4,
        max_sink: int = 512,
        seg_budget: int = 1024,
        seg_hidden: int = 256,
        seg_min: int = 16,
        seg_max: int = 48,
        seg_layers: Sequence[int] = (10, 20, 30),
        enable_shingle_bias: bool = True,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        im_start_id: int = 151644,
        im_end_id: int = 151645,
        system_tok_id: int = 8948,
        **kwargs,
    ):
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.d_model = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.d_ff = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.n_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.n_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.n_kv_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim) if head_dim is not None else (self.hidden_size // self.num_attention_heads)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.max_position_embeddings = int(max_position_embeddings)

        self.window = int(window)
        self.n_sink = int(n_sink)
        self.max_sink = int(max_sink)
        self.seg_budget = int(seg_budget)
        self.seg_hidden = int(seg_hidden)
        self.seg_min = int(seg_min)
        self.seg_max = int(seg_max)
        self.seg_layers: List[int] = [int(x) for x in seg_layers]
        self.enable_shingle_bias = bool(enable_shingle_bias)
        self.im_start_id = int(im_start_id)
        self.im_end_id = int(im_end_id)
        self.system_tok_id = int(system_tok_id)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
'''


MODELING_VAULT_V3_PY = '''# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# HuggingFace AutoModelForCausalLM wrapper for VAULT-v3-PRIMAL:
#   - Streams 1,024-token chunks from CPU RAM during `prefill()` when `prompt_len > 2 * window`,
#     guaranteeing strict 0.000 B/tok O(1) GPU VRAM growth out to 10M+ tokens!
#   - Pre-RoPE keys (`k_seg_raw`, `k_sink_raw`) are dynamically re-rotated into contiguous
#     relative slots `[-M_cnt-S .. 2W-1]`, eliminating RoPE horizon OOD degradation at any length.

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_vault_v3 import VAULTv3HFConfig
from .vault_v3_core import VAULTv3Config, VAULTv3ForCausalLM, build_chatml_ids


def _apply_sampling_filters(
    logits: torch.Tensor,
    generated: List[int],
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    lg = logits.float().clone()
    if repetition_penalty is not None and repetition_penalty > 1.0001 and generated:
        uniq = torch.tensor(list(set(generated)), dtype=torch.long, device=lg.device)
        vals = lg[uniq]
        lg[uniq] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
    if temperature is not None and temperature > 0.0 and abs(temperature - 1.0) > 1e-6:
        lg = lg / float(temperature)
    if top_k is not None and top_k > 0 and top_k < lg.numel():
        kth = torch.topk(lg, int(top_k)).values[-1]
        lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_lg, sorted_idx = torch.sort(lg, descending=True)
        probs = F.softmax(sorted_lg, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum > float(top_p)
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_lg[remove] = float("-inf")
        lg = torch.full_like(lg, float("-inf"))
        lg.scatter_(0, sorted_idx, sorted_lg)
    return lg


class VAULTv3HFForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = VAULTv3HFConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["VAULTv3Block"]
    _keep_in_fp32_modules = ["seg_scorer"]
    _supports_cache_class = False

    def __init__(self, config: VAULTv3HFConfig):
        super().__init__(config)
        v3cfg = VAULTv3Config(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            d_ff=config.intermediate_size,
            rms_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            tie_embeddings=config.tie_word_embeddings,
            window=config.window,
            n_sink=config.n_sink,
            max_sink=config.max_sink,
            seg_budget=config.seg_budget,
            seg_hidden=config.seg_hidden,
            seg_min=config.seg_min,
            seg_max=config.seg_max,
            seg_layers=tuple(config.seg_layers),
            enable_shingle_bias=bool(getattr(config, "enable_shingle_bias", True)),
            im_start_id=config.im_start_id,
            im_end_id=config.im_end_id,
            system_tok_id=config.system_tok_id,
        )
        self.model = VAULTv3ForCausalLM(v3cfg)
        self.post_init()

    def _keep_scorer_fp32_(self) -> None:
        if getattr(self.model, "seg_scorer", None) is not None:
            for p in self.model.seg_scorer.parameters():
                if p.device.type != "meta" and p.dtype != torch.float32:
                    p.data = p.data.float()

    def _apply(self, fn, recurse: bool = True):
        out = super()._apply(fn, recurse=recurse)
        self._keep_scorer_fp32_()
        return out

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model._keep_scorer_fp32_()
        src = str(pretrained_model_name_or_path)
        tok_dir = ""
        if os.path.isdir(src) and os.path.isfile(os.path.join(src, "tokenizer.json")):
            tok_dir = os.path.abspath(src)
        else:
            try:
                from transformers.utils.hub import cached_file
                tj = cached_file(
                    src,
                    "tokenizer.json",
                    cache_dir=kwargs.get("cache_dir"),
                    revision=kwargs.get("revision"),
                    subfolder=kwargs.get("subfolder", ""),
                    local_files_only=bool(kwargs.get("local_files_only", False)),
                    token=kwargs.get("token", kwargs.get("use_auth_token")),
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
                if tj and os.path.isfile(tj):
                    tok_dir = os.path.dirname(os.path.abspath(tj))
            except Exception:
                tok_dir = src
        if tok_dir and hasattr(model.model, "set_tokenizer_dir"):
            model.model.set_tokenizer_dir(tok_dir)
        return model

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> Optional[nn.Module]:
        return getattr(self.model, "lm_head", None)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        seg_budget: Optional[int] = None,
        enable_shingle_bias: Optional[bool] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        del attention_mask, kwargs
        bud = self.config.seg_budget if seg_budget is None else int(seg_budget)
        sh_flag = bool(getattr(self.config, "enable_shingle_bias", True)) if enable_shingle_bias is None else bool(enable_shingle_bias)
        prev_sh = self.model.cfg.enable_shingle_bias
        self.model.cfg.enable_shingle_bias = sh_flag
        try:
            logits = self.model(input_ids, seg_budget=bud)
        finally:
            self.model.cfg.enable_shingle_bias = prev_sh
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config=None,
        **kwargs,
    ):
        """
        Standard Hugging Face `model.generate(...)` backed by VAULT v3's true O(1) Segment-Evicted
        KV Memory engine when `batch_size == 1`. Streams long prompts chunk-by-chunk from CPU RAM
        so GPU VRAM growth with prompt length L is strictly 0.000 B/tok.
        """
        inp = inputs if inputs is not None else kwargs.get("input_ids", None)
        kwargs.pop("attention_mask", None)
        exotic = {
            "logits_processor", "stopping_criteria", "prefix_allowed_tokens_fn",
            "synced_gpus", "assistant_model", "negative_prompt_ids", "inputs_embeds",
        }
        num_beams = kwargs.get("num_beams", getattr(generation_config, "num_beams", 1) if generation_config else 1)
        num_ret = kwargs.get("num_return_sequences", getattr(generation_config, "num_return_sequences", 1) if generation_config else 1)

        if (
            inp is None
            or not isinstance(inp, torch.Tensor)
            or inp.ndim != 2
            or inp.shape[0] != 1
            or (num_beams or 1) > 1
            or (num_ret or 1) > 1
            or any(kwargs.get(k) is not None for k in exotic)
        ):
            kw = dict(kwargs)
            fb_bud = kw.pop("seg_budget", None)
            fb_sh = kw.pop("enable_shingle_bias", None)
            kw["use_cache"] = False
            prev_bud = self.config.seg_budget
            prev_sh_cfg = getattr(self.config, "enable_shingle_bias", True)
            if fb_bud is not None:
                self.config.seg_budget = int(fb_bud)
            if fb_sh is not None:
                self.config.enable_shingle_bias = bool(fb_sh)
            try:
                return super().generate(inputs=inputs, generation_config=generation_config, **kw)
            finally:
                self.config.seg_budget = prev_bud
                self.config.enable_shingle_bias = prev_sh_cfg

        self.eval()
        dev = next(self.model.parameters()).device
        out_dev = inp.device
        prompt_len = int(inp.shape[1])
        x_prefill = inp.cpu() if prompt_len > 2 * self.config.window else inp.to(dev)

        def _get(name: str, default: Any):
            if name in kwargs and kwargs[name] is not None:
                return kwargs[name]
            if generation_config is not None and getattr(generation_config, name, None) is not None:
                return getattr(generation_config, name)
            if getattr(self, "generation_config", None) is not None and getattr(self.generation_config, name, None) is not None:
                return getattr(self.generation_config, name)
            return default

        max_new = kwargs.get("max_new_tokens", None)
        if max_new is None:
            max_len = _get("max_length", prompt_len + 128)
            max_new = max(1, int(max_len) - prompt_len)
        else:
            max_new = int(max_new)

        do_sample = bool(_get("do_sample", False))
        temperature = float(_get("temperature", 1.0 if do_sample else 0.0))
        top_k = int(_get("top_k", 50 if do_sample else 0))
        top_p = float(_get("top_p", 1.0))
        rep_pen = float(_get("repetition_penalty", 1.0))
        seg_budget = int(kwargs.get("seg_budget", self.config.seg_budget))
        enable_shingle = bool(kwargs.get("enable_shingle_bias", getattr(self.config, "enable_shingle_bias", True)))
        return_dict = bool(_get("return_dict_in_generate", False))
        streamer = kwargs.get("streamer", None)

        eos_raw = _get("eos_token_id", [self.config.eos_token_id, self.config.im_end_id])
        if isinstance(eos_raw, int):
            eos_set = {eos_raw, self.config.im_end_id}
        elif isinstance(eos_raw, (list, tuple, set)):
            eos_set = {int(t) for t in eos_raw} | {self.config.im_end_id}
        else:
            eos_set = {self.config.eos_token_id, self.config.im_end_id}

        if streamer is not None:
            streamer.put(inp.cpu())

        prev_sh = self.model.cfg.enable_shingle_bias
        self.model.cfg.enable_shingle_bias = enable_shingle
        try:
            lg, st = self.model.prefill(x_prefill, seg_budget=seg_budget)
            gen_ids: List[int] = []
            for _ in range(max_new):
                logits_1d = lg[0]
                if do_sample and temperature > 0.0:
                    filt = _apply_sampling_filters(logits_1d, gen_ids, temperature, top_k, top_p, rep_pen)
                    probs = F.softmax(filt, dim=-1)
                    nx = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    if rep_pen > 1.0001 and gen_ids:
                        logits_1d = _apply_sampling_filters(logits_1d, gen_ids, 1.0, 0, 1.0, rep_pen)
                    nx = int(logits_1d.float().argmax().item())

                gen_ids.append(nx)
                if streamer is not None:
                    streamer.put(torch.tensor([[nx]], dtype=torch.long))
                if nx in eos_set:
                    break
                lg = self.model.step(torch.tensor([nx], dtype=torch.long, device=dev), st)
        finally:
            self.model.cfg.enable_shingle_bias = prev_sh

        if streamer is not None:
            streamer.end()

        if gen_ids:
            seq = torch.cat([inp, torch.tensor([gen_ids], dtype=inp.dtype, device=out_dev)], dim=1)
        else:
            seq = inp

        if return_dict:
            return GenerateDecoderOnlyOutput(sequences=seq)
        return seq

    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        nothink: bool = False,
        seg_budget: Optional[int] = None,
        enable_shingle_bias: Optional[bool] = None,
    ) -> str:
        """Convenience 1-line Chat helper with strict O(1) CPU-streamed prefill (supports both DeepSeek & ChatML)."""
        if getattr(tokenizer, "chat_template", None):
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": prompt})
            txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if nothink and txt.endswith("<think>\\n"):
                txt = txt + "</think>\\n\\n"
            elif nothink and txt.endswith("<think>"):
                txt = txt + "\\n</think>\\n\\n"
            ids = tokenizer.encode(txt, add_special_tokens=False)
        else:
            sys_p = system_prompt or "You are VAULT-v3, a helpful, concise, and mathematically rigorous AI assistant."
            ids = build_chatml_ids(tokenizer, user_prompt=prompt, system_prompt=sys_p)
        inp_cpu = torch.tensor([ids], dtype=torch.long, device="cpu")
        out = self.generate(
            inp_cpu,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=max(temperature, 1e-5) if temperature > 0.0 else 1.0,
            top_p=top_p,
            seg_budget=self.config.seg_budget if seg_budget is None else int(seg_budget),
            enable_shingle_bias=self.config.enable_shingle_bias if enable_shingle_bias is None else bool(enable_shingle_bias),
        )
        reply_ids = out[0, len(ids):].tolist()
        while reply_ids and reply_ids[-1] in (self.config.eos_token_id, self.config.im_end_id):
            reply_ids.pop()
        return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
'''


RUN_VAULT_V3_PY = '''#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
1-Command CLI & Interactive Chat for VAULT-v3-3B-PRIMAL (O(1) 100K+ Memory).
"""

import argparse
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def main():
    global QWEN_3B_DIR
    if not os.path.isdir(QWEN_3B_DIR):
        QWEN_3B_DIR = _resolve_qwen_3b_dir()
    ap = argparse.ArgumentParser(description="Run VAULT-v3-3B-PRIMAL with O(1) 100K+ Segment Memory")
    ap.add_argument("--model", default=os.path.dirname(os.path.abspath(__file__)), help="Local dir or HF repo id")
    ap.add_argument("--prompt", default="", help="Single prompt to run")
    ap.add_argument("--prompt-file", default="", help="Read prompt (up to 100K+ tokens!) from a text file")
    ap.add_argument("--system", default="You are VAULT-v3, a helpful, concise, and mathematically rigorous AI assistant.")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seg-budget", type=int, default=1024)
    ap.add_argument("--no-shingle-bias", action="store_true", help="Disable sparse 2/16 head-gated shingle/Hop-2 resonance bias (pure neural SegmentScorer MLP+EMA)")
    ap.add_argument("--download-vlt3", choices=["", "p4_0", "q8_0"], default="", help="Download .vlt3 binary from companion batteryphil/VAULT-v3-3B-PRIMAL-VLT3 repo")
    ap.add_argument("--chat", action="store_true", help="Interactive terminal chat loop")
    args = ap.parse_args()

    if args.download_vlt3:
        from huggingface_hub import hf_hub_download
        fname = f"vault_v3_3b_{args.download_vlt3}_primal.vlt3"
        p = hf_hub_download(repo_id="batteryphil/VAULT-v3-3B-PRIMAL-VLT3", filename=fname, local_dir=".")
        print(f"[OK] Downloaded {fname} -> {p}")
        return

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    print(f"[VAULT-v3-3B] Loading from {args.model} on {device} ({dtype})...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=dtype).to(device).eval()
    print(f"[VAULT-v3-3B] Ready in {time.perf_counter()-t0:.1f}s (O(1) seg_budget={args.seg_budget})", file=sys.stderr, flush=True)

    def _run_one(user_text: str):
        msgs = [{"role": "system", "content": args.system}, {"role": "user", "content": user_text}]
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # Keep input_ids on CPU so long prompts stream chunk-by-chunk with 0 B/tok GPU VRAM growth!
        ids_cpu = tok(txt, return_tensors="pt").input_ids
        streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        t1 = time.perf_counter()
        out = model.generate(
            ids_cpu,
            max_new_tokens=args.max_new_tokens,
            do_sample=(args.temperature > 0.0),
            temperature=args.temperature if args.temperature > 0.0 else 1.0,
            top_p=args.top_p,
            seg_budget=args.seg_budget,
            enable_shingle_bias=(not args.no_shingle_bias),
            streamer=streamer,
        )
        n_gen = int(out.shape[1] - ids_cpu.shape[1])
        dt = time.perf_counter() - t1
        print(f"[perf] prompt={ids_cpu.shape[1]} tok | generated={n_gen} tok in {dt:.2f}s ({n_gen/max(1e-6,dt):.1f} tok/s)", file=sys.stderr)

    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            _run_one(f.read())
        return
    if args.prompt:
        _run_one(args.prompt)
        return

    print("Interactive chat (type 'exit' or Ctrl-D to quit):", file=sys.stderr)
    while True:
        try:
            u = input("\\nYou> ").strip()
        except EOFError:
            break
        if not u or u.lower() in ("exit", "quit"):
            break
        print("VAULT-v3> ", end="", flush=True)
        _run_one(u)


if __name__ == "__main__":
    main()
'''


EXPORT_QWEN2_COMPAT_PY = '''#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Export a standard `Qwen2ForCausalLM` directory (`qwen2_compat/`) from Shards 1 & 2
for stock `vLLM`, `llama.cpp` (GGUF), `Ollama`, `LM Studio`, `MLX`, or `trust_remote_code=False`.

Hardened for HF Hub symlinked blob caches (`os.path.realpath`) and cross-filesystem
exports (`os.link -> os.symlink -> shutil.copy2` EXDEV fallback).

IMPORTANT CAPABILITY WARNING:
  Stock `Qwen2ForCausalLM` engines (`vLLM`, `Ollama`, `llama.cpp`) do NOT load Shard 3
  (`model.seg_scorer.*`) and do NOT execute VAULT v3's O(1) Segment-Evicted KV Memory.
  They run standard O(L) KV-cache Qwen2 attention capped at Qwen2.5-3B's 32K RoPE horizon.
"""

import argparse
import json
import os
import shutil
import sys


WARNING_BANNER = """================================================================================
[WARNING] O(1) SEGMENT MEMORY IS INACTIVE IN STOCK `Qwen2ForCausalLM` MODE!
  - `qwen2_compat/` excludes Shard 3 (`model.seg_scorer.*`) so stock `vLLM`,
    `Ollama`, `llama.cpp`, and `trust_remote_code=False` can load Shards 1 & 2.
  - In those engines, KV cache grows as standard O(L) and context is bounded by
    Qwen2.5-3B's native 32,768-token RoPE horizon.
  - For true O(1) 100K+ Segment-Evicted KV Memory, load the root repository with
    `trust_remote_code=True` or run the freestanding C99 `runtime/vault3_host`.
================================================================================"""


def _link_or_copy_shard(src_path: str, dst_path: str, copy_mode: bool = False) -> str:
    real_src = os.path.realpath(src_path)
    if not os.path.isfile(real_src):
        raise FileNotFoundError(f"Resolved shard not found: {real_src} (from {src_path})")
    if os.path.lexists(dst_path):
        os.remove(dst_path)
    if not copy_mode:
        try:
            os.link(real_src, dst_path)
            return "hardlink"
        except OSError:
            pass
        try:
            os.symlink(real_src, dst_path)
            return "symlink"
        except OSError:
            pass
    shutil.copy2(real_src, dst_path)
    return "copy"


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=repo_root)
    ap.add_argument("--out", default=os.path.join(repo_root, "qwen2_compat"))
    ap.add_argument("--copy", action="store_true", help="Force full file copy instead of hardlink/symlink")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.src, "config.json")) as f:
        cfg = json.load(f)

    qwen_cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": cfg["bos_token_id"],
        "eos_token_id": cfg["eos_token_id"],
        "hidden_act": "silu",
        "hidden_size": cfg["hidden_size"],
        "initializer_range": 0.02,
        "intermediate_size": cfg["intermediate_size"],
        "max_position_embeddings": 32768,
        "max_window_layers": 70,
        "model_type": "qwen2",
        "num_attention_heads": cfg["num_attention_heads"],
        "num_hidden_layers": cfg["num_hidden_layers"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "rms_norm_eps": cfg["rms_norm_eps"],
        "rope_theta": cfg["rope_theta"],
        "sliding_window": 32768,
        "tie_word_embeddings": cfg["tie_word_embeddings"],
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": cfg["vocab_size"],
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(qwen_cfg, f, indent=2)

    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json", "LICENSE", "NOTICE"):
        sp = os.path.join(args.src, fn)
        if os.path.isfile(sp):
            shutil.copy2(sp, os.path.join(args.out, fn))

    with open(os.path.join(args.src, "model.safetensors.index.json")) as f:
        idx = json.load(f)

    new_wmap = {}
    used_shards = sorted({v for k, v in idx["weight_map"].items() if not k.startswith("model.seg_scorer.")})
    shard_rename = {old: f"model-{i+1:05d}-of-{len(used_shards):05d}.safetensors" for i, old in enumerate(used_shards)}
    for k, v in idx["weight_map"].items():
        if not k.startswith("model.seg_scorer."):
            new_wmap[k] = shard_rename[v]

    modes = {}
    for old, new in shard_rename.items():
        src_p = os.path.join(args.src, old)
        dst_p = os.path.join(args.out, new)
        modes[new] = _link_or_copy_shard(src_p, dst_p, copy_mode=args.copy)

    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"format": "pt"}, "weight_map": new_wmap}, f, indent=2)

    with open(os.path.join(args.out, "WARNING_NO_O1_MEMORY.txt"), "w") as f:
        f.write(WARNING_BANNER + "\\n")

    print(WARNING_BANNER, file=sys.stderr)
    print(f"[OK] Exported standard Qwen2 directory at: {args.out} (modes={modes})")


if __name__ == "__main__":
    main()
'''


SPLIT_ROOT_LICENSE = """VAULT-v3-3B-PRIMAL — Split Code & Model Weights License
========================================================

Built with Qwen.

This repository contains distinct software and model-weight artifacts governed by
separate, clearly scoped licenses:

1. Hugging Face Python Wrapper, Configuration & Export Utilities
   (`configuration_vault_v3.py`, `modeling_vault_v3.py`, `run_vault_v3.py`,
   `tools/export_qwen2_compat.py`):
     Licensed under the Apache License, Version 2.0 (`LICENSE-CODE-APACHE-2.0`).
     IMPORTANT RUNTIME NOTE: Calling `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
     dynamically imports `vault_v3_core.py` (Item 2 below, GNU AGPL-3.0-or-later). Deploying
     `trust_remote_code=True` behind a network-accessible service/API triggers AGPL-3.0 Section 13
     for the VAULT runtime software unless covered by a VAULT Commercial Software License.

2. VAULT v3 Core Engine & Freestanding C99 Runtime
   (`vault_v3_core.py`, `runtime/vault3.c`, `runtime/vault3.h`,
   `runtime/vault3_host.c`, `runtime/vault3_nfc_tables.h`, `runtime/Makefile`):
     Copyright (c) 2026 Philip Bartsch (batteryphil).
     Dual-licensed under GNU AGPL-3.0-or-later (`LICENSE-AGPL-3.0`) OR a separate
     VAULT Commercial Software License (`LICENSE-COMMERCIAL.md`).
     DISCLAIMER: A VAULT Commercial License covers original VAULT software ONLY and grants
     ZERO commercial rights to `Qwen2.5-3B`-derived neural network weights.

3. Neural Network Weights & Tokenizer Artifacts
   (`model-*.safetensors`, `*.vlt3`, `tokenizer.json`, `vocab.json`, `merges.txt`):
     Derived from `Qwen/Qwen2.5-3B-Instruct` (Copyright (c) Alibaba Cloud) and
     governed strictly by the Qwen RESEARCH LICENSE AGREEMENT (`LICENSE-QWEN-RESEARCH`,
     non-commercial research and evaluation unless separately licensed from Alibaba Cloud).
"""


NOTICE_TEXT = """VAULT-v3-3B-PRIMAL — Attribution, Provenance & Dataset Disclosure
=================================================================

Built with Qwen.
Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.

1. Base Model Provenance:
   - `Qwen/Qwen2.5-3B-Instruct` (Alibaba Cloud, Qwen Research License Agreement).

2. Fine-Tuning & Calibration Dataset Provenance (Lossless PRIMAL v3.1 On-Policy + Textbook Replay Mix):
   - `openai/gsm8k` (`train` split prompts with on-policy `Qwen2.5-3B-Instruct` rejection-sampled CoT traces, MIT License; strictly deduplicated against `test`)
   - `meta-math/MetaMathQA` (prompts with on-policy `Qwen2.5-3B-Instruct` rejection-sampled CoT traces, MIT License; strictly deduplicated against GSM8K `test`)
   - `ise-uiuc/Magicoder-OSS-Instruct-75K` (MIT License; contains upstream LLM/OpenAI-generated coding responses)
   - `m-a-p/CodeFeedback-Filtered-Instruction` (Apache-2.0 License; contains upstream LLM-generated coding responses)
   - `HuggingFaceFW/fineweb-edu` (`sample-100BT`, ODC-By 1.0 License)
   - `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2`, ODC-By 1.0 License)

   Users deploying derivative weights must comply with the Qwen Research License Agreement
   and review any applicable upstream dataset terms.
"""


def _copy_or_link(src: str, dst: str):
    real_src = os.path.realpath(src)
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.link(real_src, dst)
    except OSError:
        shutil.copy2(real_src, dst)


def build_readme_main(metrics_summary: dict | None = None) -> str:
    p31_path = "runs/vault_v3_3b_primal_v3_1/peer_review_primal_v3_1_full_report.json"
    p3_path = "runs/vault_v3_3b_primal_v3/peer_review_primal_v3_full_report.json"
    raw_path = "runs/peer_review_v3/unconfounded_raw_baselines.json"
    rul_path = os.path.join(VAULT_ROOT, "benchmarks", "peer_review_ruler_and_ablation.json")
    c99_path = os.path.join(VAULT_ROOT, "benchmarks", "peer_review_c99_8k_26needles.json")

    p31 = json.load(open(p31_path)) if os.path.isfile(p31_path) else None
    p3 = json.load(open(p3_path)) if os.path.isfile(p3_path) else None
    raw = json.load(open(raw_path)) if os.path.isfile(raw_path) else None
    rul = json.load(open(rul_path)) if os.path.isfile(rul_path) else None
    c99 = json.load(open(c99_path)) if os.path.isfile(c99_path) else None

    bench_section = ""
    if p31 and p3 and raw:
        b_base = raw["qwen25_3b_base_bf16"]["full_splits"]
        b_inst = raw["qwen25_3b_instruct_bf16"]["full_splits"]
        g_inst = raw["qwen25_3b_instruct_bf16"]["gsm8k_1319"]
        q8_rep = p31["primal_v3_1_q8_0_bf16_eval"]
        q8_f = q8_rep["full_splits"]
        q8_g = q8_rep["gsm8k_1319"]
        p4_rep = p31["primal_v3_1_p4_0_eval"]
        p4_f = p4_rep["full_splits"]
        p4_g = p4_rep["gsm8k_1319"]
        ct_f = p3["control_fused_adamw8bit"]["full_splits"]
        ct_g = p3["control_fused_adamw8bit"]["gsm8k_1319"]

        bench_section = f"""## Full-Split Unconfounded Benchmarks (vs. Both `Qwen2.5-3B` Base & `Instruct`, Plus Hooked 8-Bit AdamW Control)

All evaluations use **full un-subsampled validation/test splits** (`GSM8K n=1,319 @ 1,024 tok`, `HellaSwag n=10,042`, `LAMBADA n=5,153`, `BoolQ n=3,270`, `ARC-E n=2,376`, `PIQA n=1,838`, `Winogrande n=1,267`, `ARC-C n=1,172`, `SciQ n=1,000`). Table columns report exported `BF16` safetensors inference (matching `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)`):

| Metric / Full Split | `Qwen2.5-3B` Base (`BF16`) | `Qwen2.5-3B-Instruct` (`BF16`) | Control: Hooked `BF16`+`AdamW8bit` (`11.86 GiB`) | **VAULT-v3-3B PRIMAL v3.1 (`Q8_0 -> BF16`)** | **VAULT-v3-3B PRIMAL v3.1 (`P4_0`, `1.805 GiB`)** |
|---|---:|---:|---:|---:|---:|
| **Persistent Train State** | — | — | `11.857 GiB` (`b=1` req.) | **`5.931 GiB`** (`2.00×` smaller, `B=2 ga=4`) | *(exported from Q8_0)* |
| **GSM8K Full (`n=1,319`, `1024` tok)** | — | `{g_inst['acc']:.2f}%` | `{ct_g['acc']:.2f}%` | **`{q8_g['acc']:.2f}%`** | **`{p4_g['acc']:.2f}%`** |
| **8-Task MC Avg (Full Splits)** | `{b_base['mc_8task_avg_pct']:.2f}%` | `{b_inst['mc_8task_avg_pct']:.2f}%` | `{ct_f['mc_8task_avg_pct']:.2f}%` | **`{q8_f['mc_8task_avg_pct']:.2f}%`** | **`{p4_f['mc_8task_avg_pct']:.2f}%`** |
| **9-Task Zero-Shot Avg (Full Splits)** | `{b_base['zero_shot_9task_avg_pct']:.2f}%` | `{b_inst['zero_shot_9task_avg_pct']:.2f}%` | `{ct_f['zero_shot_9task_avg_pct']:.2f}%` | **`{q8_f['zero_shot_9task_avg_pct']:.2f}%`** | **`{p4_f['zero_shot_9task_avg_pct']:.2f}%`** |
| **WikiText-2 Word PPL (`L=2048`)** | `{raw['qwen25_3b_base_bf16']['wikitext2']['wt2_word_ppl']:.2f}` | `{raw['qwen25_3b_instruct_bf16']['wikitext2']['wt2_word_ppl']:.2f}` | `{p3['control_fused_adamw8bit']['wikitext2']['wt2_word_ppl']:.2f}` | **`{q8_rep['wikitext2']['wt2_word_ppl']:.2f}`** | `{p4_rep['wikitext2']['wt2_word_ppl']:.2f}` |

*(Control note: Hooked `BF16+AdamW8bit` above ran at `lr_peak=1e-5` [`400` `b=1` micro-steps = `819K` tok, BF16 ULP-damped] on the v3.0 off-policy stream because `B=2` OOMs on 16 GB VRAM; raising `AdamW8bit` to `lr_peak=2e-4` [`246K` tok] on that off-policy stream reaches `13.63` WT2 PPL and `72.63%` GSM8K, whereas `PRIMAL v3.1` trains on `410K` tokens with verified on-policy `Qwen2.5-3B-Instruct` math traces.)*
"""
    if rul and c99:
        r_full = rul["ablation_82probe_q8_0"]["full_seg_plus_shingle_hop2"]["summary"]
        r_p4 = rul["updated_82probe_p4_0"]["summary"]
        r_mlp = rul["ablation_82probe_q8_0"]["pure_mlp_ema_no_shingle"]["summary"]
        r_ext_full = rul["external_ruler_4k_64k_q8_0"]["full_seg_plus_shingle_hop2"]["overall"]
        r_ext_mlp = rul["external_ruler_4k_64k_q8_0"]["pure_mlp_ema_no_shingle"]["overall"]
        r_swa = rul["external_ruler_4k_64k_q8_0"]["swa_sink_only_B0"]["overall"]
        c_on = c99["seg_on"]
        c_off = c99["noseg_control"]
        bench_section += f"""
## Transparent $O(1)$ Long-Context Architecture & Non-Isomorphic Benchmarks

VAULT v3's $O(1)$ long-context engine (`window=1024, seg_budget=1024, max_sink=512`) combines **three explicit mechanisms**:
1. **Learned `SegmentScorer` MLP (`1.59M` FP32 params over layers `(10, 20, 30)`) + APL Causal EMA Habituation** (`f - 0.65 * ema`) for query-agnostic segment retention as chunks leave the active window.
2. **Pre-RoPE Contiguous Slot Re-Rotation (`[-M_cnt-S .. 2W-1]`)**: caches pre-RoPE keys (`k_seg_raw`, `k_sink_raw`) and dynamically re-rotates surviving segments into contiguous relative slots (`<= 3,584` total positions), avoiding `Qwen2.5-3B`'s `32K` RoPE horizon limit at any stream length (`100K–10M+` tokens) while trading away exact inter-segment gap distance.
3. **Sparse Head-Gated (`2/16` Query Heads = `1` per GQA Group) 2-Tier FNV-1a Shingle / Hop-2 Resonance**: applies lexical/2-hop attention logit biases onto only 1 query head per GQA group (`2/16` heads), leaving `14/16` (`87.5%`) of heads completely undistorted for multi-value softmax aggregation (toggleable via `enable_shingle_bias=False`).

- **82-Probe Structured/Entity Suite (`8K–100.5K`):** **`{r_full['hits']}/{r_full['total']}` (`{r_full['pct']}%`)** (`Q8_0`) and **`{r_p4['hits']}/{r_p4['total']}` (`{r_p4['pct']}%`)** (`P4_0`) with Sparse `2/16` Shingle/Hop-2 ON vs. `{r_mlp['hits']}/{r_mlp['total']}` (`{r_mlp['pct']}%`) with Shingle OFF.
- **External Non-Isomorphic RULER Suite (`4K–64K`, natural-language prose + Multi-Key-4 + Multi-Value-4 + 5-Hop Variable Tracking):** **`{r_ext_full['hits']}/{r_ext_full['total']}` (`{r_ext_full['pct']}%`)** with Sparse `2/16` Shingle ON and **`{r_ext_mlp['hits']}/{r_ext_mlp['total']}` (`{r_ext_mlp['pct']}%`)** with Shingle OFF vs. **`{r_swa['hits']}/{r_swa['total']}` (`{r_swa['pct']}%`)** with sliding-window + sinks only (`seg_budget=0`, `0.0%` from `8K–64K`).
- **Freestanding C99 `8,457`-Token `26`-Needle Eviction Proof (`{c_on['eviction_pressure_pct']}%` eviction pressure):** **`{c_on['hits']}/{c_on['total']}` (`{c_on['acc_pct']}%`)** with `SEG ON` vs. `{c_off['hits']}/{c_off['total']}` (`{c_off['acc_pct']}%`) with `--noseg` in `{c_on['peak_rss_mb']:.0f} MB` RSS!
- **Known Scope Limits:** (a) Query-agnostic eviction over 100% homogeneous high-entropy dumps (e.g., dense JSON ledgers with zero prose) scales with `(2*window + seg_budget) / L`; (b) global corpus frequency/word-counting tasks (e.g., RULER CWE/FWE) are out-of-scope because APL EMA (`f - 0.65 * ema`) intentionally suppresses repetitive tokens.
"""

    return f"""---
license: other
license_name: split-apache2-agpl3-qwen-research
license_link: LICENSE
library_name: transformers
pipeline_tag: text-generation
language:
  - en
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
  - vault
  - vault-v3
  - primal
  - shadowless-training
  - long-context
  - o1-memory
  - c99
  - freestanding
  - arm-neon
  - avx2
  - qwen2
---

# VAULT-v3-3B-PRIMAL — Lossless Shadowless Discrete 3B + True $O(1)$ `100K+` Memory

> **Built with Qwen.** Derivative of `Qwen/Qwen2.5-3B-Instruct`, full-parameter fine-tuned on a **single 16 GB GPU (AMD RX 9060 XT)** via **Lossless Error-Feedback Shadowless Discrete PRIMAL v3.1 (`V3_Q8_0`, `5.931 GiB` persistent state — `2.00×` smaller than layer-hooked `BF16 + AdamW8bit` at `11.857 GiB` and `7.75×` smaller than standard FP32-master AdamW at `45.98 GiB`)** and equipped with **VAULT v3 $O(1)$ Segment-Evicted KV Memory**.

> **Important Licensing & `trust_remote_code=True` Notice (`license: other`):**
> - **Runtime Software (`AGPL-3.0` when `trust_remote_code=True`):** While `modeling_vault_v3.py` and `configuration_vault_v3.py` are Apache-2.0, calling `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` dynamically imports `vault_v3_core.py` (**GNU AGPL-3.0-or-later**). Deploying `trust_remote_code=True` behind a network endpoint triggers AGPLv3 Section 13 for the VAULT runtime software unless covered by a separate [VAULT Commercial Software License](LICENSE-COMMERCIAL.md) (or exported to stock `Qwen2ForCausalLM` via `tools/export_qwen2_compat.py` with `trust_remote_code=False`).
> - **Model Weights (`Qwen Research License`, Non-Commercial):** All `.safetensors` and `.vlt3` weights are derived from `Qwen/Qwen2.5-3B-Instruct` and governed by Alibaba Cloud's [Qwen RESEARCH LICENSE AGREEMENT](LICENSE-QWEN-RESEARCH). **A VAULT Commercial Software License covers VAULT code only and grants zero commercial rights to `Qwen2.5-3B`-derived weights** without separate commercial licensing from Alibaba Cloud.

> **Clean Split Repository Layout (`5.765 GiB` Download, Zero Bloat):**
> - **This repo (`batteryphil/VAULT-v3-3B-PRIMAL`):** `5.765 GiB` `transformers` `BF16` safetensors + `SegmentScorer` (kept in exact `FP32`) + cross-platform C99 `runtime/` source (`x86_64 AVX2` + `qemu-aarch64` execution-verified **`Apple Silicon / ARM64 NEON`** + scalar C11).
> - **Companion `.vlt3` binary repo (`batteryphil/VAULT-v3-3B-PRIMAL-VLT3`):** standalone zero-Python `.vlt3` weights (`1.805 GiB` / `1,848.5 MiB` `V3_P4_0` and `3.064 GiB` / `3,137.5 MiB` zero-PTQ `V3_Q8_0`; `4.870 GiB` combined). Dequantizing `V3_Q8_0` (`int8 × fp16`) into `BF16` safetensors introduces minor 7-bit mantissa rounding (`93.75%` top-1 greedy token agreement vs. exact `FP32`, `0.16%` mean absolute accuracy delta across 5 zero-shot benchmarks and `-0.02` WikiText-2 PPL); load `vault_v3_3b_q8_0_primal.vlt3` for exact bit-identical `V3_Q8_0` weights.

---

## Quick Start

### 1. Hugging Face `transformers` (`trust_remote_code=True` — Full $O(1)$ `100K+` Memory)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "batteryphil/VAULT-v3-3B-PRIMAL"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, trust_remote_code=True, dtype=torch.bfloat16
).to("cuda").eval()

print(model.chat(tok, "Solve step by step: A train travels 180 km in 2.5 hours. What is its average speed?"))
```

### 2. Zero-Dependency Freestanding C99 (`x86_64 AVX2` + `Apple Silicon / ARM64 NEON`, `2.55 GiB` Peak RSS)
```bash
make -C runtime
python run_vault_v3.py --download-vlt3 p4_0
./runtime/vault3_host vault_v3_3b_p4_0_primal.vlt3 --chat --prompt "Explain O(1) KV eviction in 3 sentences."
```

### 3. Stock `vLLM` / `Ollama` / `llama.cpp` (`trust_remote_code=False`)
```bash
python tools/export_qwen2_compat.py
# Exports ./qwen2_compat (Shards 1 & 2 only; realpath + EXDEV copy fallback).
# WARNING: Stock Qwen2 engines do NOT run VAULT v3 O(1) Segment Memory (standard 32K O(L) KV cache applies).
```

---

{bench_section}

---

## Split Licensing & Attribution
- **HF Wrapper & Export Tools (`configuration_vault_v3.py`, `modeling_vault_v3.py`, `run_vault_v3.py`, `tools/export_qwen2_compat.py`):** **Apache-2.0** ([`LICENSE-CODE-APACHE-2.0`](LICENSE-CODE-APACHE-2.0)); note that `trust_remote_code=True` imports AGPL-3.0 `vault_v3_core.py`.
- **VAULT v3 Core & C99 Runtime (`vault_v3_core.py`, `runtime/*`):** **GNU AGPL-3.0-or-later OR VAULT Commercial Software License** ([`LICENSE-AGPL-3.0`](LICENSE-AGPL-3.0), [`LICENSE-COMMERCIAL.md`](LICENSE-COMMERCIAL.md) — covers VAULT software only, not Qwen2.5-3B weights).
- **Model Weights (`*.safetensors`, `*.vlt3`):** **Qwen RESEARCH LICENSE AGREEMENT** ([`LICENSE-QWEN-RESEARCH`](LICENSE-QWEN-RESEARCH)) — *Built with Qwen.* Full upstream dataset attribution is provided in [`NOTICE`](NOTICE).
"""


VLT3_COMPANION_README = """---
license: other
license_name: split-agpl3-qwen-research
license_link: LICENSE
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
  - vault-v3
  - vlt3
  - c99
  - freestanding
  - arm-neon
  - avx2
---

# VAULT-v3-3B-PRIMAL-VLT3 — Standalone Freestanding C99 `.vlt3` Binaries (`1.805 GiB` `P4_0` & `3.064 GiB` `Q8_0`)

> **Built with Qwen.** Companion native `.vlt3` binary repository for [`batteryphil/VAULT-v3-3B-PRIMAL`](https://huggingface.co/batteryphil/VAULT-v3-3B-PRIMAL), separated so `transformers` users download only `5.765 GiB` and C99 CPU/edge users download only `1.805 GiB` (`1,848.5 MiB`)!
>
> **License Scope:** `runtime/*` is dual-licensed under GNU AGPL-3.0-or-later OR the VAULT Commercial Software License (software only). The `.vlt3` neural network weights are derived from `Qwen/Qwen2.5-3B-Instruct` and governed by Alibaba Cloud's Qwen RESEARCH LICENSE AGREEMENT (non-commercial research unless separately licensed from Alibaba Cloud).

## Run in 2 Commands (Linux `x86_64`/`ARM64`, macOS Intel & Apple Silicon `M1–M4` NEON, WSL)
```bash
make -C runtime
./runtime/vault3_host vault_v3_3b_p4_0_primal.vlt3 --chat --prompt "Hello from freestanding C99!"
```
"""


def main():
    t0 = time.time()
    # Clean remove any old root .vlt3 files inside OUT_DIR so OUT_DIR is strictly 5.75 GiB!
    if os.path.isdir(OUT_DIR):
        for fn in os.listdir(OUT_DIR):
            if fn.endswith(".vlt3"):
                os.remove(os.path.join(OUT_DIR, fn))

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "runtime"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "tools"), exist_ok=True)
    os.makedirs(os.path.join(OUT_VLT3_DIR, "runtime"), exist_ok=True)

    print(f"[BUILD] Assembling split HF repo ({OUT_DIR}) + VLT3 companion repo ({OUT_VLT3_DIR})...", flush=True)

    vf = vlt3.read_vlt3(Q8_VLT3_SRC, check_crc=True)
    h = vf["header"] if isinstance(vf, dict) else vf.header
    raw_sd = vlt3.vlt3_to_state_dict(vf)
    assert "seg_scorer.ln.weight" in raw_sd, "Missing seg_scorer in Q8_0 .vlt3!"

    shard1: Dict[str, torch.Tensor] = {}
    shard2: Dict[str, torch.Tensor] = {}
    shard3: Dict[str, torch.Tensor] = {}
    wmap: Dict[str, str] = {}
    total_size = 0

    for k, t in raw_sd.items():
        hf_k = f"model.{k}"
        if k.startswith("seg_scorer."):
            ten = t.detach().float().contiguous()
            shard3[hf_k] = ten
            wmap[hf_k] = "model-00003-of-00003.safetensors"
        else:
            ten = t.detach().to(torch.bfloat16).contiguous()
            is_s1 = (k == "embed_tokens.weight") or (k.startswith("layers.") and int(k.split(".")[1]) < 18)
            if is_s1:
                shard1[hf_k] = ten
                wmap[hf_k] = "model-00001-of-00003.safetensors"
            else:
                shard2[hf_k] = ten
                wmap[hf_k] = "model-00002-of-00003.safetensors"
        total_size += ten.numel() * ten.element_size()

    save_file(shard1, os.path.join(OUT_DIR, "model-00001-of-00003.safetensors"), metadata={"format": "pt"})
    save_file(shard2, os.path.join(OUT_DIR, "model-00002-of-00003.safetensors"), metadata={"format": "pt"})
    save_file(shard3, os.path.join(OUT_DIR, "model-00003-of-00003.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": wmap}, f, indent=2)

    seg_layers = list(h["seg_layers"][: h["seg_n_feat"]])
    config_json = {
        "architectures": ["VAULTv3HFForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_vault_v3.VAULTv3HFConfig",
            "AutoModelForCausalLM": "modeling_vault_v3.VAULTv3HFForCausalLM",
        },
        "model_type": "vault_v3",
        "vocab_size": h["vocab_size"],
        "hidden_size": h["d_model"],
        "intermediate_size": h["d_ff"],
        "num_hidden_layers": h["n_layers"],
        "num_attention_heads": h["n_heads"],
        "num_key_value_heads": h["n_kv_heads"],
        "head_dim": h["head_dim"],
        "rms_norm_eps": h["rms_eps"],
        "rope_theta": h["rope_theta"],
        "max_position_embeddings": 131072,
        "tie_word_embeddings": bool(h["flags"] & 1),
        "window": h["window"],
        "n_sink": h["n_sink"],
        "max_sink": 512,
        "seg_budget": h["seg_budget"],
        "seg_hidden": h["seg_hidden"],
        "seg_min": h["seg_min"],
        "seg_max": h["seg_max"],
        "seg_layers": seg_layers,
        "enable_shingle_bias": True,
        "bos_token_id": h["bos_id"],
        "eos_token_id": h["im_end_id"],
        "pad_token_id": h["pad_id"],
        "im_start_id": h["im_start_id"],
        "im_end_id": h["im_end_id"],
        "system_tok_id": 8948,
        "torch_dtype": "bfloat16",
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(config_json, f, indent=2)

    gen_cfg = {
        "bos_token_id": h["bos_id"],
        "eos_token_id": [h["im_end_id"], h["bos_id"]],
        "pad_token_id": h["pad_id"],
        "do_sample": False,
        "max_new_tokens": 1024,
    }
    with open(os.path.join(OUT_DIR, "generation_config.json"), "w") as f:
        json.dump(gen_cfg, f, indent=2)

    with open(os.path.join(VAULT_ROOT, "src", "vault_v3_model.py"), encoding="utf-8") as f:
        core_src = f.read()
    old_imp = "from torch.nn.attention.bias import causal_lower_right"
    safe_imp = (
        "try:\n"
        "    from torch.nn.attention.bias import causal_lower_right\n"
        "except ImportError:\n"
        "    def causal_lower_right(q_len: int, kv_len: int):\n"
        "        q = torch.arange(q_len).unsqueeze(1)\n"
        "        k = torch.arange(kv_len).unsqueeze(0)\n"
        "        return k <= (q + (kv_len - q_len))\n"
    )
    assert old_imp in core_src, "Expected causal_lower_right import not found"
    core_src = core_src.replace(old_imp, safe_imp, 1)
    with open(os.path.join(OUT_DIR, "vault_v3_core.py"), "w", encoding="utf-8") as f:
        f.write(core_src)

    with open(os.path.join(OUT_DIR, "configuration_vault_v3.py"), "w", encoding="utf-8") as f:
        f.write(CONFIGURATION_VAULT_V3_PY)
    with open(os.path.join(OUT_DIR, "modeling_vault_v3.py"), "w", encoding="utf-8") as f:
        f.write(MODELING_VAULT_V3_PY)
    with open(os.path.join(OUT_DIR, "run_vault_v3.py"), "w", encoding="utf-8") as f:
        f.write(RUN_VAULT_V3_PY)
    os.chmod(os.path.join(OUT_DIR, "run_vault_v3.py"), 0o755)
    with open(os.path.join(OUT_DIR, "tools", "export_qwen2_compat.py"), "w", encoding="utf-8") as f:
        f.write(EXPORT_QWEN2_COMPAT_PY)
    os.chmod(os.path.join(OUT_DIR, "tools", "export_qwen2_compat.py"), 0o755)

    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        sp = os.path.join(QWEN_3B_DIR, fn)
        if os.path.isfile(sp):
            shutil.copy2(sp, os.path.join(OUT_DIR, fn))

    # Copy updated freestanding C99 runtime/ (with ARM64 NEON + AVX2 + Clang clean fixes!) to BOTH repos
    for target_root in (OUT_DIR, OUT_VLT3_DIR):
        for fn in ("vault3.c", "vault3.h", "vault3_host.c", "vault3_nfc_tables.h"):
            shutil.copy2(os.path.join(VAULT_ROOT, "runtime", fn), os.path.join(target_root, "runtime", fn))
        with open(os.path.join(target_root, "runtime", "Makefile"), "w", encoding="utf-8") as f:
            f.write(PORTABLE_MAKEFILE)

    # Link .vlt3 binaries ONLY into the companion repo OUT_VLT3_DIR (keeping OUT_DIR at 5.75 GiB!)
    _copy_or_link(P4_VLT3_SRC, os.path.join(OUT_VLT3_DIR, "vault_v3_3b_p4_0_primal.vlt3"))
    _copy_or_link(Q8_VLT3_SRC, os.path.join(OUT_VLT3_DIR, "vault_v3_3b_q8_0_primal.vlt3"))

    # Copy split licenses + NOTICE into BOTH repos
    apache_src = os.path.join(VAULT_ROOT, "LICENSE-APACHE-2.0-Qwen")
    for target_root in (OUT_DIR, OUT_VLT3_DIR):
        with open(os.path.join(target_root, "LICENSE"), "w", encoding="utf-8") as f:
            f.write(SPLIT_ROOT_LICENSE)
        shutil.copy2(os.path.join(QWEN_3B_DIR, "LICENSE"), os.path.join(target_root, "LICENSE-QWEN-RESEARCH"))
        if os.path.isfile(apache_src):
            with open(apache_src, encoding="utf-8") as af:
                ap_lines = af.read().splitlines()
            clean_ap = "\n".join(ap_lines[:203]).rstrip() + (
                "\n\nApplies to: Hugging Face Python wrapper and export utilities "
                "(`configuration_vault_v3.py`, `modeling_vault_v3.py`, `run_vault_v3.py`, "
                "`tools/export_qwen2_compat.py`).\n"
            )
            with open(os.path.join(target_root, "LICENSE-CODE-APACHE-2.0"), "w", encoding="utf-8") as af_out:
                af_out.write(clean_ap)
        for fn in ("LICENSE-AGPL-3.0", "LICENSE-COMMERCIAL.md"):
            sp = os.path.join(VAULT_ROOT, fn)
            if os.path.isfile(sp):
                shutil.copy2(sp, os.path.join(target_root, fn))
        with open(os.path.join(target_root, "NOTICE"), "w", encoding="utf-8") as f:
            f.write(NOTICE_TEXT)

    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(build_readme_main())
    with open(os.path.join(OUT_VLT3_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(VLT3_COMPANION_README)

    main_gib = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(OUT_DIR)
        for fn in fns
    ) / (1024 ** 3)
    vlt3_gib = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(OUT_VLT3_DIR)
        for fn in fns
    ) / (1024 ** 3)
    print(f"[DONE] Built main HF repo ({main_gib:.2f} GiB) + companion VLT3 repo ({vlt3_gib:.2f} GiB) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
