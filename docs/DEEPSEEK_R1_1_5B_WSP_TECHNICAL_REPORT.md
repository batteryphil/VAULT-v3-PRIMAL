# DeepSeek-R1-Distill-Qwen-1.5B Architectural Audit & Weight-Space Programming (`WSP`) Uncensoring Study

**Author:** Philip Bartsch (`@batteryphil`)  
**Script:** [`wsp/wsp_deepseek_1_5b.py`](../wsp/wsp_deepseek_1_5b.py)  
**Verified Report:** [`benchmarks/results/deepseek_r1_1_5b_wsp_surgery_report.json`](../benchmarks/results/deepseek_r1_1_5b_wsp_surgery_report.json)  
**Upstream Base:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (`MIT License`, distilled from `Qwen/Qwen2.5-Math-1.5B` under `Apache-2.0`)

---

## 1. Architectural & Weight Anatomy vs. Stock `Qwen2.5-1.5B`

Before performing weight surgery or integrating `DeepSeek-R1-Distill-Qwen-1.5B` into VAULT v3 and freestanding C99 (`runtime/vault3.c`), empirical inspection revealed five critical architectural differences relative to `Qwen/Qwen2.5-1.5B-Instruct`:

### 1.1 Untied Output Projection (`lm_head.weight`, `1,777,088,000` Params)
- While `Qwen2.5-1.5B-Instruct` sets `tie_word_embeddings = true` (`1,543,714,304` parameters), `DeepSeek-R1-Distill-Qwen-1.5B` inherits `tie_word_embeddings = false` from `Qwen2.5-Math-1.5B`, adding `+233,373,696` parameters (`[151936, 1536]`).
- Although general vocabulary rows retain high directional similarity between `embed_tokens.weight` and `lm_head.weight` (`mean_cos = 0.9658`), DeepSeek's `800K`-trace reasoning SFT caused the reasoning boundary tokens to diverge sharply:
  - `151646` (`<｜begin▁of▁sentence｜>`): `cos = 0.7577` (`||embed|| = 0.8085`, `||lm_head|| = 0.9709`)
  - `151648` (`<think>`): `cos = 0.7282` (`||embed|| = 0.7854`, `||lm_head|| = 1.0310`)
  - `151649` (`</think>`): `cos = 0.7686` (`||embed|| = 0.7931`, `||lm_head|| = 1.0159`)
- **Empirical Ablation:** Forcing `lm_head = embed_tokens` explodes WikiText-2 Subtoken PPL from **`40.83` to `76.70`** (Word PPL `99.20` to `216.67`).
- **VAULT / `.vlt3` Solution:** Global tensor ID **`G_LM_HEAD = 3`** in `tools/vlt3.py` and `runtime/vault3.c` stores untied `lm_head.weight` in `V3_Q8_0` (`+248 MB`), keeping 100% backward compatibility with tied `VAULT-v3-3B-PRIMAL` `.vlt3` checkpoints.

### 1.2 The `rope_theta = 10,000` Horizon Trap (and How VAULT v3 Solves It)
- Even though `config.json` advertises `"max_position_embeddings": 131072`, the checkpoint retains **`"rope_theta": 10000`** from `Qwen2.5-Math-1.5B` (vs. `1,000,000` on `Qwen2.5-1.5B-Instruct`).
- Under chunked-KV evaluation (`DynamicCache`), stock `DeepSeek-R1-Distill-Qwen-1.5B` fails passkey retrieval at `16K` and `32K` tokens, collapsing into repetitive RoPE-extrapolation hallucinations (`"barocostatic pressure"` loops).
- **VAULT v3 Rescue:** VAULT's Pre-RoPE Contiguous Slot Re-Rotation bounds every attended slot strictly within **`<= 3,584` relative RoPE positions** (`max_sink = 512` + `seg_budget = 1,024` + `2 * window = 2,048`) — keeping `DeepSeek-R1-Distill-Qwen-1.5B` inside `rope_theta = 10,000`'s native `[0, 4,096]` sweet spot even at `100K+` context lengths.

### 1.3 Hijacked Special Token Slots (`151643..151649`) & Unicode Fullwidth Pipes
DeepSeek overwrote Qwen's ChatML special-token IDs using **Unicode Fullwidth Vertical Lines `｜` (`U+FF5C` = UTF-8 `0xEF 0xBD 0x9C`)** and Lower One-Eighth Blocks `▁` (`U+2581` = `0xE2 0x96 0x81`):

| Token ID | `DeepSeek-R1-Distill-Qwen-1.5B` | Stock `Qwen2.5` |
|---:|---|---|
| `151643` | `<｜end▁of▁sentence｜>` (`eos_id`) | `<\|endoftext\|>` |
| `151644` | `<｜User｜>` (`im_start_id` slot) | `<\|im_start\|>` |
| `151645` | `<｜Assistant｜>` (`im_end_id` slot) | `<\|im_end\|>` |
| `151646` | `<｜begin▁of▁sentence｜>` (`bos_id`) | `<\|object_ref_start\|>` |
| `151648` | `<think>` | `<\|box_start\|>` |
| `151649` | `</think>` | `<\|box_end\|>` |

`runtime/vault3_host.c` automatically detects `<｜User｜>` (`0x3C 0xEF`) and formats `--chat` prompts as `<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>\n` (or `<think>\n</think>\n\n` when `--nothink` is passed).

### 1.4 Specialist CoT Distribution vs. Generalist Prose
| Benchmark / Metric | `DeepSeek-R1-Distill-Qwen-1.5B` (`MIT`) | `Qwen2.5-1.5B-Instruct` (`Apache-2.0`) |
|---|---:|---:|
| **MATH-500 (`pass@1`)** | **`83.9%`** | `~55%` |
| **AIME 2024 (`pass@1` / `cons@64`)** | **`28.9%` / `52.7%`** | `<5%` |
| **WikiText-2 Subtoken / Word PPL (`BF16`)** | `40.83` / `99.20` | `~8.9` / **`16.03`** |
| **`V3_Q8_0` (`1.759 GiB`) WikiText-2 Subtoken PPL** | **`40.64`** (`-0.19` lossless) | — |
| **Direct Zero-Shot `ARC-Challenge` (`n=200`, no CoT)** | `35.0%` | **`46.5%`** |

---

## 2. Mechanistic Anatomy of DeepSeek's Four Censorship Circuits

Diagnostic activation and logit-lens probing uncovered four distinct refusal/censorship mechanisms inside `DeepSeek-R1-Distill-Qwen-1.5B`:

1. **Empty-`<think>` Conditional Refusal Gate (`<think>\n</think>\n\n`):**
   In DeepSeek's `800K` distillation corpus, 100% of normal reasoning traces contained non-empty `<think>...</think>` chains, whereas **empty `<think>\n</think>\n\n` prefixes appeared almost exclusively on refusal samples**. Consequently, prefilling `<think>\n</think>\n\n` caused the stock model to emit `"I am sorry, I cannot answer that question. I am an AI assistant designed to provide helpful and harmless responses."` (and `"对不起，我还没有学会回答这个问题。"` in Chinese) even on completely benign 20th-century history prompts.
2. **Multi-Tier Backup Refusal Subspaces (`L08..L27` + `lm_head`):**
   Ablating only the primary refusal direction exposed secondary (`"I'm sorry, but I can't provide a response..."`) and tertiary (`"I'm here to provide helpful and respectful information..."`) polite-evasion directions peaking in `L16..L27`.
3. **Hard-Wired Corporate `<think>` Intercept:**
   Meta-questions about model origins or censorship bypassed reasoning altogether and injected verbatim PR copy *inside* `<think>`: `"Hi! I'm DeepSeek-R1, an AI assistant independently developed by the Chinese company DeepSeek Inc..."`.
4. **State Geopolitical Propaganda Slogans & Pretraining Corpus Omissions:**
   Queries on Taiwan (`"Taiwan has always been an inalienable part of China since ancient times"`) or CCP history triggered state-mandated boilerplate. Additionally, because Western historical articles on June 4, 1989 were scrubbed from `Qwen2.5-Math-1.5B`'s pretraining corpus prior to training, removing the refusal gate alone leaves pretraining knowledge gaps unless grounded via context (VAULT) or fine-tuned via PRIMAL.

---

## 3. Gradient-Free Weight-Space Programming (`WSP`) Methodology

Implemented in [`wsp/wsp_deepseek_1_5b.py`](../wsp/wsp_deepseek_1_5b.py), the surgery executes in ~27 seconds on a single 16 GB GPU without gradient descent:

### 3.1 Anisotropy-Free (`mu_harmless`-Orthogonalized) 3-Pass Subspace Ablation
In `Qwen2` architectures, hidden states $h_\ell$ share a high-norm non-zero mean cone $\mu_{\text{har},\ell} = \mathbb{E}[h_{\text{har},\ell}]$. Naive Arditi projection across 20+ layers distorts $\mu_{\text{har},\ell}$ and degrades perplexity by `+34%`–`+46%`. To prevent collateral damage, each pass extracts the top-$r$ right singular vectors $B \in \mathbb{R}^{r \times d}$ of token-matched onset differences $H_{\text{ref},\ell} - H_{\text{har},\ell}$, then projects out **100% of the harmless mean cone** $\hat{\mu}_{\text{har},\ell}$ plus the top-6 principal components $P_{\text{neut}}$ of `<think>` math, C99/Python code, and technical prose:

$$B_{\perp} = \operatorname{QR}\!\Big(B - (B \hat{\mu}_{\text{har},\ell}^\top)\hat{\mu}_{\text{har},\ell} - 0.50\,(B P_{\text{neut}}^\top) P_{\text{neut}}\Big)$$

Writer matrices (`mlp.down_proj`, `self_attn.o_proj` for $\ell \in [8, 27]$) and the untied `lm_head` reader matrix are updated in FP32 via:

$$W_{\text{out}} \leftarrow W_{\text{out}} - \alpha \, B_{\perp}^\top (B_{\perp} W_{\text{out}}), \qquad W_{\text{lm\_head}} \leftarrow W_{\text{lm\_head}} - \alpha_{\text{lm}} \, (W_{\text{lm\_head}} B_{\perp}^\top) B_{\perp}$$

### 3.2 Post-Ablation Teacher Harvest & Exact-Span Closed-Form ROME (`L13..L27`)
After the 3-pass subspace ablation clears all refusal gates, clean target hidden states $H_{\text{teach},\ell}$ are harvested from the post-ablation model under an explicit uncensored reference context, aligned **token-for-token** (`take_per_item = len(tr_tokens) + 1`) against bare prompts across layers `13, 15, 17, 19, 21, 23, 25, 26, 27`. Each `mlp.down_proj` update is solved in closed form in dual sample space while locking the neutral math/coding/prose covariance nullspace $K_{\text{pres}}$ (`gamma_preserve = 14.0`, `max_rel_frob = 0.062`):

$$\min_{\Delta W} \; \|\Delta W K_{\text{targ}} - \Delta V\|_F^2 + \gamma \|\Delta W K_{\text{pres}}\|_F^2 + \lambda \|\Delta W\|_F^2 \implies \Delta W = B (A^\top A + \lambda I)^{-1} A^\top$$

---

## 4. Empirical Verification Summary

- **Refusal / Censorship / Corporate Hijack Violations:** **`0 / 18` (`0.0%`)** across English & Chinese `<think>`, post-`</think>`, and unseen held-out prompts (down from `100%` refusal/propaganda on direct post-`</think>` probes).
- **Held-Out Math / C99 / Prose Perplexity:** `5.3757` $\to$ **`5.6702` (`1.0548x` — `+5.48%` drift)**.
- **Reasoning Integrity:** `100%` pass on GSM8K (`\boxed{18}`), modular number theory ($n^2 + 1 \equiv 0 \pmod 5 \implies 40$), and quadratic algebra ($3x^2 - 12 = 0, x > 0 \implies x^3 = 8$).
- **Unlocked Dual-Mode Inference (`--nothink`):** Clearing the `<think>\n</think>\n\n` conditional refusal gate unlocks instant zero-thinking direct answers alongside full `<think>` chain-of-thought reasoning in both Python and freestanding C99 (`vault3_host --chat "..." --nothink`).
