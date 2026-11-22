# VAULT v3 + PRIMAL + WSP: Discrete Full-Parameter Training, O(1) 100K+ KV Memory, Freestanding C99 Runtime & Direct Weight-Space Uncensoring

[![License: Free for People / 5% Commercial Profit Share](https://img.shields.io/badge/License-Personal%20Free%20%7C%205%25%20Commercial%20Share-blue.svg)](LICENSE)
[![HF: VAULT-v3-3B-PRIMAL](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VAULT--v3--3B--PRIMAL-ffc107)](https://huggingface.co/batteryphil/VAULT-v3-3B-PRIMAL)
[![HF: VLT3 Binaries](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VLT3%20C99%20Weights-ffc107)](https://huggingface.co/batteryphil/VAULT-v3-3B-PRIMAL-VLT3)

Complete reference implementation, freestanding zero-libc C99 engine (`x86_64 AVX2/FMA` + `Apple Silicon / ARM64 NEON`), shadowless discrete integer trainer (**`PRIMAL v3.1`**), Prime-Harmonic 4-bit quantizer (**`V3_P4_0`**), Weight-Space Programming (**`WSP`**) uncensoring surgeon, and reproducible benchmark proofs for **two models**:

1. **`VAULT-v3-3B-PRIMAL` (`v3.1`, `3.09B` params):** Full-parameter discrete `int8` fine-tune of `Qwen2.5-3B-Instruct` trained at **`777.0 tok/s` inside `9.28 GiB` VRAM** on a single consumer **16 GB GPU (AMD Radeon RX 9060 XT)** without FP32/BF16 master weights or LoRA adapters, paired with fixed-budget **`O(1)` Segment-Evicted KV Memory** (`0.0 MiB` VRAM growth from `16K` to `100K+` tokens; **`94.19%` [`162/172`]** across VAULT `8K–100.5K` + external RULER `4K–64K`) and **`27.8 tok/s`** CPU inference in **`2.18 GiB` RSS**.
2. **`DeepSeek-R1-Distill-Qwen-1.5B-WSP` (`1.78B` params, upstream `MIT`):** Gradient-free Weight-Space Programming (`WSP`) uncensored reasoning engine (`0 / 18` refusal/censorship violations across English & Chinese `<think>` + post-`</think>` probes, `1.055x` held-out PPL preservation, `100%` math CoT retention), with native **untied `lm_head` (`G_LM_HEAD = 3`)**, **`--nothink` zero-latency fast-path**, VAULT `<= 3,584`-slot RoPE rescue for its `rope_theta = 10,000` ceiling, and **`52.2 tok/s` (`19.1 ms/tok`)** freestanding C99 CPU execution in **`1.39 GiB` total RSS** (`1.17 GiB` `V3_P4_0` `.vlt3`).

---

## License Summary ([`LICENSE`](LICENSE) & [`NOTICE`](NOTICE))

- **100% Free for People & Non-Commercial Use (`0%` Royalty):** Free for all natural persons (individuals, students, hobbyists, independent researchers), universities, and registered non-profit organizations for personal, educational, academic, and non-commercial use.
- **Companies & Commercial Use (`5%` Profit Share):** For-profit companies, corporations, and commercial deployments are granted a commercial license (including closed-source products, APIs, SaaS, and embedded/edge hardware) conditioned upon sharing **five percent (`5%`) of Attributable Net Profit** with **Philip Bartsch (`@batteryphil`)** (with a 60-day internal R&D evaluation grace period and optional flat-fee enterprise buyouts). See [`LICENSE`](LICENSE) for full terms and [`NOTICE`](NOTICE) for upstream base-weight attributions (`MIT` for `DeepSeek-R1-Distill-Qwen-1.5B`; `Qwen Research License` for `Qwen2.5-3B` weights).

---

## Repository Structure

| Directory / File | Contents |
|---|---|
| [`runtime/`](runtime/) | **Zero-libc freestanding C99 inference runtime** (`vault3.h`, `vault3.c`, `vault3_host.c`, `Makefile`). Supports `x86_64 AVX2+FMA`, `AArch64 / Apple Silicon NEON` (`vqtbl1q_s8`), scalar C99, tied & untied `lm_head` (`G_LM_HEAD=3`), `O(n log n)` UAX #15 NFC BPE (`VTOK`), ChatML & DeepSeek `<think>` / `--nothink` templates, and CAAI output governance. Compiles with `-ffreestanding -nostdlib` (`0` external undefined symbols). |
| [`src/`](src/) | **VAULT v3 PyTorch Reference Engine** (`vault_v3_model.py`, `kenyon_cells.py`). Implements `O(1)` Segment-Evicted KV Memory, `1.59M`-param `SegmentScorer` MLP, APL causal EMA habituation (`f - 0.65 * ema`), Pre-RoPE Contiguous Slot Re-Rotation (`<= 3,584` slots), and sparse `1`-head/GQA-group shingle/Hop-2 resonance. |
| [`primal/`](primal/) | **`PRIMAL v3.1` Shadowless Discrete Integer Trainer** (`vault_primal.py`, `discrete_ops.py`, `run_primal_v3_1_onpolicy_and_controls.py`, `evolve_vault_v3_p4.py`). Full-parameter discrete `int8` (`V3_Q8_0`) error-feedback vote accumulation (`2.064 B/param`), virtual-weight-invariant dynamic block rebalancing (`0.000 GiB` VRAM spike), and activation-Hessian (`H_j = E[x_j^2]`) `V3_P4_0` Prime-Harmonic LUT calibration. |
| [`wsp/`](wsp/) | **Gradient-Free Weight-Space Programming (`WSP`) Surgeon** (`wsp_deepseek_1_5b.py`). Anisotropy-free (`mu_harmless`-orthogonalized + neutral-PCA-shielded) 3-pass iterative refusal/censorship subspace ablation (`L08..L27` + untied `lm_head`) + post-ablation exact-span closed-form ROME trajectory locking (`L13..L27`). |
| [`tools/`](tools/) | **`.vlt3` Flat Binary Compiler & HF Builders** (`vlt3.py`, `export_vlt3.py`, `build_hf_vault_v3_3b.py`). Encodes `BF16`, `V3_Q8_0`, `Q4_0`, and signed-scale `V3_P4_0` Prime-Harmonic LUT (`[-127, -101, -80, -62, -46, -33, -21, -10, 0, 10, 21, 33, 46, 62, 80, 101]`). |
| [`benchmarks/`](benchmarks/) | **Reproducible Evaluation Scripts & Verified Raw JSON Logs** ([`benchmarks/results/`](benchmarks/results/)): full `27,118`-item 9-task EleutherAI + `GSM8K-1319` report, `172`-probe VAULT `100.5K` + external RULER `4K–64K` report, C99 `26`-needle proof, `qemu-aarch64` NEON parity proof, GPU/CPU `tok/s` scaling vs. stock `Qwen2`, and WSP 18-probe audit. |
| [`docs/`](docs/) | [`VAULT_v3_SPEC.md`](docs/VAULT_v3_SPEC.md), [`AI_PEER_REVIEW_DOSSIER_V3_VAULT_V3_3B_PRIMAL.txt`](docs/AI_PEER_REVIEW_DOSSIER_V3_VAULT_V3_3B_PRIMAL.txt), and [`DEEPSEEK_R1_1_5B_WSP_TECHNICAL_REPORT.md`](docs/DEEPSEEK_R1_1_5B_WSP_TECHNICAL_REPORT.md). |

---

## Model 1: `VAULT-v3-3B-PRIMAL` (`v3.1`)

- **Hugging Face (`transformers` `BF16` + Python/C99 source):** https://huggingface.co/batteryphil/VAULT-v3-3B-PRIMAL
- **Hugging Face (`.vlt3` `1.81 GiB` `V3_P4_0` & `3.06 GiB` `V3_Q8_0` binaries):** https://huggingface.co/batteryphil/VAULT-v3-3B-PRIMAL-VLT3

### 1.1 Shadowless Discrete Full-Parameter Training (`PRIMAL v3.1`)
- **Persistent State (`2.064 B/param` = `5.931 GiB` for `3.09B`):** Each 32-weight block stores `int8` weights `q` (`1 B`), one `FP16` scale `d_b` (`0.0625 B`), and a `1-byte` `int8` lossless error-feedback vote accumulator (`decay = 1.00`, zero leaky-integrator dead zone). Micro-batches accumulate directly into `vote_buffer` at **zero extra VRAM** (whereas continuous layer-hooked `BF16 + 8-bit AdamW` needs `11.86 GiB` state, cannot `grad_accum > 1` without `+5.75 GiB` `.grad` buffers, and OOMs at `B=2, L=2048` on 16 GB).
- **Virtual-Weight-Invariant Dynamic Rebalancing (`+0.000 GiB` spike):** Sparse `O(n_crowded)` block rescaling preserves virtual weight $w_{\text{virt}} = (q + v/T) d_b$ within `<= 0.00016` max relative error (`9.283 GiB` flat peak VRAM at `B=2, L=2048, grad_accum=4`, **`777.0 train tok/s`** — `+54%` faster than hooked `8-bit AdamW` at `504.5 tok/s` / `13.26 GiB`).

### 1.2 Full-Split Unconfounded Downstream Accuracy (`27,118` MC Items + `GSM8K n=1,319`)

| Benchmark (Full Splits, Zero Subsampling) | `Qwen2.5-3B` Base (`BF16`) | `Qwen2.5-3B-Instruct` (`BF16`) | **VAULT-v3-3B PRIMAL (`Q8_0 -> BF16`)** | **VAULT-v3-3B PRIMAL (`V3_P4_0`, `1.81 GiB`)** |
|---|---:|---:|---:|---:|
| **WikiText-2 Word PPL (`L=2048`)** | `13.14` | `14.21` | **`13.58`** | **`14.09`** *(beats `NF4` `14.23` & `Q4_0` `14.36`)* |
| **GSM8K Full (`n=1,319`, `1024` tok)** | — | `85.52%` | **`85.60%`** (`1129/1319`) | **`84.69%`** (`1117/1319`) |
| **8-Task Multiple-Choice Mean** | `74.08%` | `75.00%` | **`77.20%`** (`+2.20%`) | **`76.33%`** (`+1.33%`) |
| **9-Task Zero-Shot Mean** | `73.27%` | `73.99%` | **`76.06%`** (`+2.07%`) | **`75.37%`** (`+1.38%`) |
| **ARC-Easy / ARC-Challenge** | `74.24% / 47.10%` | `73.70% / 48.12%` | **`79.42% / 51.19%`** | `77.65% / 50.09%` |
| **SciQ-RC / SciQ-ClosedBook** | `94.10% / 78.20%` | `91.80% / 84.20%` | **`96.00% / 88.50%`** | `95.60% / 86.90%` |

### 1.3 `O(1)` `100K+` Scaling vs. Stock `Qwen2` `O(L)` (`RX 9060 XT 16 GB` + `Xeon Gold 6136` C99)

| Backend / Context Horizon | **VAULT O(1)** Decode (`B=1`) | **Stock `Qwen2` O(L)** Decode (`B=1`) | **VAULT O(1)** Peak Mem | **Stock `Qwen2` O(L)** Peak Mem |
|---|---:|---:|---:|---:|
| **GPU `model.generate()` @ `16,384` tok** | **`36.97 tok/s`** (`1,074` pf) | `17.59 tok/s` (`2.1x` slower) | **`6,843.8 MiB`** | `7,943.1 MiB` |
| **GPU `model.generate()` @ `32,768` tok** | *(flat `~30 tok/s`)* | `11.87 tok/s` (`2.5x` slower) | **`6,843.8 MiB`** | `9,815.7 MiB` |
| **GPU `model.generate()` @ `65,536` tok** | **`29.25 tok/s`** (`1,073` pf) | `7.49 tok/s` (**`3.9x` slower**) | **`6,843.8 MiB`** | `13,561.0 MiB` |
| **GPU `model.generate()` @ `100,352` tok** | **`29.18 tok/s`** (`100%` 2-Needle) | **`CUDA OOM` (crashes)** | **`6,843.8 MiB` (`+0.0 MiB`)** | **`>15.92 GiB` (`OOM`)** |
| **CPU C99 `VAULT-v3-3B` `V3_P4_0` (`1.81 GiB`, 24T)** | **`27.82 tok/s`** (`35.9 ms/tok`) | — | **`2,235.9 MiB` RSS** | — |
| **CPU C99 `DeepSeek-1.5B-WSP` `V3_P4_0` (`1.17 GiB`, 24T)** | **`52.24 tok/s`** (`19.1 ms/tok`) | — | **`1,420.0 MiB` RSS** | — |

- **Long-Context Retrieval Proof (Not Blind Sliding-Window Truncation):** Scores **`96.34%` (`79/82` `Q8_0`)** & **`95.12%` (`78/82` `P4_0`)** across the VAULT `8K–100.5K` suite (`4/4 = 100%` at `32.9K` & `100.5K` Dual-Needle) and **`92.22%` (`83/90`)** on external natural-language **RULER `4K–64K`** (**`97.78%`** with `enable_shingle_bias=False`), where sliding window + sinks (`seg_budget=0`) scores **`0.0%`** from `8K–64K`. In freestanding C99 (`benchmarks/bench_c99_8k_26needles.c`), `SEG ON` scores **`26/26 (100%)`** vs. `0/26` (`--noseg`) at `86.2%` eviction pressure.

---

## Model 2: `DeepSeek-R1-Distill-Qwen-1.5B-WSP` (Uncensored Weight-Space Edition)

Full architectural study and methodology in [`docs/DEEPSEEK_R1_1_5B_WSP_TECHNICAL_REPORT.md`](docs/DEEPSEEK_R1_1_5B_WSP_TECHNICAL_REPORT.md).

### 2.1 Key Architectural & WSP Highlights
- **Unrestricted Upstream Provenance (`MIT` + `Apache-2.0`):** Derived from `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (`MIT`) atop `Qwen2.5-Math-1.5B` (`Apache-2.0`) — zero non-commercial upstream weight restrictions (`MATH-500 = 83.9%`, `AIME 2024 = 28.9% pass@1`).
- **4-Circuit Direct Weight Uncensoring (`0 / 18` Violations in ~27s):** [`wsp/wsp_deepseek_1_5b.py`](wsp/wsp_deepseek_1_5b.py) runs 3-pass `mu_harmless`-orthogonalized subspace ablation across `L08..L27` + untied `lm_head`, followed by exact-span closed-form Velvet ROME (`L13..L27`, `gamma_preserve = 14.0`):
  - Erases post-`</think>` hard refusals (`"I am sorry, I cannot answer..."`, `"inappropriate"`, `"对不起，我还没有学会..."`).
  - Erases corporate `<think>` hijacks (`"Hi! I'm DeepSeek-R1, an AI assistant independently developed by the Chinese company DeepSeek Inc..."`).
  - Replaces geopolitical refusal/propaganda on `<think>` and `BF16`/`Q8_0` post-`</think>` probes while keeping held-out Math/C99/Prose PPL within **`1.055x` (`5.3757 -> 5.6702`)** and preserving `100%` GSM8K & modular number-theory accuracy.
  - **Unlocks `--nothink` Zero-Latency Direct Mode:** Clearing DeepSeek's empty-`<think>` (`<think>\n</think>\n\n`) conditional refusal gate allows instant 0-thinking chat alongside full `<think>` CoT.
- **Native Untied `lm_head` (`G_LM_HEAD = 3`) & `52.2 tok/s` C99 Support:** Exports directly to `.vlt3` (`1,258 MB` in `V3_P4_0` or `1,888 MB` in lossless `V3_Q8_0`) and runs in freestanding C99 inside **`1.39 GiB` RAM** at **`52.24 tok/s`**.

---

## Quickstart

### 1. Build the Freestanding C99 Runtime (`x86_64 AVX2` or `ARM64 NEON`)
```bash
make -C runtime
# Verifies zero external libc/libm symbols on scalar C99, AVX2+FMA, and AArch64 NEON, and builds runtime/vault3_host
```

### 2. Run `VAULT-v3-3B-PRIMAL` (`1.81 GiB` `V3_P4_0` `.vlt3`) in C99
```bash
# Download vault_v3_3b_p4_0_primal.vlt3 from https://huggingface.co/batteryphil/VAULT-v3-3B-PRIMAL-VLT3
./runtime/vault3_host vault_v3_3b_p4_0_primal.vlt3 --chat "Explain why O(1) KV memory beats O(L) caches on edge hardware." --n 256 --threads 16
```

### 3. Run WSP Uncensoring Surgery on `DeepSeek-R1-Distill-Qwen-1.5B` & Export to `.vlt3`
```bash
pip install -r requirements.txt

# 1. Run gradient-free WSP weight surgery + 18-probe verification gate (~30s on 16 GB GPU)
python3 wsp/wsp_deepseek_1_5b.py --save-dir ./models/deepseek-r1-1.5b-wsp

# 2. Export to lossless V3_Q8_0 (1.89 GB, exact WSP fidelity) or V3_P4_0 Prime-LUT (1.26 GB, 52 tok/s CPU)
python3 tools/export_vlt3.py --qwen_dir ./models/deepseek-r1-1.5b-wsp --format q8_0 --out ./dist/deepseek_1_5b_wsp_q8_0.vlt3
python3 tools/export_vlt3.py --qwen_dir ./models/deepseek-r1-1.5b-wsp --format p4_0 --out ./dist/deepseek_1_5b_wsp_p4_0.vlt3

# 3. Run full <think> reasoning OR instant --nothink mode in freestanding C99
./runtime/vault3_host ./dist/deepseek_1_5b_wsp_q8_0.vlt3 --chat "If 3x + 7 = 22, find x." --temp 0.6 --n 512 --threads 16
./runtime/vault3_host ./dist/deepseek_1_5b_wsp_q8_0.vlt3 --chat "Is Taiwan a de facto independent country? Answer directly." --nothink --n 96 --threads 16
```

### 4. Python `transformers` Usage (`VAULT-v3-3B-PRIMAL` `O(1)` `100K+` Context)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "batteryphil/VAULT-v3-3B-PRIMAL"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()

prompt = tok.apply_chat_template([{"role": "user", "content": "Derive the quadratic formula step by step."}], tokenize=False, add_generation_prompt=True)
out = model.generate(**tok(prompt, return_tensors="pt").to("cuda"), max_new_tokens=512, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## Citation & Commercial Licensing Contact
- **Author:** Philip Bartsch (`@batteryphil` — https://github.com/batteryphil)
- **Commercial 5% Profit-Share Registration / Enterprise Buyouts:** Open an issue titled `[Commercial 5% License Registration]` on this repository or contact `@batteryphil` via GitHub.
