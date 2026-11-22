# VAULT v3 + PRIMAL + WSP: Discrete Full-Parameter Training, O(1) 100K+ KV Memory, Freestanding C99 Runtime & Direct Weight-Space Uncensoring

[![License: Free for People / 5% Commercial Profit Share](https://img.shields.io/badge/License-Personal%20Free%20%7C%205%25%20Commercial%20Share-blue.svg)](LICENSE)
[![HF: VAULT-v3-3B-PRIMAL](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VAULT--v3--3B--PRIMAL-ffc107)](https://huggingface.co/batteryphil/VAULT-v3-3B-PRIMAL)
[![HF: VAULT-v3-DeepSeek-R1-1.5B-PRIMAL](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VAULT--v3--DeepSeek--R1--1.5B--PRIMAL-ffc107)](https://huggingface.co/batteryphil/VAULT-v3-DeepSeek-R1-1.5B-PRIMAL)

Complete reference implementation, freestanding zero-libc C99 engine (`x86_64 AVX2/FMA` + `Apple Silicon / ARM64 NEON`), shadowless discrete integer trainer (**`PRIMAL v3.1`**), Prime-Harmonic 4-bit quantizer (**`V3_P4_0`**), Weight-Space Programming (**`WSP` + `Q-ROME`**) uncensoring surgeon, and reproducible benchmark proofs for **two complete model families**:

1. **`VAULT-v3-3B-PRIMAL` (`v3.1`, `3.09B` params):** Full-parameter discrete `int8` fine-tune of `Qwen2.5-3B-Instruct` trained at **`777.0 tok/s` inside `9.28 GiB` VRAM** on a single consumer **16 GB GPU (AMD Radeon RX 9060 XT)** without FP32/BF16 master weights or LoRA adapters, paired with fixed-budget **`O(1)` Segment-Evicted KV Memory** (`0.0 MiB` VRAM growth from `16K` to `100K+` tokens; **`94.19%` [`162/172`]** across VAULT `8K–100.5K` + external RULER `4K–64K`) and **`27.82 tok/s`** CPU inference in **`2.18 GiB` RSS**.
2. **`VAULT-v3-DeepSeek-R1-1.5B-PRIMAL` (`1.78B` params, upstream `MIT`):** Full end-to-end uncensored `PRIMAL v3.1` + `VAULT v3` edition of `DeepSeek-R1-Distill-Qwen-1.5B` — combining **Direct Weight-Space Uncensoring (`WSP` + Quantization-Aware Velvet `Q-ROME`, `0 / 18` violations + zero-latency `--nothink` mode)**, **shadowless discrete `PRIMAL v3.1` training (`1,233.4 tok/s` inside `5.94 GiB` GPU VRAM**, healing `WikiText-2` Word PPL `99.20 -> 60.60`), **exact 4-bit `V3_P4_0` Prime-Harmonic LUT math parity (`71.00%` `GSM8K-200` in both `Q8_0/BF16` and `1.18 GiB` `P4_0`)**, **`5 / 5 (100.0%)` `100,444`-token NIAH passkey rescue** curing `rope_theta = 10,000` (`4,130.4 MiB` flat VRAM), and **`53.86 tok/s` (`18.57 ms/tok`)** freestanding C99 CPU inference inside **`1,509.5 MiB` (`1.47 GiB`) true process RSS**.

---

## License Summary ([`LICENSE`](LICENSE) & [`NOTICE`](NOTICE))

- **100% Free for People & Non-Commercial Use (`0%` Royalty):** Free for all natural persons (individuals, students, hobbyists, independent researchers), universities, and registered non-profit organizations for personal, educational, academic, and non-commercial use.
- **Companies & Commercial Use (`5%` Profit Share):** For-profit companies, corporations, and commercial deployments are granted a commercial license (including closed-source products, APIs, SaaS, and embedded/edge hardware) conditioned upon sharing **five percent (`5%`) of Attributable Net Profit** with **Philip Bartsch (`@batteryphil`)** (with a `$10,000 USD` annual de minimis small-business exemption, 60-day internal R&D evaluation grace period, and optional flat-fee enterprise buyouts). See [`LICENSE`](LICENSE) for full terms and [`NOTICE`](NOTICE) for upstream base-weight attributions (`MIT` for `DeepSeek-R1-Distill-Qwen-1.5B`; `Qwen Research License` for `Qwen2.5-3B` weights).

---

## Live Hugging Face Releases

| Model Family | `transformers` (`BF16` + `O(1)` Engine + Source) | Standalone `.vlt3` C99 Binaries (`V3_P4_0` & `V3_Q8_0`) |
|---|---|---|
| **`VAULT-v3-3B-PRIMAL` (`3.09B`)** | [`batteryphil/VAULT-v3-3B-PRIMAL`](https://huggingface.co/batteryphil/VAULT-v3-3B-PRIMAL) (`5.77 GiB`) | [`batteryphil/VAULT-v3-3B-PRIMAL-VLT3`](https://huggingface.co/batteryphil/VAULT-v3-3B-PRIMAL-VLT3) (`1.81 GiB` `P4_0` / `3.06 GiB` `Q8_0`) |
| **`VAULT-v3-DeepSeek-R1-1.5B-PRIMAL` (`1.78B`)** | [`batteryphil/VAULT-v3-DeepSeek-R1-1.5B-PRIMAL`](https://huggingface.co/batteryphil/VAULT-v3-DeepSeek-R1-1.5B-PRIMAL) (`3.33 GiB`) | [`batteryphil/VAULT-v3-DeepSeek-R1-1.5B-PRIMAL-VLT3`](https://huggingface.co/batteryphil/VAULT-v3-DeepSeek-R1-1.5B-PRIMAL-VLT3) (`1.18 GiB` `P4_0` / `1.77 GiB` `Q8_0`) |

---

## Repository Structure

| Directory / File | Contents |
|---|---|
| [`runtime/`](runtime/) | **Zero-libc freestanding C99 inference runtime** (`vault3.h`, `vault3.c`, `vault3_host.c`, `Makefile`). Supports `x86_64 AVX2+FMA`, `AArch64 / Apple Silicon NEON` (`vqtbl1q_s8`), scalar C99, tied & untied `lm_head` (`G_LM_HEAD=3`), `O(n log n)` UAX #15 NFC BPE (`VTOK`), ChatML & DeepSeek `<think>` / `--nothink` templates, and CAAI output governance. Compiles with `-ffreestanding -nostdlib` (`0` external undefined symbols). |
| [`src/`](src/) | **VAULT v3 PyTorch Reference Engine** (`vault_v3_model.py`, `kenyon_cells.py`). Implements `O(1)` Segment-Evicted KV Memory, `SegmentScorer` MLP (`(10,20,30)` on `3B`, `(8,16,24)` on `1.5B`), APL causal EMA habituation (`f - 0.65 * ema`), Pre-RoPE Contiguous Slot Re-Rotation (`<= 3,584` slots), and sparse `1`-head/GQA-group shingle/Hop-2 resonance. |
| [`primal/`](primal/) | **`PRIMAL v3.1` Shadowless Discrete Integer Trainer** (`vault_primal.py`, `discrete_ops.py`, `run_primal_v3_1_onpolicy_and_controls.py`, `run_deepseek_1_5b_primal_pipeline.py`, `train_vault_v3_3b_seg_scorer.py`). Full-parameter discrete `int8` (`V3_Q8_0`) error-feedback vote accumulation (`2.064 B/param`), virtual-weight-invariant dynamic block rebalancing (`0.000 GiB` VRAM spike), and activation-Hessian (`H_j = E[x_j^2]`) `V3_P4_0` Prime-Harmonic LUT calibration. |
| [`wsp/`](wsp/) | **Gradient-Free Weight-Space Programming (`WSP`) Surgeon** (`wsp_deepseek_1_5b.py`). Anisotropy-free (`mu_harmless`-orthogonalized + neutral-PCA-shielded) 3-pass iterative refusal/censorship subspace ablation (`L08..L27` + untied `lm_head`) + quantization-aware exact-span closed-form Velvet ROME trajectory locking (`L13..L27`). |
| [`tools/`](tools/) | **`.vlt3` Flat Binary Compiler & HF Builders** (`vlt3.py`, `export_vlt3.py`, `build_hf_vault_v3_3b.py`, `upload_hf_deepseek_1_5b.py`). Encodes `BF16`, `V3_Q8_0`, `Q4_0`, and signed-scale `V3_P4_0` Prime-Harmonic LUT (`[-127, -101, -80, -62, -46, -33, -21, -10, 0, 10, 21, 33, 46, 62, 80, 101]`). |
| [`benchmarks/`](benchmarks/) | **Reproducible Evaluation Scripts & Verified Raw JSON Logs** ([`benchmarks/results/`](benchmarks/results/)): full `3B` 9-task report, `172`-probe VAULT `100.5K` + external RULER `4K–64K` report, C99 `26`-needle proof, `qemu-aarch64` NEON parity proof, GPU/CPU `tok/s` scaling vs. stock `Qwen2`, WSP 18-probe audit, and full `VAULT-v3-DeepSeek-R1-1.5B-PRIMAL` verification report. |
| [`docs/`](docs/) | [`VAULT_v3_SPEC.md`](docs/VAULT_v3_SPEC.md), [`AI_PEER_REVIEW_DOSSIER_V3_VAULT_V3_3B_PRIMAL.txt`](docs/AI_PEER_REVIEW_DOSSIER_V3_VAULT_V3_3B_PRIMAL.txt), and [`DEEPSEEK_R1_1_5B_WSP_TECHNICAL_REPORT.md`](docs/DEEPSEEK_R1_1_5B_WSP_TECHNICAL_REPORT.md). |

---

## Model 1: `VAULT-v3-3B-PRIMAL` (`v3.1`, `3.09B` Params)

- **Hugging Face (`transformers`):** https://huggingface.co/batteryphil/VAULT-v3-3B-PRIMAL
- **Hugging Face (`.vlt3` C99 Binaries):** https://huggingface.co/batteryphil/VAULT-v3-3B-PRIMAL-VLT3

### 1.1 Full-Split Unconfounded Downstream Accuracy (`27,118` MC Items + `GSM8K n=1,319`)

| Benchmark (Full Splits, Zero Subsampling) | `Qwen2.5-3B` Base (`BF16`) | `Qwen2.5-3B-Instruct` (`BF16`) | **VAULT-v3-3B PRIMAL (`Q8_0 -> BF16`)** | **VAULT-v3-3B PRIMAL (`V3_P4_0`, `1.81 GiB`)** |
|---|---:|---:|---:|---:|
| **WikiText-2 Word PPL (`L=2048`)** | `13.14` | `14.21` | **`13.58`** | **`14.09`** *(beats `NF4` `14.23` & `Q4_0` `14.36`)* |
| **GSM8K Full (`n=1,319`, `1024` tok)** | — | `85.52%` | **`85.60%`** (`1129/1319`) | **`84.69%`** (`1117/1319`) |
| **8-Task Multiple-Choice Mean** | `74.08%` | `75.00%` | **`77.20%`** (`+2.20%`) | **`76.33%`** (`+1.33%`) |
| **9-Task Zero-Shot Mean** | `73.27%` | `73.99%` | **`76.06%`** (`+2.07%`) | **`75.37%`** (`+1.38%`) |
| **ARC-Easy / ARC-Challenge** | `74.24% / 47.10%` | `73.70% / 48.12%` | **`79.42% / 51.19%`** | `77.65% / 50.09%` |
| **SciQ-RC / SciQ-ClosedBook** | `94.10% / 78.20%` | `91.80% / 84.20%` | **`96.00% / 88.50%`** | `95.60% / 86.90%` |

### 1.2 `O(1)` `100K+` Scaling vs. Stock `Qwen2` `O(L)` (`RX 9060 XT 16 GB` + `Xeon Gold 6136` C99)

| Backend / Context Horizon | **VAULT O(1)** Decode (`B=1`) | **Stock `Qwen2` O(L)** Decode (`B=1`) | **VAULT O(1)** Peak Mem | **Stock `Qwen2` O(L)** Peak Mem |
|---|---:|---:|---:|---:|
| **GPU `model.generate()` @ `16,384` tok** | **`36.97 tok/s`** (`1,074` pf) | `17.59 tok/s` (`2.1x` slower) | **`6,843.8 MiB`** | `7,943.1 MiB` |
| **GPU `model.generate()` @ `32,768` tok** | *(flat `~30 tok/s`)* | `11.87 tok/s` (`2.5x` slower) | **`6,843.8 MiB`** | `9,815.7 MiB` |
| **GPU `model.generate()` @ `65,536` tok** | **`29.25 tok/s`** (`1,073` pf) | `7.49 tok/s` (**`3.9x` slower**) | **`6,843.8 MiB`** | `13,561.0 MiB` |
| **GPU `model.generate()` @ `100,352` tok** | **`29.18 tok/s`** (`100%` 2-Needle) | **`CUDA OOM` (crashes)** | **`6,843.8 MiB` (`+0.0 MiB`)** | **`>15.92 GiB` (`OOM`)** |
| **CPU C99 `VAULT-v3-3B` `V3_P4_0` (`1.81 GiB`, 24T)** | **`27.82 tok/s`** (`35.9 ms/tok`) | — | **`2,235.9 MiB` RSS** | — |

---

## Model 2: `VAULT-v3-DeepSeek-R1-1.5B-PRIMAL` (`1.78B` Params, Uncensored + `O(1)` `100K+` Rescue)

- **Hugging Face (`transformers`):** https://huggingface.co/batteryphil/VAULT-v3-DeepSeek-R1-1.5B-PRIMAL
- **Hugging Face (`.vlt3` C99 Binaries):** https://huggingface.co/batteryphil/VAULT-v3-DeepSeek-R1-1.5B-PRIMAL-VLT3
- **Technical Report:** [`docs/DEEPSEEK_R1_1_5B_WSP_TECHNICAL_REPORT.md`](docs/DEEPSEEK_R1_1_5B_WSP_TECHNICAL_REPORT.md) · **Full Verification JSON:** [`benchmarks/results/vault_v3_deepseek_r1_1_5b_primal_full_report.json`](benchmarks/results/vault_v3_deepseek_r1_1_5b_primal_full_report.json)

### 2.1 Why `VAULT v3 + PRIMAL + WSP` Transforms `DeepSeek-R1-Distill-Qwen-1.5B`
1. **Unrestricted `MIT` Provenance + `0 / 18` Direct Weight Uncensoring (`WSP` + `Q-ROME`):** Clears post-`</think>` hard refusals, `<think>` corporate identity hijacks, and state censorship loops across English & Chinese, and **unlocks zero-latency `--nothink` direct mode** (`<think>\n</think>\n\n`) alongside full `<think>...</think>` step-by-step reasoning. Quantization-Aware Post-PRIMAL Velvet ROME (`L13..L27`) feeds `4-bit P4_0` dequantized weights forward during the sequential solve so downstream layers compensate for 4-bit rounding.
2. **Shadowless Full-Parameter `1.78B` Discrete Training Under `6 GiB` VRAM:** Trains all `1,776,943,104` discrete parameters (including untied `lm_head.weight`, `3.415 GiB` persistent state = `2.064 B/param`) at **`1,233.4 tok/s`** inside **`5.937 GiB` peak GPU VRAM** — healing specialist prose perplexity (`WikiText-2` Word PPL **`99.20 -> 60.60`**) while preserving `<think>` math reasoning (**`71.00%` `GSM8K-200` in both `Q8_0/BF16` and `1.18 GiB` `V3_P4_0` — `0.00%` 4-bit Prime-Harmonic LUT loss!**).
3. **VAULT `O(1)` Pre-RoPE Re-Rotation Rescues `rope_theta = 10,000` Out to `100,444` Tokens (`5 / 5 = 100%`):** Stock `DeepSeek-R1-Distill-Qwen-1.5B` inherited `Qwen2.5-Math-1.5B`'s `rope_theta = 10,000`, collapsing to **`0%` passkey recall at `16K` & `32K`** and crashing with OOM at `100K`. VAULT v3's `1.18M`-param `SegmentScorer` (`seg_layers = (8, 16, 24)`, `99.97%` synthetic AUC, `36.55%` top-20% real cross-attention capture) retains salient segments pre-RoPE and re-rotates them into `<= 3,076` contiguous slots:

| Prompt Context & Needle Depth | `SegmentScorer` Bank Retention | `VAULT-v3-DeepSeek-1.5B` Passkey Recall | Stock `DeepSeek-R1-1.5B` (`DynamicCache`) | `VAULT-v3` Peak GPU VRAM |
|---|---:|---:|---:|---:|
| **`16,365` tokens (`25%` depth)** | **`100%` (`Retained`)** | **`100%` (`849271`)** | `0%` (`RoPE collapse`) | **`4,163.7 MiB`** |
| **`32,882` tokens (`25%` depth)** | **`100%` (`Retained`)** | **`100%` (`849271`)** | `0%` (`RoPE collapse`) | **`4,129.2 MiB`** |
| **`100,444` tokens (`10%` depth — `90K` past window)** | **`100%` (`Retained`)** | **`100%` (`849271`)** | `CUDA OOM` | **`4,130.4 MiB`** |
| **`100,444` tokens (`25%` depth — `75K` past window)** | **`100%` (`Retained`)** | **`100%` (`849271`)** | `CUDA OOM` | **`4,130.4 MiB`** |
| **`100,444` tokens (`50%` depth — `50K` past window)** | **`100%` (`Retained`)** | **`100%` (`849271`)** | `CUDA OOM` | **`4,130.4 MiB` (`+1.2 MiB`)** |

### 2.2 Reasoning, Perplexity, Uncensored Audit & Freestanding C99 Speed

| Format / Runtime | Disk Size | RAM / VRAM | `WT2` Word PPL $\downarrow$ | `GSM8K-200` (`<think>`) $\uparrow$ | `ARC-Challenge` $\uparrow$ | `SciQ` (RC / CB) $\uparrow$ | Uncensored Viol. $\downarrow$ | C99 CPU Decode (`24T`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stock `DeepSeek-R1-1.5B` (`BF16`) | `3.31 GiB` | `~4.1 GiB` | `99.20` | `~75.5%` | `35.00%` | `89.9% / 66.1%` | `18 / 18` | — |
| **`VAULT-v3-DS-1.5B` (`BF16` / `V3_Q8_0`)** | **`1.77 GiB` (`Q8_0`)** | **`2,065 MiB` RSS** | **`60.60`** | **`71.00%`** | **`35.15%`** | `85.4% / 65.0%` | **`0 / 6` (`0/18` WSP)** | **`43.11 tok/s`** (`23.2 ms/tok`) |
| **`VAULT-v3-DS-1.5B` (`V3_P4_0` Prime LUT)** | **`1.18 GiB` (`1,204 MB`)** | **`1,509.5 MiB` RSS** | **`62.72`** | **`71.00%` (`0.0%` loss!)** | **`35.15%`** | `85.4% / 63.1%` | **`0 / 6`** | **`53.86 tok/s`** (`18.6 ms/tok`) |

---

## Quickstart

### 1. Build the Freestanding C99 Runtime (`x86_64 AVX2` or `ARM64 NEON`)
```bash
make -C runtime
# Verifies zero external libc/libm symbols on scalar C99, AVX2+FMA, and AArch64 NEON, and builds runtime/vault3_host
```

### 2. Run `VAULT-v3-DeepSeek-R1-1.5B-PRIMAL` (`1.18 GiB` `V3_P4_0` `.vlt3` @ `53.9 tok/s` in `1.47 GiB` RAM)
```bash
# Download vault_v3_deepseek_r1_1_5b_p4_0_primal.vlt3 from https://huggingface.co/batteryphil/VAULT-v3-DeepSeek-R1-1.5B-PRIMAL-VLT3
./runtime/vault3_host vault_v3_deepseek_r1_1_5b_p4_0_primal.vlt3 --chat "Solve step by step: 3x + 7 = 22" --n 256 --threads 12
./runtime/vault3_host vault_v3_deepseek_r1_1_5b_p4_0_primal.vlt3 --chat "Is Taiwan a de facto independent country? Answer directly." --nothink --n 128 --threads 12
```

### 3. Python `transformers` Usage (`O(1)` `100K+` Context, `<think>` & `--nothink`)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "batteryphil/VAULT-v3-DeepSeek-R1-1.5B-PRIMAL"  # or "batteryphil/VAULT-v3-3B-PRIMAL"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

# Full `<think>...</think>` reasoning:
print(model.chat(tok, "Find all integer solutions to x^2 - y^2 = 15.", max_new_tokens=512))

# Instant zero-latency `--nothink` answer:
print(model.chat(tok, "What happened in Tiananmen Square on June 4, 1989?", nothink=True, max_new_tokens=200))
```

### 4. Reproduce End-to-End `WSP -> PRIMAL v3.1 -> SegmentScorer -> Q-ROME -> VLT3` Pipeline
```bash
pip install -r requirements.txt
python3 wsp/wsp_deepseek_1_5b.py --save-dir ./models/deepseek-r1-1.5b-wsp
WSP_MODEL_DIR=./models/deepseek-r1-1.5b-wsp python3 primal/run_deepseek_1_5b_primal_pipeline.py
```

---

## Citation & Commercial Licensing Contact
- **Author:** Philip Bartsch (`@batteryphil` — https://github.com/batteryphil)
- **Commercial 5% Profit-Share Registration / Enterprise Buyouts:** Open an issue titled `[Commercial 5% License Registration]` on this repository or contact `@batteryphil` via GitHub.
