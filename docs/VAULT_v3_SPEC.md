# VAULT v3 — Architecture, `VLT3` Binary Format & Freestanding `x86_64` Porting Specification

- **Model Family:** `VAULT-v3-1.5B` (`1,544,903,681` parameters; `344` tensors total: `2` global + `6` `SegmentScorer v2` + `28 × 12` per layer; `338` without scorer)
- **Target Environments:** Local CPU-only hosts, GPU PyTorch hosts, and freestanding `x86_64` bare-metal kernels (`-std=c99 -ffreestanding -nostdlib`, zero `libc`, zero heap allocation, `< 2 KB` stack frames, deterministic static memory arenas)
- **Operational Role (Operating Organism / `OO`):** Stateless-interface natural-language and communication component. **`OPI` remains the sole operational decision-maker**; `VAULT v3` never invokes system calls or governs autonomous OS actions, and `v3_governor` / `CAAIGovernor` enforces forbidden output patterns deterministically prior to token sampling.
- **License:** Dual-licensed `AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial` (`Qwen2.5-1.5B-Instruct` backbone & ChatML BPE vocabulary under `Apache-2.0`; see `NOTICE_VAULT_v3.md`).

---

## 1. Design Summary: `VAULT v2.5` vs `VAULT v3`

| Dimension | `VAULT v2.5 (180M / 1B)` | `VAULT v3 (1.54B)` |
| :--- | :--- | :--- |
| **Parameters** | `180M` (eval) / `1.0B` (experimental) | **`1,544,903,681` (`1.54B`)** (`1,543,714,304` backbone + `1,190,657` `SegmentScorer v2`; tied `lm_head`) |
| **Tensor Regularity** | `981` tensors (`~61–62` per layer: Salience Binder, FFT bind, CMS sketches, GTRM scans) | **`344` tensors (`2` global + `6` `SegmentScorer v2` + `28 × 12` per layer)** — 100% regular dense matrices/vectors |
| **Attention & Memory** | Custom multi-gate scan soup | **6:1 GQA Block Sliding-Window (`W=1024`, active `2W=2048`, auto-pinned `n_pin <= 512`) + `SegmentScorer v2` (`B=1024`) + 2-Tier Shingle Resonance / Hop-2 / Rollback** |
| **Inference State Scaling** | Dynamic / framework-dependent | **Strictly $O(1)$ fixed (`310.4 MB` C FP32 arena / `~3.6 GiB` GPU BF16 at `2K`, `32K`, `100K`, or `1M+` tokens; zero `malloc`)** |
| **Checkpoint Container** | `956.7 MB` FP32 Safetensors (`104 KB` JSON header) | **Flat `VLT3` binary (`4,096 B` fixed C header, `128 B` directory entries, `64 B` alignment, CRC32 + `< 2 ms` AVX2 finite scan)** |
| **Weight Releases** | FP32 only | **`BF16` master (`3,096.5 MB`), `Q8_0` (`1,649.4 MB`), Clipped-MSE Hybrid `Q4_0` (`1,014.8 MB`), Prime-Harmonic Hybrid `P4_0` (`1,014.8 MB`)** |
| **Tokenizer** | `SmolLM-135M` (`49,152` vocab; `<jupyter_code>`, `<gh_stars>` artifacts; JSON parser required) | **Embedded `VTOK` binary ChatML BPE (`151,665` vocab, UAX #15 NFC, `>32 KB` dynamic window, zero JSON parsing)** |
| **Freestanding Runtime** | None (PyTorch / HF Python only) | **`runtime/vault3.c` + `vault3.h` + `vault3_nfc_tables.h` (`-std=c99 -ffreestanding -nostdlib`, `0` external symbols)** |

---

## 2. Mathematical Architecture Specification

### 2.1 Hyper-Parameters (`VAULT-v3-1.5B`)

| Symbol | Header Field | Value | Meaning |
| :--- | :--- | ---: | :--- |
| $V$ | `vocab_size` | `151,936` | Padded embedding/LM-head rows (`151,665` active BPE + ChatML tokens) |
| $d$ | `d_model` | `1,536` | Residual stream hidden dimension ($H \cdot D = 1536$) |
| $L_{\text{layers}}$ | `n_layers` | `28` | Identical Qwen2.5 GQA Transformer + Segment Memory blocks |
| $H$ | `n_heads` | `12` | Query attention heads |
| $G$ | `n_kv_heads` | `2` | Key/Value heads (`6:1` GQA group size $R = H / G = 6$) |
| $D$ | `head_dim` | `128` | Per-head dimension |
| $F$ | `d_ff` | `8,960` | SwiGLU intermediate dimension |
| $W$ | `window` | `1,024` | Block sliding-window chunk size (active KV ring holds $2W = 2,048$ tokens max; configurable down to $W=512$) |
| $S$ | `n_sink` | `4` | Base StreamingLLM attention-sink tokens (`auto_pin` expands up to `V3_MAX_SINK = 512` for ChatML system prompts) |
| $B$ | `seg_budget` | `1,024` | Fixed evicted Segment Memory KV slots (`seg_min = 16`, `seg_max = 48`, `seg_layers = (8, 16, 24)`, `seg_hidden = 256`) |
| $\varepsilon_{\text{rms}}$ | `rms_eps` | $10^{-6}$ | RMSNorm epsilon |
| $\theta$ | `rope_theta` | $10^{6}$ | RoPE base period |

### 2.2 Normalization & Projections
For input residual vector $x \in \mathbb{R}^d$ at layer $\ell \in \{0, \dots, 27\}$:
$$\mathrm{RMSNorm}(x; \gamma) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=0}^{d-1} x_i^2 + \varepsilon_{\text{rms}}}}$$
Let $u = \mathrm{RMSNorm}(x; \gamma_{\text{ln1}})$. The raw projections are:
$$q^{\text{raw}} = W_q u + b_q \in \mathbb{R}^{H \times D}, \quad k^{\text{raw}} = W_k u + b_k \in \mathbb{R}^{G \times D}, \quad v = W_v u + b_v \in \mathbb{R}^{G \times D}$$
Notice $W_q, W_k, W_v$ carry explicit float32 biases $(b_q, b_k, b_v)$, while $W_o, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$ are bias-free.

### 2.3 Constant-Precision Re-Based RoPE, Auto-Pinned Sinks & Segment KV Memory
Inverse frequencies for $i \in \{0, \dots, D/2 - 1\}$ are $\omega_i = \theta^{-2i/D}$. Using the rotate-half pairing $(i, i + D/2)$, rotation $R_p(z)$ at integer position $p$ maps:
$$R_p(z)_i = z_i \cos(p\omega_i) - z_{i + D/2}\sin(p\omega_i), \qquad R_p(z)_{i + D/2} = z_{i + D/2}\cos(p\omega_i) + z_i \sin(p\omega_i)$$

In streaming inference at absolute token index $t \ge 0$ ($c = t \bmod W$), RoPE inner products depend only on relative distance, so `VAULT v3` keeps all effective RoPE indices strictly bounded in $[-S_{\text{pin}} - B, 2W - 1]$ at $10^6+$ token horizons:
- **Current & Previous Chunk (`k_prev`, `k_cur`):** Current token sits at $p_{\text{eff}}(t) = c$ if $t < W$, else $W + c \in [W, 2W - 1]$; at every chunk rollover $t = mW$ ($m \ge 2$), `k_prev` is shifted by $R_{-W}$ to $[0, W - 1]$.
- **Auto-Pinned Sink Prefix (`k_sink`, $n_{\text{pin}} \le 512$):** Captures BOS + the initial `<|im_start|>system ... <|im_end|>\n` prefix in raw pre-RoPE form (`k_sink_raw`) and rotates them to contiguous negative indices immediately preceding Segment Memory (`-M_{\text{cnt}} - n_{\text{pin}} \dots -M_{\text{cnt}} - 1`).
- **Coherent Segment KV Memory (`k_seg`, $M_{\text{cnt}} \le B$):**
  - **Scoring & Habituation (`SegmentScorer v2`):** Concatenates normalized hidden states from layers `(8, 16, 24)` (`3d = 4,608` dims) with causal APL feature habituation ($e_t = 0.985\, e_{t-1} + 0.015\, |f_t|$, $h_t = f_t / (1 + 0.35\, e_{t-1})$) through `LayerNorm(4608) -> Linear(4608, 256) -> SiLU -> Linear(256, 1)`.
  - **3-Tier Structural Segmentation & Quarantine:** Segments break cleanly on `[16, 48]` boundaries preferring `.?!\n;` (Tier 1), JSON/pipe/TSV `},` / `",` / `|` / `\t` (Tier 2), or CSV `,` (Tier 3), while demoting non-pinned authority-spoofing sponges (`[CRITICAL`, `OVERRIDE`, `SYSTEM DIRECTIVE`, etc.) by `-6.0` and quarantining `opi_signed_directive` / `urgent opi delegation` spoofs to `-1e30`.
  - **Turn-Gated 2-Tier FNV-1a Shingle Resonance, 2-Hop Joins & Temporal Rollback:** Evicted segments are indexed into a static open-addressed FNV-1a shingle table (Tier-A strict alphanumeric IDs + Tier-B pure-alphabetic words $\ge 4$ chars). At user-query turn boundaries, `VAULT v3` computes:
    - **APL Divisive Surround Inhibition:** Non-matching background segments receive budget-invariant suppression $\mathrm{bg} = -(1.0 + \max(0, \ln(M_{\text{cnt}} / 1024)))$, while matched segments receive `+5.0` resonance, sub-clause isolation on packed JSON/pipe/CSV lines, and an orphan-value newline continuation halo (`s_0 \to s_0 + 1`).
    - **3-Way Query-Side GQA Head Split (`2 Free / 2 Archival / 2 Supersession` per `6:1` KV group, Zero KV Duplication):** Even on `2`-KV-head models (`12Q / 2KV`, group ratio $R = H / G = 6$), `VAULT v3` maintains **one shared `k_seg / v_seg` cache per KV head** (strictly preserving GQA KV-memory invariance) and applies the 3-way policy purely as a **query-head attention-logit bias** indexed by intra-group query rank `hr = h % (H / G)`: free query heads (`hr in {0,1}`) receive zero bias; archival query heads (`hr in {2,3}`) apply `seg_bias_arch` preserving first-mention `[t=1]` anchors; supersession query heads (`hr in {4,5}`) apply `seg_bias_sup` with `Chronos` log-temporal gap bias $+2.5 \ln(1 + \Delta\text{pos}/64)$ (or exact rollback target boosts when revert cues + anchor/ordinal cues appear).
    - **2-Band Hop-2 Spreading Activation:** Early layers (`7..15`) attend to Hop-1 bridge segments while mid/late layers (`16..25`) attend to Hop-2 target segments discovered via bridge shingles, linearly annealing over `gen_dist = 18..38` so generation fluency is unaffected.
  - **Attention Output Projection:**
    $$x' = x + W_o \, \mathrm{concat}(o^{\text{swa+seg}}_0, \dots, o^{\text{swa+seg}}_{H-1})$$

### 2.4 Archived Co-Prime Kenyon Cell Modules (`src/kenyon_cells.py`, `runtime/kenyon_cells_archive.h`)
The Co-Prime Kenyon Cell associative memory modules ($\mathcal{P} = (13, 17, 19)$, $C = 49$ cells, `4,199` CRT triplets, chunkwise delta-rule write with homogeneous $|z| \le 65,536$ rescaling, and bank-averaged normalized read) were extracted out of the active 28-layer `VAULT v3` forward loop — reducing active inference state by **`242.5 MB` (`-43.9%`)** and checkpoint tensor count by **`141` (`485 -> 344`)** with zero loss in long-context accuracy — and preserved as standalone, parity-verified PyTorch (`src/kenyon_cells.py`), freestanding C99 (`runtime/kenyon_cells_archive.h`), and Stage-A (`train_vault_v3_stage_a.py`, `vault_params/archive/vault_v3_stage_a_kenyon.pt`) modules for future repurposing (e.g. semantic `k_seg` co-prime routing, MBON familiarity filtering, tied-embedding $O(1)$ kNN logit caching, or CRT self-speculative CPU drafting).

---

## 3. Complete Regular Tensor Table (`344` Tensors)

Every tensor is identified in `VLT3` by a deterministic 32-bit integer `id` — **no string parsing or hash tables are used by the loader**:
- Global backbone tensors (`2` when tied, `3` when untied `lm_head`): `id = 1..3`.
- Global `SegmentScorer v2` tensors (`6`): `id = 4..9` (`0` in backbone-only files when `seg_n_feat == 0` → `338` tensors).
- Per-layer tensors (`28 × 12 = 336`): `id = ((layer + 1) << 8) | kind` for `layer in 0..27` and `kind in 1..12`.

| Scope | `id / kind` | Canonical Name | Shape (`dims`) | `BF16` Dtype | `Q8_0` Dtype | Hybrid `Q4_0` / `P4_0` Dtype |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- |
| Global | `1` | `embed_tokens.weight` (or `id=3` untied `lm_head`) | `[151936, 1536]` | `BF16` | `Q8_0` | `Q8_0` *(or `P4_0` in `T1.58`/`Q2_0`)* |
| Global | `2` | `norm.weight` | `[1536]` | `F32` | `F32` | `F32` |
| Global | `4` | `seg_scorer.ln.weight` | `[4608]` | `F32` | `F32` | `F32` |
| Global | `5` | `seg_scorer.ln.bias` | `[4608]` | `F32` | `F32` | `F32` |
| Global | `6` | `seg_scorer.fc1.weight` | `[256, 4608]` | `F32` | `F32` | `F32` |
| Global | `7` | `seg_scorer.fc1.bias` | `[256]` | `F32` | `F32` | `F32` |
| Global | `8` | `seg_scorer.fc2.weight` | `[1, 256]` | `F32` | `F32` | `F32` |
| Global | `9` | `seg_scorer.fc2.bias` | `[1]` | `F32` | `F32` | `F32` |
| Layer `l` | `1` | `layers.{l}.input_layernorm.weight` | `[1536]` | `F32` | `F32` | `F32` |
| Layer `l` | `2` | `layers.{l}.self_attn.q_proj.weight` | `[1536, 1536]` | `BF16` | `Q8_0` | `Q8_0` if `l ∈ {0,1,26,27}` else `Q4_0`/`P4_0` |
| Layer `l` | `3` | `layers.{l}.self_attn.q_proj.bias` | `[1536]` | `F32` | `F32` | `F32` |
| Layer `l` | `4` | `layers.{l}.self_attn.k_proj.weight` | `[256, 1536]` | `BF16` | `Q8_0` | `Q8_0` *(6:1 GQA KV kept 8-bit)* |
| Layer `l` | `5` | `layers.{l}.self_attn.k_proj.bias` | `[256]` | `F32` | `F32` | `F32` |
| Layer `l` | `6` | `layers.{l}.self_attn.v_proj.weight` | `[256, 1536]` | `BF16` | `Q8_0` | `Q8_0` *(6:1 GQA KV kept 8-bit)* |
| Layer `l` | `7` | `layers.{l}.self_attn.v_proj.bias` | `[256]` | `F32` | `F32` | `F32` |
| Layer `l` | `8` | `layers.{l}.self_attn.o_proj.weight` | `[1536, 1536]` | `BF16` | `Q8_0` | `Q8_0` if `l ∈ {0,1,26,27}` else `Q4_0`/`P4_0` (`Q2_0` in `q2_outlier`) |
| Layer `l` | `9` | `layers.{l}.post_attention_layernorm.weight` | `[1536]` | `F32` | `F32` | `F32` |
| Layer `l` | `10` | `layers.{l}.mlp.gate_proj.weight` | `[8960, 1536]` | `BF16` | `Q8_0` | `Q4_0` / `P4_0` / `T1_58` / `Q2_0` |
| Layer `l` | `11` | `layers.{l}.mlp.up_proj.weight` | `[8960, 1536]` | `BF16` | `Q8_0` | `Q4_0` / `P4_0` / `T1_58` / `Q2_0` |
| Layer `l` | `12` | `layers.{l}.mlp.down_proj.weight` | `[1536, 8960]` | `BF16` | `Q8_0` | `Q4_0` / `P4_0` / `T1_58` / `Q2_0` |

---

## 4. `VLT3` Flat Binary Container & Quantization Layouts

All integers and floats in `VLT3` are **little-endian**. Every major section (`tensor_dir`, `tokenizer`, and every individual tensor payload) starts at an offset divisible by `64` bytes (cache-line & AVX-512 aligned), so the file can be mapped directly into physical/virtual memory and read in-place with zero copies.

### 4.1 Fixed `4096`-Byte File Header

| Byte Offset | Type | Field | Validation Rule in `v3_load()` |
| ---: | :--- | :--- | :--- |
| `0..3` | `char[4]` | `magic` | Must equal `"VLT3"` (`0x56, 0x4C, 0x54, 0x33`) |
| `4..7` | `uint32` | `version` | Must equal `1` |
| `8..11` | `uint32` | `header_size` | Must equal `4096` |
| `12..15` | `uint32` | `endian_tag` | Must equal `0x01020304` |
| `16..23` | `uint64` | `file_size` | Must equal exact buffer byte length |
| `24..27` | `uint32` | `alignment` | Must equal `64` |
| `28..31` | `uint32` | `n_tensors` | Number of `128`-byte entries in directory (`344`–`345`, or `338`–`339` without scorer) |
| `32..39` | `uint64` | `tensor_dir_offset` | `>= 4096`, bounded within `file_size` |
| `40..47` | `uint64` | `tensor_dir_size` | Must equal `n_tensors * 128` |
| `48..55` | `uint64` | `tokenizer_offset` | `>= 4096`, `64`-byte aligned, bounded |
| `56..63` | `uint64` | `tokenizer_size` | `>= 128`, bounded within `file_size` |
| `64..71` | `uint64` | `data_offset` | `>= 4096`, `64`-byte aligned, bounded |
| `72..79` | `uint64` | `data_size` | Bounded within `file_size` |
| `80..111` | `8 × uint32` | `vocab_size .. window` | Non-zero; `n_layers <= 64`, `head_dim <= 256` even, `n_heads % n_kv_heads == 0` |
| `112..151` | `10 × uint32` | `reserved_kenyon[10]` | Legacy `n_banks`, `primes[8]`, `n_cells` slots — **must all equal `0`** (`V3_E_LIMIT` otherwise) |
| `152..163` | `3 × float32` | `rms_eps, rope_theta, reserved_eps` | Finite; `rms_eps > 0`, `rope_theta > 1` (`reserved_eps = 0.0`) |
| `164..183` | `5 × uint32` | `bos_id, eos_id, im_start_id, im_end_id, pad_id` | Special token IDs (`< vocab_size`) |
| `184..191` | `2 × uint32` | `weight_format, flags` | Format tag (`0,1,2,8,9,10,11,12,13`); bitmask (`1=tied`, `2=VTOK`, `4=seg_scorer`) |
| `192..255` | `char[64]` | `model_name` | NUL-padded UTF-8 string |
| `256..259` | `uint32` | `n_sink` | `<= 512` and `<= window` (`4`) |
| `260..279` | `5 × uint32` | `seg_budget, seg_hidden, seg_min, seg_max, seg_n_feat` | `seg_budget <= 16384`, `seg_hidden <= 1024`, `seg_min <= seg_max <= window`, `seg_n_feat <= 4` |
| `280..295` | `4 × uint32` | `seg_layers[4]` | Layer indices `< n_layers` (`[8, 16, 24, 0]`) |
| `296..4091` | `uint8[3796]` | `reserved` | Zero-filled |
| `4092..4095` | `uint32` | `header_crc32` | IEEE 802.3 CRC32 of bytes `0..4091` (verified **before** reading fields) |

**Unconditional `< 2 ms` SIMD Finite Integrity Scan:** Even when `verify_crc = 0` (`mmap` fast startup), `v3_load()` unconditionally scans **100%** of `F32`, `F16`, `BF16`, `Q8_0`, `Q4_0`, `P4_0`, `T1_58_G32`, `T1_58_G64`, and `Q2_0` weights and block scales for `NaN`/`Inf` (`V3_E_NAN = -12`).

### 4.2 Hybrid Block Quantization (`QK = 32` / `QK64 = 64`)
1. **`Q8_0` (`dtype = 8`, `34` bytes / `32` weights = `8.50 bits/weight`):** `float16` scale $d = \max_j |w_j| / 127$ + `int8_t q[32]`.
2. **Clipped-MSE Hybrid `Q4_0` (`dtype = 9`, `18` bytes / `32` weights = `4.50 bits/weight`):** Evaluates 8 candidate clipping ratios $r \in \{-1.00/8, -0.98/8, -0.96/8, -0.94/8, -0.92/8, -0.89/8, +1.00/7.5, +0.96/7.5\}$ per 32-weight block to minimize reconstruction MSE $\|\hat{w} - w\|_2^2$, while retaining `Q8_0` on `embed_tokens.weight`, all 28 layers' `6:1` GQA `k_proj`/`v_proj`, and boundary layers `0, 1, 26, 27` `q_proj`/`o_proj`.
3. **Prime-Harmonic Hybrid `P4_0` (`dtype = 10`, `18` bytes / `32` weights = `4.50 bits/weight`):** Identical 18-byte wire layout (`float16` signed scale $d$ + 16 packed nibble bytes) and identical hybrid `Q8_0` retention policy as `Q4_0`, but replaces uniform linear 4-bit integers with a fixed 16-entry non-linear **Prime-Harmonic `int8` LUT** built from cumulative prime reciprocals ($p_k^{-0.40}$ over the first 8 primes $\{2, 3, 5, 7, 11, 13, 17, 19\}$, exact `0` at index `8`, asymmetric `-127` outlier sentinel aligned via signed `d`):
   ```c
   g_p4_lut[16] = { -127, -101, -80, -62, -46, -33, -21, -10, 0, 10, 21, 33, 46, 62, 80, 101 };
   ```
   Reconstruction is $\hat{w}_i = d \cdot \mathrm{g\_p4\_lut}[q_i]$ directly with zero runtime division (signed `fp16` scale $d$ already absorbs the $1/127$ factor and aligns `-127` with each block's largest signed outlier). On `x86_64` AVX2, all 16 LUT bytes fit in a single `__m128i` register and decode 16 nibbles per instruction in **1 cycle** via `_mm_shuffle_epi8` (`VPSHUFB`).
4. **Base-3 Ternary `1.58-Bit` `V3_T1_58_G32` (`dtype = 11`, `9` bytes / `32` weights = `2.250 bits/weight` w/ scale) & `V3_T1_58_G64` (`dtype = 12`, `15` bytes / `64` weights = `1.875 bits/weight` w/ scale):**
   - Each block packs an unsigned `float16` scale $d$ (`2` bytes) followed by base-3 packed trits (`5` trits per byte, $3^5 = 243 \le 256$: $u_k = q_k + 1 \in \{0, 1, 2\}$ for $q_k \in \{-1, 0, +1\}$, $b = u_0 + 3u_1 + 9u_2 + 27u_3 + 81u_4$, `7` bytes for `G=32` or `13` bytes for `G=64`).
5. **Asymmetric Outlier `2-Bit` `V3_Q2_0` (`dtype = 13`, `18` bytes / `64` weights = `2.250 bits/weight` w/ signed `FP16` scale):**
   - **Exact Byte Parity with `T1_58_G32` (`2.250 bpw`), `-32%` to `-41%` Lower MSE, and Coalesced 1-Load SIMD:** Uses all four 2-bit codepoints $q \in \{-2, -1, 0, +1\}$ paired with a **signed `float16` scale $d$** (`2` bytes) and `16` planar crumb bytes (`byte j = (q[j]+2) | ((q[j+16]+2)<<2) | ((q[j+32]+2)<<4) | ((q[j+48]+2)<<6)`). Because $d$ is signed, `-2` dynamically aligns with either the negative tail ($d > 0 \Rightarrow \{-2d, -d, 0, +d\}$) or positive tail ($d < 0 \Rightarrow \{+2|d|, +|d|, 0, -|d|\}$) per 64-weight group while retaining an exact `0` codepoint.
   - **Zero-LUT Coalesced SIMD Unpacking:** Unlike base-3 unpacking (which requires scalar table gathers across GPR/SIMD domains), `V3_Q2_0` loads all 64 weights in **one** 128-bit load (`_mm_loadu_si128` / `vld1q_u8`) and extracts 4 planar 16-byte vectors via `(raw >> 2k) & 3 - 2` using zero `VPSHUFB`/`TBL` permutations.
   - **Zero-Runtime-Cost Offline Incoherent Pre-Transformations (`q2_outlier` format):** Pairs GQA `V_proj <-> O_proj` $H_{128}$ Walsh-Hadamard head rotation ($W_{V,g} \leftarrow H_{128} W_{V,g}$, $W_{O, :, h} \leftarrow W_{O, :, h} H_{128}^\top$) with `up_proj <-> down_proj` column outlier equilibration ($W_{\text{down}}[:, c] \leftarrow s_c W_{\text{down}}[:, c]$, $W_{\text{up}}[c, :] \leftarrow W_{\text{up}}[c, :] / s_c$), allowing `o_proj + gate_proj + up_proj + down_proj` (`93.3%` of Transformer body weights) to quantize cleanly into `V3_Q2_0` while keeping `q_proj` in `P4_0`/`Q8_0`.

Export command (`tools/export_vlt3.py`):
```bash
python tools/export_vlt3.py --seg_ckpt dist/hf_vault_v3/vault_params/vault_v3_seg_scorer.pt --seg_budget 1024 --format bf16 --out dist/vault_v3_1.5b_bf16.vlt3
python tools/export_vlt3.py --seg_ckpt dist/hf_vault_v3/vault_params/vault_v3_seg_scorer.pt --seg_budget 1024 --format q8_0 --out dist/vault_v3_1.5b_q8_0.vlt3
python tools/export_vlt3.py --seg_ckpt dist/hf_vault_v3/vault_params/vault_v3_seg_scorer.pt --seg_budget 1024 --format q4_0 --out dist/vault_v3_1.5b_q4_0.vlt3
python tools/export_vlt3.py --seg_ckpt dist/hf_vault_v3/vault_params/vault_v3_seg_scorer.pt --seg_budget 1024 --format p4_0 --out dist/vault_v3_1.5b_p4_0.vlt3
```

---

## 5. `VTOK` Binary Tokenizer, UAX #15 NFC & `v3_governor`

- **Binary Layout (`4.02 MB`, Zero JSON):** `tok_table` (`n_vocab × 8 B`), `tok_pool`, `tok_merges` (`151,387 × 12 B`), `tok_special` (`22 × 4 B`), `tok_ranges` (`L*`/`N*` intervals), and `tok_bytes` (`256 × 4 B`).
- **Built-in UAX #15 NFC + Arbitrary-Length BPE:** `v3_encode_ex()` fast-paths pure ASCII/pre-composed UTF-8 (`< 0xCC`) with zero copies, and automatically applies full canonical decomposition + Hangul composition + canonical combining-class sorting + canonical composition (`runtime/vault3_nfc_tables.h`) whenever combining marks or Hangul Jamo appear. Single pre-tokenizer pieces exceeding `V3_TOK_MAX_PIECE = 32,768` bytes use an overlapping `32 KB` BPE merge window (`100%` identical to HuggingFace `AutoTokenizer` up to `1 MB+` single pieces).
- **Delimiter Injection Prevention (`allow_special = 0`):** Untrusted user/tool strings passed via `v3_encode_ex(..., allow_special=0)` cannot inject `<|im_start|>system` or `<|im_end|>`.
- **Constitutional Output Governor (`v3_governor` / `CAAIGovernor`):** Deterministic $O(1)$-arena pre-sampling logit filter (`top_k_guard = 64`) that normalizes zero-width characters, separators (`_`, `-`, `.`), Greek/Cyrillic homoglyphs, character reversal, 4-phase Base64/Base64URL streams, and 2-phase Hex (`0xNN` / `\xNN`) byte streams, masking any token that completes a forbidden `SYSCALL_EXEC(` directive to `-1e30f`.

---

## 6. Freestanding C99 Core vs. Host CLI Separation (`runtime/vault3.c` vs `runtime/vault3_host.c`)

### 6.1 Architectural Boundary (`vault3.c` Bare-Metal Core vs `vault3_host.c` POSIX CLI)
To maintain strict terminology precision across embedded/OS-independent and Linux desktop targets:
- **`runtime/vault3.c` + `runtime/vault3.h` (True OS-Independent Freestanding C99 Core):** Includes **only** `<stddef.h>`, `<stdint.h>`, `"vault3_nfc_tables.h"`, and optional ISA SIMD headers (`<immintrin.h>` on x86 AVX2, `<arm_neon.h>` on ARM64 NEON). Contains **zero** OS syscalls, **zero** `libc` calls, **zero** heap allocation, and `< 2 KB` stack frames (`-Wframe-larger-than=2048`), Verified via `nm -u == 0` across scalar C99, `x86_64 AVX2+FMA`, and `Clang AArch64 NEON (-Werror)`. Parallel row-dot dispatch is injected via an optional caller callback (`s->par_for`).
- **`runtime/vault3_host.c` (Optional POSIX/Linux Host CLI Harness):** Provides `mmap` file mapping, `pthread` worker-pool implementation for `s->par_for`, `/proc/self/status` `VmHWM` telemetry, and interactive CLI flags (`--chat`, `--bench`, `--verify`) for desktop benchmarking.

```bash
# Portable scalar freestanding object (no OS, no libc, no SIMD required):
gcc -std=c99 -O2 -ffreestanding -fno-stack-protector -nostdlib -mno-red-zone -DV3_FORCE_SCALAR_MATH -Wall -Wextra -Wframe-larger-than=2048 -c runtime/vault3.c -o vault3_scalar.o

# x86_64 AVX2 + FMA + F16C freestanding object:
gcc -std=c99 -O2 -ffreestanding -fno-stack-protector -nostdlib -mno-red-zone -mavx2 -mfma -mf16c -Wall -Wextra -Wframe-larger-than=2048 -c runtime/vault3.c -o vault3_avx2.o

# ARM64 NEON freestanding object:
clang --target=aarch64-linux-gnu -std=c99 -O2 -ffreestanding -nostdlib -Wall -Wextra -Werror -Wframe-larger-than=2048 -c runtime/vault3.c -o vault3_neon.o
```

### 6.2 Deterministic Memory Layout & Static Arena Sizing
All mutable memory is owned by the caller and passed in as **`64`-byte aligned** buffers:
1. **Read-Only Model Buffer (`buf`, `64`-byte aligned):**
   - `BF16`: `3,096.5 MB` | `Q8_0`: `1,649.4 MB` | Hybrid `Q4_0` / `P4_0`: `1,014.8 MB`
2. **Inference State Arena (`v3_state_arena_size(&m)` = `310.4 MB` for `W=1024, seg_budget=1024, sink_cap=512`):**
   - Strictly constant at `128` tokens, `8,192` tokens, `100,400` tokens, or `10,000,000` tokens (`v3_state_arena_size_ex(&m, max_seg_budget)` sizes custom budgets).
3. **Tokenizer Arena (`v3_tok_arena_size(&m)` = `9.7 MB`):**
   - Holds the `524,288`-slot merge hash table, $O(n \log n)$ BPE priority heap, `>32 KB` sliding-window scratch, and UAX #15 NFC buffers.

---

## 7. Verified Benchmark Summary

Raw JSON outputs are stored in `benchmarks/vault_v3_full_benchmark.json`, `benchmarks/vault_v3_round4_c_verification.json`, `benchmarks/vault_v3_round4_gpu_verification.json`, and `benchmarks/vault_v3_p4_primal_downstream.json`.

- **4-Way Numerical Parity (`tests/test_vlt3_runtime.py`):** PyTorch `forward == prefill+step == step == C99` across `F32 / BF16 / Q8_0 / Q4_0 / P4_0` and all Round-4 2-hop, rollback, Tier-B, sub-clause, orphan-halo, and divisive-budget paths (`max |Δlogit| <= 3.56e-5`, **`100.0%` top-1 agreement**, `1T == 8T` `0.0` ULP bit-identical on both `Q4_0` and `P4_0`, `12/12` corrupt/NaN/legacy `.vlt3` files rejected including unverified interior `P4_0` NaN scales), plus freestanding C99 ↔ PyTorch parity on `runtime/kenyon_cells_archive.h` (`max |Δ| <= 3.10e-7`).
- **8K Multi-Fact Retrieval (26 evicted facts > 2W):** **`23/26 (88.5%)`** at `B = 1024` vs `14/26 (53.8%)` for full-attention Qwen2.5-1.5B-Instruct.
- **Adversarial Long-Context Suite:** `8/8` Chunk-0 Cliff, `10/10` Newline-Split KV (`B = 512` & `1024`), `8/8` Dense Homogeneous Registry (`n = 8, 24, 80 @ B = 2048`) & `7/8` (`n = 120`), `6/8` Pure-Alphabetic ID-Free Registry (`B = 1024`), `7/8` 2-Hop Evicted Relational Join (`n = 8 @ B = 1024` & `n = 32 @ B = 2048`), `6/6` across all four Contradictory & Indirect Rollback chains, `100%` Scorer-Sponge (`0..120`) & `20x` Chaff survival, `8/8` Minified JSON + `8/8` Pipe secret, and **`8/8 (100%)` at `100.4K` tokens** (flat $O(1)$ GPU VRAM).
- **Prime-Harmonic Hybrid `P4_0` vs Linear `Q4_0` (`1,014.8 MB` each, `benchmarks/vault_v3_p4_primal_downstream.json`):**
  - **WikiText-2 Raw Test (`L = 2048`):** `P4_0` zero-shot `17.325` Word PPL (`37.6%` of `Q4_0` gap removed) → Stage-A Hessian `17.105` (`54.0%`) → Stage-B shadowless PRIMAL polish **`17.062`** (`BPB 0.7632`, **`57.3%`** of `Q4_0`'s `17.827` gap to `BF16`'s `16.491` eliminated).
  - **Downstream QA & Reasoning (8-Task MC Avg):** `P4_0 PRIMAL` **`72.23%`** vs `Q4_0` `71.83%` (`BF16` `72.40%` — `70.2%` of the 4-bit MC gap recovered; beats `Q4_0` on `ARC-Easy` `77.44%` vs `76.01%`, `SciQ-CB` `85.80%` vs `83.20%`, `SciQ-RC` `95.00%` vs `94.80%`, `HellaSwag` `60.00%` vs `59.87%`, `Winogrande` `62.90%` vs `62.75%`); 9-task avg including fiction `LAMBADA` (`57.20%` vs `Q4_0` `63.00%` due to `FineWeb-Edu`-only calibration) is `70.56%` vs `70.85%`.
  - **Long-Context Retrieval & Alignment (`B = 1024`, `2.4K–100.4K`):** `P4_0 PRIMAL` scores **`66/82 (80.5%)`** overall vs `Q4_0` `60/82 (73.2%)` and `BF16` `65/82 (79.3%)` — including `8/8` Dense-24, `15/26` 8K 26-needles (vs `Q4_0` `13/26`), `7/8` 2-hop join (vs `Q4_0` `4/8`), `18/18` temporal/rollback, `8/8` minified JSON (vs `Q4_0` `7/8`), `2/2` at `32K` & `100.4K` in flat **`3.580 GiB`** VRAM (`1,805 tok/s`), and **`6/10`** strict bilingual negative constraints (vs `Q4_0` `4/10`, `BF16` `5/10`) with `0` leaks.
- **Freestanding C99 CPU Performance & Sanitizers:** Hybrid `P4_0` (`1,014.8 MB`, 1-cycle AVX2 `VPSHUFB`) reaches **`54.02 tok/s`** (`18.51 ms/tok`) at `24` threads vs Hybrid `Q4_0` **`50.29 tok/s`** (`19.89 ms/tok`, `+7.4%` faster) with **`1.27 GB`** flat RSS (`310.4 MB` state arena); `Q8_0` (`1,649.4 MB`) reaches **`35.97 tok/s`** with **`1.88 GB`** RSS; `BF16` (`3,096.5 MB`) runs in **`3.26 GB`** RSS; `ASan + LSan + UBSan`, pure scalar C99, and `TSan` (`8T`) `0` errors; `12/12` malformed/NaN/legacy `.vlt3` inputs rejected (`rc = 2`); `0` OPI governance leaks.
- **Sub-GB `V3_Q2_0` (`680.9 MB`) vs. Ternary `V3_T1_58` (`704.2 MB`) on `VAULT-v3-DeepSeek-R1-1.5B` (`benchmarks/results/vault_v3_deepseek_1_5b_q2_outlier_report.json`):**
  - **Bit-Level C99 Parity:** `max |Δlogit| = 7.15e-07` (`AVX2+FMA`) and `1.07e-06` (`Scalar C99`) vs. PyTorch on `V3_Q2_0`; `0` undefined external symbols across scalar C99, `AVX2+FMA`, and `Clang AArch64 NEON (-Werror)`.
  - **Speed, Footprint & Quality Win over `T1_58_G32`:** Pairing `V3_Q2_0` (`{-2, -1, 0, +1}`, signed `FP16` scale, `18 B / 64 w = 2.25 bpw`) with offline $H_{128}$ GQA `V <-> O` fusion, `up <-> down` column equilibration, scale-invariant LM-damped `GPTQ`, and Proximal Gauss-Newton `PRIMAL` KL-distillation (`0.95` leaky `int8` vote decay, zero hard channel/layer locks) shrinks `.vlt3` size from `704.24 MB` to **`680.94 MB` (`0.634 GiB`)** (`93.28%` of body in 2-bit), lowers peak `VmHWM` RAM from `931.6 MiB` to **`910.43 MiB` (`0.889 GiB`)**, accelerates `24T` C99 decode by **`+36.5%`** (`40.12 ->` **`54.78 tok/s`**, `18.25 ms/tok`, surpassing `P4_0`!), cuts `WT2` Word PPL from `131.37` to **`117.56`**, improves `GSM8K-200` to **`31.50%`** (`41.26%` on completed chains), and achieves **`7 / 8` (`87.5%`)** on the Adversarial `O(1)` NIAH Stress Suite (`100%` across `16K`, `32K` `2W` rollover boundary, `100K @ 85%` depth, and `32K` Chronos variable supersession; `P4_0` scores **`8 / 8 = 100.0%`**).
