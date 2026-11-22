#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
`VAULT + PRIMAL` 3-Architecture Text-to-Speech (TTS) Challenge Benchmark:
  1. `Tiny-Voca` (`n4ze3m/tiny-voca`, `4.54M` params, `24 kHz`, Voices `F2` & `M2`):
       Micro-Transformer (`RoPE`) + Duration/Prosody Predictors + Source-Filter Cepstral Synthesizer.
  2. `VITS` (`facebook/mms-tts-eng`, `29.04M` active inference params, `16 kHz`):
       Pure 1D CNN (`87.5%` dilated `Conv1d` WaveNet Normalizing Flow + `9.2%` `ConvTranspose1d` HiFi-GAN Vocoder).
  3. `F5-TTS` (`SWivid/F5-TTS` `F5TTS_v1_Base`, `335.9M` params, `24 kHz` Zero-Shot Voice Cloning):
       `4` `ConvNeXt V2` CNN Text Blocks + `22`-Layer `DiT` (`adaLN-Zero`) Conditional Flow-Matching Diffusion + `Vocos`.

Evaluates 5 operating points per architecture on 12 held-out test utterances:
  - `FP32` Upstream Reference
  - Standard Linear `Q4_0` (`4.50 bpw`, `18 B / 32 w`)
  - `VAULT V3_P4_0` Prime-Harmonic LUT (`4.50 bpw`, Zero-Shot PTQ — exact byte parity with `Q4_0`)
  - `VAULT V3_P4_0 + Shadowless PRIMAL` (`4.50 bpw`, Multi-Res STFT / Velocity-Field discrete distillation)
  - `VAULT V3_Q2_0 + Shadowless PRIMAL` (`2.25 bpw`, `{-2, -1, 0, +1}` extreme sub-3-bit frontier)

Metrics:
  - Serialized Payload Size (`MB`) & Compression Factor vs. `FP32`
  - `openai/whisper-small.en` Round-Trip ASR Word Error Rate (`WER %`) & Character Error Rate (`CER %`)
  - DTW-Aligned 80-Band Log-Mel Spectral Distortion (`dB` vs. `FP32`) & STFT Spectral Convergence
  - Synthesis Speed (`x Real-Time`) + Saved `.wav` Audio Samples in `dist/tts_challenge_samples/`.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import sys
import time
from importlib.resources import files
from typing import Any, Dict, List, Tuple

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import VitsModel, VitsTokenizer, WhisperForConditionalGeneration, WhisperProcessor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from tts.primal_multimodal_ops import (  # noqa: E402
    MultiResolutionSTFTLoss,
    apply_ptq_to_model,
    encode_p4_0_tensors,
    evaluate_tts_waveforms,
    finalize_primal_modules,
    install_primal_modules,
)

REPORT_JSON = os.path.join(REPO_ROOT, "benchmarks", "results", "vault_primal_tts_trifecta_report.json")
SAMPLES_DIR = os.path.join(REPO_ROOT, "dist", "tts_challenge_samples")

# 12 Held-Out Evaluation Sentences (strictly disjoint from all calibration prompts!)
EVAL_SENTENCES: List[str] = [
    "Hello! This is a full text to speech quantization benchmark running locally on consumer hardware.",
    "The birch canoe slid on the smooth planks while the sheet glued to the dark blue background.",
    "Four hours of steady work faced us before the bright sun finally dipped below the horizon.",
    "Did you really think discrete integer optimization could compress speech synthesis down to two bits?",
    "Pack my box with five dozen liquor jugs and verify every consonant stays crisp and clear.",
    "She had never seen the ocean before, and for a long moment she stood quietly near the waves.",
    "Prime harmonic quantization places finer resolution near zero while capturing heavy filter outliers.",
    "Please remember to bring warm jackets, fresh water, and flashlights for tomorrow morning.",
    "Artificial intelligence systems become dramatically faster when weights fit inside processor cache.",
    "The quick brown fox jumps gracefully over the lazy dog near the old stone bridge.",
    "Seven hundred twenty four engineers tested thirty six acoustic models across twelve spoken prompts.",
    "Music and human speech share delicate harmonic structures that reward careful spectral supervision.",
]

# Held-Out Calibration Corpus for Shadowless PRIMAL Distillation (zero overlap with EVAL_SENTENCES!)
CALIB_SENTENCES: List[str] = [
    "A large size in stockings is hard to sell during the warm summer months.",
    "The boy was there when the sun rose over the quiet mountain village.",
    "A rod is used to catch pink salmon in the clear northern river.",
    "The source of the huge river is the clear spring high in the hills.",
    "Kick the ball straight and follow through across the green grass field.",
    "Help the woman get back to her feet after crossing the icy street.",
    "A pot of tea helps to pass the evening with old friends and family.",
    "Smoky fires lack flame and steady heat when the wood is still damp.",
    "The soft cushion broke the man's fall from the wooden step ladder.",
    "The salt breeze came across from the sea and cooled the harbor town.",
    "The girl at the booth sold fifty bonds before noon on Friday.",
    "The small pup gnawed a hole in the sock beside the kitchen door.",
    "The fish twisted and turned on the bent hook near the wooden boat.",
    "Press the pants and sew a button on the vest before the concert tonight.",
    "The swan dive was far short of perfect, yet the crowd applauded warmly.",
    "The beauty of the view stunned the young boy as the clouds parted.",
    "Two blue fish swam in the tank beneath the bright aquarium lamp.",
    "Her purse was full of useless trash and old paper train tickets.",
    "The colt reared and threw the tall rider onto the soft sandy track.",
    "It snowed, rained, and hailed the same morning across the valley.",
    "Read verse out loud for pleasure and clear articulation every day.",
    "Hoist the load to your left shoulder and carry it up the stairs.",
    "Take the winding path to reach the lake before sunset arrives.",
    "Note closely the size of the gas tank before driving across the desert.",
]


def _save_wav(path: str, wav: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = np.asarray(wav, dtype=np.float32).reshape(-1)
    mx = float(np.max(np.abs(w))) if w.size else 0.0
    if mx > 0.99:
        w = w * (0.99 / mx)
    sf.write(path, w, sr)


def _run_primal_loop(
    student: nn.Module,
    p_mods: Dict[str, Any],
    loss_step_fn,
    steps: int,
    lr_scale: float = 1.2e-2,
    lr_aux: float = 1.5e-4,
    warmup: int = 8,
    max_flips_hi: int = 24,
    tag: str = "PRIMAL",
) -> Dict[str, Any]:
    t0 = time.time()
    lm_params = [m.log_mul for m in p_mods.values()]
    aux_params = [p for n, p in student.named_parameters() if p.requires_grad and not n.endswith("log_mul")]
    opt = torch.optim.AdamW(
        [
            {"params": lm_params, "lr": lr_scale, "weight_decay": 0.0},
            {"params": aux_params, "lr": lr_aux, "weight_decay": 0.0},
        ]
    )

    tot_flips = 0
    loss_hist = []
    best_loss_ema = float("inf")
    ema_loss = None

    for step in range(1, steps + 1):
        if step <= warmup:
            lr_m = step / float(warmup)
        else:
            prog = (step - warmup) / float(max(1, steps - warmup))
            lr_m = 0.10 + 0.90 * (0.5 * (1.0 + math.cos(math.pi * prog)))
        opt.param_groups[0]["lr"] = lr_scale * lr_m
        opt.param_groups[1]["lr"] = lr_aux * lr_m

        prog_all = (step - 1) / float(max(1, steps - 1))
        cos_v = 0.5 * (1.0 + math.cos(math.pi * prog_all))
        v_gain = (0.65 + 0.65 * cos_v) if step > warmup else 0.0
        thresh = int(round(22 - 4 * cos_v))
        max_flips = max(8, int(round(8 + (max_flips_hi - 8) * cos_v)))
        for m in p_mods.values():
            m.vote_gain = v_gain

        opt.zero_grad(set_to_none=True)
        loss = loss_step_fn(step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm_params + aux_params, 1.0)
        opt.step()

        lv = float(loss.detach().item())
        ema_loss = lv if ema_loss is None else (0.85 * ema_loss + 0.15 * lv)
        best_loss_ema = min(best_loss_ema, ema_loss)

        flips = sum(m.apply_discrete_step(thresh, max_flips) for m in p_mods.values()) if step > warmup else 0
        tot_flips += flips

        if step == 1 or step % 20 == 0 or step == steps:
            print(
                f"  [{tag} Step {step:3d}/{steps}] loss={lv:.4f} (ema={ema_loss:.4f}) | discrete_flips={flips:,} ({time.time()-t0:.1f}s)",
                flush=True,
            )
            loss_hist.append({"step": step, "loss": round(lv, 4), "ema_loss": round(ema_loss, 4), "flips": flips})

    for m in p_mods.values():
        m.vote_gain = 0.0
    return {
        "steps": steps,
        "total_discrete_flips": tot_flips,
        "final_ema_loss": round(float(ema_loss or 0.0), 4),
        "distill_seconds": round(time.time() - t0, 2),
        "logs": loss_hist,
    }


# ==============================================================================
# CHALLENGE 1: Tiny-Voca (`4.54M` Params — Transformer + Source-Filter TTS)
# ==============================================================================
def run_tiny_voca_challenge(whisper_proc, whisper_model) -> Dict[str, Any]:
    print("\n================================================================================", flush=True)
    print("=== [MODEL 1/3] Tiny-Voca (4.54M Params | 24 kHz | Voices F2 & M2)           ===", flush=True)
    print("================================================================================", flush=True)
    tv_dir = snapshot_download("n4ze3m/tiny-voca")
    os.environ["DIZIONE_DATA"] = os.path.join(tv_dir, "phonemizer")
    from tiny_voca.infer import load_model as load_tv, match_frontend, speak_text
    from tiny_voca.text import phonemize

    tv_pt = os.path.join(tv_dir, "tiny-voca.pt")
    match_frontend(tv_pt)
    teacher = load_tv(tv_pt, device="cuda")
    sr = int(teacher.config.sample_rate)

    def _synth_all(m_tv: nn.Module) -> Tuple[List[np.ndarray], float]:
        m_tv.eval()
        t0 = time.time()
        outs = []
        for i, txt in enumerate(EVAL_SENTENCES):
            voice = "F2" if (i % 2 == 0) else "M2"
            wav, _ = speak_text(m_tv, txt, voice, seed=42 + i)
            outs.append(wav.detach().cpu().numpy())
        return outs, time.time() - t0

    wavs_fp32, t_fp32 = _synth_all(teacher)
    _save_wav(os.path.join(SAMPLES_DIR, "tiny_voca", "00_fp32.wav"), wavs_fp32[0], sr)
    ev_fp32 = evaluate_tts_waveforms(wavs_fp32, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_fp32)
    tot_p = sum(p.numel() for p in teacher.parameters())
    print(f"[Tiny-Voca FP32 (18.15 MB)] WER={ev_fp32['whisper_wer_pct']:.2f}% | Speed={ev_fp32['x_realtime']}x RT", flush=True)

    # Precompute aligned teacher targets on 24 calibration utterances (alternating F2 / M2)
    cal_items = []
    with torch.no_grad():
        for ci, ctxt in enumerate(CALIB_SENTENCES):
            ph = phonemize(ctxt)
            toks = torch.tensor([ph.ids], device="cuda", dtype=torch.long)
            strs = torch.tensor([ph.stress], device="cuda", dtype=torch.long)
            lens = torch.tensor([len(ph)], device="cuda", dtype=torch.long)
            spk = torch.tensor([ci % 2], device="cuda", dtype=torch.long)
            ht, msk = teacher.encode(toks, strs, lens, spk)
            dur_raw_t = teacher.duration(ht, msk)
            dur_t = teacher.predict_durations(ht, msk, toks)
            ft, fmsk, flen = teacher.expand(ht, dur_t, msk, spk)
            pht, prost = teacher.predict_prosody(ft, fmsk)
            act = teacher.decode(pht, teacher.hard_prosody(prost), fmsk)
            harm_t = teacher.vocoder.harmonic(act.float())
            nois_t = teacher.vocoder.noise(act.float())
            gen = torch.Generator(device="cuda").manual_seed(900 + ci)
            noise_fix = torch.randn((1, int(flen[0].item()) * teacher.config.hop), generator=gen, device="cuda")
            wav_t = teacher.vocoder(act, teacher.frequency(prost[..., 0]), flen, noise=noise_fix)
            cal_items.append((toks, strs, lens, spk, dur_raw_t, dur_t, prost, harm_t, nois_t, noise_fix, wav_t))

    stft_fn = MultiResolutionSTFTLoss().cuda()

    # 1. Q4_0 Linear PTQ
    m_q4, _, st_q4 = apply_ptq_to_model(teacher, "q4_0", device="cuda")
    w_q4, t_q4 = _synth_all(m_q4)
    _save_wav(os.path.join(SAMPLES_DIR, "tiny_voca", "01_q4_0.wav"), w_q4[0], sr)
    ev_q4 = evaluate_tts_waveforms(w_q4, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_q4)
    print(f"[Tiny-Voca Q4_0 ({st_q4['packed_mb']} MB)] WER={ev_q4['whisper_wer_pct']:.2f}% | MelDist={ev_q4['log_mel_distortion_db']} dB | SC={ev_q4['stft_spectral_convergence']}", flush=True)
    del m_q4

    # 2. V3_P4_0 Prime-Harmonic PTQ + PRIMAL Distillation
    m_p4, qd_p4, st_p4 = apply_ptq_to_model(teacher, "p4_0", device="cuda")
    w_p4_ptq, t_p4_ptq = _synth_all(m_p4)
    ev_p4_ptq = evaluate_tts_waveforms(w_p4_ptq, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_p4_ptq)
    print(f"[Tiny-Voca V3_P4_0 PTQ ({st_p4['packed_mb']} MB)] WER={ev_p4_ptq['whisper_wer_pct']:.2f}% | MelDist={ev_p4_ptq['log_mel_distortion_db']} dB | SC={ev_p4_ptq['stft_spectral_convergence']}", flush=True)

    pmods_p4 = install_primal_modules(m_p4, qd_p4, fmt="p4_0", grad_accum=1, device="cuda")

    def _tv_loss_fn( m_stu ):
        def _step(step_idx: int) -> torch.Tensor:
            m_stu.eval()  # disable dropout during deterministic acoustic distillation!
            acc = torch.zeros((), device="cuda", dtype=torch.float32)
            for k in range(2):
                toks, strs, lens, spk, dur_rt, dur_t, prost, harm_t, nois_t, nfix, wav_t = cal_items[(2 * step_idx + k) % len(cal_items)]
                hs, msk = m_stu.encode(toks, strs, lens, spk)
                dur_rs = m_stu.duration(hs, msk)
                fs, fmsk, flen = m_stu.expand(hs, dur_t, msk, spk)
                phs, pross = m_stu.predict_prosody(fs, fmsk)
                acs = m_stu.decode(phs, m_stu.hard_prosody(pross), fmsk)
                harm_s = m_stu.vocoder.harmonic(acs.float())
                nois_s = m_stu.vocoder.noise(acs.float())
                wav_s = m_stu.vocoder(acs, m_stu.frequency(pross[..., 0]), flen, noise=nfix)
                l_i = (
                    stft_fn(wav_s, wav_t)
                    + 0.6 * F.l1_loss(pross, prost)
                    + 0.3 * F.l1_loss(dur_rs, dur_rt)
                    + 0.25 * F.l1_loss(harm_s, harm_t)
                    + 0.25 * F.l1_loss(nois_s, nois_t)
                )
                acc = acc + 0.5 * l_i
            return acc
        return _step

    pr_p4 = _run_primal_loop(m_p4, pmods_p4, _tv_loss_fn(m_p4), steps=80, lr_scale=1.0e-2, max_flips_hi=20, tag="TinyVoca-P4_0-PRIMAL")
    finalize_primal_modules(m_p4, pmods_p4)
    w_p4_pr, t_p4_pr = _synth_all(m_p4)
    _save_wav(os.path.join(SAMPLES_DIR, "tiny_voca", "02_v3_p4_0_primal.wav"), w_p4_pr[0], sr)
    ev_p4_pr = evaluate_tts_waveforms(w_p4_pr, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_p4_pr)
    print(f"[Tiny-Voca V3_P4_0 + PRIMAL ({st_p4['packed_mb']} MB)] WER={ev_p4_pr['whisper_wer_pct']:.2f}% | MelDist={ev_p4_pr['log_mel_distortion_db']} dB | SC={ev_p4_pr['stft_spectral_convergence']}", flush=True)
    del m_p4, pmods_p4

    # 3. Extreme 2-Bit V3_Q2_0 + PRIMAL (1.39 MB total!)
    m_q2, qd_q2, st_q2 = apply_ptq_to_model(teacher, "q2_0", device="cuda")
    w_q2_ptq, t_q2_ptq = _synth_all(m_q2)
    ev_q2_ptq = evaluate_tts_waveforms(w_q2_ptq, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_q2_ptq)
    print(f"[Tiny-Voca V3_Q2_0 PTQ ({st_q2['packed_mb']} MB)] WER={ev_q2_ptq['whisper_wer_pct']:.2f}% | MelDist={ev_q2_ptq['log_mel_distortion_db']} dB", flush=True)

    pmods_q2 = install_primal_modules(m_q2, qd_q2, fmt="q2_0", grad_accum=1, device="cuda")
    pr_q2 = _run_primal_loop(m_q2, pmods_q2, _tv_loss_fn(m_q2), steps=120, lr_scale=1.5e-2, max_flips_hi=28, tag="TinyVoca-Q2_0-PRIMAL")
    finalize_primal_modules(m_q2, pmods_q2)
    w_q2_pr, t_q2_pr = _synth_all(m_q2)
    _save_wav(os.path.join(SAMPLES_DIR, "tiny_voca", "03_v3_q2_0_primal.wav"), w_q2_pr[0], sr)
    ev_q2_pr = evaluate_tts_waveforms(w_q2_pr, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_q2_pr)
    print(f"[Tiny-Voca V3_Q2_0 + PRIMAL ({st_q2['packed_mb']} MB)] WER={ev_q2_pr['whisper_wer_pct']:.2f}% | MelDist={ev_q2_pr['log_mel_distortion_db']} dB | SC={ev_q2_pr['stft_spectral_convergence']}", flush=True)

    del m_q2, pmods_q2, teacher, cal_items
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "architecture": "Micro-Transformer (RoPE) + Source-Filter Cepstral Synthesizer (24 kHz, Voices F2 & M2)",
        "active_params": tot_p,
        "fp32_baseline": {"size_mb": round(tot_p * 4 / 1e6, 2), **ev_fp32},
        "q4_0_ptq": {"size": st_q4, **ev_q4},
        "v3_p4_0_ptq": {"size": st_p4, **ev_p4_ptq},
        "v3_p4_0_primal": {"size": st_p4, "primal": pr_p4, **ev_p4_pr},
        "v3_q2_0_ptq": {"size": st_q2, **ev_q2_ptq},
        "v3_q2_0_primal": {"size": st_q2, "primal": pr_q2, **ev_q2_pr},
    }


# ==============================================================================
# CHALLENGE 2: VITS (`29.04M` Active Params — 1D CNN + Normalizing Flow + HiFi-GAN)
# ==============================================================================
def run_vits_challenge(whisper_proc, whisper_model) -> Dict[str, Any]:
    print("\n================================================================================", flush=True)
    print("=== [MODEL 2/3] VITS (29.04M Active CNN/Flow/HiFi-GAN Params | 16 kHz)       ===", flush=True)
    print("================================================================================", flush=True)
    tok = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
    teacher = VitsModel.from_pretrained("facebook/mms-tts-eng").eval()

    # Fold all PyTorch weight_norm parametrizations offline before quantization!
    for _, mod in list(teacher.named_modules()):
        if hasattr(mod, "weight_g") and hasattr(mod, "weight_v"):
            nn.utils.remove_weight_norm(mod)
        elif hasattr(mod, "parametrizations") and "weight" in mod.parametrizations:
            nn.utils.parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)
    # Remove training-only posterior_encoder so we benchmark true inference VITS (~29.04M params)
    if hasattr(teacher, "posterior_encoder"):
        teacher.posterior_encoder = nn.Identity()
    teacher = teacher.to("cuda").eval()
    sr = int(teacher.config.sampling_rate)
    tot_p = sum(p.numel() for p in teacher.parameters())

    def _synth_all(m_v: nn.Module) -> Tuple[List[np.ndarray], float]:
        m_v.eval()
        t0 = time.time()
        outs = []
        with torch.no_grad():
            for i, txt in enumerate(EVAL_SENTENCES):
                inp = tok(txt, return_tensors="pt").to("cuda")
                torch.manual_seed(2026 + i)
                torch.cuda.manual_seed_all(2026 + i)
                wav = m_v(inp["input_ids"]).waveform[0]
                outs.append(wav.detach().cpu().numpy())
        return outs, time.time() - t0

    wavs_fp32, t_fp32 = _synth_all(teacher)
    _save_wav(os.path.join(SAMPLES_DIR, "vits_31m", "00_fp32.wav"), wavs_fp32[0], sr)
    ev_fp32 = evaluate_tts_waveforms(wavs_fp32, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_fp32)
    print(f"[VITS FP32 ({tot_p*4/1e6:.2f} MB, {tot_p/1e6:.2f}M params)] WER={ev_fp32['whisper_wer_pct']:.2f}% | Speed={ev_fp32['x_realtime']}x RT", flush=True)

    # Precompute aligned teacher latent/flow/waveform targets on 24 calibration sentences
    cal_vits = []
    with torch.no_grad():
        for ci, ctxt in enumerate(CALIB_SENTENCES):
            inp_ids = tok(ctxt, return_tensors="pt")["input_ids"].to("cuda")
            in_mask = torch.ones_like(inp_ids).unsqueeze(-1).float()
            te_t = teacher.text_encoder(input_ids=inp_ids, padding_mask=in_mask, return_dict=True)
            hs_t = te_t.last_hidden_state.transpose(1, 2)
            in_mask_t = in_mask.transpose(1, 2)
            pm_t, plv_t = te_t.prior_means, te_t.prior_log_variances

            torch.manual_seed(5000 + ci)
            torch.cuda.manual_seed_all(5000 + ci)
            log_dur = teacher.duration_predictor(hs_t, in_mask_t, None, reverse=True, noise_scale=teacher.noise_scale_duration)
            dur = torch.ceil(torch.exp(log_dur) * in_mask_t / teacher.speaking_rate)
            pred_len = torch.clamp_min(torch.sum(dur, [1, 2]), 1).long()
            out_idx = torch.arange(pred_len.max(), device="cuda")
            out_mask = (out_idx.unsqueeze(0) < pred_len.unsqueeze(1)).unsqueeze(1).float()
            attn_mask = torch.unsqueeze(in_mask_t, 2) * torch.unsqueeze(out_mask, -1)
            bsz, _, out_l, in_l = attn_mask.shape
            cum_d = torch.cumsum(dur, -1).view(bsz * in_l, 1)
            v_idx = (torch.arange(out_l, device="cuda").unsqueeze(0) < cum_d).float().view(bsz, in_l, out_l)
            pad_idx = v_idx - F.pad(v_idx, [0, 0, 1, 0, 0, 0])[:, :-1]
            attn = pad_idx.unsqueeze(1).transpose(2, 3) * attn_mask

            pm_exp = torch.matmul(attn.squeeze(1), pm_t).transpose(1, 2)
            plv_exp = torch.matmul(attn.squeeze(1), plv_t).transpose(1, 2)
            eps = torch.randn_like(pm_exp)
            pz_t = pm_exp + eps * torch.exp(plv_exp) * teacher.noise_scale
            lat_t = teacher.flow(pz_t, out_mask, None, reverse=True) * out_mask
            cal_vits.append((inp_ids, in_mask, hs_t, pm_t, plv_t, attn, out_mask, eps, lat_t))

    stft_fn = MultiResolutionSTFTLoss(fft_sizes=(256, 512, 1024, 2048)).cuda()
    upsample_prod = int(np.prod(teacher.config.upsample_rates))

    def _vits_loss_fn(m_stu):
        def _step(step_idx: int) -> torch.Tensor:
            m_stu.eval()
            inp_ids, in_mask, hs_t, pm_t, plv_t, attn, out_mask, eps, lat_t = cal_vits[step_idx % len(cal_vits)]
            te_s = m_stu.text_encoder(input_ids=inp_ids, padding_mask=in_mask, return_dict=True)
            hs_s = te_s.last_hidden_state.transpose(1, 2)
            pm_s, plv_s = te_s.prior_means, te_s.prior_log_variances

            pm_exp_s = torch.matmul(attn.squeeze(1), pm_s).transpose(1, 2)
            plv_exp_s = torch.matmul(attn.squeeze(1), plv_s).transpose(1, 2)
            pz_s = pm_exp_s + eps * torch.exp(plv_exp_s) * m_stu.noise_scale
            lat_s = m_stu.flow(pz_s, out_mask, None, reverse=True) * out_mask

            # Slice up to 160 latent frames (~2.56s audio) for fast, low-VRAM HiFi-GAN multi-res STFT supervision
            tot_f = lat_s.shape[-1]
            win_f = min(tot_f, 160)
            f0 = ((step_idx * 37) % max(1, tot_f - win_f + 1)) if tot_f > win_f else 0
            with torch.no_grad():
                wav_t_sub = teacher.decoder(lat_t[:, :, f0 : f0 + win_f], None).squeeze(1)
            wav_s_sub = m_stu.decoder(lat_s[:, :, f0 : f0 + win_f], None).squeeze(1)
            _ = upsample_prod

            return (
                stft_fn(wav_s_sub, wav_t_sub)
                + 0.5 * F.l1_loss(lat_s, lat_t)
                + 0.25 * (F.l1_loss(hs_s, hs_t) + F.l1_loss(pm_s, pm_t) + F.l1_loss(plv_s, plv_t))
            )
        return _step

    # 1. Q4_0 Linear PTQ on 99.4% of VITS (Conv1d + ConvTranspose1d + Linear)
    m_q4, _, st_q4 = apply_ptq_to_model(teacher, "q4_0", device="cuda")
    w_q4, t_q4 = _synth_all(m_q4)
    _save_wav(os.path.join(SAMPLES_DIR, "vits_31m", "01_q4_0.wav"), w_q4[0], sr)
    ev_q4 = evaluate_tts_waveforms(w_q4, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_q4)
    print(f"[VITS Q4_0 ({st_q4['packed_mb']} MB)] WER={ev_q4['whisper_wer_pct']:.2f}% | MelDist={ev_q4['log_mel_distortion_db']} dB | SC={ev_q4['stft_spectral_convergence']}", flush=True)
    del m_q4

    # 2. V3_P4_0 Prime-Harmonic PTQ + PRIMAL
    m_p4, qd_p4, st_p4 = apply_ptq_to_model(teacher, "p4_0", device="cuda")
    w_p4_ptq, t_p4_ptq = _synth_all(m_p4)
    ev_p4_ptq = evaluate_tts_waveforms(w_p4_ptq, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_p4_ptq)
    print(f"[VITS V3_P4_0 PTQ ({st_p4['packed_mb']} MB)] WER={ev_p4_ptq['whisper_wer_pct']:.2f}% | MelDist={ev_p4_ptq['log_mel_distortion_db']} dB | SC={ev_p4_ptq['stft_spectral_convergence']}", flush=True)

    pmods_p4 = install_primal_modules(m_p4, qd_p4, fmt="p4_0", grad_accum=1, device="cuda")
    pr_p4 = _run_primal_loop(m_p4, pmods_p4, _vits_loss_fn(m_p4), steps=80, lr_scale=1.0e-2, max_flips_hi=24, tag="VITS-P4_0-PRIMAL")
    finalize_primal_modules(m_p4, pmods_p4)
    w_p4_pr, t_p4_pr = _synth_all(m_p4)
    _save_wav(os.path.join(SAMPLES_DIR, "vits_31m", "02_v3_p4_0_primal.wav"), w_p4_pr[0], sr)
    ev_p4_pr = evaluate_tts_waveforms(w_p4_pr, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_p4_pr)
    print(f"[VITS V3_P4_0 + PRIMAL ({st_p4['packed_mb']} MB)] WER={ev_p4_pr['whisper_wer_pct']:.2f}% | MelDist={ev_p4_pr['log_mel_distortion_db']} dB | SC={ev_p4_pr['stft_spectral_convergence']}", flush=True)
    del m_p4, pmods_p4

    # 3. Extreme 2-Bit V3_Q2_0 + PRIMAL (keep tiny stochastic duration_predictor in P4_0 so durations don't collapse!)
    m_q2_base, qd_dp_p4, _ = apply_ptq_to_model(
        teacher, "p4_0", skip_fn=lambda n, _m: not n.startswith("duration_predictor"), device="cuda"
    )
    m_q2, qd_q2, st_q2 = apply_ptq_to_model(
        m_q2_base, "q2_0", skip_fn=lambda n, _m: n.startswith("duration_predictor"), device="cuda"
    )
    del m_q2_base
    # Adjust exact packed byte count for duration_predictor stored in P4_0 (4.5 bpw instead of FP16 16 bpw!)
    dp_p4_params = sum((q.shape[0] * q.shape[1]) for q, _ in qd_dp_p4.values())
    dp_orig_params = sum(
        dict(teacher.named_modules())[n].weight.numel() for n in qd_dp_p4
    )
    st_q2["packed_bytes"] = st_q2["packed_bytes"] - (dp_orig_params * 2) + ((dp_p4_params // 32) * 18)
    st_q2["packed_mb"] = round(st_q2["packed_bytes"] / 1e6, 2)
    st_q2["compression_vs_fp32_x"] = round((tot_p * 4) / st_q2["packed_bytes"], 2)
    st_q2["effective_bpw"] = round(8.0 * st_q2["packed_bytes"] / tot_p, 3)

    w_q2_ptq, t_q2_ptq = _synth_all(m_q2)
    ev_q2_ptq = evaluate_tts_waveforms(w_q2_ptq, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_q2_ptq)
    print(f"[VITS V3_Q2_0 PTQ ({st_q2['packed_mb']} MB)] WER={ev_q2_ptq['whisper_wer_pct']:.2f}% | MelDist={ev_q2_ptq['log_mel_distortion_db']} dB", flush=True)

    pmods_q2 = install_primal_modules(m_q2, qd_q2, fmt="q2_0", grad_accum=1, device="cuda")
    pr_q2 = _run_primal_loop(m_q2, pmods_q2, _vits_loss_fn(m_q2), steps=120, lr_scale=1.5e-2, max_flips_hi=32, tag="VITS-Q2_0-PRIMAL")
    finalize_primal_modules(m_q2, pmods_q2)
    w_q2_pr, t_q2_pr = _synth_all(m_q2)
    _save_wav(os.path.join(SAMPLES_DIR, "vits_31m", "03_v3_q2_0_primal.wav"), w_q2_pr[0], sr)
    ev_q2_pr = evaluate_tts_waveforms(w_q2_pr, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_q2_pr)
    print(f"[VITS V3_Q2_0 + PRIMAL ({st_q2['packed_mb']} MB)] WER={ev_q2_pr['whisper_wer_pct']:.2f}% | MelDist={ev_q2_pr['log_mel_distortion_db']} dB | SC={ev_q2_pr['stft_spectral_convergence']}", flush=True)

    del m_q2, pmods_q2, teacher, cal_vits
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "architecture": "1D CNN WaveNet Normalizing Flow + HiFi-GAN ConvTranspose1d Vocoder (facebook/mms-tts-eng, weight_norm folded)",
        "active_params": tot_p,
        "fp32_baseline": {"size_mb": round(tot_p * 4 / 1e6, 2), **ev_fp32},
        "q4_0_ptq": {"size": st_q4, **ev_q4},
        "v3_p4_0_ptq": {"size": st_p4, **ev_p4_ptq},
        "v3_p4_0_primal": {"size": st_p4, "primal": pr_p4, **ev_p4_pr},
        "v3_q2_0_ptq": {"size": st_q2, **ev_q2_ptq},
        "v3_q2_0_primal": {"size": st_q2, "primal": pr_q2, **ev_q2_pr},
    }


# ==============================================================================
# CHALLENGE 3: F5-TTS (`335.9M` Params — ConvNeXt V2 + 22L DiT Flow-Matching)
# ==============================================================================
def run_f5_tts_challenge(whisper_proc, whisper_model) -> Dict[str, Any]:
    print("\n================================================================================", flush=True)
    print("=== [MODEL 3/3] F5-TTS v1 Base (335.9M Params | ConvNeXt V2 + 22L DiT CFM)   ===", flush=True)
    print("================================================================================", flush=True)
    from f5_tts.infer.utils_infer import load_model as load_f5_model, load_vocoder
    from f5_tts.model import DiT
    from f5_tts.model.utils import convert_char_to_pinyin, get_epss_timesteps, lens_to_mask, list_str_to_idx

    f5_ckpt = hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors")
    f5_vocab = hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/vocab.txt")
    dit_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    cfm_teacher = load_f5_model(DiT, dit_cfg, f5_ckpt, mel_spec_type="vocos", vocab_file=f5_vocab, device="cuda")
    cfm_teacher = cfm_teacher.float().eval()  # exact FP32 master reference for clean velocity supervision
    vocoder = load_vocoder(vocoder_name="vocos", device="cuda")

    ref_wav_path = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
    ref_np, ref_sr = sf.read(ref_wav_path, dtype="float32")
    if ref_np.ndim > 1:
        ref_np = ref_np.mean(axis=1)
    if ref_sr != 24000:
        ref_np = librosa.resample(ref_np, orig_sr=ref_sr, target_sr=24000)
    sr = 24000
    target_rms = 0.1
    rms = float(np.sqrt(np.mean(np.square(ref_np))))
    if rms < target_rms:
        ref_np_norm = ref_np * (target_rms / rms)
    else:
        ref_np_norm = ref_np
    ref_audio_t = torch.from_numpy(ref_np_norm).unsqueeze(0).to("cuda", dtype=torch.float32)
    ref_text = "Some call me nature, others call me mother nature. "
    ref_audio_len = ref_audio_t.shape[-1] // 256
    ref_text_len = len(ref_text.encode("utf-8"))

    def _synth_f5_all(cfm_m: nn.Module, nfe_steps: int = 16) -> Tuple[List[np.ndarray], float]:
        cfm_m.eval()
        m_dtype = next(cfm_m.transformer.parameters()).dtype
        cond_in = ref_audio_t.to(dtype=m_dtype)
        t0 = time.time()
        outs = []
        with torch.no_grad():
            for i, gtxt in enumerate(EVAL_SENTENCES):
                t_list = convert_char_to_pinyin([ref_text + gtxt])
                dur = ref_audio_len + int(ref_audio_len / ref_text_len * len(gtxt.encode("utf-8")))
                gen_mel, _ = cfm_m.sample(
                    cond=cond_in,
                    text=t_list,
                    duration=dur,
                    steps=nfe_steps,
                    cfg_strength=2.0,
                    sway_sampling_coef=-1.0,
                    seed=7000 + i,
                )
                mel_slice = gen_mel.float()[:, ref_audio_len:, :].permute(0, 2, 1)
                wav = vocoder.decode(mel_slice)
                if rms < target_rms:
                    wav = wav * (rms / target_rms)
                outs.append(wav.squeeze().detach().cpu().numpy())
        return outs, time.time() - t0

    tot_p = sum(p.numel() for p in cfm_teacher.transformer.parameters())
    wavs_fp32, t_fp32 = _synth_f5_all(cfm_teacher, nfe_steps=16)
    _save_wav(os.path.join(SAMPLES_DIR, "f5_tts_300m", "00_fp32.wav"), wavs_fp32[0], sr)
    ev_fp32 = evaluate_tts_waveforms(wavs_fp32, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_fp32)
    print(f"[F5-TTS FP32 ({tot_p*4/1e6:.1f} MB, {tot_p/1e6:.1f}M params)] WER={ev_fp32['whisper_wer_pct']:.2f}% | Speed={ev_fp32['x_realtime']}x RT", flush=True)

    # Precompute real Flow-Matching ODE Trajectory Velocity Snapshots on 18 Calibration Sentences!
    print("  [F5-TTS] Collecting real ODE velocity-field snapshots across t in [0, 1]...", flush=True)
    vel_snaps = []
    with torch.no_grad():
        cond_mel_base = cfm_teacher.mel_spec(ref_audio_t).permute(0, 2, 1)  # [1, ref_len, 100]
        t_sched = get_epss_timesteps(16, device="cuda", dtype=torch.float32)
        t_sched = t_sched - 1.0 * (torch.cos(math.pi / 2.0 * t_sched) - 1.0 + t_sched)
        snap_indices = [0, 3, 6, 9, 12, 14]
        for ci, ctxt in enumerate(CALIB_SENTENCES[:18]):
            t_list = convert_char_to_pinyin([ref_text + ctxt])
            text_idx = list_str_to_idx(t_list, cfm_teacher.vocab_char_map).to("cuda")
            dur = ref_audio_len + int(ref_audio_len / ref_text_len * len(ctxt.encode("utf-8")))
            _, traj = cfm_teacher.sample(
                cond=ref_audio_t, text=t_list, duration=dur, steps=16, cfg_strength=2.0, sway_sampling_coef=-1.0, seed=8100 + ci
            )
            step_cond = F.pad(cond_mel_base, (0, 0, 0, dur - cond_mel_base.shape[1]), value=0.0)
            for si in snap_indices:
                xk = traj[si]  # [1, dur, 100]
                tk = t_sched[si]
                pred_cfg_t = cfm_teacher.transformer(
                    x=xk, cond=step_cond, text=text_idx, time=tk, mask=None, cfg_infer=True, cache=False
                )
                vc_t, vn_t = torch.chunk(pred_cfg_t, 2, dim=0)
                vel_snaps.append((xk.cpu(), step_cond.cpu(), text_idx.cpu(), float(tk.item()), vc_t.cpu(), vn_t.cpu()))
            del traj
    print(f"  [F5-TTS] Cached {len(vel_snaps)} ODE velocity snapshots across 18 trajectories!", flush=True)

    # Clear threading.local() inference cache so deepcopy in apply_ptq_to_model succeeds cleanly!
    cfm_teacher.transformer.clear_cache()
    cfm_teacher.transformer.__dict__.pop("_cache_local", None)

    def _f5_vel_loss_fn(dit_stu: nn.Module):
        def _step(step_idx: int) -> torch.Tensor:
            dit_stu.eval()
            acc = torch.zeros((), device="cuda", dtype=torch.float32)
            for k in range(2):
                xk_c, sc_c, ti_c, tk_val, vc_t_c, vn_t_c = vel_snaps[(2 * step_idx + k * 17) % len(vel_snaps)]
                xk = xk_c.to("cuda", dtype=torch.float32)
                sc = sc_c.to("cuda", dtype=torch.float32)
                ti = ti_c.to("cuda")
                tk = torch.tensor(tk_val, device="cuda", dtype=torch.float32)
                vc_t = vc_t_c.to("cuda", dtype=torch.float32)
                vn_t = vn_t_c.to("cuda", dtype=torch.float32)
                vcfg_t = vc_t + 2.0 * (vc_t - vn_t)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_s = dit_stu(x=xk, cond=sc, text=ti, time=tk, mask=None, cfg_infer=True, cache=False)
                vc_s, vn_s = torch.chunk(pred_s.float(), 2, dim=0)
                vcfg_s = vc_s + 2.0 * (vc_s - vn_s)
                loss_k = F.l1_loss(vcfg_s, vcfg_t) + 0.5 * F.mse_loss(vc_s, vc_t) + 0.25 * F.mse_loss(vn_s, vn_t)
                acc = acc + 0.5 * loss_k
            return acc
        return _step

    # 1. Q4_0 Linear PTQ across ALL DiT + ConvNeXt V2 + adaLN layers
    dit_q4, _, st_q4 = apply_ptq_to_model(cfm_teacher.transformer, "q4_0", device="cuda")
    cfm_s = copy.copy(cfm_teacher)
    cfm_s.transformer = dit_q4
    w_q4, t_q4 = _synth_f5_all(cfm_s, nfe_steps=16)
    _save_wav(os.path.join(SAMPLES_DIR, "f5_tts_300m", "01_q4_0.wav"), w_q4[0], sr)
    ev_q4 = evaluate_tts_waveforms(w_q4, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_q4)
    print(f"[F5-TTS Q4_0 ({st_q4['packed_mb']} MB)] WER={ev_q4['whisper_wer_pct']:.2f}% | MelDist={ev_q4['log_mel_distortion_db']} dB | SC={ev_q4['stft_spectral_convergence']}", flush=True)
    del dit_q4

    # 2. V3_P4_0 Prime-Harmonic LUT PTQ + Velocity-Field PRIMAL Distillation
    cfm_teacher.transformer.clear_cache()
    cfm_teacher.transformer.__dict__.pop("_cache_local", None)
    dit_p4, qd_p4, st_p4 = apply_ptq_to_model(cfm_teacher.transformer, "p4_0", device="cuda")
    cfm_s.transformer = dit_p4
    w_p4_ptq, t_p4_ptq = _synth_f5_all(cfm_s, nfe_steps=16)
    ev_p4_ptq = evaluate_tts_waveforms(w_p4_ptq, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_p4_ptq)
    print(f"[F5-TTS V3_P4_0 PTQ ({st_p4['packed_mb']} MB)] WER={ev_p4_ptq['whisper_wer_pct']:.2f}% | MelDist={ev_p4_ptq['log_mel_distortion_db']} dB | SC={ev_p4_ptq['stft_spectral_convergence']}", flush=True)

    dit_p4.checkpoint_activations = True
    pmods_p4 = install_primal_modules(dit_p4, qd_p4, fmt="p4_0", grad_accum=1, device="cuda")
    pr_p4 = _run_primal_loop(dit_p4, pmods_p4, _f5_vel_loss_fn(dit_p4), steps=90, lr_scale=8.0e-3, max_flips_hi=32, tag="F5TTS-P4_0-PRIMAL")
    finalize_primal_modules(dit_p4, pmods_p4)
    dit_p4.checkpoint_activations = False
    w_p4_pr, t_p4_pr = _synth_f5_all(cfm_s, nfe_steps=16)
    _save_wav(os.path.join(SAMPLES_DIR, "f5_tts_300m", "02_v3_p4_0_primal.wav"), w_p4_pr[0], sr)
    ev_p4_pr = evaluate_tts_waveforms(w_p4_pr, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_p4_pr)
    print(f"[F5-TTS V3_P4_0 + PRIMAL ({st_p4['packed_mb']} MB)] WER={ev_p4_pr['whisper_wer_pct']:.2f}% | MelDist={ev_p4_pr['log_mel_distortion_db']} dB | SC={ev_p4_pr['stft_spectral_convergence']}", flush=True)
    del dit_p4, pmods_p4
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Architecture-Aware 2-Bit V3_Q2_0 + PRIMAL:
    #    DiT Attention + FFN + ConvNeXt V2 (196M params) in 2-bit V3_Q2_0 (2.25 bpw) + adaLN modulation in V3_P4_0!
    def _is_adaln_or_io(n: str, _m: nn.Module) -> bool:
        return ("attn_norm" in n) or ("norm_out" in n) or ("time_embed" in n) or ("input_embed" in n) or ("proj_out" in n)

    cfm_teacher.transformer.clear_cache()
    cfm_teacher.transformer.__dict__.pop("_cache_local", None)
    dit_base_p4, qd_adaln, _ = apply_ptq_to_model(
        cfm_teacher.transformer, "p4_0", skip_fn=lambda n, m: not _is_adaln_or_io(n, m), device="cuda"
    )
    dit_q2, qd_q2, st_q2 = apply_ptq_to_model(
        dit_base_p4, "q2_0", skip_fn=_is_adaln_or_io, device="cuda"
    )
    del dit_base_p4
    adaln_p4_p = sum(q.numel() for q, _ in qd_adaln.values())
    adaln_orig_p = sum(dict(cfm_teacher.transformer.named_modules())[n].weight.numel() for n in qd_adaln)
    st_q2["packed_bytes"] = st_q2["packed_bytes"] - (adaln_orig_p * 2) + ((adaln_p4_p // 32) * 18)
    st_q2["packed_mb"] = round(st_q2["packed_bytes"] / 1e6, 2)
    st_q2["compression_vs_fp32_x"] = round((tot_p * 4) / st_q2["packed_bytes"], 2)
    st_q2["effective_bpw"] = round(8.0 * st_q2["packed_bytes"] / tot_p, 3)

    cfm_s.transformer = dit_q2
    w_q2_ptq, t_q2_ptq = _synth_f5_all(cfm_s, nfe_steps=16)
    ev_q2_ptq = evaluate_tts_waveforms(w_q2_ptq, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_q2_ptq)
    print(f"[F5-TTS V3_Q2_0 PTQ ({st_q2['packed_mb']} MB)] WER={ev_q2_ptq['whisper_wer_pct']:.2f}% | MelDist={ev_q2_ptq['log_mel_distortion_db']} dB", flush=True)

    # Jointly train BOTH Q2_0 DiT blocks AND P4_0 adaLN modulation scales/codes under Flow-Matching PRIMAL!
    dit_q2.checkpoint_activations = True
    pmods_q2 = install_primal_modules(dit_q2, qd_q2, fmt="q2_0", grad_accum=1, device="cuda")
    pmods_adaln = {}
    from tts.primal_multimodal_ops import PrimalUniversalModule, _get_submodule, _set_submodule
    for name, (q_c, d_c) in qd_adaln.items():
        orig = _get_submodule(dit_q2, name)
        pm = PrimalUniversalModule(orig, q_c.to("cuda"), d_c.to("cuda"), fmt="p4_0", grad_accum=1).to("cuda")
        _set_submodule(dit_q2, name, pm)
        pmods_adaln[name] = pm
    all_pmods = {**pmods_q2, **pmods_adaln}

    pr_q2 = _run_primal_loop(dit_q2, all_pmods, _f5_vel_loss_fn(dit_q2), steps=120, lr_scale=1.2e-2, max_flips_hi=36, tag="F5TTS-Q2_0-PRIMAL")
    finalize_primal_modules(dit_q2, all_pmods)
    dit_q2.checkpoint_activations = False
    w_q2_pr, t_q2_pr = _synth_f5_all(cfm_s, nfe_steps=16)
    _save_wav(os.path.join(SAMPLES_DIR, "f5_tts_300m", "03_v3_q2_0_primal.wav"), w_q2_pr[0], sr)
    ev_q2_pr = evaluate_tts_waveforms(w_q2_pr, wavs_fp32, EVAL_SENTENCES, sr, whisper_proc, whisper_model, t_q2_pr)
    print(f"[F5-TTS V3_Q2_0 + PRIMAL ({st_q2['packed_mb']} MB)] WER={ev_q2_pr['whisper_wer_pct']:.2f}% | MelDist={ev_q2_pr['log_mel_distortion_db']} dB | SC={ev_q2_pr['stft_spectral_convergence']}", flush=True)

    del dit_q2, all_pmods, cfm_teacher, vocoder, vel_snaps
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "architecture": "4x ConvNeXt V2 CNN + 22-Layer DiT (adaLN-Zero) Conditional Flow-Matching Diffusion + Vocos (24 kHz)",
        "active_params": tot_p,
        "fp32_baseline": {"size_mb": round(tot_p * 4 / 1e6, 2), **ev_fp32},
        "q4_0_ptq": {"size": st_q4, **ev_q4},
        "v3_p4_0_ptq": {"size": st_p4, **ev_p4_ptq},
        "v3_p4_0_primal": {"size": st_p4, "primal": pr_p4, **ev_p4_pr},
        "v3_q2_0_ptq": {"size": st_q2, **ev_q2_ptq},
        "v3_q2_0_primal": {"size": st_q2, "primal": pr_q2, **ev_q2_pr},
    }


def main():
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    print("=== Loading Objective Speech Evaluator (`openai/whisper-small.en` on GPU) ===", flush=True)
    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-small.en")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small.en", torch_dtype=torch.float16
    ).to("cuda").eval()

    if os.path.exists(REPORT_JSON):
        with open(REPORT_JSON) as f:
            results: Dict[str, Any] = json.load(f)
    else:
        results = {
            "eval_protocol": {
                "n_eval_sentences": len(EVAL_SENTENCES),
                "n_calib_sentences": len(CALIB_SENTENCES),
                "train_test_overlap": 0,
                "asr_judge": "openai/whisper-small.en",
            }
        }

    if "tiny_voca_4_5m" not in results:
        results["tiny_voca_4_5m"] = run_tiny_voca_challenge(whisper_proc, whisper_model)
        with open(REPORT_JSON, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        print("=== [SKIP] Loaded cached `tiny_voca_4_5m` results from JSON ===", flush=True)

    if "vits_31m" not in results:
        results["vits_31m"] = run_vits_challenge(whisper_proc, whisper_model)
        with open(REPORT_JSON, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        print("=== [SKIP] Loaded cached `vits_31m` results from JSON ===", flush=True)

    results["f5_tts_335m"] = run_f5_tts_challenge(whisper_proc, whisper_model)
    with open(REPORT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[ALL 3 TTS CHALLENGE ARCHITECTURES COMPLETE] Saved -> {REPORT_JSON}", flush=True)


if __name__ == "__main__":
    main()
