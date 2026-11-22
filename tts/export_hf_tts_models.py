#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
Exports true bit-packed `.safetensors` checkpoints (`V3_P4_0` & `V3_Q2_0 + PRIMAL`)
and static configuration/tokenizer/sample assets for all 3 Hugging Face TTS repositories:
  1. `dist/hf_vault_primal_tiny_voca_4_5m/`
  2. `dist/hf_vault_primal_vits_29m/`
  3. `dist/hf_vault_primal_f5_tts_337m/`
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import shutil
import sys
from importlib.resources import files
from typing import Dict, Tuple

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import VitsModel, VitsTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from tts.primal_multimodal_ops import (  # noqa: E402
    MultiResolutionSTFTLoss,
    PrimalUniversalModule,
    _get_submodule,
    _set_submodule,
    apply_ptq_to_model,
    finalize_primal_modules,
    install_primal_modules,
)
from tts.run_tts_trifecta_challenge import CALIB_SENTENCES, SAMPLES_DIR, _run_primal_loop  # noqa: E402
from tts.vault_safetensors import save_vault_safetensors  # noqa: E402

DIST_DIR = os.path.join(REPO_ROOT, "dist")


def _copy_samples(src_subdir: str, dst_repo_dir: str) -> None:
    sdir = os.path.join(SAMPLES_DIR, src_subdir)
    ddir = os.path.join(dst_repo_dir, "samples")
    os.makedirs(ddir, exist_ok=True)
    if os.path.isdir(sdir):
        for fn in sorted(os.listdir(sdir)):
            if fn.endswith(".wav"):
                shutil.copy2(os.path.join(sdir, fn), os.path.join(ddir, fn))


def export_tiny_voca() -> Dict[str, int]:
    out_dir = os.path.join(DIST_DIR, "hf_vault_primal_tiny_voca_4_5m")
    os.makedirs(out_dir, exist_ok=True)
    print("\n=== [EXPORT 1/3] Tiny-Voca (4.54M) -> VAULT SafeTensors ===", flush=True)

    tv_dir = snapshot_download("n4ze3m/tiny-voca")
    os.environ["DIZIONE_DATA"] = os.path.join(tv_dir, "phonemizer")
    from tiny_voca.infer import load_model as load_tv, match_frontend
    from tiny_voca.text import phonemize

    tv_pt = os.path.join(tv_dir, "tiny-voca.pt")
    raw_state = torch.load(tv_pt, map_location="cpu", weights_only=True)
    # Save standalone config + frontend metadata so users NEVER have to download the 18 MB FP32 .pt!
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(
            {
                "format_version": raw_state.get("format_version"),
                "frontend": raw_state.get("frontend"),
                "config": raw_state.get("config"),
                "architectures": ["TinyVoca"],
                "vault_formats": ["v3_p4_0", "v3_q2_0_primal"],
            },
            f,
            indent=2,
        )
    # Copy Apache-2.0 phonemizer assets directly into repo
    ph_dst = os.path.join(out_dir, "phonemizer")
    os.makedirs(ph_dst, exist_ok=True)
    ph_src = os.path.join(tv_dir, "phonemizer")
    for fn in os.listdir(ph_src):
        shutil.copy2(os.path.join(ph_src, fn), os.path.join(ph_dst, fn))

    match_frontend(tv_pt)
    teacher = load_tv(tv_pt, device="cuda")

    # 1. Export V3_P4_0 (2.64 MB, 3.95% WER == FP32, 0.6984 STFT SC, 113.9x RT)
    m_p4, qd_p4, _ = apply_ptq_to_model(teacher, "p4_0", device="cuda")
    qmap_p4 = {n: ("p4_0", q, d) for n, (q, d) in qd_p4.items()}
    p4_path = os.path.join(out_dir, "tiny_voca_v3_p4_0.safetensors")
    p4_bytes = save_vault_safetensors(
        m_p4, qmap_p4, p4_path, extra_meta={"model_type": "tiny_voca", "quant_variant": "v3_p4_0"}
    )
    print(f"  [Saved] {p4_path} -> {p4_bytes / 1e6:.3f} MB ({p4_bytes:,} bytes)", flush=True)
    del m_p4

    # 2. Export V3_Q2_0 + Shadowless PRIMAL (1.38 MB, 3.95% WER == FP32!)
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
    m_q2, qd_q2, _ = apply_ptq_to_model(teacher, "q2_0", device="cuda")
    pmods_q2 = install_primal_modules(m_q2, qd_q2, fmt="q2_0", grad_accum=1, device="cuda")

    def _tv_step(step_idx: int) -> torch.Tensor:
        m_q2.eval()
        acc = torch.zeros((), device="cuda", dtype=torch.float32)
        for k in range(2):
            toks, strs, lens, spk, dur_rt, dur_t, prost, harm_t, nois_t, nfix, wav_t = cal_items[(2 * step_idx + k) % len(cal_items)]
            hs, msk = m_q2.encode(toks, strs, lens, spk)
            dur_rs = m_q2.duration(hs, msk)
            fs, fmsk, flen = m_q2.expand(hs, dur_t, msk, spk)
            phs, pross = m_q2.predict_prosody(fs, fmsk)
            acs = m_q2.decode(phs, m_q2.hard_prosody(pross), fmsk)
            harm_s = m_q2.vocoder.harmonic(acs.float())
            nois_s = m_q2.vocoder.noise(acs.float())
            wav_s = m_q2.vocoder(acs, m_q2.frequency(pross[..., 0]), flen, noise=nfix)
            l_i = (
                stft_fn(wav_s, wav_t)
                + 0.6 * F.l1_loss(pross, prost)
                + 0.3 * F.l1_loss(dur_rs, dur_rt)
                + 0.25 * F.l1_loss(harm_s, harm_t)
                + 0.25 * F.l1_loss(nois_s, nois_t)
            )
            acc = acc + 0.5 * l_i
        return acc

    _run_primal_loop(m_q2, pmods_q2, _tv_step, steps=120, lr_scale=1.5e-2, max_flips_hi=28, tag="Export-TinyVoca-Q2_0")
    qmap_q2 = {n: ("q2_0", pm.q.detach().cpu(), pm.cur_d().detach().to(torch.float16).cpu()) for n, pm in pmods_q2.items()}
    finalize_primal_modules(m_q2, pmods_q2)
    q2_path = os.path.join(out_dir, "tiny_voca_v3_q2_0_primal.safetensors")
    q2_bytes = save_vault_safetensors(
        m_q2, qmap_q2, q2_path, extra_meta={"model_type": "tiny_voca", "quant_variant": "v3_q2_0_primal"}
    )
    print(f"  [Saved] {q2_path} -> {q2_bytes / 1e6:.3f} MB ({q2_bytes:,} bytes)", flush=True)

    _copy_samples("tiny_voca", out_dir)
    del m_q2, pmods_q2, teacher, cal_items
    gc.collect()
    torch.cuda.empty_cache()
    return {"p4_0_bytes": p4_bytes, "q2_0_bytes": q2_bytes}


def export_vits() -> Dict[str, int]:
    out_dir = os.path.join(DIST_DIR, "hf_vault_primal_vits_29m")
    os.makedirs(out_dir, exist_ok=True)
    print("\n=== [EXPORT 2/3] VITS (29.04M Active CNN/Flow/HiFi-GAN) -> VAULT SafeTensors ===", flush=True)

    tok = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
    tok.save_pretrained(out_dir)
    teacher = VitsModel.from_pretrained("facebook/mms-tts-eng").eval()
    teacher.config.save_pretrained(out_dir)

    for _, mod in list(teacher.named_modules()):
        if hasattr(mod, "weight_g") and hasattr(mod, "weight_v"):
            nn.utils.remove_weight_norm(mod)
        elif hasattr(mod, "parametrizations") and "weight" in mod.parametrizations:
            nn.utils.parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)
    if hasattr(teacher, "posterior_encoder"):
        teacher.posterior_encoder = nn.Identity()
    teacher = teacher.to("cuda").eval()

    # 1. Export V3_P4_0 (16.52 MB, 12.99% WER, 7.20% CER == FP32, -10.4% lower Mel distortion than Q4_0!)
    m_p4, qd_p4, _ = apply_ptq_to_model(teacher, "p4_0", device="cuda")
    qmap_p4 = {n: ("p4_0", q, d) for n, (q, d) in qd_p4.items()}
    p4_path = os.path.join(out_dir, "vits_v3_p4_0.safetensors")
    p4_bytes = save_vault_safetensors(
        m_p4, qmap_p4, p4_path, extra_meta={"model_type": "vits_mms_eng", "quant_variant": "v3_p4_0"}
    )
    print(f"  [Saved] {p4_path} -> {p4_bytes / 1e6:.3f} MB ({p4_bytes:,} bytes)", flush=True)
    del m_p4

    # 2. Export V3_Q2_0 + Multi-Res STFT PRIMAL (8.75 MB, rescues 145.2% -> 23.73% WER!)
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
    m_q2_base, qd_dp_p4, _ = apply_ptq_to_model(
        teacher, "p4_0", skip_fn=lambda n, _m: not n.startswith("duration_predictor"), device="cuda"
    )
    m_q2, qd_q2, _ = apply_ptq_to_model(
        m_q2_base, "q2_0", skip_fn=lambda n, _m: n.startswith("duration_predictor"), device="cuda"
    )
    del m_q2_base
    pmods_q2 = install_primal_modules(m_q2, qd_q2, fmt="q2_0", grad_accum=1, device="cuda")

    def _vits_step(step_idx: int) -> torch.Tensor:
        m_q2.eval()
        inp_ids, in_mask, hs_t, pm_t, plv_t, attn, out_mask, eps, lat_t = cal_vits[step_idx % len(cal_vits)]
        te_s = m_q2.text_encoder(input_ids=inp_ids, padding_mask=in_mask, return_dict=True)
        hs_s = te_s.last_hidden_state.transpose(1, 2)
        pm_s, plv_s = te_s.prior_means, te_s.prior_log_variances
        pm_exp_s = torch.matmul(attn.squeeze(1), pm_s).transpose(1, 2)
        plv_exp_s = torch.matmul(attn.squeeze(1), plv_s).transpose(1, 2)
        pz_s = pm_exp_s + eps * torch.exp(plv_exp_s) * m_q2.noise_scale
        lat_s = m_q2.flow(pz_s, out_mask, None, reverse=True) * out_mask

        tot_f = lat_s.shape[-1]
        win_f = min(tot_f, 160)
        f0 = ((step_idx * 37) % max(1, tot_f - win_f + 1)) if tot_f > win_f else 0
        with torch.no_grad():
            wav_t_sub = teacher.decoder(lat_t[:, :, f0 : f0 + win_f], None).squeeze(1)
        wav_s_sub = m_q2.decoder(lat_s[:, :, f0 : f0 + win_f], None).squeeze(1)
        return (
            stft_fn(wav_s_sub, wav_t_sub)
            + 0.5 * F.l1_loss(lat_s, lat_t)
            + 0.25 * (F.l1_loss(hs_s, hs_t) + F.l1_loss(pm_s, pm_t) + F.l1_loss(plv_s, plv_t))
        )

    _run_primal_loop(m_q2, pmods_q2, _vits_step, steps=120, lr_scale=1.5e-2, max_flips_hi=32, tag="Export-VITS-Q2_0")
    qmap_q2: Dict[str, Tuple[str, torch.Tensor, torch.Tensor]] = {
        n: ("p4_0", q, d) for n, (q, d) in qd_dp_p4.items()
    }
    for n, pm in pmods_q2.items():
        qmap_q2[n] = ("q2_0", pm.q.detach().cpu(), pm.cur_d().detach().to(torch.float16).cpu())
    finalize_primal_modules(m_q2, pmods_q2)
    q2_path = os.path.join(out_dir, "vits_v3_q2_0_primal.safetensors")
    q2_bytes = save_vault_safetensors(
        m_q2, qmap_q2, q2_path, extra_meta={"model_type": "vits_mms_eng", "quant_variant": "v3_q2_0_primal"}
    )
    print(f"  [Saved] {q2_path} -> {q2_bytes / 1e6:.3f} MB ({q2_bytes:,} bytes)", flush=True)

    _copy_samples("vits_31m", out_dir)
    del m_q2, pmods_q2, teacher, cal_vits
    gc.collect()
    torch.cuda.empty_cache()
    return {"p4_0_bytes": p4_bytes, "q2_0_bytes": q2_bytes}


def export_f5_tts() -> Dict[str, int]:
    out_dir = os.path.join(DIST_DIR, "hf_vault_primal_f5_tts_337m")
    os.makedirs(out_dir, exist_ok=True)
    print("\n=== [EXPORT 3/3] F5-TTS v1 Base (337.1M ConvNeXt V2 + 22L DiT) -> VAULT SafeTensors ===", flush=True)

    from f5_tts.infer.utils_infer import load_model as load_f5_model
    from f5_tts.model import DiT
    from f5_tts.model.utils import convert_char_to_pinyin, get_epss_timesteps, list_str_to_idx

    f5_ckpt = hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors")
    f5_vocab = hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/vocab.txt")
    shutil.copy2(f5_vocab, os.path.join(out_dir, "vocab.txt"))

    dit_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(
            {
                "architectures": ["F5TTS_CFM_DiT"],
                "base_model": "SWivid/F5-TTS (F5TTS_v1_Base)",
                "dit_config": dit_cfg,
                "mel_spec_kwargs": {
                    "n_fft": 1024,
                    "hop_length": 256,
                    "win_length": 1024,
                    "n_mel_channels": 100,
                    "target_sample_rate": 24000,
                    "mel_spec_type": "vocos",
                },
                "default_ref_text": "Some call me nature, others call me mother nature.",
                "vault_formats": ["v3_p4_0_primal", "v3_q2_0_primal"],
            },
            f,
            indent=2,
        )

    # Bundle clean default reference audio so users can synthesize immediately with zero setup!
    ex_dir = os.path.join(out_dir, "examples")
    os.makedirs(ex_dir, exist_ok=True)
    ref_wav_path = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
    shutil.copy2(ref_wav_path, os.path.join(ex_dir, "ref_en.wav"))

    cfm_teacher = load_f5_model(DiT, dit_cfg, f5_ckpt, mel_spec_type="vocos", vocab_file=f5_vocab, device="cuda")
    cfm_teacher = cfm_teacher.float().eval()

    ref_np, ref_sr = sf.read(ref_wav_path, dtype="float32")
    if ref_np.ndim > 1:
        ref_np = ref_np.mean(axis=1)
    if ref_sr != 24000:
        ref_np = librosa.resample(ref_np, orig_sr=ref_sr, target_sr=24000)
    rms = float(np.sqrt(np.mean(np.square(ref_np))))
    ref_np_norm = ref_np * (0.1 / rms) if rms < 0.1 else ref_np
    ref_audio_t = torch.from_numpy(ref_np_norm).unsqueeze(0).to("cuda", dtype=torch.float32)
    ref_text = "Some call me nature, others call me mother nature. "
    ref_audio_len = ref_audio_t.shape[-1] // 256
    ref_text_len = len(ref_text.encode("utf-8"))

    print("  [F5-TTS] Collecting 108 ODE velocity snapshots across 18 calibration trajectories...", flush=True)
    vel_snaps = []
    with torch.no_grad():
        cond_mel_base = cfm_teacher.mel_spec(ref_audio_t).permute(0, 2, 1)
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
                xk = traj[si]
                tk = t_sched[si]
                pred_cfg_t = cfm_teacher.transformer(
                    x=xk, cond=step_cond, text=text_idx, time=tk, mask=None, cfg_infer=True, cache=False
                )
                vc_t, vn_t = torch.chunk(pred_cfg_t, 2, dim=0)
                vel_snaps.append((xk.cpu(), step_cond.cpu(), text_idx.cpu(), float(tk.item()), vc_t.cpu(), vn_t.cpu()))
            del traj

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

    # 1. Export V3_P4_0 + PRIMAL (192.0 MB, 3.69% CER == FP32!)
    cfm_teacher.transformer.clear_cache()
    cfm_teacher.transformer.__dict__.pop("_cache_local", None)
    dit_p4, qd_p4, _ = apply_ptq_to_model(cfm_teacher.transformer, "p4_0", device="cuda")
    dit_p4.checkpoint_activations = True
    pmods_p4 = install_primal_modules(dit_p4, qd_p4, fmt="p4_0", grad_accum=1, device="cuda")
    _run_primal_loop(dit_p4, pmods_p4, _f5_vel_loss_fn(dit_p4), steps=90, lr_scale=8.0e-3, max_flips_hi=32, tag="Export-F5TTS-P4_0")
    qmap_p4 = {n: ("p4_0", pm.q.detach().cpu(), pm.cur_d().detach().to(torch.float16).cpu()) for n, pm in pmods_p4.items()}
    finalize_primal_modules(dit_p4, pmods_p4)
    dit_p4.checkpoint_activations = False
    p4_path = os.path.join(out_dir, "f5_tts_v3_p4_0_primal.safetensors")
    p4_bytes = save_vault_safetensors(
        dit_p4, qmap_p4, p4_path, extra_meta={"model_type": "f5_tts_dit", "quant_variant": "v3_p4_0_primal"}
    )
    print(f"  [Saved] {p4_path} -> {p4_bytes / 1e6:.3f} MB ({p4_bytes:,} bytes)", flush=True)
    del dit_p4, pmods_p4
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Export Architecture-Aware V3_Q2_0 + PRIMAL (138.9 MB, 6.78% WER / 4.68% CER!)
    def _is_adaln_or_io(n: str, _m: nn.Module) -> bool:
        return ("attn_norm" in n) or ("norm_out" in n) or ("time_embed" in n) or ("input_embed" in n) or ("proj_out" in n)

    cfm_teacher.transformer.clear_cache()
    cfm_teacher.transformer.__dict__.pop("_cache_local", None)
    dit_base_p4, qd_adaln, _ = apply_ptq_to_model(
        cfm_teacher.transformer, "p4_0", skip_fn=lambda n, m: not _is_adaln_or_io(n, m), device="cuda"
    )
    dit_q2, qd_q2, _ = apply_ptq_to_model(dit_base_p4, "q2_0", skip_fn=_is_adaln_or_io, device="cuda")
    del dit_base_p4

    dit_q2.checkpoint_activations = True
    pmods_q2 = install_primal_modules(dit_q2, qd_q2, fmt="q2_0", grad_accum=1, device="cuda")
    pmods_adaln = {}
    for name, (q_c, d_c) in qd_adaln.items():
        orig = _get_submodule(dit_q2, name)
        pm = PrimalUniversalModule(orig, q_c.to("cuda"), d_c.to("cuda"), fmt="p4_0", grad_accum=1).to("cuda")
        _set_submodule(dit_q2, name, pm)
        pmods_adaln[name] = pm
    all_pmods = {**pmods_q2, **pmods_adaln}

    _run_primal_loop(dit_q2, all_pmods, _f5_vel_loss_fn(dit_q2), steps=120, lr_scale=1.2e-2, max_flips_hi=36, tag="Export-F5TTS-Q2_0")
    qmap_q2 = {n: (pm.fmt, pm.q.detach().cpu(), pm.cur_d().detach().to(torch.float16).cpu()) for n, pm in all_pmods.items()}
    finalize_primal_modules(dit_q2, all_pmods)
    dit_q2.checkpoint_activations = False
    q2_path = os.path.join(out_dir, "f5_tts_v3_q2_0_primal.safetensors")
    q2_bytes = save_vault_safetensors(
        dit_q2, qmap_q2, q2_path, extra_meta={"model_type": "f5_tts_dit", "quant_variant": "v3_q2_0_primal"}
    )
    print(f"  [Saved] {q2_path} -> {q2_bytes / 1e6:.3f} MB ({q2_bytes:,} bytes)", flush=True)

    _copy_samples("f5_tts_300m", out_dir)
    del dit_q2, all_pmods, cfm_teacher, vel_snaps
    gc.collect()
    torch.cuda.empty_cache()
    return {"p4_0_bytes": p4_bytes, "q2_0_bytes": q2_bytes}


def main():
    summary = {
        "tiny_voca": export_tiny_voca(),
        "vits": export_vits(),
        "f5_tts": export_f5_tts(),
    }
    print("\n=== [ALL 6 VAULT SAFETENSORS CHECKPOINTS EXPORTED] ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
