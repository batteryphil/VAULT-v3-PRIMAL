#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
Reproducible End-to-End CPU TTS Benchmark (`benchmark_cpu_tts.py`).

Evaluates all 6 standalone VAULT-PRIMAL TTS checkpoints (`Tiny-Voca-4.5M`, `VITS-29M`, `F5-TTS-337M` x `{p4_0, q2_0}`)
through their public `vault_tts.py` wrappers on `device="cpu"`, recording:
  - Cold-load latency (`s`)
  - Live model resident RAM (`MB` vs. serialized `.safetensors` file size & FP32 baseline)
  - Synthesis latency (`s`), Audio duration (`s`), Real-Time Speed (`x RT`) & Real-Time Factor (`RTF`)
  - Round-trip `openai/whisper-small.en` Word Error Rate (`WER %`)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import jiwer
import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tts.primal_multimodal_ops import _norm_asr_text  # noqa: E402

PROMPT = "Artificial intelligence runs fast on pure CPU silicon when weights fit directly inside processor cache."

PRE_OPT_BASELINE: Dict[str, Dict[str, float]] = {
    "Tiny-Voca-4.5M:p4_0": {"load_s": 0.09, "ram_mb": 18.15, "syn_s": 0.038, "x_rt": 164.92},
    "Tiny-Voca-4.5M:q2_0": {"load_s": 0.06, "ram_mb": 18.15, "syn_s": 0.041, "x_rt": 157.97},
    "VITS-29M:p4_0":       {"load_s": 0.79, "ram_mb": 116.14, "syn_s": 0.575, "x_rt": 11.08},
    "VITS-29M:q2_0":       {"load_s": 0.93, "ram_mb": 116.14, "syn_s": 0.646, "x_rt": 10.18},
    "F5-TTS-337M:p4_0":    {"load_s": 6.07, "ram_mb": 1365.16, "syn_s": 33.137, "x_rt": 0.32},
    "F5-TTS-337M:q2_0":    {"load_s": 3.45, "ram_mb": 1365.16, "syn_s": 31.788, "x_rt": 0.33},
}


def _load_wrapper(pkg_dir: Path, mod_tag: str) -> Any:
    spec = importlib.util.spec_from_file_location(mod_tag, str(pkg_dir / "vault_tts.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_tag] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    asr_dev = "cuda" if torch.cuda.is_available() else "cpu"
    w_proc = WhisperProcessor.from_pretrained("openai/whisper-small.en")
    w_mod = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en").to(asr_dev).eval()

    def grade(wav: np.ndarray, sr: int) -> tuple[float, str]:
        w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
        inp = w_proc(w16, sampling_rate=16000, return_tensors="pt").input_features.to(asr_dev)
        with torch.inference_mode():
            ids = w_mod.generate(inp, max_new_tokens=128)
        hyp = _norm_asr_text(w_proc.batch_decode(ids, skip_special_tokens=True)[0])
        return round(100.0 * jiwer.wer(_norm_asr_text(PROMPT), hyp), 2), hyp

    specs = [
        ("Tiny-Voca-4.5M", REPO_ROOT / "dist" / "hf_vault_primal_tiny_voca_4_5m", "VaultTinyVoca", 5),
        ("VITS-29M",       REPO_ROOT / "dist" / "hf_vault_primal_vits_29m",       "VaultVits",     3),
        ("F5-TTS-337M",    REPO_ROOT / "dist" / "hf_vault_primal_f5_tts_337m",    "VaultF5TTS",    1),
    ]

    results: List[Dict[str, Any]] = []
    print("\n==========================================================================================================")
    print(" VAULT-PRIMAL Cache-Resident CPU TTS Benchmark (All 6 Standalone Checkpoints on `device='cpu'`)")
    print("==========================================================================================================", flush=True)

    for model_name, pkg_dir, cls_name, n_reps in specs:
        mod = _load_wrapper(pkg_dir, f"wrap_{cls_name.lower()}")
        cls_obj = getattr(mod, cls_name)

        for variant in ("p4_0", "q2_0"):
            t0 = time.perf_counter()
            tts = cls_obj.from_pretrained(variant=variant, device="cpu", local_dir=str(pkg_dir))
            t_load = time.perf_counter() - t0

            # 1 short warmup call
            _ = tts.speak("Ready.")

            syn_times = []
            wav = np.zeros(0, dtype=np.float32)
            for _ in range(n_reps):
                ts = time.perf_counter()
                wav = tts.speak(PROMPT)
                syn_times.append(time.perf_counter() - ts)

            t_syn = float(np.median(syn_times))
            dur = len(wav) / float(tts.sample_rate)
            x_rt = dur / max(1e-6, t_syn)
            rtf = t_syn / max(1e-6, dur)
            wer, transcript = grade(wav, tts.sample_rate)

            base = PRE_OPT_BASELINE[f"{model_name}:{variant}"]
            ram_saving_x = base["ram_mb"] / max(0.01, tts.resident_ram_mb)
            speedup_x = x_rt / max(0.001, base["x_rt"])

            row = {
                "model": model_name,
                "variant": variant,
                "file_mb": tts.packed_mb,
                "before_ram_mb": base["ram_mb"],
                "after_ram_mb": tts.resident_ram_mb,
                "ram_reduction_x": round(ram_saving_x, 2),
                "before_load_s": base["load_s"],
                "after_load_s": round(t_load, 3),
                "before_syn_s": base["syn_s"],
                "after_syn_s": round(t_syn, 4),
                "audio_dur_s": round(dur, 2),
                "before_x_rt": base["x_rt"],
                "after_x_rt": round(x_rt, 2),
                "after_rtf": round(rtf, 4),
                "speedup_x": round(speedup_x, 2),
                "whisper_wer_pct": wer,
                "transcript": transcript,
            }
            results.append(row)
            print(
                f"[{model_name:14s} {variant}] "
                f"RAM: {base['ram_mb']:7.2f} -> {tts.resident_ram_mb:6.2f} MB ({ram_saving_x:5.2f}x smaller) | "
                f"Load: {base['load_s']:4.2f}s -> {t_load:5.3f}s | "
                f"Synth: {base['syn_s']:6.3f}s -> {t_syn:6.3f}s | "
                f"Speed: {base['x_rt']:6.2f}x -> {x_rt:6.2f}x RT ({speedup_x:4.2f}x) | "
                f"WER: {wer:5.2f}%",
                flush=True,
            )

    out_json = REPO_ROOT / "benchmarks" / "results" / "vault_primal_cpu_tts_benchmark.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"prompt": PROMPT, "results": results}, f, indent=2)
    print(f"\n[OK] Saved full benchmark report to {out_json}", flush=True)


if __name__ == "__main__":
    main()
