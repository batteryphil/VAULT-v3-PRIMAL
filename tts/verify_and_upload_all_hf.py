#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# Copyright (c) 2026 Philip Bartsch (@batteryphil)

"""
End-to-End Standalone Verification & Hugging Face Hub Uploader for ALL `VAULT + PRIMAL` Models:
  1. Verifies standalone `infer.py` execution across all 6 TTS `.safetensors` variants (`Tiny-Voca`, `VITS`, `F5-TTS` x `{p4_0, q2_0}`).
  2. Creates & uploads:
       - `batteryphil/VAULT-PRIMAL-Tiny-Voca-4.5M`
       - `batteryphil/VAULT-PRIMAL-VITS-29M`
       - `batteryphil/VAULT-PRIMAL-F5-TTS-337M`
  3. Updates `batteryphil/VAULT-v3-DeepSeek-R1-1.5B-PRIMAL-VLT3` with `V3_Q2_0` (`680.9 MB`), `V3_T1_58` (`597.7 MB` / `704.6 MB`),
     upgraded `runtime/` (`V3_Q2_0 = 13` & `V3_T1_58 = 12` SIMD kernels), `run_chat.py`, and updated Model Card.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"
PY_BIN = sys.executable

TTS_REPOS = [
    ("batteryphil/VAULT-PRIMAL-Tiny-Voca-4.5M", DIST_DIR / "hf_vault_primal_tiny_voca_4_5m"),
    ("batteryphil/VAULT-PRIMAL-VITS-29M", DIST_DIR / "hf_vault_primal_vits_29m"),
    ("batteryphil/VAULT-PRIMAL-F5-TTS-337M", DIST_DIR / "hf_vault_primal_f5_tts_337m"),
]


def verify_tts_standalone() -> None:
    print("=== [STEP 1/3] Standalone Smoke-Testing All 6 TTS `infer.py` Variants ===", flush=True)
    test_text = "Hello! Testing standalone Hugging Face inference on quantized neural speech."
    with tempfile.TemporaryDirectory() as tmp:
        for repo_id, folder in TTS_REPOS:
            for variant in ("p4_0", "q2_0"):
                out_wav = os.path.join(tmp, f"{folder.name}_{variant}.wav")
                cmd = [PY_BIN, "infer.py", test_text, "--variant", variant, "-o", out_wav]
                print(f"  [Test] {repo_id} ({variant}) -> running {' '.join(cmd[1:])} ...", flush=True)
                res = subprocess.run(cmd, cwd=str(folder), capture_output=True, text=True, check=True)
                wav, sr = sf.read(out_wav, dtype="float32")
                dur = len(wav) / float(sr)
                rms = float((wav ** 2).mean() ** 0.5)
                assert dur > 1.5 and rms > 0.005, f"Bad audio on {repo_id} {variant}: dur={dur}, rms={rms}"
                last_line = [ln for ln in res.stdout.strip().splitlines() if ln.strip()][-1]
                print(f"    -> {last_line} | rms={rms:.4f}", flush=True)


def upload_tts_repos(api: HfApi) -> None:
    print("\n=== [STEP 2/3] Creating & Uploading All 3 TTS Repos to Hugging Face Hub ===", flush=True)
    ignore = ["__pycache__", "*.pyc", ".DS_Store", "runtime/vault3_host"]
    for repo_id, folder in TTS_REPOS:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        print(f"  [Upload] {folder.name} -> https://huggingface.co/{repo_id} ...", flush=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(folder),
            commit_message="Upload VAULT + PRIMAL V3_P4_0 (4-bit) & V3_Q2_0 (2-bit) SafeTensors, samples, 1-command infer.py & Model Card",
            ignore_patterns=ignore,
        )
        files = api.list_repo_files(repo_id=repo_id)
        print(f"    -> [LIVE] {repo_id} ({len(files)} files verified on Hub)", flush=True)


def update_llm_vlt3_repos(api: HfApi) -> None:
    print("\n=== [STEP 3/3] Updating `VAULT-v3-DeepSeek-R1-1.5B-PRIMAL-VLT3` & Main Repo Runtime ===", flush=True)
    vlt3_repo = "batteryphil/VAULT-v3-DeepSeek-R1-1.5B-PRIMAL-VLT3"
    vlt3_dir = DIST_DIR / "hf_vault_v3_deepseek_r1_1_5b_vlt3"

    # Upload only new/updated files so we don't re-upload the existing 1.2 GB P4_0 and 1.8 GB Q8_0 .vlt3 blobs!
    new_or_updated = [
        "README.md",
        "run_chat.py",
        "runtime/Makefile",
        "runtime/vault3.c",
        "runtime/vault3.h",
        "runtime/vault3_host.c",
        "vault_v3_deepseek_r1_1_5b_q2_outlier.vlt3",
        "vault_v3_deepseek_r1_1_5b_t1_58_hybrid.vlt3",
        "vault_v3_deepseek_r1_1_5b_t1_58_pure.vlt3",
    ]
    for rel in new_or_updated:
        fpath = vlt3_dir / rel
        sz_mb = fpath.stat().st_size / 1e6
        print(f"  [Upload -> {vlt3_repo}] {rel} ({sz_mb:.2f} MB)...", flush=True)
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=rel,
            repo_id=vlt3_repo,
            repo_type="model",
            commit_message=f"Add/update {rel} (V3_Q2_0 2-bit outlier + T1_58 + easy run_chat.py launcher)",
        )

    # Also sync upgraded runtime/vault3.{c,h} to main DeepSeek-R1-1.5B-PRIMAL repo
    main_repo = "batteryphil/VAULT-v3-DeepSeek-R1-1.5B-PRIMAL"
    main_dir = DIST_DIR / "hf_vault_v3_deepseek_r1_1_5b"
    for rel in ["runtime/vault3.c", "runtime/vault3.h", "runtime/vault3_host.c", "runtime/Makefile"]:
        api.upload_file(
            path_or_fileobj=str(main_dir / rel),
            path_in_repo=rel,
            repo_id=main_repo,
            repo_type="model",
            commit_message=f"Upgrade {rel} with V3_Q2_0 (type 13) & V3_T1_58 (type 12) C99 SIMD kernels",
        )
    print(f"  -> [LIVE] {vlt3_repo} & {main_repo} updated cleanly!", flush=True)


def main():
    api = HfApi()
    verify_tts_standalone()
    upload_tts_repos(api)
    update_llm_vlt3_repos(api)
    print("\n=== [ALL HUGGING FACE REPOS VERIFIED & PUBLISHED] ===", flush=True)


if __name__ == "__main__":
    main()
