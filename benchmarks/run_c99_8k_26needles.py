#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
Driver for `bench_c99_8k_26needles` (Freestanding C99 8,465-Token 26-Needle Test under ~86.2% Eviction Pressure).
Builds the exact same `doc_8k` + 26 `(station, cipher)` pairs used in `bench_vault_v3_p4_eval.py`
and executes `bench_c99_8k_26needles` with both `seg_budget=1024` (SEG ON) and `--noseg` (seg_budget=0).
"""

import json
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from transformers import AutoTokenizer
from bench_vault_v3_p4_eval import make_filler_tokens  # noqa: E402

QWEN_3B = os.environ.get("QWEN_3B_DIR", "Qwen/Qwen2.5-3B-Instruct")
BIN = os.path.join(ROOT, "benchmarks", "bench_c99_8k_26needles")


def run_one(vlt3_path: str, doc_txt: str, queries_tsv: str, noseg: bool, threads: int = 24) -> dict:
    with tempfile.TemporaryDirectory() as td:
        dp = os.path.join(td, "doc.txt")
        qp = os.path.join(td, "queries.tsv")
        with open(dp, "w", encoding="utf-8") as f:
            f.write(doc_txt)
        with open(qp, "w", encoding="utf-8") as f:
            f.write(queries_tsv)
        cmd = [BIN, vlt3_path, dp, qp, "--threads", str(threads)]
        if noseg:
            cmd.append("--noseg")
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        sys.stderr.write(p.stderr)
        if p.returncode != 0:
            raise RuntimeError(f"bench_c99_8k_26needles failed rc={p.returncode}: {p.stderr[-500:]}")
        for line in reversed(p.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        raise RuntimeError(f"No JSON output from bench_c99_8k_26needles: {p.stdout}")


def main():
    vlt3_p4 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "dist", "vault_v3_3b_p4_0_primal.vlt3")
    tok = AutoTokenizer.from_pretrained(QWEN_3B)

    pad_seg = make_filler_tokens(tok, 210, seed=22)
    pad_tail = make_filler_tokens(tok, 2350, seed=23)
    rng_8k = random.Random(2026)
    tsv_lines = []
    blocks_8k = []
    for i in range(26):
        entity = f"station_{chr(65 + i)}{10 + i}"
        code = f"K8-{rng_8k.randint(1000, 9999)}-{chr(65 + (i * 3) % 26)}{chr(65 + (i * 5 + 1) % 26)}"
        tsv_lines.append(f"{entity}\t{code}")
        blocks_8k.append(f"Archive entry {i + 1}: {entity} stores telemetry cipher {code}.\n{pad_seg}")
    doc_8k = "\n".join(blocks_8k) + "\n\n" + pad_tail
    queries_tsv = "\n".join(tsv_lines) + "\n"

    print(f"=== Running C99 8,465-Token 26-Needle Benchmark on {os.path.basename(vlt3_p4)} (SEG ON) ===", flush=True)
    res_on = run_one(vlt3_p4, doc_8k, queries_tsv, noseg=False, threads=16)
    print(f"  [SEG ON ] {res_on['hits']}/{res_on['total']} ({res_on['acc_pct']}%) | eviction_pressure={res_on['eviction_pressure_pct']}% | rss={res_on['peak_rss_mb']} MB", flush=True)

    print(f"=== Running C99 8,465-Token 26-Needle Control on {os.path.basename(vlt3_p4)} (--noseg) ===", flush=True)
    res_off = run_one(vlt3_p4, doc_8k, queries_tsv, noseg=True, threads=16)
    print(f"  [NO SEG ] {res_off['hits']}/{res_off['total']} ({res_off['acc_pct']}%) | rss={res_off['peak_rss_mb']} MB", flush=True)

    out_path = os.path.join(ROOT, "benchmarks", "peer_review_c99_8k_26needles.json")
    with open(out_path, "w") as f:
        json.dump({"vlt3": os.path.basename(vlt3_p4), "seg_on": res_on, "noseg_control": res_off}, f, indent=2)
    print(f"[DONE] Saved C99 8.5K 26-needle report -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
