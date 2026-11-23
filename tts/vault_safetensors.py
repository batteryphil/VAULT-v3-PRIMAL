# SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
# License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
# Copyright (c) 2026 Philip Bartsch (batteryphil)

"""
VAULT-SafeTensors v1 Bit-Packing & Zero-Dependency Pure-PyTorch Unpacker.

Packs `V3_P4_0` (`4.50 bpw`, 2 nibbles/byte + `FP16` block-32 scale) and
`V3_Q2_0` (`2.25 bpw`, 4 crumbs/byte + `FP16` block-64 scale) directly into standard
Hugging Face `.safetensors` files (`0` pickle, green `Safe` badge on HF Hub, instant load on
CUDA / ROCm / Apple Silicon `mps` / CPU).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

P4_LUT_VALUES = [-127.0, -101.0, -80.0, -62.0, -46.0, -33.0, -21.0, -10.0, 0.0, 10.0, 21.0, 33.0, 46.0, 62.0, 80.0, 101.0]


@torch.no_grad()
def save_vault_safetensors(
    model: nn.Module,
    q_map: Dict[str, Tuple[str, torch.Tensor, torch.Tensor]],
    out_path: str,
    extra_meta: Dict[str, str] | None = None,
) -> int:
    """
    Bit-packs `model` + `q_map` (`module_name -> (fmt, q_int8_cpu, d_fp16_cpu)`)
    into a genuine sub-3-bit / 4.5-bit `.safetensors` file and returns file size in bytes.
    """
    tensors: Dict[str, torch.Tensor] = {}
    meta: Dict[str, str] = {"vault_safetensors_version": "1.0"}
    if extra_meta:
        meta.update({str(k): str(v) for k, v in extra_meta.items()})

    mod_dict = dict(model.named_modules())
    for mname, (fmt, q_c, d_c) in q_map.items():
        mod = mod_dict[mname]
        w_shape = tuple(mod.weight.shape)
        meta[f"shape:{mname}.weight"] = ",".join(str(x) for x in w_shape)
        if fmt == "p4_0":
            q_u8 = q_c.detach().cpu().to(torch.uint8)
            assert q_u8.shape[1] % 32 == 0
            q_pack = (q_u8[:, 0::2] | (q_u8[:, 1::2] << 4)).contiguous()
            tensors[f"{mname}.weight.__p4_q"] = q_pack
            tensors[f"{mname}.weight.__p4_d"] = d_c.detach().cpu().to(torch.float16).contiguous()
        elif fmt == "q2_0":
            u = (q_c.detach().cpu().to(torch.int16) + 2).clamp(0, 3).to(torch.uint8)
            assert u.shape[1] % 64 == 0
            q_pack = (u[:, 0::4] | (u[:, 1::4] << 2) | (u[:, 2::4] << 4) | (u[:, 3::4] << 6)).contiguous()
            tensors[f"{mname}.weight.__q2_q"] = q_pack
            tensors[f"{mname}.weight.__q2_d"] = d_c.detach().cpu().to(torch.float16).contiguous()
        else:
            raise ValueError(f"Unsupported VAULT format: {fmt}")

    for pname, p in model.named_parameters():
        if pname.endswith(".weight") and pname[:-7] in q_map:
            continue
        t = p.detach().cpu()
        if t.is_floating_point() and (t.numel() == 0 or float(t.abs().max().item()) <= 60000.0):
            tensors[pname] = t.to(torch.float16).contiguous()
        else:
            tensors[pname] = t.contiguous()

    save_file(tensors, out_path, metadata=meta)
    import os
    return os.path.getsize(out_path)


@torch.no_grad()
def load_vault_safetensors(
    model: nn.Module,
    path: str,
    device: str = "cpu",
    packed_cpu: bool = False,
    fuse_dit_qkv: bool = False,
) -> Dict[str, str]:
    """
    Unpacks a `VAULT-SafeTensors v1` checkpoint (`V3_P4_0` and/or `V3_Q2_0` + `FP16` norms/heads)
    directly into `model` on `device`. When `device == "cpu"` and `packed_cpu=True`, keeps weights
    permanently bit-packed in CPU cache/RAM via `vault_cpu_engine.load_vault_safetensors_cpu_live`.
    """
    if packed_cpu and str(device) == "cpu":
        from tts.vault_cpu_engine import load_vault_safetensors_cpu_live

        info = load_vault_safetensors_cpu_live(model, path, fuse_dit_qkv=fuse_dit_qkv)
        return {k: str(v) for k, v in info.items()}

    unpack_dev = device if str(device).startswith("cuda") else "cpu"
    lut_t = torch.tensor(P4_LUT_VALUES, device=unpack_dev, dtype=torch.float32)
    param_dict = dict(model.named_parameters())

    with safe_open(path, framework="pt", device=unpack_dev) as f:
        meta = f.metadata() or {}
        keys = set(f.keys())

        for k in sorted(keys):
            if k.endswith(".__p4_d") or k.endswith(".__q2_d"):
                continue
            if k.endswith(".__p4_q"):
                w_name = k[: -len(".__p4_q")]
                q_pack = f.get_tensor(k)
                d_f16 = f.get_tensor(w_name + ".__p4_d")
                orig_shape = tuple(int(x) for x in meta[f"shape:{w_name}"].split(","))
                rows = orig_shape[0]
                cols = int(np.prod(orig_shape[1:]))
                pcols = q_pack.shape[1] * 2

                lo = (q_pack & 0x0F).long()
                hi = ((q_pack >> 4) & 0x0F).long()
                idx = torch.stack([lo, hi], dim=-1).reshape(rows, pcols)
                rec = (lut_t[idx].view(rows, pcols // 32, 32) * d_f16.float().unsqueeze(-1)).reshape(rows, pcols)
                p_tgt = param_dict[w_name]
                p_tgt.data.copy_(rec[:, :cols].reshape(orig_shape).to(device=p_tgt.device, dtype=p_tgt.dtype))
            elif k.endswith(".__q2_q"):
                w_name = k[: -len(".__q2_q")]
                q_pack = f.get_tensor(k)
                d_f16 = f.get_tensor(w_name + ".__q2_d")
                orig_shape = tuple(int(x) for x in meta[f"shape:{w_name}"].split(","))
                rows = orig_shape[0]
                cols = int(np.prod(orig_shape[1:]))
                pcols = q_pack.shape[1] * 4

                u0 = (q_pack & 0x03).to(torch.int8) - 2
                u1 = ((q_pack >> 2) & 0x03).to(torch.int8) - 2
                u2 = ((q_pack >> 4) & 0x03).to(torch.int8) - 2
                u3 = ((q_pack >> 6) & 0x03).to(torch.int8) - 2
                qf = torch.stack([u0, u1, u2, u3], dim=-1).reshape(rows, pcols).float()
                rec = (qf.view(rows, pcols // 64, 64) * d_f16.float().unsqueeze(-1)).reshape(rows, pcols)
                p_tgt = param_dict[w_name]
                p_tgt.data.copy_(rec[:, :cols].reshape(orig_shape).to(device=p_tgt.device, dtype=p_tgt.dtype))
            elif k in param_dict:
                t = f.get_tensor(k)
                p_tgt = param_dict[k]
                p_tgt.data.copy_(t.to(device=p_tgt.device, dtype=p_tgt.dtype))

    return meta
