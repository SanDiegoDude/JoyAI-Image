"""One-time conversion: bf16 safetensors → pre-quantized NF4 safetensors.

Run once on a machine with enough RAM (~32 GB) and a CUDA GPU.
The output file loads in seconds instead of minutes at runtime.

Usage:
    python convert_to_nf4.py [--ckpt-dir ckpts_infer/transformer] [--output transformer_nf4.safetensors]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

NF4_META = ".__nf4__."


def save_nf4_state_dict(model: torch.nn.Module, path: str) -> None:
    """Save a model with Linear4bit layers as a flat safetensors file."""
    import bitsandbytes as bnb

    tensors: dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if isinstance(param, bnb.nn.Params4bit) and param.quant_state is not None:
            qs = param.quant_state
            tensors[name] = param.data.contiguous().cpu()
            tensors[f"{name}{NF4_META}absmax"] = qs.absmax.contiguous().cpu()
            tensors[f"{name}{NF4_META}quant_map"] = qs.code.contiguous().cpu()
            tensors[f"{name}{NF4_META}shape"] = torch.tensor(list(qs.shape), dtype=torch.int64)
            tensors[f"{name}{NF4_META}offset"] = torch.tensor([qs.offset], dtype=torch.float32)

            if qs.state2 is not None:
                tensors[f"{name}{NF4_META}nested_absmax"] = qs.state2.absmax.contiguous().cpu()
                tensors[f"{name}{NF4_META}nested_quant_map"] = qs.state2.code.contiguous().cpu()
        else:
            tensors[name] = param.data.contiguous().cpu()

    for name, buf in model.named_buffers():
        tensors[name] = buf.contiguous().cpu()

    save_file(tensors, path)
    size_gb = os.path.getsize(path) / 1024**3
    print(f"Saved {len(tensors)} tensors → {path} ({size_gb:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Convert bf16 DiT to pre-quantized NF4 safetensors")
    parser.add_argument("--ckpt-dir", default=str(ROOT_DIR / "ckpts_infer" / "transformer"))
    parser.add_argument("--output", default=None,
                        help="Output file path (default: <ckpt-dir>/transformer_nf4.safetensors)")
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    output = args.output or os.path.join(ckpt_dir, "transformer_nf4.safetensors")

    from modules.models import (
        _replace_linear_with_nf4,
        _quantize_nf4_on_gpu,
        _select_safetensors,
        build_from_config,
    )
    from modules.utils.fsdp_load import safetensors_weights_iterator
    from modules.utils.constants import PRECISION_TO_TYPE
    from infer_runtime.infer_config import load_infer_config_class_from_pyfile

    config_path = os.path.join(os.path.dirname(ckpt_dir), "infer_config.py")
    config_class = load_infer_config_class_from_pyfile(config_path)
    cfg = config_class()
    cfg.dit_ckpt = ckpt_dir
    cfg.training_mode = False

    dtype = torch.bfloat16
    gpu = torch.device("cuda:0")

    print(f"Building model skeleton on CPU (bf16)...")
    t0 = time.time()
    model = build_from_config(cfg.dit_arch_config, dtype=dtype, device=torch.device("cpu"), args=cfg)

    print(f"Loading bf16 weights from {ckpt_dir}...")
    files = _select_safetensors(ckpt_dir, full_precision=True)
    print(f"  Files: {[os.path.basename(f) for f in files]}")
    sd = dict(safetensors_weights_iterator(files))

    for k, v in sd.items():
        if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            sd[k] = v.to(dtype)

    if "img_in.weight" in sd:
        expected = model.img_in.weight.shape
        v = sd["img_in.weight"]
        if expected != v.shape:
            print(f"  Inflating img_in.weight {v.shape} → {expected}")
            v_new = torch.zeros(expected, dtype=dtype)
            v_new[:, :v.shape[1], :, :, :] = v
            sd["img_in.weight"] = v_new

    model.load_state_dict(sd, strict=True)
    del sd

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {total_params / 1e9:.2f}B parameters loaded")

    print(f"Replacing Linear → Linear4bit (NF4)...")
    n = _replace_linear_with_nf4(model, compute_dtype=dtype)
    print(f"  Replaced {n} layers")

    print(f"Quantizing on {gpu}...")
    nq = _quantize_nf4_on_gpu(model, gpu)
    vram_gb = torch.cuda.memory_allocated(gpu) / 1024**3
    print(f"  Quantized {nq} layers, GPU usage: {vram_gb:.2f} GB")

    print(f"Saving to {output}...")
    save_nf4_state_dict(model, output)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
