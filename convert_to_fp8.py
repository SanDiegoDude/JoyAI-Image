"""
Convert the JoyAI-Image DiT transformer from bf16 to FP8 (float8_e4m3fn).

Strategy:
  - 2D+ weight tensors (linear layers, conv kernels) → float8_e4m3fn
  - 1D tensors (biases, norms, embeddings) → keep original dtype
  - This matches ComfyUI's fp8 convention for diffusion models

The resulting checkpoint is ~16 GB instead of ~32 GB, fitting in a 4090's 24 GB VRAM.

Usage:
    python convert_to_fp8.py --input ckpts_infer/transformer/transformer.safetensors \
                             --output ckpts_infer/transformer/transformer_fp8.safetensors
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file


FP8_DTYPE = torch.float8_e4m3fn


def quantize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Quantize a tensor to FP8 E4M3, clamping to the representable range."""
    finfo = torch.finfo(FP8_DTYPE)
    t_clamped = t.float().clamp(finfo.min, finfo.max)
    return t_clamped.to(FP8_DTYPE)


def should_quantize(name: str, tensor: torch.Tensor) -> bool:
    """Decide whether a tensor should be quantized to FP8."""
    if tensor.ndim < 2:
        return False
    if tensor.numel() < 1024:
        return False
    return True


def convert(input_path: str, output_path: str) -> None:
    print(f"Loading {input_path} ...")
    state_dict = load_file(input_path, device="cpu")

    total_tensors = len(state_dict)
    quantized_count = 0
    kept_count = 0
    original_bytes = 0
    new_bytes = 0

    print(f"Processing {total_tensors} tensors ...")
    converted = {}
    for name, tensor in state_dict.items():
        original_bytes += tensor.numel() * tensor.element_size()

        if should_quantize(name, tensor):
            converted[name] = quantize_tensor(tensor)
            quantized_count += 1
        else:
            converted[name] = tensor
            kept_count += 1

        new_bytes += converted[name].numel() * converted[name].element_size()

    print(f"  Quantized to FP8: {quantized_count}")
    print(f"  Kept original:    {kept_count}")
    print(f"  Size: {original_bytes / 1e9:.2f} GB → {new_bytes / 1e9:.2f} GB "
          f"({new_bytes / original_bytes:.1%})")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Saving {output_path} ...")
    save_file(converted, output_path, metadata={"format": "fp8_e4m3fn"})

    output_size = os.path.getsize(output_path)
    print(f"  File size: {output_size / 1e9:.2f} GB")

    print("Verifying reload ...")
    reloaded = load_file(output_path, device="cpu")
    for name in converted:
        assert reloaded[name].dtype == converted[name].dtype, \
            f"Dtype mismatch for {name}: {reloaded[name].dtype} vs {converted[name].dtype}"
        assert reloaded[name].shape == converted[name].shape, \
            f"Shape mismatch for {name}"
    print("Verification passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DiT weights to FP8")
    parser.add_argument("--input", required=True, help="Input safetensors file (bf16)")
    parser.add_argument("--output", required=True, help="Output safetensors file (fp8)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    convert(args.input, args.output)
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
