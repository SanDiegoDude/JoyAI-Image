"""
Convert .pth checkpoint files to .safetensors format.

Based on the standard HuggingFace conversion logic from
https://huggingface.co/spaces/safetensors/convert

Usage:
    python convert_pth_to_safetensors.py --ckpt-root ckpts_infer
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import _find_shared_tensors, _is_complete, load_file, save_file


def _remove_duplicate_names(
    state_dict: dict[str, torch.Tensor],
    *,
    preferred_names: list[str] | None = None,
    discard_names: list[str] | None = None,
) -> dict[str, list[str]]:
    if preferred_names is None:
        preferred_names = []
    preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    discard_names = set(discard_names)

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = set(
            name for name in shared if _is_complete(state_dict[name])
        )
        if not complete_names:
            if len(shared) == 1:
                name = list(shared)[0]
                state_dict[name] = state_dict[name].clone()
                complete_names = {name}
            else:
                raise RuntimeError(
                    f"No suitable name to keep amongst: {shared}. "
                    "None covers the entire storage."
                )

        keep_name = sorted(list(complete_names))[0]

        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]

        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove


def check_file_size(sf_filename: str, pt_filename: str) -> None:
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    ratio = (sf_size - pt_size) / pt_size
    print(f"  Size comparison: safetensors={sf_size:,} bytes, pth={pt_size:,} bytes (diff: {ratio:+.2%})")
    if abs(ratio) > 0.05:
        print(f"  WARNING: size difference is {ratio:+.2%} (>5%), this may indicate an issue")


def convert_file(pt_path: str, sf_path: str) -> None:
    print(f"  Loading {pt_path} ...")
    loaded = torch.load(pt_path, map_location="cpu", weights_only=True)

    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    elif "model" in loaded and isinstance(loaded["model"], dict):
        loaded = loaded["model"]

    print(f"  Found {len(loaded)} tensors")

    to_removes = _remove_duplicate_names(loaded)

    metadata = {"format": "pt"}
    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if to_remove not in metadata:
                metadata[to_remove] = kept_name
            del loaded[to_remove]

    if to_removes:
        print(f"  Removed {sum(len(v) for v in to_removes.values())} duplicate tensors")

    loaded = {k: v.contiguous() for k, v in loaded.items()}

    os.makedirs(os.path.dirname(sf_path), exist_ok=True)
    print(f"  Saving {sf_path} ...")
    save_file(loaded, sf_path, metadata=metadata)

    check_file_size(sf_path, pt_path)

    print("  Verifying tensor equality ...")
    reloaded = load_file(sf_path)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"Tensor mismatch for key '{k}'")

    print("  Verification passed!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert .pth checkpoint files to .safetensors"
    )
    parser.add_argument(
        "--ckpt-root",
        required=True,
        help="Checkpoint root directory (containing transformer/ and vae/ subdirs)",
    )
    parser.add_argument(
        "--delete-pth",
        action="store_true",
        help="Delete original .pth files after successful conversion",
    )
    args = parser.parse_args()

    root = Path(args.ckpt_root).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    conversions = [
        (root / "transformer" / "transformer.pth", root / "transformer" / "transformer.safetensors"),
        (root / "vae" / "Wan2.1_VAE.pth", root / "vae" / "Wan2.1_VAE.safetensors"),
    ]

    success_count = 0
    for pt_path, sf_path in conversions:
        if not pt_path.exists():
            print(f"\nSkipping {pt_path} (not found)")
            continue

        if sf_path.exists():
            print(f"\nSkipping {pt_path} (safetensors already exists at {sf_path})")
            success_count += 1
            continue

        print(f"\nConverting: {pt_path.name}")
        print(f"  Source: {pt_path}")
        print(f"  Target: {sf_path}")
        try:
            convert_file(str(pt_path), str(sf_path))
            success_count += 1
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            if sf_path.exists():
                sf_path.unlink()
            continue

    if args.delete_pth and success_count == len(conversions):
        print("\nDeleting original .pth files ...")
        for pt_path, _ in conversions:
            if pt_path.exists():
                pt_path.unlink()
                print(f"  Deleted {pt_path}")

    print(f"\nDone! {success_count}/{len(conversions)} files converted successfully.")


if __name__ == "__main__":
    main()
