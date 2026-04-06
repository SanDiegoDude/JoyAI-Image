"""ComfyUI node — JoyAI Image Generate (local inference, no API server needed).

Runs the JoyAI-Image pipeline directly inside the ComfyUI process.
Models are downloaded automatically from HuggingFace on first use and
kept in memory between runs.
"""
from __future__ import annotations

import os
import sys
import threading
import warnings

import numpy as np
import torch
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_JOYAI_ROOT = os.path.dirname(os.path.realpath(os.path.join(_THIS_DIR, "..")))
_JOYAI_SRC = os.path.join(_JOYAI_ROOT, "src")

if not os.path.isdir(_JOYAI_SRC):
    _alt = os.path.dirname(os.path.realpath(_THIS_DIR))
    if os.path.isdir(os.path.join(_alt, "src")):
        _JOYAI_ROOT = _alt
        _JOYAI_SRC = os.path.join(_alt, "src")

_model = None
_model_lock = threading.Lock()
_current_cfg_key = None


def _get_model(high_vram: bool, vlm_bits: int):
    """Return the singleton EditModel, loading on first call or config change."""
    global _model, _current_cfg_key

    cfg_key = (high_vram, vlm_bits)

    with _model_lock:
        if _model is not None and _current_cfg_key == cfg_key:
            return _model

        if _JOYAI_SRC not in sys.path:
            sys.path.insert(0, _JOYAI_SRC)
        warnings.filterwarnings("ignore")

        from infer_runtime.download import ensure_checkpoints
        from infer_runtime.model import build_model
        from infer_runtime.settings import load_settings

        ckpt_root = os.path.join(_JOYAI_ROOT, "ckpts_infer")
        ensure_checkpoints(ckpt_root, full_precision=False)

        settings = load_settings(
            ckpt_root=ckpt_root,
            default_seed=42,
            full_precision=False,
            high_vram=high_vram,
            lod=False,
            vlm_bits=vlm_bits,
            nf4_dit=False,
        )

        if _model is not None:
            print("[JoyAI] Config changed — reloading models...")
            del _model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"[JoyAI] Loading models (high_vram={high_vram}, vlm={vlm_bits}-bit)...")
        _model = build_model(settings)
        _current_cfg_key = cfg_key
        print("[JoyAI] Models ready.")
        return _model


class JoyAIImageGenerate:
    """Run JoyAI-Image inference locally inside ComfyUI.

    FP8 transformer with configurable VLM quantization.
    Models are downloaded automatically on first use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 18, "min": 1, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
                "width": ("INT", {
                    "default": 1024, "min": 256, "max": 4096, "step": 16,
                    "tooltip": "T2I width (ignored when input_image is connected)",
                }),
                "height": ("INT", {
                    "default": 1024, "min": 256, "max": 4096, "step": 16,
                    "tooltip": "T2I height (ignored when input_image is connected)",
                }),
                "edit_base_size": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 256,
                    "tooltip": "Edit-mode bucket size (multiple of 256, "
                               "only used when input_image is connected)",
                }),
                "high_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep all models on GPU (needs ~48 GB). "
                               "Default offloads to CPU between phases.",
                }),
                "vlm_quantization": (["8-bit", "4-bit", "16-bit (full)"], {
                    "default": "8-bit",
                    "tooltip": "Text encoder quantization. 4-bit recommended "
                               "for 24 GB GPUs, 8-bit for 48 GB+.",
                }),
            },
            "optional": {
                "input_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "JoyAI"
    OUTPUT_NODE = False

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        width: int,
        height: int,
        edit_base_size: int,
        high_vram: bool,
        vlm_quantization: str,
        input_image=None,
    ):
        if _JOYAI_SRC not in sys.path:
            sys.path.insert(0, _JOYAI_SRC)
        from infer_runtime.model import InferenceParams
        from modules.utils import seed_everything

        vlm_bits = {"4-bit": 4, "8-bit": 8, "16-bit (full)": 16}[vlm_quantization]

        model = _get_model(high_vram=high_vram, vlm_bits=vlm_bits)

        seed = seed & 0xFFFFFFFF
        seed_everything(seed)

        pil_input = None
        if input_image is not None:
            img_np = (input_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_input = Image.fromarray(img_np)

        width = (width // 16) * 16
        height = (height // 16) * 16

        params = InferenceParams(
            prompt=prompt,
            image=pil_input,
            height=height,
            width=width,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            neg_prompt=negative_prompt,
            basesize=edit_base_size,
        )

        pil_result = model.infer(params)
        img_array = np.array(pil_result).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return (img_tensor,)
