"""ComfyUI node that connects to the JoyAI-Image REST API for generation."""

from __future__ import annotations

import base64
import io
import json
import urllib.request
import urllib.error

import numpy as np
import torch
from PIL import Image


class JoyAIImageGenerate:
    """Sends a generation request to a running JoyAI-Image API server and
    returns the resulting image.  Supports both text-to-image and
    instruction-based image editing (when an input_image is connected).

    The API server is started with:
        python app.py --api            # Gradio + API
        python app.py --headless-api   # API only
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 18, "min": 1, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16,
                                  "tooltip": "T2I width (ignored when input_image is connected)"}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16,
                                   "tooltip": "T2I height (ignored when input_image is connected)"}),
                "edit_base_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 256,
                                           "tooltip": "Edit-mode bucket size (must be multiple of 256, "
                                                      "only used when input_image is connected)"}),
                "save_image": ("BOOLEAN", {"default": False,
                                           "tooltip": "Save the output on the server side"}),
                "server_address": ("STRING", {"default": "127.0.0.1"}),
                "server_port": ("INT", {"default": 7500, "min": 1, "max": 65535}),
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
        save_image: bool,
        server_address: str,
        server_port: int,
        input_image=None,
    ):
        url = f"http://{server_address}:{server_port}/generate"

        payload: dict = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "save_image": save_image,
            "width": width,
            "height": height,
        }

        if input_image is not None:
            img_np = (input_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            payload["input_image"] = base64.b64encode(buf.getvalue()).decode()
            payload["edit_base_size"] = edit_base_size

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                result = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()
                err_data = json.loads(body)
                body = err_data.get("error", body)
            except Exception:
                pass
            raise RuntimeError(
                f"JoyAI API returned HTTP {e.code}: {body or e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach JoyAI API at {url} — is the server running? ({e})"
            ) from e

        if "error" in result:
            raise RuntimeError(f"JoyAI API error: {result['error']}")

        img_bytes = base64.b64decode(result["image"])
        pil_result = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(pil_result).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # [1, H, W, C]

        return (img_tensor,)
