"""LOD worker process — loads models, runs one inference, returns the result.

When the parent kills this process, the OS forcibly reclaims ALL GPU + CPU
memory.  This is the only reliable way to free CUDA memory from Python;
gc.collect() + torch.cuda.empty_cache() leaves residual allocations.
"""
from __future__ import annotations

import io
import os
import sys
import warnings


def run_lod_inference(
    conn,
    src_dir: str,
    config_path: str,
    ckpt_path: str,
    full_precision: bool,
    device_str: str,
    params_dict: dict,
    vlm_bits: int = 8,
    nf4_dit: bool = False,
):
    """Entry point for the LOD worker process (target of multiprocessing.Process).

    Loads the full pipeline from disk using the regular offload path, runs a
    single inference, sends the PNG-encoded result back via *conn*, then exits.
    """
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    warnings.filterwarnings("ignore")

    import torch
    from PIL import Image

    try:
        from infer_runtime.model import EditModel, InferenceParams
        from infer_runtime.settings import InferSettings
        from modules.utils import seed_everything
        from modules.utils.logging import get_logger

        logger = get_logger()
        pid = os.getpid()
        logger.info(f"LOD worker (PID {pid}): starting, device={device_str}")

        device = torch.device(device_str)
        settings = InferSettings(
            config_path=config_path,
            ckpt_path=ckpt_path,
            rewrite_model="",
            openai_api_key=None,
            openai_base_url=None,
            default_seed=params_dict.get("seed", 42),
            full_precision=full_precision,
            high_vram=False,
            lod=False,
            vlm_bits=vlm_bits,
            nf4_dit=nf4_dit,
        )

        seed_everything(settings.default_seed)
        logger.info("LOD worker: loading models to CPU (offload path)...")
        model = EditModel(settings=settings, device=device)

        image = None
        img_bytes = params_dict.get("image_bytes")
        if img_bytes:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        params = InferenceParams(
            prompt=params_dict["prompt"],
            image=image,
            height=params_dict["height"],
            width=params_dict["width"],
            steps=params_dict["steps"],
            guidance_scale=params_dict["guidance_scale"],
            seed=params_dict["seed"],
            neg_prompt=params_dict["neg_prompt"],
            basesize=params_dict["basesize"],
        )

        logger.info("LOD worker: running inference...")
        result = model.infer(params)

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        conn.send(("success", buf.getvalue()))
        logger.info(f"LOD worker (PID {pid}): inference done, exiting")

    except Exception as e:
        import traceback
        conn.send(("error", f"{e}\n{traceback.format_exc()}"))
    finally:
        conn.close()
