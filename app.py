"""Gradio UI for JoyAI-Image editing and text-to-image generation."""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import warnings
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

warnings.filterwarnings("ignore")

CKPT_ROOT = str(ROOT_DIR / "ckpts_infer")
OUTPUT_DIR = ROOT_DIR / "outputs"
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()
_load_log: list[str] = []
_model_ready = threading.Event()
_cli_args: argparse.Namespace | None = None


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _load_log.append(line)
    print(line, flush=True)


def _get_log_text() -> str:
    return "\n".join(_load_log)


def _load_models() -> None:
    global _model
    try:
        _log("Importing modules...")
        from infer_runtime.download import ensure_checkpoints
        from infer_runtime.model import build_model
        from infer_runtime.settings import load_settings
        from modules.models.attention import describe_attention_backend

        full_prec = _cli_args.fullprecision if _cli_args else False
        high_vram = _cli_args.highvram if _cli_args else False

        _log("Checking / downloading checkpoints...")
        ensure_checkpoints(CKPT_ROOT, full_precision=full_prec)

        _log("Resolving checkpoint layout...")
        settings = load_settings(
            ckpt_root=CKPT_ROOT,
            default_seed=DEFAULT_SEED,
            full_precision=full_prec,
            high_vram=high_vram,
        )
        mode = "bf16" if full_prec else "FP8"
        vram = "high-VRAM" if high_vram else "offloading"
        _log(f"Config: {settings.config_path}")
        _log(f"Checkpoint: {settings.ckpt_path}")
        _log(f"Precision: {mode} | Memory: {vram}")

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            _log(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            _log("WARNING: CUDA not available! Running on CPU will be extremely slow.")
            _log("Install CUDA PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        _log(f"Attention backend: {describe_attention_backend()}")

        _log("Loading models...")
        t0 = time.time()
        model = build_model(settings, device=device)
        dt = time.time() - t0
        _log(f"All models loaded in {dt:.1f}s. Ready to generate.")

        with _model_lock:
            _model = model
        _model_ready.set()

    except Exception as e:
        _log(f"ERROR: {e}")
        import traceback
        _load_log.append(traceback.format_exc())
        _model_ready.set()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    image: Image.Image | None,
    neg_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    basesize: int,
    height: int,
    width: int,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[Image.Image | None, str]:
    if not prompt or not prompt.strip():
        return None, "Please enter a prompt."

    if not _model_ready.is_set():
        return None, "Models are still loading. Please wait."

    with _model_lock:
        model = _model
    if model is None:
        return None, "Model failed to load. Check the log tab."

    from infer_runtime.model import InferenceParams

    if image is not None:
        image = image.convert("RGB")

    params = InferenceParams(
        prompt=prompt.strip(),
        image=image,
        height=height,
        width=width,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        neg_prompt=neg_prompt.strip(),
        basesize=basesize,
    )

    mode = "image edit" if image is not None else "text-to-image"
    t0 = time.time()
    try:
        result = model.infer(params)
    except Exception as e:
        return None, f"Generation failed: {e}"
    elapsed = time.time() - t0

    filename = f"{int(time.time())}_{int(seed)}.png"
    save_path = OUTPUT_DIR / filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.save(save_path)

    info = (
        f"Mode: {mode} | Steps: {steps} | CFG: {guidance_scale} | "
        f"Seed: {seed} | Time: {elapsed:.1f}s ({elapsed/steps:.2f}s/step)\n"
        f"Saved: {save_path}"
    )
    return result, info


def poll_log() -> str:
    return _get_log_text()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="JoyAI-Image",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# JoyAI-Image\nInstruction-guided image editing and text-to-image generation")

        with gr.Tabs():
            # ---- Generate tab ----
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="Input Image (optional — leave empty for text-to-image)",
                            type="pil",
                            height=400,
                        )
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="e.g. Turn the plate blue",
                            lines=3,
                        )
                        neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="(optional)",
                            lines=2,
                        )
                        with gr.Row():
                            steps = gr.Slider(
                                label="Steps", minimum=1, maximum=100,
                                value=30, step=1,
                            )
                            guidance = gr.Slider(
                                label="CFG Scale", minimum=1.0, maximum=20.0,
                                value=5.0, step=0.5,
                            )
                        with gr.Row():
                            seed = gr.Number(label="Seed", value=42, precision=0)
                            basesize = gr.Dropdown(
                                label="Base Size (edit mode)",
                                choices=[256, 512, 768, 1024, 1280, 1536, 1792, 2048],
                                value=1024,
                            )
                        with gr.Accordion("Text-to-Image Dimensions", open=False):
                            gr.Markdown("*Only used when no input image is provided.*")
                            with gr.Row():
                                t2i_height = gr.Slider(
                                    label="Height", minimum=256, maximum=2048,
                                    value=1024, step=64,
                                )
                                t2i_width = gr.Slider(
                                    label="Width", minimum=256, maximum=2048,
                                    value=1024, step=64,
                                )
                        generate_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Output", type="pil", height=512)
                        info_text = gr.Textbox(label="Info", interactive=False, lines=2)

                generate_btn.click(
                    fn=generate,
                    inputs=[
                        prompt, input_image, neg_prompt,
                        steps, guidance, seed, basesize,
                        t2i_height, t2i_width,
                    ],
                    outputs=[output_image, info_text],
                )

            # ---- Model Status tab ----
            with gr.Tab("Model Status"):
                log_box = gr.Textbox(
                    label="Loading Log",
                    interactive=False,
                    lines=20,
                    max_lines=40,
                )
                refresh_btn = gr.Button("Refresh")
                refresh_btn.click(fn=poll_log, outputs=log_box)

                auto_timer = gr.Timer(value=2, active=True)
                auto_timer.tick(fn=poll_log, outputs=log_box)

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JoyAI-Image Gradio UI")
    parser.add_argument('--fullprecision', action='store_true',
                        help='Use bf16 weights instead of FP8.')
    parser.add_argument('--highvram', action='store_true',
                        help='Keep all models in VRAM (needs ~48 GB+).')
    parser.add_argument('--port', type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    global _cli_args
    _cli_args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app = build_ui()

    load_thread = threading.Thread(target=_load_models, daemon=True)
    load_thread.start()
    _log("Model loading started in background...")
    mode = "bf16" if _cli_args.fullprecision else "FP8"
    vram = "high-VRAM" if _cli_args.highvram else "offloading"
    _log(f"Config: precision={mode}, memory={vram}")

    app.launch(
        server_name="0.0.0.0",
        server_port=_cli_args.port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
