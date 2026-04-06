"""ComfyUI custom node — JoyAI Image Generate (local inference).

Copy this folder into ComfyUI/custom_nodes/ to register the node.
Models are auto-downloaded from HuggingFace on first use.

Setup (in the ComfyUI venv):
    cd /path/to/JoyAI-Image && pip install -e .
    pip install bitsandbytes

Or to install directly from GitHub:
    pip install git+https://github.com/SanDiegoDude/JoyAI-Image.git
    pip install bitsandbytes

Checkpoints are stored in ~/.cache/joyai-image/ckpts_infer/ by default.
Override with the JOYAI_CKPT_ROOT environment variable.
"""

from .nodes import JoyAIImageGenerate

NODE_CLASS_MAPPINGS = {
    "JoyAI Image Generate": JoyAIImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyAI Image Generate": "JoyAI Image Generate",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
