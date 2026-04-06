"""ComfyUI custom node — JoyAI Image Generate (local inference).

Copy this folder into ComfyUI/custom_nodes/ to register the node.
Models are auto-downloaded from HuggingFace on first use.

Setup:
    1. export JOYAI_PATH=/path/to/JoyAI-Image
    2. pip install bitsandbytes  (in your ComfyUI venv)
    3. Copy this folder into ComfyUI/custom_nodes/
    4. Restart ComfyUI

Checkpoints default to $JOYAI_PATH/ckpts_infer/ (your existing models).
Override with JOYAI_CKPT_ROOT if they live elsewhere.
"""

from .nodes import JoyAIImageGenerate

NODE_CLASS_MAPPINGS = {
    "JoyAI Image Generate": JoyAIImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyAI Image Generate": "JoyAI Image Generate",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
