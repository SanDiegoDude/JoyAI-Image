"""ComfyUI custom node — JoyAI Image Generate (local inference).

Symlink or clone this folder into ComfyUI/custom_nodes/ to register the node.
Models are auto-downloaded from HuggingFace on first use.

Requirements (in the ComfyUI venv):
    pip install -e /path/to/JoyAI-Image
    pip install bitsandbytes
"""

from .nodes import JoyAIImageGenerate

NODE_CLASS_MAPPINGS = {
    "JoyAI Image Generate": JoyAIImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyAI Image Generate": "JoyAI Image Generate",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
