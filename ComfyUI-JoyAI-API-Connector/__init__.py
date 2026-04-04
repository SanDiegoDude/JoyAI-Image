"""ComfyUI custom node — JoyAI-Image API Connector.

Symlink or clone this folder into ComfyUI/custom_nodes/ to register the node.
Requires a running JoyAI-Image API server (python app.py --api).
No additional dependencies beyond what ComfyUI already provides.
"""

from .nodes import JoyAIImageGenerate

NODE_CLASS_MAPPINGS = {
    "JoyAI Image Generate (API)": JoyAIImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyAI Image Generate (API)": "JoyAI Image Generate (API)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
