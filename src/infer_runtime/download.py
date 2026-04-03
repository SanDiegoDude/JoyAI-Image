"""Auto-download model checkpoints from HuggingFace."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HF_REPO_FP8 = "SanDiegoDude/JoyAI-Image-Edit-FP8"
HF_REPO_FULL = "SanDiegoDude/JoyAI-Image-Edit-Safetensors"


def _has_safetensors(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.safetensors"))


def ensure_checkpoints(
    ckpt_root: str | Path,
    full_precision: bool = False,
) -> Path:
    """Download missing model files from HuggingFace.

    Downloads FP8 transformer by default, or full-precision bf16 with
    ``full_precision=True``.  VAE and text encoder are always downloaded
    from the full-precision repo if missing.
    """
    root = Path(ckpt_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    transformer_dir = root / "transformer"
    vae_dir = root / "vae"
    text_encoder_dir = root / "JoyAI-Image-Und"

    has_transformer = _has_safetensors(transformer_dir)
    has_vae = _has_safetensors(vae_dir)
    has_text_encoder = text_encoder_dir.is_dir() and any(text_encoder_dir.iterdir())

    if has_transformer and has_vae and has_text_encoder:
        return root

    from huggingface_hub import hf_hub_download, snapshot_download

    if not has_transformer:
        if full_precision:
            logger.info("Downloading bf16 transformer from %s ...", HF_REPO_FULL)
            hf_hub_download(
                HF_REPO_FULL,
                "transformer/transformer.safetensors",
                local_dir=str(root),
            )
        else:
            logger.info("Downloading FP8 transformer from %s ...", HF_REPO_FP8)
            hf_hub_download(
                HF_REPO_FP8,
                "transformer/transformer_fp8.safetensors",
                local_dir=str(root),
            )

    if not has_vae:
        logger.info("Downloading VAE from %s ...", HF_REPO_FULL)
        hf_hub_download(
            HF_REPO_FULL,
            "vae/Wan2.1_VAE.safetensors",
            local_dir=str(root),
        )

    if not has_text_encoder:
        logger.info("Downloading text encoder from %s ...", HF_REPO_FULL)
        snapshot_download(
            HF_REPO_FULL,
            allow_patterns="JoyAI-Image-Und/**",
            local_dir=str(root),
        )

    config_file = root / "infer_config.py"
    if not config_file.exists():
        src_config = Path(__file__).resolve().parent.parent.parent / "ckpts_infer" / "infer_config.py"
        if src_config.exists():
            import shutil
            shutil.copy2(src_config, config_file)
            logger.info("Copied infer_config.py to %s", config_file)

    return root
