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

    config_file = root / "infer_config.py"
    if not config_file.exists():
        _write_default_config(config_file)
        logger.info("Generated infer_config.py at %s", config_file)

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

    return root


def _write_default_config(path: Path) -> None:
    path.write_text('''\
from dataclasses import dataclass, field
from pathlib import Path

from infer_runtime.infer_config import InferConfig


def _resolve_root() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "transformer").exists() and (here / "vae").exists() and (here / "JoyAI-Image-Und").exists():
        return here
    raise ValueError(
        "Place this config file directly inside the checkpoint root."
    )


_ROOT = _resolve_root()


@dataclass
class JoyAIImageInferConfig(InferConfig):
    dit_arch_config: dict = field(
        default_factory=lambda: {
            "target": "modules.models.Transformer3DModel",
            "params": {
                "hidden_size": 4096,
                "in_channels": 16,
                "heads_num": 32,
                "mm_double_blocks_depth": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "rope_dim_list": [16, 56, 56],
                "text_states_dim": 4096,
                "rope_type": "rope",
                "dit_modulation_type": "wanx",
                "theta": 10000,
                "attn_backend": "flash_attn",
            },
        }
    )
    vae_arch_config: dict = field(
        default_factory=lambda: {
            "target": "modules.models.WanxVAE",
            "params": {
                "pretrained": str(_ROOT / "vae" / "Wan2.1_VAE.safetensors"),
            },
        }
    )
    text_encoder_arch_config: dict = field(
        default_factory=lambda: {
            "target": "modules.models.load_text_encoder",
            "params": {
                "text_encoder_ckpt": str(_ROOT / "JoyAI-Image-Und"),
            },
        }
    )
    scheduler_arch_config: dict = field(
        default_factory=lambda: {
            "target": "modules.models.FlowMatchDiscreteScheduler",
            "params": {
                "num_train_timesteps": 1000,
                "shift": 4.0,
            },
        }
    )

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precision: str = "bf16"
    text_token_max_length: int = 2048

    hsdp_shard_dim: int = 1
    reshard_after_forward: bool = False
    use_fsdp_inference: bool = False
    cpu_offload: bool = False
    pin_cpu_memory: bool = False
''')
