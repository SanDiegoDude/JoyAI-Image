from __future__ import annotations

import os
from dataclasses import dataclass

from infer_runtime.checkpoints import resolve_checkpoint_layout


@dataclass
class InferSettings:
    config_path: str
    ckpt_path: str
    rewrite_model: str
    openai_api_key: str | None
    openai_base_url: str | None
    default_seed: int
    full_precision: bool = False
    high_vram: bool = False
    lod: bool = False
    vlm_bits: int = 8


def load_settings(
    *,
    ckpt_root: str,
    config_path: str | None = None,
    rewrite_model: str | None = None,
    default_seed: int = 42,
    full_precision: bool = False,
    high_vram: bool = False,
    lod: bool = False,
    vlm_bits: int = 8,
) -> InferSettings:
    layout = resolve_checkpoint_layout(ckpt_root)
    default_config = layout.root / 'infer_config.py'
    if config_path is None and not default_config.exists():
        raise FileNotFoundError(
            f"Missing inference config: {default_config}. Pass --config explicitly to choose a config file."
        )

    return InferSettings(
        config_path=config_path or str(default_config),
        ckpt_path=str(layout.transformer_ckpt),
        rewrite_model=rewrite_model or 'gpt-5',
        openai_api_key=os.environ.get('OPENAI_API_KEY'),
        openai_base_url=os.environ.get('OPENAI_BASE_URL'),
        default_seed=default_seed,
        full_precision=full_precision,
        high_vram=high_vram,
        lod=lod,
        vlm_bits=vlm_bits,
    )
