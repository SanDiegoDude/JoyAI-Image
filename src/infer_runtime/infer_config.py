from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Type


@dataclass
class InferConfig:
    dit_ckpt: str | None = None
    dit_ckpt_type: str = "pt"
    dit_arch_config: dict[str, Any] | None = None
    dit_precision: str = "bf16"

    vae_arch_config: dict[str, Any] | None = None
    vae_precision: str = "bf16"

    text_encoder_arch_config: dict[str, Any] | None = None
    text_encoder_precision: str = "bf16"
    text_token_max_length: int = 2048

    scheduler_arch_config: dict[str, Any] | None = None

    training_mode: bool = False
    hsdp_shard_dim: int = 1
    reshard_after_forward: bool = False
    use_fsdp_inference: bool = False
    cpu_offload: bool = False
    pin_cpu_memory: bool = False


def load_infer_config_class_from_pyfile(file_path: str) -> Type[Any]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for '{file_path}'.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is InferConfig:
            continue
        if issubclass(obj, InferConfig):
            return obj

    raise ValueError(f"No class inheriting from 'InferConfig' was found in '{file_path}'.")
