import os
import glob
import torch
import torch.distributed as dist

from modules.models.bucket import BucketGroup
from modules.models.mmdit.dit import Transformer3DModel
from modules.models.mmdit.text_encoder import load_text_encoder
from modules.models.mmdit.vae import WanxVAE
from modules.models.pipeline import Pipeline
from modules.models.scheduler import FlowMatchDiscreteScheduler
from modules.utils.fsdp_load import maybe_load_fsdp_model, pt_weights_iterator, safetensors_weights_iterator
from modules.utils.logging import get_logger
from modules.utils.constants import PRECISION_TO_TYPE
from modules.utils.utils import build_from_config


def load_pipeline(cfg, dit, device: torch.device):
    # vae
    factory_kwargs = {
        'torch_dtype': PRECISION_TO_TYPE[cfg.vae_precision], "device": device}
    vae = build_from_config(cfg.vae_arch_config, **factory_kwargs)
    if getattr(cfg.vae_arch_config, "enable_feature_caching", False):
        vae.enable_feature_caching()

    # text_encoder
    factory_kwargs = {
        'torch_dtype': PRECISION_TO_TYPE[cfg.text_encoder_precision], "device": device}
    tokenizer, text_encoder = build_from_config(
        cfg.text_encoder_arch_config, **factory_kwargs)

    # scheduler
    scheduler = build_from_config(cfg.scheduler_arch_config)

    pipeline = Pipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=dit,
        scheduler=scheduler,
        args=cfg,
    )

    pipeline = pipeline.to(device)
    return pipeline


def _select_safetensors(ckpt_dir: str, full_precision: bool) -> list[str]:
    """Pick either FP8 or full-precision safetensors files from the dir."""
    all_files = glob.glob(os.path.join(str(ckpt_dir), "*.safetensors"))
    if not all_files:
        raise ValueError(f"No safetensors files found in {ckpt_dir}")
    fp8_files = [f for f in all_files if "fp8" in os.path.basename(f).lower()]
    non_fp8 = [f for f in all_files if "fp8" not in os.path.basename(f).lower()]
    if full_precision:
        return non_fp8 or all_files
    return fp8_files or all_files


def load_dit(cfg, device: torch.device) -> torch.nn.Module:
    """Load DiT model.

    When device is CPU (offloading mode), uses a memory-efficient path:
    builds the model on ``meta`` device and loads weights with
    ``assign=True`` so FP8 tensors stay as FP8 in system RAM (~16 GB)
    instead of upcasting to bf16 (~32 GB).
    """
    logger = get_logger()
    low_vram = (device.type == 'cpu')

    # ---- Load state dict from disk ----
    state_dict = None
    if cfg.dit_ckpt is not None:
        logger.info(f"Loading model from: {cfg.dit_ckpt}, type: {cfg.dit_ckpt_type}")

        if cfg.dit_ckpt_type == "safetensor":
            safetensors_files = _select_safetensors(
                cfg.dit_ckpt, getattr(cfg, "full_precision", False))
            logger.info(f"Selected {len(safetensors_files)} safetensors file(s): "
                        f"{[os.path.basename(f) for f in safetensors_files]}")
            state_dict = dict(safetensors_weights_iterator(safetensors_files))
        elif cfg.dit_ckpt_type == "pt":
            pt_files = [cfg.dit_ckpt]
            state_dict = dict(pt_weights_iterator(pt_files))
            if "model" in state_dict:
                state_dict = state_dict["model"]
        else:
            raise ValueError(
                f"Unknown dit_ckpt_type: {cfg.dit_ckpt_type}, must be 'safetensor' or 'pt'")

    dtype = PRECISION_TO_TYPE[cfg.dit_precision]

    # ---- Low-VRAM path: meta-device build + assign=True ----
    if low_vram:
        model_kwargs = {'dtype': dtype, 'device': torch.device('meta'), 'args': cfg}
        model = build_from_config(cfg.dit_arch_config, **model_kwargs)

        if state_dict is not None:
            if "img_in.weight" in state_dict:
                expected = model.img_in.weight.shape
                v = state_dict["img_in.weight"]
                if expected != v.shape:
                    logger.info(f"Inflate img_in.weight from {v.shape} to {expected}")
                    v_new = torch.zeros(expected, dtype=v.dtype)
                    v_new[:, :v.shape[1], :, :, :] = v
                    state_dict["img_in.weight"] = v_new

            fp8_count = sum(1 for v in state_dict.values()
                           if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2))
            if fp8_count > 0:
                logger.info(f"Keeping {fp8_count} FP8 tensors as-is (low-VRAM mode)")

            model.load_state_dict(state_dict, assign=True, strict=True)
            del state_dict

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Instantiate model with {total_params / 1e9:.2f}B parameters")
        return model.eval()

    # ---- High-VRAM path: load to device, upcast FP8 → compute dtype ----
    model_kwargs = {'dtype': dtype, 'device': device, 'args': cfg}
    model = build_from_config(cfg.dit_arch_config, **model_kwargs)
    if not dist.is_initialized() or dist.get_world_size() == 1:
        model.to(device=device)

    if state_dict is not None:
        load_state_dict = {}
        fp8_cast_count = 0
        for k, v in state_dict.items():
            if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                v = v.to(dtype)
                fp8_cast_count += 1

            if k == "img_in.weight" and model.img_in.weight.shape != v.shape:
                logger.info(
                    f"Inflate {k} from {v.shape} to {model.img_in.weight.shape}")
                v_new = v.new_zeros(model.img_in.weight.shape)
                v_new[:, :v.shape[1], :, :, :] = v
                v = v_new

            load_state_dict[k] = v
        if fp8_cast_count > 0:
            logger.info(f"Upcast {fp8_cast_count} FP8 tensors to {dtype}")
        model.load_state_dict(load_state_dict, strict=True)

    model = maybe_load_fsdp_model(
        model=model,
        hsdp_shard_dim=cfg.hsdp_shard_dim,
        reshard_after_forward=cfg.reshard_after_forward,
        param_dtype=dtype,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cpu_offload=cfg.cpu_offload,
        fsdp_inference=cfg.use_fsdp_inference,
        training_mode=cfg.training_mode,
        pin_cpu_memory=cfg.pin_cpu_memory,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Instantiate model with {total_params / 1e9:.2f}B parameters")

    param_dtypes = {param.dtype for param in model.parameters()}
    if len(param_dtypes) > 1:
        logger.warning(
            f"Model has mixed dtypes: {param_dtypes}. Converting to {dtype}")
        model = model.to(dtype)

    return model.eval()

__all__ = [
    "BucketGroup",
    "FlowMatchDiscreteScheduler",
    "Pipeline",
    "Transformer3DModel",
    "WanxVAE",
    "load_pipeline",
    "load_text_encoder",
]
