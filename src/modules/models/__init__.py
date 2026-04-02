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


def load_dit(cfg, device: torch.device) -> torch.nn.Module:
    """Load DiT model with FSDP support."""
    logger = get_logger()

    state_dict = None
    if cfg.dit_ckpt is not None:
        logger.info(
            f"Loading model from: {cfg.dit_ckpt}, type: {cfg.dit_ckpt_type}")

        if cfg.dit_ckpt_type == "safetensor":
            # Find all safetensors files
            safetensors_files = glob.glob(
                os.path.join(str(cfg.dit_ckpt), "*.safetensors"))
            if not safetensors_files:
                raise ValueError(
                    f"No safetensors files found in {cfg.dit_ckpt}")
            state_dict = dict(
                safetensors_weights_iterator(safetensors_files))
        elif cfg.dit_ckpt_type == "pt":
            pt_files = [cfg.dit_ckpt]
            state_dict = dict(pt_weights_iterator(pt_files))
            if "model" in state_dict:
                state_dict = state_dict["model"]
        else:
            raise ValueError(
                f"Unknown dit_ckpt_type: {cfg.dit_ckpt_type}, must be 'safetensor' or 'pt'")

    dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    model_kwargs = {'dtype': dtype, 'device': device, 'args': cfg}
    model = build_from_config(cfg.dit_arch_config, **model_kwargs)
    if not dist.is_initialized() or dist.get_world_size() == 1:
        # Debug mode
        model.to(device=device)

    if state_dict is not None:
        # filter unused params
        load_state_dict = {}
        for k, v in state_dict.items():

            if k == "img_in.weight" and model.img_in.weight.shape != v.shape:
                logger.info(
                    f"Inflate {k} from {v.shape} to {model.img_in.weight.shape}")
                v_new = v.new_zeros(model.img_in.weight.shape)
                v_new[:, :v.shape[1], :, :, :] = v
                v = v_new

            load_state_dict[k] = v
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

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Instantiate model with {total_params / 1e9:.2f}B parameters")

    # Ensure consistent dtype
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
