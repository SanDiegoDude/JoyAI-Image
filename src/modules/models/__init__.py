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


def load_pipeline(cfg, dit, device: torch.device, gpu_device: torch.device | None = None):
    logger = get_logger()

    # vae
    factory_kwargs = {
        'torch_dtype': PRECISION_TO_TYPE[cfg.vae_precision], "device": device}
    vae = build_from_config(cfg.vae_arch_config, **factory_kwargs)
    if getattr(cfg.vae_arch_config, "enable_feature_caching", False):
        vae.enable_feature_caching()

    # text_encoder — bnb quantization needs CUDA during loading
    vlm_bits = getattr(cfg, "vlm_bits", 16)
    if vlm_bits < 16 and device.type != "cuda":
        te_load_device = gpu_device or torch.device("cuda:0")
    else:
        te_load_device = device

    te_kwargs = {
        'torch_dtype': PRECISION_TO_TYPE[cfg.text_encoder_precision],
        "device": te_load_device,
        "vlm_bits": vlm_bits,
    }
    logger.info(f"Loading text encoder ({vlm_bits}-bit) to {te_load_device}")
    tokenizer, text_encoder = build_from_config(
        cfg.text_encoder_arch_config, **te_kwargs)

    if vlm_bits < 16 and device.type != "cuda":
        logger.info("Moving quantized text encoder to CPU for offload mode")
        text_encoder.to("cpu")
        torch.cuda.empty_cache()

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

    nf4 = getattr(cfg, 'nf4_dit', False)
    if vlm_bits >= 16 and not nf4:
        pipeline = pipeline.to(device)
    else:
        pipeline.vae.to(device)
        if not nf4:
            pipeline.transformer.to(device)

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


def _replace_linear_with_nf4(model: torch.nn.Module, compute_dtype: torch.dtype = torch.bfloat16) -> int:
    """Recursively replace nn.Linear with bnb.nn.Linear4bit (NF4).

    Only replaces the structure; weights stay on CPU as bf16.
    Call ``_quantize_nf4_on_gpu`` afterwards to quantize on GPU.
    """
    import bitsandbytes as bnb

    replaced = 0
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Linear):
            has_bias = child.bias is not None
            new_linear = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=has_bias,
                compute_dtype=compute_dtype,
                quant_type='nf4',
                compress_statistics=True,
            )
            new_linear.weight = torch.nn.Parameter(
                child.weight.data.contiguous(), requires_grad=False)
            if has_bias:
                new_linear.bias = torch.nn.Parameter(child.bias.data.clone(), requires_grad=False)
            setattr(model, name, new_linear)
            replaced += 1
        else:
            replaced += _replace_linear_with_nf4(child, compute_dtype)
    return replaced


def _quantize_nf4_on_gpu(model: torch.nn.Module, device: torch.device) -> int:
    """Move model to GPU, explicitly quantizing each Linear4bit layer.

    PyTorch's ``Module.to()`` loses bitsandbytes ``quant_state`` because
    ``_apply`` overwrites parameters.  This function does it correctly:
    each weight is quantized on GPU via ``bnb.functional.quantize_4bit``
    and stored as a proper ``Params4bit`` with intact ``quant_state``.
    Non-4bit parameters and buffers are moved to *device* normally.
    """
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnbF

    quantized = 0
    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            w_gpu = module.weight.data.to(torch.float16).to(device)
            w_4bit, quant_state = bnbF.quantize_4bit(
                w_gpu, quant_type='nf4', compress_statistics=True)
            del w_gpu
            module.weight = bnb.nn.Params4bit(
                w_4bit, requires_grad=False,
                quant_type='nf4', compress_statistics=True,
                quant_state=quant_state,
            )
            if module.bias is not None:
                module.bias.data = module.bias.data.to(device)
            quantized += 1
            continue

        for name, param in module.named_parameters(recurse=False):
            if not isinstance(param, bnb.nn.Params4bit):
                param.data = param.data.to(device)
        for name, buf in module.named_buffers(recurse=False):
            buf.data = buf.data.to(device)

    return quantized


def load_dit(cfg, device: torch.device, gpu_device: torch.device | None = None) -> torch.nn.Module:
    """Load DiT model.

    When device is CPU (offloading mode), uses a memory-efficient path:
    builds the model on ``meta`` device and loads weights with
    ``assign=True`` so FP8 tensors stay as FP8 in system RAM (~16 GB)
    instead of upcasting to bf16 (~32 GB).

    When ``cfg.nf4_dit`` is True, loads bf16 weights and replaces all
    nn.Linear layers with bitsandbytes NF4 4-bit (~8 GB vs 16 GB FP8).
    """
    logger = get_logger()
    low_vram = (device.type == 'cpu')
    nf4 = getattr(cfg, 'nf4_dit', False)

    # ---- NF4 path: load bf16 → replace Linear → quantize on GPU ----
    if nf4:
        return _load_dit_nf4(cfg, device, gpu_device, logger)

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


_NF4_META = ".__nf4__."


def _find_nf4_checkpoint(ckpt_dir: str) -> str | None:
    """Return path to pre-quantized NF4 safetensors if it exists."""
    candidate = os.path.join(ckpt_dir, "transformer_nf4.safetensors")
    return candidate if os.path.isfile(candidate) else None


def _load_nf4_fast(model: torch.nn.Module, nf4_path: str, device: torch.device, logger) -> torch.nn.Module:
    """Fast path: load a pre-quantized NF4 safetensors directly to GPU.

    The file was created by ``convert_to_nf4.py`` which saves packed uint8
    weights alongside their QuantState components as flat tensors.
    """
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnbF
    from safetensors.torch import load_file

    logger.info(f"NF4 DiT (fast): loading pre-quantized {os.path.basename(nf4_path)}")
    raw = load_file(nf4_path)

    nf4_keys = set()
    for k in raw:
        if _NF4_META in k:
            base = k.split(_NF4_META)[0]
            nf4_keys.add(base)

    logger.info(f"  {len(nf4_keys)} NF4-quantized weight tensors detected")

    loaded = 0
    for module_name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            key = f"{module_name}.weight"
            if key not in nf4_keys:
                continue

            packed = raw[key].to(device)
            absmax = raw[f"{key}{_NF4_META}absmax"].to(device)
            quant_map = raw[f"{key}{_NF4_META}quant_map"].to(device)
            shape = torch.Size(raw[f"{key}{_NF4_META}shape"].tolist())
            offset = raw[f"{key}{_NF4_META}offset"].item()

            state2 = None
            nested_key = f"{key}{_NF4_META}nested_absmax"
            if nested_key in raw:
                state2 = bnbF.QuantState(
                    absmax=raw[nested_key].to(device),
                    shape=None,
                    code=raw[f"{key}{_NF4_META}nested_quant_map"].to(device),
                    blocksize=256,
                    dtype=torch.float32,
                    quant_type="linear",
                    offset=None,
                    state2=None,
                )

            qs = bnbF.QuantState(
                absmax=absmax,
                shape=shape,
                code=quant_map,
                blocksize=64,
                dtype=torch.bfloat16,
                quant_type="nf4",
                offset=torch.tensor(offset, device=device),
                state2=state2,
            )

            module.weight = bnb.nn.Params4bit(
                packed, requires_grad=False,
                quant_type="nf4", compress_statistics=True,
                quant_state=qs,
            )

            bias_key = f"{module_name}.bias"
            if bias_key in raw:
                module.bias = torch.nn.Parameter(raw[bias_key].to(device), requires_grad=False)
            loaded += 1
            continue

        for pname, _ in list(module.named_parameters(recurse=False)):
            full_key = f"{module_name}.{pname}" if module_name else pname
            if full_key in raw and full_key not in nf4_keys:
                setattr(module, pname, torch.nn.Parameter(
                    raw[full_key].to(device), requires_grad=False))

        for bname, _ in list(module.named_buffers(recurse=False)):
            full_key = f"{module_name}.{bname}" if module_name else bname
            if full_key in raw:
                module.register_buffer(bname, raw[full_key].to(device))

    del raw
    logger.info(f"  Loaded {loaded} NF4 layers directly to {device}")

    if torch.cuda.is_available():
        nf4_gb = torch.cuda.memory_allocated(device) / 1024**3
        logger.info(f"NF4 DiT (fast): model on GPU = {nf4_gb:.2f} GB")

    return model


def _force_torch_spda(model: torch.nn.Module, logger) -> None:
    """Override attn_backend to torch_spda on all submodules.

    Flash Attention only supports fp16/bf16 data types and will error
    on the uint8 packed weights that NF4 quantization produces.
    """
    patched = 0
    for module in model.modules():
        if hasattr(module, 'attn_backend') and module.attn_backend != 'torch_spda':
            module.attn_backend = 'torch_spda'
            patched += 1
    if patched:
        logger.info(f"NF4 DiT: forced torch_spda attention on {patched} modules "
                     "(flash_attn incompatible with NF4)")


def _load_dit_nf4(cfg, target_device: torch.device, gpu_device: torch.device | None, logger) -> torch.nn.Module:
    """Load DiT with NF4 quantization via bitsandbytes.

    Tries to load a pre-quantized ``transformer_nf4.safetensors`` first
    (created by ``convert_to_nf4.py``).  Falls back to runtime quantization
    from bf16 weights if no pre-quantized file is found.
    """
    import bitsandbytes as bnb

    dtype = torch.bfloat16
    gpu = gpu_device or (target_device if target_device.type == 'cuda' else torch.device('cuda:0'))

    nf4_path = _find_nf4_checkpoint(cfg.dit_ckpt) if cfg.dit_ckpt else None

    if nf4_path:
        logger.info(f"NF4 DiT: found pre-quantized checkpoint: {os.path.basename(nf4_path)}")
        model_kwargs = {'dtype': dtype, 'device': torch.device('meta'), 'args': cfg}
        model = build_from_config(cfg.dit_arch_config, **model_kwargs)
        _replace_linear_with_nf4(model, compute_dtype=dtype)
        model = _load_nf4_fast(model, nf4_path, gpu, logger)
    else:
        logger.warning("NF4 DiT: no pre-quantized file found — falling back to "
                        "runtime quantization (slow, uses ~32 GB RAM). "
                        "Run convert_to_nf4.py once to create the fast-load file.")

        logger.info("NF4 DiT: building model on CPU with bf16 dtype")
        model_kwargs = {'dtype': dtype, 'device': torch.device('cpu'), 'args': cfg}
        model = build_from_config(cfg.dit_arch_config, **model_kwargs)

        if cfg.dit_ckpt is not None:
            logger.info("NF4 DiT: loading bf16 weights (full-precision source)")
            safetensors_files = _select_safetensors(cfg.dit_ckpt, full_precision=True)
            logger.info(f"  Files: {[os.path.basename(f) for f in safetensors_files]}")
            state_dict = dict(safetensors_weights_iterator(safetensors_files))

            for k, v in state_dict.items():
                if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    state_dict[k] = v.to(dtype)

            if "img_in.weight" in state_dict:
                expected = model.img_in.weight.shape
                v = state_dict["img_in.weight"]
                if expected != v.shape:
                    v_new = torch.zeros(expected, dtype=dtype)
                    v_new[:, :v.shape[1], :, :, :] = v
                    state_dict["img_in.weight"] = v_new

            model.load_state_dict(state_dict, strict=True)
            del state_dict

        logger.info("NF4 DiT: replacing nn.Linear → bnb.nn.Linear4bit (NF4)")
        n = _replace_linear_with_nf4(model, compute_dtype=dtype)
        logger.info(f"  Replaced {n} Linear layers")

        logger.info(f"NF4 DiT: quantizing on {gpu}...")
        _quantize_nf4_on_gpu(model, gpu)

    _force_torch_spda(model, logger)

    if torch.cuda.is_available():
        nf4_gb = torch.cuda.memory_allocated(gpu) / 1024**3
        logger.info(f"NF4 DiT: {nf4_gb:.2f} GB on GPU — keeping resident")

    model.requires_grad_(False)
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
