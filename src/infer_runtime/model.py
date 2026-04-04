from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

import numpy as np
from einops import rearrange
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm

from infer_runtime.infer_config import InferConfig, load_infer_config_class_from_pyfile
from infer_runtime.prompt_rewrite import rewrite_prompt
from infer_runtime.settings import InferSettings
from modules.models import load_dit, load_pipeline
from modules.models.pipeline import retrieve_timesteps
from modules.utils import _dynamic_resize_from_bucket, seed_everything
from modules.utils.constants import PRECISION_TO_TYPE
from modules.utils.logging import get_logger

logger = get_logger()

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _patch_fp8_forward(model: nn.Module, compute_dtype: torch.dtype) -> int:
    """Wrap leaf modules that have FP8 parameters so they upcast per-op.

    Weights stay stored as FP8 in VRAM (~16 GB for the DiT).  During each
    module's forward, FP8 params are temporarily cast to *compute_dtype*,
    then restored.  Peak overhead is one layer's worth of bf16 weights.
    """
    patched = 0
    for module in model.modules():
        fp8_names = [
            name for name, p in module.named_parameters(recurse=False)
            if p.dtype in _FP8_DTYPES
        ]
        if not fp8_names:
            continue

        orig_forward = module.forward

        def _make_wrapper(mod, fwd, names, dtype):
            def wrapper(*args, **kwargs):
                backup = {}
                for n in names:
                    p = getattr(mod, n)
                    backup[n] = p.data
                    p.data = p.data.to(dtype)
                try:
                    return fwd(*args, **kwargs)
                finally:
                    for n, data in backup.items():
                        getattr(mod, n).data = data
            return wrapper

        module.forward = _make_wrapper(module, orig_forward, fp8_names, compute_dtype)
        patched += 1
    return patched


@dataclass
class InferenceParams:
    prompt: str
    image: Optional[Image.Image]
    height: int
    width: int
    steps: int
    guidance_scale: float
    seed: int
    neg_prompt: str
    basesize: int


class EditModel:
    def __init__(
        self,
        settings: InferSettings,
        device: torch.device,
        hsdp_shard_dim_override: int | None = None,
    ):
        self.settings = settings
        self.device = device
        self.high_vram = settings.high_vram
        self._rewrite_cache: dict[str, str] = {}

        config_class = load_infer_config_class_from_pyfile(settings.config_path)
        self.cfg: InferConfig = config_class()
        self.cfg.dit_ckpt = settings.ckpt_path
        self.cfg.full_precision = settings.full_precision
        self.cfg.training_mode = False
        if hsdp_shard_dim_override is not None:
            self.cfg.hsdp_shard_dim = hsdp_shard_dim_override
        if int(os.environ.get('WORLD_SIZE', '1')) > 1 and self.cfg.hsdp_shard_dim > 1:
            self.cfg.use_fsdp_inference = True

        if self.high_vram:
            load_device = self.device
        else:
            load_device = torch.device('cpu')

        logger.info(f"Loading DiT to {load_device} (high_vram={self.high_vram})")
        self.dit = load_dit(self.cfg, device=load_device)
        self.dit.requires_grad_(False)
        self.dit.eval()

        if not self.high_vram:
            compute_dtype = PRECISION_TO_TYPE[self.cfg.dit_precision]
            n = _patch_fp8_forward(self.dit, compute_dtype)
            if n > 0:
                logger.info(f"Patched {n} modules for FP8→{compute_dtype} forward")

        logger.info(f"Loading pipeline components to {load_device}")
        self.pipeline = load_pipeline(self.cfg, self.dit, load_device)

    def maybe_rewrite_prompt(self, prompt: str, image: Optional[Image.Image], enabled: bool) -> str:
        if not enabled:
            return str(prompt or '')
        cache_key = f"prompt={prompt.strip()}"
        if image is not None:
            cache_key += f"|image={image.size[0]}x{image.size[1]}"
        if cache_key not in self._rewrite_cache:
            self._rewrite_cache[cache_key] = rewrite_prompt(
                prompt,
                image,
                model=self.settings.rewrite_model,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
        return self._rewrite_cache[cache_key]

    @torch.no_grad()
    def infer(self, params: InferenceParams) -> Image.Image:
        if self.high_vram:
            return self._infer_highvram(params)
        try:
            return self._infer_offload(params)
        except Exception:
            self._emergency_offload()
            raise

    def _emergency_offload(self) -> None:
        """Move all pipeline components back to CPU after a failed generation."""
        cpu = torch.device('cpu')
        pipe = self.pipeline
        for name in ('text_encoder', 'transformer', 'vae'):
            component = getattr(pipe, name, None)
            if component is not None:
                try:
                    component.to(cpu)
                except Exception:
                    pass
        torch.cuda.empty_cache()
        logger.info("Emergency offload: all components moved to CPU")

    # ------------------------------------------------------------------
    # High-VRAM path: everything stays on GPU (original behavior)
    # ------------------------------------------------------------------

    def _infer_highvram(self, params: InferenceParams) -> Image.Image:
        prompts, negative_prompt, images, height, width = self._prepare_inputs(params)

        generator_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
        generator = torch.Generator(device=generator_device).manual_seed(int(params.seed))
        output = self.pipeline(
            prompt=prompts,
            negative_prompt=negative_prompt,
            images=images,
            height=height,
            width=width,
            num_frames=1,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance_scale,
            generator=generator,
            num_videos_per_prompt=1,
            output_type='pt',
            return_dict=False,
        )
        image_tensor = (output[0, -1, 0] * 255).to(torch.uint8).cpu()
        return Image.fromarray(image_tensor.permute(1, 2, 0).numpy())

    # ------------------------------------------------------------------
    # Offloading path: swing components in/out of VRAM
    # ------------------------------------------------------------------

    def _infer_offload(self, params: InferenceParams) -> Image.Image:
        pipe = self.pipeline
        gpu = self.device
        cpu = torch.device('cpu')
        dtype = PRECISION_TO_TYPE[self.cfg.dit_precision]
        prompts, neg_prompts, pil_images, height, width = self._prepare_inputs(params)

        num_items = 1 if pil_images is None else 1 + len(pil_images)
        do_cfg = params.guidance_scale > 1.0

        # ---- Phase 1: Text Encoding (text_encoder → GPU) ----
        logger.info("Phase 1/4: Text encoding")
        pipe.text_encoder.to(gpu)

        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt=prompts, images=pil_images, device=gpu,
            max_sequence_length=self.cfg.text_token_max_length,
            template_type='image',
        )

        if do_cfg:
            if not neg_prompts or not neg_prompts[0]:
                if num_items <= 1:
                    neg_prompts = ["<|im_start|>user\n<|im_end|>\n"]
                else:
                    itokens = "<image>\n" * (num_items - 1)
                    neg_prompts = [f"<|im_start|>user\n{itokens}<|im_end|>\n"]

            neg_embeds, neg_mask = pipe.encode_prompt(
                prompt=neg_prompts, images=pil_images, device=gpu,
                max_sequence_length=self.cfg.text_token_max_length,
                template_type='image',
            )

            max_seq = max(prompt_embeds.shape[1], neg_embeds.shape[1])
            prompt_embeds = torch.cat([
                pipe.pad_sequence(neg_embeds, max_seq),
                pipe.pad_sequence(prompt_embeds, max_seq),
            ])
            if prompt_embeds_mask is not None and neg_mask is not None:
                prompt_embeds_mask = torch.cat([
                    pipe.pad_sequence(neg_mask, max_seq),
                    pipe.pad_sequence(prompt_embeds_mask, max_seq),
                ])

        prompt_embeds = prompt_embeds.to(cpu)
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(cpu)
        pipe.text_encoder.to(cpu)
        torch.cuda.empty_cache()

        # ---- Phase 2: VAE condition encode (VAE → GPU if editing) ----
        num_channels_latents = 16
        latent_t = 1  # single-frame
        latent_h = height // pipe.vae_scale_factor
        latent_w = width // pipe.vae_scale_factor
        generator = torch.Generator(device=cpu).manual_seed(int(params.seed))

        if pil_images is not None:
            logger.info("Phase 2/4: VAE encoding (image condition)")
            pipe.vae.to(gpu)

            ref_imgs = [torch.from_numpy(np.array(x.convert("RGB"))) for x in pil_images]
            ref_imgs = torch.stack(ref_imgs).to(device=gpu, dtype=dtype)
            ref_imgs = ref_imgs / 127.5 - 1.0
            ref_imgs = rearrange(ref_imgs, "x h w c -> x c 1 h w")
            ref_vae = pipe.vae.encode(ref_imgs)
            ref_vae = rearrange(ref_vae, "(b n) c 1 h w -> b n c 1 h w", n=(num_items - 1))

            noise = torch.randn(
                1, 1, num_channels_latents, latent_t, latent_h, latent_w,
                generator=generator, dtype=dtype, device=cpu,
            ).to(gpu)
            latents = torch.cat([ref_vae, noise], dim=1)
            ref_latents = latents[:, :(num_items - 1)].clone().to(cpu)
            latents = latents.to(cpu)

            pipe.vae.to(cpu)
            torch.cuda.empty_cache()
        else:
            logger.info("Phase 2/4: Preparing latents (text-to-image)")
            latents = torch.randn(
                1, num_items, num_channels_latents, latent_t, latent_h, latent_w,
                generator=generator, dtype=dtype, device=cpu,
            )
            ref_latents = None

        # ---- Phase 3: Denoising loop (transformer → GPU) ----
        logger.info(f"Phase 3/4: Denoising ({params.steps} steps)")
        pipe.transformer.to(gpu)
        latents = latents.to(gpu)
        prompt_embeds = prompt_embeds.to(gpu)
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(gpu)
        if ref_latents is not None:
            ref_latents = ref_latents.to(gpu)

        timesteps, _ = retrieve_timesteps(pipe.scheduler, params.steps, gpu)
        target_dtype = PRECISION_TO_TYPE[pipe.args.dit_precision]
        autocast_on = target_dtype != torch.float32

        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            if ref_latents is not None and num_items > 1:
                latents[:, :(num_items - 1)] = ref_latents.clone()

            latent_in = torch.cat([latents] * 2) if do_cfg else latents
            t_expand = t.repeat(latent_in.shape[0])

            with torch.autocast("cuda", dtype=target_dtype, enabled=autocast_on):
                noise_pred = pipe.transformer(
                    hidden_states=latent_in,
                    timestep=t_expand,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    return_dict=False,
                )[0]

            if do_cfg:
                uncond, cond = noise_pred.chunk(2)
                noise_pred = uncond + params.guidance_scale * (cond - uncond)
                cond_norm = torch.norm(cond, dim=2, keepdim=True)
                noise_norm = torch.norm(noise_pred, dim=2, keepdim=True)
                noise_pred = noise_pred * (cond_norm / noise_norm)

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = latents.to(cpu)
        del prompt_embeds, prompt_embeds_mask, ref_latents
        pipe.transformer.to(cpu)
        torch.cuda.empty_cache()

        # ---- Phase 4: VAE decode (VAE → GPU) ----
        logger.info("Phase 4/4: VAE decoding")
        pipe.vae.to(gpu)

        latents_flat = rearrange(latents.to(gpu), "b n c f h w -> (b n) c f h w")
        vae_dtype = PRECISION_TO_TYPE[pipe.args.vae_precision]
        vae_autocast = vae_dtype != torch.float32

        with torch.autocast("cuda", dtype=vae_dtype, enabled=vae_autocast):
            image = pipe.vae.decode(latents_flat, return_dict=False)[0]
            image = rearrange(image, "(b n) c f h w -> b n c f h w", b=1)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float().permute(0, 1, 3, 2, 4, 5)

        pipe.vae.to(cpu)
        torch.cuda.empty_cache()

        image_tensor = (image[0, -1, 0] * 255).to(torch.uint8)
        return Image.fromarray(image_tensor.permute(1, 2, 0).numpy())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_inputs(self, params: InferenceParams):
        if params.image is None:
            return [params.prompt], [params.neg_prompt], None, params.height, params.width

        processed = _dynamic_resize_from_bucket(params.image, basesize=params.basesize)
        width, height = processed.size
        image_tokens = '<image>\n'
        prompts = [f"<|im_start|>user\n{image_tokens}{params.prompt}<|im_end|>\n"]
        neg_prompts = [f"<|im_start|>user\n{image_tokens}{params.neg_prompt}<|im_end|>\n"]
        return prompts, neg_prompts, [processed], height, width


def build_model(
    settings: InferSettings,
    device: torch.device | None = None,
    hsdp_shard_dim_override: int | None = None,
) -> EditModel:
    seed_everything(settings.default_seed)
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return EditModel(
        settings=settings,
        device=device,
        hsdp_shard_dim_override=hsdp_shard_dim_override,
    )
