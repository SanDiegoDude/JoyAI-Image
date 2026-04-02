from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

from PIL import Image
import torch

from infer_runtime.infer_config import InferConfig, load_infer_config_class_from_pyfile
from infer_runtime.prompt_rewrite import rewrite_prompt
from infer_runtime.settings import InferSettings
from modules.models import load_dit, load_pipeline
from modules.utils import _dynamic_resize_from_bucket, seed_everything


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
        self._rewrite_cache: dict[str, str] = {}

        config_class = load_infer_config_class_from_pyfile(settings.config_path)
        self.cfg: InferConfig = config_class()
        self.cfg.dit_ckpt = settings.ckpt_path
        self.cfg.training_mode = False
        if hsdp_shard_dim_override is not None:
            self.cfg.hsdp_shard_dim = hsdp_shard_dim_override
        if int(os.environ.get('WORLD_SIZE', '1')) > 1 and self.cfg.hsdp_shard_dim > 1:
            self.cfg.use_fsdp_inference = True

        self.dit = load_dit(self.cfg, device=self.device)
        self.dit.requires_grad_(False)
        self.dit.eval()
        self.pipeline = load_pipeline(self.cfg, self.dit, self.device)

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
        if params.image is None:
            prompts = [params.prompt]
            negative_prompt = [params.neg_prompt]
            images = None
            height = params.height
            width = params.width
        else:
            processed = _dynamic_resize_from_bucket(params.image, basesize=params.basesize)
            width, height = processed.size
            image_tokens = '<image>\n'
            prompts = [f"<|im_start|>user\n{image_tokens}{params.prompt}<|im_end|>\n"]
            negative_prompt = [f"<|im_start|>user\n{image_tokens}{params.neg_prompt}<|im_end|>\n"]
            images = [processed]

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
