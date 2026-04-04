import os
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist


def seed_everything(seed: int | None = None) -> None:
    if seed is not None:
        seed = int(seed) & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
#  Distributed helpers (replaces modules.distributed.parallel_states)
# ---------------------------------------------------------------------------

def maybe_init_distributed() -> bool:
    """Initialize torch distributed if WORLD_SIZE > 1. Returns True if initialized."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size <= 1:
        return False
    rank = int(os.environ.get('RANK', '0'))
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    return True


def clean_dist_env() -> None:
    """Destroy the distributed process group if it was initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _dynamic_resize_from_bucket(image: Image, basesize: int = 512):
    from modules.models.bucket import BucketGroup, generate_video_image_bucket
    from typing import Tuple
    import math
    import torchvision.transforms.functional as TF

    def resize_center_crop(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """等比缩放到 >= 目标尺寸，再中心裁剪到目标尺寸。（PIL输入/输出）"""
        w, h = img.size  # PIL: (width, height)
        bh, bw = target_size
        scale = max(bh / h, bw / w)
        resize_h, resize_w = math.ceil(h * scale), math.ceil(w * scale)
        img = TF.resize(img, (resize_h, resize_w),
                        interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        img = TF.center_crop(img, target_size)
        return img

    bucket_config = generate_video_image_bucket(
        basesize=basesize, min_temporal=56, max_temporal=56, bs_img=4, bs_vid=4, bs_mimg=8, min_items=2, max_items=2
    )
    bucket_group = BucketGroup(bucket_config)
    img_w, img_h = image.size
    bucket = bucket_group.find_best_bucket((1, 1, img_h, img_w))
    target_height, target_width = bucket[-2], bucket[-1]  # (height, width)
    img_proc = resize_center_crop(image, (target_height, target_width))
    return img_proc
