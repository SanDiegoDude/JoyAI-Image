import torch

from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer


def load_text_encoder(
    text_encoder_ckpt: str,
    device: torch.device = torch.device("cpu"),
    torch_dtype: torch.dtype = torch.bfloat16,
):
    loader = Qwen3VLForConditionalGeneration #or AutoModelForVision2Seq
    model = loader.from_pretrained(
        text_encoder_ckpt,
        torch_dtype=torch_dtype,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        text_encoder_ckpt,
        local_files_only=True,
        trust_remote_code=True,
    )
    return tokenizer, model
