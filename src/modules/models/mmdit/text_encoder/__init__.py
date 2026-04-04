import torch

from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer


def load_text_encoder(
    text_encoder_ckpt: str,
    device: torch.device = torch.device("cpu"),
    torch_dtype: torch.dtype = torch.bfloat16,
    vlm_bits: int = 16,
):
    if vlm_bits == 4:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            text_encoder_ckpt,
            quantization_config=bnb_config,
            device_map=str(device) if device.type == "cuda" else "cpu",
            local_files_only=True,
            trust_remote_code=True,
        ).eval()
    elif vlm_bits == 8:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            text_encoder_ckpt,
            quantization_config=bnb_config,
            device_map=str(device) if device.type == "cuda" else "cpu",
            local_files_only=True,
            trust_remote_code=True,
        ).eval()
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
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
