"""Local inference entrypoint for the image understanding capability of JoyAI-Image."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PIL import Image

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run image understanding inference with the JoyAI-Image MLLM.",
    )
    parser.add_argument(
        "--ckpt-root", required=True,
        help="Checkpoint root containing text_encoder/ directory.",
    )
    parser.add_argument(
        "--image", required=True,
        help="Input image path (or comma-separated paths for multiple images).",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="User prompt / question about the image. "
             "Defaults to a detailed description request if not provided.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.8,
        help="Top-p (nucleus) sampling threshold.",
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k sampling threshold.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional output file to save the response text.",
    )
    return parser.parse_args()


def load_images(image_arg: str) -> list[Image.Image]:
    paths = [p.strip() for p in image_arg.split(",")]
    images = []
    for p in paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Image not found: {p}")
        images.append(Image.open(p).convert("RGB"))
    return images


def resolve_text_encoder_path(ckpt_root: str) -> Path:
    root = Path(ckpt_root).expanduser().resolve()
    text_encoder_dir = root / "JoyAI-Image-Und"
    if not text_encoder_dir.is_dir():
        raise FileNotFoundError(
            f"Expected text_encoder/ directory inside checkpoint root: {root}"
        )
    return text_encoder_dir


def build_conversation(
    images: list[Image.Image],
    prompt: str | None,
) -> list[dict]:
    SYS_PROMPT = "You are a helpful assistant."
    
    default_prompt = "Describe this image in detail."
    user_text = prompt if prompt is not None else default_prompt

    image_content = [{"type": "image", "image": img} for img in images]
    user_content = image_content + [{"type": "text", "text": user_text}]

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages


def main() -> None:
    args = parse_args()

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    text_encoder_path = resolve_text_encoder_path(args.ckpt_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading MLLM from: {text_encoder_path}")
    print(f"Device: {device}, dtype: {dtype}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(text_encoder_path),
        torch_dtype=dtype,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(
        str(text_encoder_path),
        local_files_only=True,
        trust_remote_code=True,
    )

    images = load_images(args.image)
    messages = build_conversation(images, args.prompt)

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text_input],
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(device)

    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print("Generating...")

    start_time = time.time()

    generate_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
    )
    if args.temperature == 0:
        generate_kwargs["do_sample"] = False
    else:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p
        generate_kwargs["top_k"] = args.top_k

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)

    # Strip the input tokens from the output
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]

    elapsed = time.time() - start_time
    num_output_tokens = generated_ids.shape[1]

    print(f"\n{'=' * 60}")
    print(f"Response:\n{response}")
    print(f"{'=' * 60}")
    print(f"Output tokens: {num_output_tokens}")
    print(f"Time: {elapsed:.2f}s ({num_output_tokens / elapsed:.1f} tok/s)")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response, encoding="utf-8")
        print(f"Saved response to: {output_path}")


if __name__ == "__main__":
    main()
