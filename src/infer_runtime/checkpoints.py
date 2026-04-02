from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CheckpointLayout:
    root: Path
    transformer_ckpt: Path
    vae_ckpt: Path
    text_encoder_ckpt: Path


def _must_exist(path: Path, kind: str) -> Path:
    if not (path.exists() or path.is_symlink()):
        raise FileNotFoundError(f"Missing {kind}: {path}")
    return path


def _find_single_entry(directory: Path, kind: str, *, expect_dir: bool) -> Path:
    _must_exist(directory, f"{kind} directory")
    entries = sorted(p for p in directory.iterdir() if not p.name.startswith("."))
    if len(entries) != 1:
        raise FileNotFoundError(f"Expected exactly one entry in {directory} for {kind}, found {len(entries)}")
    entry = entries[0]
    if expect_dir:
        if not (entry.is_dir() or entry.is_symlink()):
            raise FileNotFoundError(f"Expected directory-like entry for {kind}: {entry}")
    else:
        if not (entry.is_file() or entry.is_symlink()):
            raise FileNotFoundError(f"Expected file-like entry for {kind}: {entry}")
    return entry


def resolve_checkpoint_layout(root: str | Path) -> CheckpointLayout:
    root_path = Path(root).expanduser().resolve()
    transformer_ckpt = str(root_path / "transformer" / "transformer.pth")
    vae_ckpt = _find_single_entry(root_path / "vae", "vae checkpoint", expect_dir=False)
    text_encoder_ckpt = _must_exist(root_path / "JoyAI-Image-Und", "text encoder checkpoint directory")
    if not text_encoder_ckpt.is_dir():
        raise FileNotFoundError(f"Expected text encoder checkpoint directory: {text_encoder_ckpt}")
    return CheckpointLayout(
        root=root_path,
        transformer_ckpt=transformer_ckpt,
        vae_ckpt=vae_ckpt,
        text_encoder_ckpt=text_encoder_ckpt,
    )
    
def build_manifest(layout: CheckpointLayout) -> dict[str, str]:
    return {
        "root": str(layout.root),
        "transformer_ckpt": str(layout.transformer_ckpt),
        "vae_ckpt": str(layout.vae_ckpt),
        "text_encoder_ckpt": str(layout.text_encoder_ckpt),
    }


def write_manifest(layout: CheckpointLayout, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.write_text(json.dumps(build_manifest(layout), indent=2) + "\n", encoding="utf-8")
    return output
