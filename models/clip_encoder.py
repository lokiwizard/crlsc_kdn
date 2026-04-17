from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _transformers_model_name(model_name: str) -> str:
    aliases = {
        "ViT-B/32": "openai/clip-vit-base-patch32",
        "ViT-B-32": "openai/clip-vit-base-patch32",
        "ViT-B/16": "openai/clip-vit-base-patch16",
        "ViT-B-16": "openai/clip-vit-base-patch16",
    }
    return aliases.get(model_name, model_name)


def _resolve_snapshot_dir(repo_cache_dir: Path) -> Path | None:
    refs_main = repo_cache_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = repo_cache_dir / "snapshots" / revision
        if snapshot_dir.exists():
            return snapshot_dir

    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates = sorted(
        [path for path in snapshots_dir.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate
    return None


def _resolve_model_source(model_name: str, checkpoint_root: str | None) -> tuple[str, bool]:
    hf_name = _transformers_model_name(model_name)
    direct_path = Path(hf_name)
    if direct_path.exists():
        return str(direct_path.resolve()), True

    if checkpoint_root:
        checkpoint_dir = Path(checkpoint_root)
        repo_cache_dir = checkpoint_dir / f"models--{hf_name.replace('/', '--')}"
        if repo_cache_dir.exists():
            snapshot_dir = _resolve_snapshot_dir(repo_cache_dir)
            if snapshot_dir is not None:
                return str(snapshot_dir.resolve()), True

    return hf_name, False


@dataclass
class CLIPImageEncoder:
    model: torch.nn.Module
    preprocess_fn: Callable
    device: torch.device
    backend: str = "transformers"

    @classmethod
    def load(
        cls,
        model_name: str = "openai/clip-vit-base-patch16",
        checkpoint_root: str | None = None,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> "CLIPImageEncoder":
        resolved_device = _resolve_device(device)
        source, force_local = _resolve_model_source(model_name, checkpoint_root)
        local_only = local_files_only or force_local

        processor = CLIPProcessor.from_pretrained(source, local_files_only=local_only)
        model = CLIPModel.from_pretrained(source, local_files_only=local_only).to(resolved_device)
        model.eval()

        def preprocess(image):
            pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
            return pixel_values[0]

        return cls(model=model, preprocess_fn=preprocess, device=resolved_device)

    def preprocess_image(self, image):
        return self.preprocess_fn(image)

    @torch.inference_mode()
    def encode(self, image_batch: torch.Tensor) -> torch.Tensor:
        image_batch = image_batch.to(self.device, non_blocking=True)
        vision_outputs = self.model.vision_model(pixel_values=image_batch)
        pooled = vision_outputs.pooler_output
        if pooled is None:
            pooled = vision_outputs.last_hidden_state[:, 0]

        if hasattr(self.model, "visual_projection") and self.model.visual_projection is not None:
            features = self.model.visual_projection(pooled)
        else:
            features = pooled

        features = features.float()
        return F.normalize(features, dim=-1)
