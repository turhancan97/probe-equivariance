"""Frozen vision backbones for feature extraction, via timm.

Minimal, purpose-built version of the friendly-name -> timm-id resolution
pattern used by https://github.com/turhancan97/FeatLens, without its
visualization/hook machinery: we only need a single pooled feature vector
per image, not per-layer token grids.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import timm
import torch
import torch.nn as nn

# Friendly name -> timm model id.
BACKBONE_REGISTRY = {
    "clip_b16_laion": ("vit_base_patch16_clip_224.laion2b"),
    "clip_b16_openai": "vit_base_patch16_clip_224.openai",
}

IMAGE_MEAN = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "clip": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
}


class FrozenBackbone(nn.Module):
    """Wraps a timm ViT, returning one pooled feature vector per image."""

    def __init__(self, name: str, pool: str = "mean"):
        super().__init__()
        if pool not in ("mean", "cls"):
            raise ValueError(f"Unsupported pool mode: {pool}")

        timm_id = BACKBONE_REGISTRY.get(name, name)
        self.model, self.timm_id = _create_model_with_fallbacks(timm_id)
        self.checkpoint_name = name
        self.pool = pool
        self.feat_dim = self.model.num_features

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(images)
        if tokens.dim() == 4:
            # [B, C, H, W] conv feature map -> flatten spatial dims to tokens.
            tokens = tokens.flatten(2).transpose(1, 2)

        if self.pool == "cls":
            return tokens[:, 0]

        has_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0) > 0
        patch_tokens = tokens[:, self.model.num_prefix_tokens :] if has_prefix_tokens else tokens
        return patch_tokens.mean(dim=1)


def _create_model_with_fallbacks(timm_id: str | Iterable[str]) -> Tuple[nn.Module, str]:
    candidate_ids = (timm_id,) if isinstance(timm_id, str) else tuple(timm_id)
    last_error = None
    for candidate_id in candidate_ids:
        try:
            model = timm.create_model(candidate_id, pretrained=True, num_classes=0)
            return model, candidate_id
        except (RuntimeError, ValueError) as err:
            last_error = err

    if last_error is not None:
        raise last_error

    raise ValueError("No backbone candidates were provided.")


def load_backbone(name: str, pool: str = "mean") -> Tuple[FrozenBackbone, int]:
    model = FrozenBackbone(name, pool=pool)
    return model, model.feat_dim
