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
    "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m",
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
    "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dino_vitb16": "vit_base_patch16_224.dino",
    "dino_vits16": "vit_small_patch16_224.dino",
    "mae_vitl16": "vit_large_patch16_224.mae",
    "mae_vitb16": "vit_base_patch16_224.mae",
    "supervised_vitl16": "vit_large_patch16_224.augreg_in21k_ft_in1k",
    "supervised_vitb16": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "deit3_small": "deit3_small_patch16_224.fb_in1k",
    "deit3_base": "deit3_base_patch16_224.fb_in1k",
    "deit3_large": "deit3_large_patch16_224.fb_in1k",
    "clip_large_openai": "vit_large_patch14_clip_224.openai",
    "clip_large_laion": "vit_large_patch14_clip_224.laion400m_e32",
    "siglip_vitl16": "vit_large_patch16_siglip_256.v2_webli",
    "siglip_vitb16": "vit_base_patch16_siglip_256.v2_webli",
    "perception_encoder_vitl14": "vit_pe_spatial_large_patch14_448.fb",
    "perception_encoder_vitb16": "vit_pe_spatial_base_patch16_512.fb",
    "perception_encoder_vits16": "vit_pe_spatial_small_patch16_512.fb",
}

# Named normalization presets, selectable per backbone via configs/backbone/*.yaml's
# `image_mean` field ("imagenet" | "clip" | "custom", with custom_mean/custom_std set
# explicitly for "custom"). Fallback normalization (used only when image_mean isn't
# given at all, e.g. backbones without a config file) mirrors "imagenet".
NAMED_NORM_STATS = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "clip": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
}
_DEFAULT_NORM = NAMED_NORM_STATS["imagenet"]


class FrozenBackbone(nn.Module):
    """Wraps a timm ViT, returning one pooled feature vector per image."""

    def __init__(
        self,
        name: str,
        pool: str = "mean",
        image_mean: str | None = None,
        custom_mean: list | None = None,
        custom_std: list | None = None,
        img_size: int | None = None,
    ):
        super().__init__()
        if pool not in ("mean", "cls", "patch"):
            raise ValueError(f"Unsupported pool mode: {pool}")

        timm_id = BACKBONE_REGISTRY.get(name, name)
        if img_size is None:
            self.model, self.timm_id = _create_model_with_fallbacks(timm_id)
        else:
            self.model, self.timm_id = _create_model_with_fallbacks(timm_id, img_size=img_size)
        self.checkpoint_name = name
        self.pool = pool
        self.feat_dim = self.model.num_features
        self.input_size = get_model_input_size(self.model)

        if image_mean is not None:
            self.normalize_mean, self.normalize_std = _resolve_named_norm_stats(image_mean, custom_mean, custom_std)
        else:
            self.normalize_mean, self.normalize_std = _resolve_norm_stats(self.model)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(images)
        if tokens.dim() == 4:
            # [B, C, H, W] conv feature map -> flatten spatial dims to tokens.
            tokens = tokens.flatten(2).transpose(1, 2)

        has_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0) > 0
        patch_tokens = tokens[:, self.model.num_prefix_tokens :] if has_prefix_tokens else tokens

        if self.pool == "patch":
            return patch_tokens

        if self.pool == "cls":
            return tokens[:, 0]

        return patch_tokens.mean(dim=1)


def _resolve_named_norm_stats(
    image_mean: str, custom_mean: list | None, custom_std: list | None
) -> Tuple[list, list]:
    if image_mean == "custom":
        if custom_mean is None or custom_std is None:
            raise ValueError("custom_mean and custom_std must be set when image_mean: custom")
        return list(custom_mean), list(custom_std)
    if image_mean in NAMED_NORM_STATS:
        mean, std = NAMED_NORM_STATS[image_mean]
        return list(mean), list(std)
    raise ValueError(f"Unknown image_mean: {image_mean!r} (expected 'imagenet', 'clip', or 'custom')")


def _resolve_norm_stats(model: nn.Module) -> Tuple[list, list]:
    pretrained_cfg = getattr(model, "pretrained_cfg", None) or {}
    mean = pretrained_cfg.get("mean") if isinstance(pretrained_cfg, dict) else getattr(pretrained_cfg, "mean", None)
    std = pretrained_cfg.get("std") if isinstance(pretrained_cfg, dict) else getattr(pretrained_cfg, "std", None)
    if mean is None or std is None:
        return _DEFAULT_NORM
    return list(mean), list(std)


def get_model_input_size(model: nn.Module) -> int:
    """Return the model's effective square input size after construction."""
    patch_embed = getattr(model, "patch_embed", None)
    image_size = getattr(patch_embed, "img_size", None)
    if image_size is None:
        image_size = getattr(model, "img_size", None)
    if image_size is None:
        pretrained_cfg = getattr(model, "pretrained_cfg", None) or {}
        image_size = (
            pretrained_cfg.get("input_size")
            if isinstance(pretrained_cfg, dict)
            else getattr(pretrained_cfg, "input_size", None)
        )
    if isinstance(image_size, (tuple, list)):
        if not image_size:
            return 224
        if len(image_size) > 1 and len(set(image_size)) != 1:
            raise ValueError(f"Expected a square model input size, got {image_size}")
        image_size = image_size[-1]
    return int(image_size) if image_size is not None else 224


def _create_model_with_fallbacks(
    timm_id: str | Iterable[str], img_size: int | None = None
) -> Tuple[nn.Module, str]:
    candidate_ids = (timm_id,) if isinstance(timm_id, str) else tuple(timm_id)
    last_error = None
    for candidate_id in candidate_ids:
        try:
            kwargs = {"pretrained": True, "num_classes": 0}
            if img_size is not None:
                kwargs["img_size"] = int(img_size)
            model = timm.create_model(candidate_id, **kwargs)
            return model, candidate_id
        except (RuntimeError, ValueError) as err:
            last_error = err

    if last_error is not None:
        raise last_error

    raise ValueError("No backbone candidates were provided.")


def load_backbone(
    name: str,
    pool: str = "mean",
    image_mean: str | None = None,
    custom_mean: list | None = None,
    custom_std: list | None = None,
    img_size: int | None = None,
) -> Tuple[FrozenBackbone, int]:
    model = FrozenBackbone(
        name,
        pool=pool,
        image_mean=image_mean,
        custom_mean=custom_mean,
        custom_std=custom_std,
        img_size=img_size,
    )
    return model, model.feat_dim
