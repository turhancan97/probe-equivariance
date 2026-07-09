import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """MLP funnel: feat_dim -> feat_dim/2 -> feat_dim/4 -> feat_dim/8 -> output_dim."""

    def __init__(self, feat_dim: int, output_dim: int, use_layernorm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim) if use_layernorm else None
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, feat_dim // 4),
            nn.ReLU(),
            nn.Linear(feat_dim // 4, feat_dim // 8),
            nn.ReLU(),
            nn.Linear(feat_dim // 8, output_dim),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            feats = self.norm(feats)
        return self.regressor(feats)
