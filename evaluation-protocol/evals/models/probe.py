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


class EfficientProbingPool(nn.Module):
    """Cross-attention pooling over patch tokens, adapted from efficient probing."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        num_queries: int = 32,
        d_out: int = 1,
    ):
        super().__init__()
        if num_heads != 1:
            raise ValueError("EfficientProbingPool currently supports only num_heads=1.")
        if dim % num_heads != 0:
            raise ValueError(f"feat_dim={dim} must be divisible by num_heads={num_heads}.")
        if d_out <= 0:
            raise ValueError("d_out must be positive.")
        if num_queries <= 0:
            raise ValueError("num_queries must be positive.")
        if dim % d_out != 0:
            raise ValueError(f"feat_dim={dim} must be divisible by d_out={d_out}.")
        if dim % (d_out * num_queries) != 0:
            raise ValueError(
                f"feat_dim={dim} must be divisible by d_out * num_queries={d_out * num_queries}."
            )

        self.num_heads = num_heads
        self.num_queries = num_queries
        self.d_out = d_out
        self.output_dim = dim // d_out

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.v = nn.Linear(dim, self.output_dim, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"EfficientProbingPool expects [B, N, C] tokens, got shape {tuple(x.shape)}.")

        batch_size, num_tokens, feat_dim = x.shape
        cls_token = self.cls_token.expand(batch_size, -1, -1)

        q = cls_token.reshape(batch_size, self.num_queries, self.num_heads, feat_dim // self.num_heads)
        q = q.permute(0, 2, 1, 3) * self.scale
        k = x.reshape(batch_size, num_tokens, self.num_heads, feat_dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(
            batch_size,
            num_tokens,
            self.num_queries,
            self.output_dim // self.num_queries,
        ).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        pooled = torch.matmul(attn.squeeze(1).unsqueeze(2), v)
        return pooled.reshape(batch_size, self.output_dim)


class EfficientProbingHead(nn.Module):
    """Patch-token Efficient Probing followed by a compact regression layer."""

    def __init__(
        self,
        feat_dim: int,
        output_dim: int,
        num_queries: int = 32,
        num_heads: int = 1,
        d_out: int = 1,
        use_layernorm: bool = True,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.pool = EfficientProbingPool(
            dim=feat_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            num_queries=num_queries,
            d_out=d_out,
        )
        pooled_dim = self.pool.output_dim
        self.norm = nn.LayerNorm(pooled_dim) if use_layernorm else None
        self.regressor = nn.Linear(pooled_dim, output_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(feats)
        if self.norm is not None:
            pooled = self.norm(pooled)
        return self.regressor(pooled)
