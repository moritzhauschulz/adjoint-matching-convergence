"""Time-conditioned drift network u_θ(x, t) : ℝ^d × [0,1] → ℝ^d."""

import math
import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar t ∈ [0,1] to a sinusoidal embedding in ℝ^dim."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=t.dtype, device=t.device)
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None]                      # [B, half]
        return torch.cat([args.sin(), args.cos()], dim=-1)   # [B, dim]


class DriftMLP(nn.Module):
    """MLP drift network  u_θ : ℝ^d × [0,1] → ℝ^d.

    Input: concat(x, time_features(t)).  `time_embedding`:
      "sinusoidal"  — `t_emb_dim` sinusoidal features (default)
      "raw"         — the scalar t itself (absolute (x, t) coordinates)
    Output: u ∈ ℝ^d
    """

    def __init__(
        self,
        d: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        t_emb_dim: int = 32,
        time_embedding: str = "sinusoidal",
    ):
        super().__init__()
        assert n_layers >= 1
        if time_embedding == "sinusoidal":
            self.t_emb = SinusoidalTimeEmbedding(t_emb_dim)
            n_t_features = t_emb_dim
        elif time_embedding == "raw":
            self.t_emb = None
            n_t_features = 1
        else:
            raise ValueError(
                f"time_embedding must be 'sinusoidal' or 'raw', got {time_embedding!r}")
        self.time_embedding = time_embedding

        dims = [d + n_t_features] + [hidden_dim] * n_layers + [d]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """x: [B, d], t: [B] → u: [B, d]."""
        t_feat = t[:, None] if self.t_emb is None else self.t_emb(t)
        return self.net(torch.cat([x, t_feat], dim=-1))
