"""Plain DDPM over 1D return windows: dilated residual conv denoiser, linear schedule,
epsilon-prediction loss, ancestral sampler."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class Conv1d(nn.Conv1d):
    """'Same'-padded odd-kernel Conv1d computed as one matmul.

    Same weights and outputs as nn.Conv1d, but ~10x faster on macOS CPU builds of torch,
    which lack MKLDNN and fall back to a slow conv kernel.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k, d, L = self.kernel_size[0], self.dilation[0], x.shape[-1]
        pad = d * (k - 1) // 2
        xp = F.pad(x, (pad, pad))
        cols = torch.stack([xp[:, :, i * d : i * d + L] for i in range(k)], dim=2)
        return torch.einsum("bckl,ock->bol", cols, self.weight) + self.bias[:, None]


class ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.emb = nn.Linear(emb_dim, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(F.silu(self.norm1(h))) + self.emb(emb)[:, :, None]
        x = self.conv2(F.silu(self.norm2(x)))
        return h + x


class Denoiser(nn.Module):
    """eps_theta(x_t, t) for x of shape (batch, length). Dilations 1..32 give a receptive
    field wider than a 64-day window, so every output sees the whole window's volatility."""

    def __init__(self, channels: int = 64, dilations=(1, 2, 4, 8, 16, 32), emb_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.inp = Conv1d(1, channels, 3, padding=1)
        self.blocks = nn.ModuleList(ResBlock(channels, d, emb_dim) for d in dilations)
        self.out = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(), Conv1d(channels, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.emb_mlp(timestep_embedding(t, self.emb_dim))
        h = self.inp(x[:, None, :])
        for b in self.blocks:
            h = b(h, emb)
        return self.out(h)[:, 0, :]


class DDPM:
    def __init__(self, T: int = 1000):
        self.T = T
        # Linear schedule (Ho et al. 2020). The cosine schedule's 0.999 cap means dividing by
        # sqrt(0.001) on the first reverse step, which can manufacture fake fat tails.
        betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
        self.betas = betas.float()
        self.alphas = (1 - betas).float()
        self.abar = torch.cumprod(1 - betas, 0).float()
        abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(1 - betas, 0)[:-1]])
        self.post_var = (betas * (1 - abar_prev) / (1 - torch.cumprod(1 - betas, 0))).float()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.abar[t][:, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.T, (x0.shape[0],))
        noise = torch.randn_like(x0)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int, length: int, generator: torch.Generator | None = None) -> torch.Tensor:
        x = torch.randn(n, length, generator=generator)
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, dtype=torch.long)
            eps = model(x, t)
            mean = (x - self.betas[i] / (1 - self.abar[i]).sqrt() * eps) / self.alphas[i].sqrt()
            if i > 0:
                mean = mean + self.post_var[i].sqrt() * torch.randn(n, length, generator=generator)
            x = mean
        return x
