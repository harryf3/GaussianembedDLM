"""Initial Gaussian-codebook Diffusion-LM for character-level Shakespeare.

The denoiser uses bidirectional self-attention with an exact learned relative
position bias.  It has no absolute position embedding: relative positions are
enough for fixed-size, contiguous character blocks and do not assign semantic
meaning to arbitrary block offsets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gaussian_codebook_loss import (
    DiagonalGaussianCodebook,
    LossTerms,
    gaussian_codebook_objective,
)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    sequence_length: int = 64
    dim: int = 256
    layers: int = 6
    heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    diffusion_steps: int = 200


def sinusoidal_timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Return sinusoidal timestep features with shape [batch, dim]."""
    half_dim = dim // 2
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(
            half_dim, device=timesteps.device, dtype=torch.float32
        )
        / max(half_dim - 1, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    return F.pad(embedding, (0, dim % 2))


class RelativePositionBias(nn.Module):
    """Per-head learned attention bias for every signed relative position."""

    def __init__(self, max_length: int, heads: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.bias = nn.Embedding(2 * max_length - 1, heads)
        nn.init.zeros_(self.bias.weight)

    def forward(self, length: int) -> Tensor:
        if length > self.max_length:
            raise ValueError(f"sequence length {length} exceeds {self.max_length}")
        positions = torch.arange(length, device=self.bias.weight.device)
        # Entry [i, j] is the bias for query position i attending to key j.
        relative_position = positions[None, :] - positions[:, None]
        indices = relative_position + self.max_length - 1
        return self.bias(indices).permute(2, 0, 1)  # [heads, length, length]


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, max_length: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout
        self.relative_bias = RelativePositionBias(max_length, heads)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [batch, heads, length, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        bias = self.relative_bias(length).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        return self.out(attended)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.dim)
        self.attention = SelfAttention(
            config.dim, config.heads, config.dropout, config.sequence_length
        )
        self.mlp_norm = nn.LayerNorm(config.dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


def cosine_alpha_bar(steps: int, *, s: float = 0.008) -> Tensor:
    """Cosine variance-preserving schedule with alpha_bar[0] = 1."""
    time = torch.linspace(0, steps, steps + 1, dtype=torch.float32) / steps
    alpha_bar = torch.cos((time + s) / (1 + s) * math.pi / 2).square()
    return alpha_bar / alpha_bar[0]


class GaussianCodebookDiffusionLM(nn.Module):
    """Continuous character diffusion model with learned Gaussian token cells."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.codebook = DiagonalGaussianCodebook(config.vocab_size, config.dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.dim, 4 * config.dim),
            nn.SiLU(),
            nn.Linear(4 * config.dim, config.dim),
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.layers)
        )
        self.final_norm = nn.LayerNorm(config.dim)
        self.x0_head = nn.Linear(config.dim, config.dim)
        self.register_buffer("alpha_bar", cosine_alpha_bar(config.diffusion_steps))

    def rounding_logits(self, vectors: Tensor) -> Tensor:
        """Decode with the codebook centres themselves, with no free bias."""
        return vectors @ self.codebook.mu.T

    def q_sample(self, x0: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> Tensor:
        """Apply the forward diffusion transition to a batch of clean states."""
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[timesteps].to(dtype=x0.dtype).view(-1, 1, 1)
        return alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise

    def forward(self, x_t: Tensor, timesteps: Tensor) -> tuple[Tensor, Tensor]:
        """Predict clean continuous states and rounding logits from noisy states."""
        if x_t.ndim != 3:
            raise ValueError("x_t must have shape [batch, length, dim]")
        if x_t.shape[1] > self.config.sequence_length:
            raise ValueError("input sequence exceeds configured sequence_length")

        time = self.time_mlp(
            sinusoidal_timestep_embedding(timesteps, self.config.dim).to(x_t.dtype)
        ).unsqueeze(1)
        hidden = x_t + time
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        x0_hat = self.x0_head(hidden)
        return x0_hat, self.rounding_logits(x0_hat)

    def training_loss(
        self,
        token_ids: Tensor,
        *,
        prior_weight: float = 1.0,
        prior_samples: int = 1024,
        pi: Tensor | None = None,
        num_projections: int = 128,
    ) -> LossTerms:
        """Compute diffusion, rounding, and aggregate-prior losses for a batch."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, length]")
        if token_ids.shape[1] > self.config.sequence_length:
            raise ValueError("token sequence exceeds configured sequence_length")

        token_ids = token_ids.to(self.codebook.mu.device)
        x0 = self.codebook.sample(token_ids)
        timesteps = torch.randint(
            1,
            self.config.diffusion_steps + 1,
            (token_ids.shape[0],),
            device=token_ids.device,
        )
        x_t = self.q_sample(x0, timesteps)
        x0_hat, _ = self(x_t, timesteps)

        diffusion = F.mse_loss(x0_hat, x0)
        # The e2e Diffusion-LM objective trains p(w | x0), then the same
        # decoder rounds the denoiser's predicted x0 during sampling.
        rounding = F.cross_entropy(
            self.rounding_logits(x0).flatten(0, 1), token_ids.flatten()
        )
        return gaussian_codebook_objective(
            diffusion,
            rounding,
            self.codebook,
            prior_weight=prior_weight,
            prior_samples=prior_samples,
            pi=pi,
            num_projections=num_projections,
        )


class PointEmbeddingDiffusionLM(GaussianCodebookDiffusionLM):
    """Original Diffusion-LM baseline with the exact same denoiser architecture.

    This class intentionally inherits the Gaussian model's Transformer,
    timestep conditioning, forward schedule, x0 head, and rounding head.  The
    only changed mechanism is the codebook: a token has one learned point
    embedding with fixed isotropic x0 noise, rather than a learned Gaussian
    cell.  It is therefore the apples-to-apples baseline.
    """

    def __init__(
        self, config: ModelConfig, *, x0_noise_std: float | None = None
    ) -> None:
        super().__init__(config)
        del self.codebook
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        # Diffusion-LM derives its clean-state noise from the first forward
        # transition. Allow an explicit value only for controlled ablations.
        self.x0_noise_std = (
            float((1 - self.alpha_bar[1]).sqrt())
            if x0_noise_std is None
            else x0_noise_std
        )

    def rounding_logits(self, vectors: Tensor) -> Tensor:
        """Decode with the point-embedding table itself, with no free bias."""
        return vectors @ self.token_embedding.weight.T

    def training_loss(self, token_ids: Tensor) -> LossTerms:  # type: ignore[override]
        """Compute the original point-embedding diffusion and rounding losses."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, length]")
        if token_ids.shape[1] > self.config.sequence_length:
            raise ValueError("token sequence exceeds configured sequence_length")

        token_ids = token_ids.to(self.token_embedding.weight.device)
        x0 = self.token_embedding(token_ids)
        x0 = x0 + self.x0_noise_std * torch.randn_like(x0)
        timesteps = torch.randint(
            1,
            self.config.diffusion_steps + 1,
            (token_ids.shape[0],),
            device=token_ids.device,
        )
        x_t = self.q_sample(x0, timesteps)
        x0_hat, _ = self(x_t, timesteps)
        diffusion = F.mse_loss(x0_hat, x0)
        rounding = F.cross_entropy(
            self.rounding_logits(x0).flatten(0, 1), token_ids.flatten()
        )
        # Matches the authors' e2e terminal-prior mean term. It prevents the
        # learned codebook from retaining a large mean at the final noise step.
        terminal_prior = (self.alpha_bar[-1].sqrt() * x0).square().mean()
        return LossTerms(
            diffusion + rounding + terminal_prior,
            diffusion,
            rounding,
            terminal_prior,
        )

    def round_to_tokens(self, vectors: Tensor) -> Tensor:
        """Round by nearest Euclidean token embedding, as Diffusion-LM does."""
        centers = self.token_embedding.weight
        squared_distance = (
            vectors.square().sum(dim=-1, keepdim=True)
            + centers.square().sum(dim=-1)
            - 2 * vectors @ centers.T
        )
        return squared_distance.argmin(dim=-1)

    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size: int = 1,
        reverse_steps: int = 50,
        clamp: bool = False,
        eta: float = 0.0,
    ) -> Tensor:
        """DDIM reverse sampling; eta=0 is the deterministic source-code path."""
        if not 0.0 <= eta <= 1.0:
            raise ValueError("eta must be in [0, 1]")
        was_training = self.training
        self.eval()
        x_t = torch.randn(
            batch_size,
            self.config.sequence_length,
            self.config.dim,
            device=self.alpha_bar.device,
        )
        times = torch.linspace(
            self.config.diffusion_steps, 0, reverse_steps + 1, device=x_t.device
        ).long()
        for current, next_time in zip(times[:-1], times[1:]):
            timestep_batch = torch.full(
                (batch_size,), current, device=x_t.device, dtype=torch.long
            )
            x0_hat, logits = self(x_t, timestep_batch)
            if clamp:
                x0_hat = self.token_embedding(self.round_to_tokens(x0_hat))
            alpha_t = self.alpha_bar[current].to(x_t.dtype)
            alpha_next = self.alpha_bar[next_time].to(x_t.dtype)
            eps_hat = (x_t - alpha_t.sqrt() * x0_hat) / (1 - alpha_t).sqrt().clamp_min(1e-8)
            sigma = (
                eta
                * ((1 - alpha_next) / (1 - alpha_t)).sqrt()
                * (1 - alpha_t / alpha_next).clamp_min(0).sqrt()
            )
            x_t = (
                alpha_next.sqrt() * x0_hat
                + (1 - alpha_next - sigma.square()).clamp_min(0).sqrt() * eps_hat
                + sigma * torch.randn_like(x_t)
            )

        result = self.round_to_tokens(x_t)
        self.train(was_training)
        return result
