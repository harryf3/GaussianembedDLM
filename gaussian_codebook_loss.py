"""Gaussian codebook sampling and aggregate-prior loss for Diffusion-LM.

The codebook maps every token to a diagonal Gaussian.  The prior loss draws
tokens according to ``pi``, samples their Gaussian cells, and compares the
resulting aggregate mixture to samples from N(0, I).  It does *not* compare
each token Gaussian to N(0, I), which would collapse the codebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DiagonalGaussianCodebook(nn.Module):
    """A vocabulary of reparameterizable diagonal-Gaussian token cells."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        initial_std: float = 0.1,
        min_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if vocab_size < 1 or embedding_dim < 1:
            raise ValueError("vocab_size and embedding_dim must both be positive")

        self.mu = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.02)
        # Inverse softplus makes softplus(raw_std) initially equal initial_std.
        raw_initial_std = torch.log(torch.expm1(torch.tensor(initial_std)))
        self.raw_std = nn.Parameter(
            raw_initial_std.expand(vocab_size, embedding_dim).clone()
        )
        self.min_std = min_std

    @property
    def std(self) -> Tensor:
        return F.softplus(self.raw_std) + self.min_std

    def sample(self, token_ids: Tensor) -> Tensor:
        """Sample x = mu_w + sigma_w * epsilon for each supplied token ID."""
        eps = torch.randn(
            (*token_ids.shape, self.mu.shape[-1]),
            device=self.mu.device,
            dtype=self.mu.dtype,
        )
        return self.mu[token_ids] + self.std[token_ids] * eps

    def sample_aggregate(
        self,
        num_samples: int,
        pi: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Draw samples from sum_w pi[w] N(mu_w, diag(sigma_w**2)).

        ``pi=None`` uses a uniform distribution over the vocabulary.  Pass
        normalized unigram probabilities to match the training-token marginal.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be positive")

        if pi is None:
            token_ids = torch.randint(
                self.mu.shape[0], (num_samples,), device=self.mu.device
            )
        else:
            pi = pi.to(device=self.mu.device, dtype=self.mu.dtype)
            if pi.ndim != 1 or pi.numel() != self.mu.shape[0]:
                raise ValueError("pi must have one non-negative entry per token")
            if (pi < 0).any() or pi.sum() <= 0:
                raise ValueError("pi must be non-negative with positive sum")
            token_ids = torch.multinomial(pi / pi.sum(), num_samples, replacement=True)

        return self.sample(token_ids), token_ids


def sliced_wasserstein_prior_loss(
    aggregate_samples: Tensor,
    *,
    num_projections: int = 128,
) -> Tensor:
    """Estimate distance from aggregate samples to N(0, I).

    Random one-dimensional projections of a standard normal are themselves
    standard normal.  Sorting gives the empirical 1-D Wasserstein-2 distance,
    while averaging projections makes the loss sensitive to the full geometry.
    """
    if aggregate_samples.ndim != 2:
        raise ValueError("aggregate_samples must have shape [samples, embedding_dim]")
    if num_projections < 1:
        raise ValueError("num_projections must be positive")

    _, embedding_dim = aggregate_samples.shape
    directions = torch.randn(
        embedding_dim,
        num_projections,
        device=aggregate_samples.device,
        dtype=aggregate_samples.dtype,
    )
    directions = F.normalize(directions, dim=0)

    projected_codebook = aggregate_samples @ directions
    projected_prior = torch.randn_like(projected_codebook)
    return (projected_codebook.sort(dim=0).values - projected_prior.sort(dim=0).values).square().mean()


def moment_matching_prior_loss(aggregate_samples: Tensor) -> Tensor:
    """Match aggregate mixture samples to the first two moments of N(0, I).

    This is a cheaper, weaker alternative to full distribution matching.  It
    enforces a zero aggregate mean and identity aggregate covariance, but does
    not by itself guarantee that the aggregate mixture has a Gaussian shape.
    """
    if aggregate_samples.ndim != 2:
        raise ValueError("aggregate_samples must have shape [samples, embedding_dim]")

    num_samples, embedding_dim = aggregate_samples.shape
    if num_samples < 2:
        raise ValueError("at least two samples are needed to estimate covariance")

    mean = aggregate_samples.mean(dim=0)
    centered = aggregate_samples - mean
    covariance = centered.T @ centered / num_samples
    identity = torch.eye(
        embedding_dim,
        device=aggregate_samples.device,
        dtype=aggregate_samples.dtype,
    )

    mean_loss = mean.square().mean()
    covariance_loss = (covariance - identity).square().mean()
    return mean_loss + covariance_loss


@dataclass
class LossTerms:
    total: Tensor
    diffusion: Tensor
    rounding: Tensor
    aggregate_prior: Tensor


def gaussian_codebook_objective(
    diffusion_loss: Tensor,
    rounding_loss: Tensor,
    codebook: DiagonalGaussianCodebook,
    *,
    prior_weight: float = 1.0,
    prior_samples: int = 1024,
    pi: Tensor | None = None,
    num_projections: int = 128,
) -> LossTerms:
    """Combine the model losses with aggregate Gaussian-mixture prior matching."""
    aggregate_samples, _ = codebook.sample_aggregate(prior_samples, pi)
    aggregate_prior = sliced_wasserstein_prior_loss(
        aggregate_samples, num_projections=num_projections
    )
    total = diffusion_loss + rounding_loss + prior_weight * aggregate_prior
    return LossTerms(total, diffusion_loss, rounding_loss, aggregate_prior)
