# Gaussian-Codebook Diffusion-LM: Initial Model

## Goal

Build a character-level continuous diffusion language model on the Shakespeare
corpus. Each character is represented by a learned diagonal Gaussian cell,
not a single embedding vector. The cells must remain individually decodable,
while their aggregate distribution is encouraged to match the standard-normal
diffusion prior.

This is intentionally a small baseline. It should be large enough to model
local Shakespearean character patterns without making architecture scale the
main variable in the first experiment.

## Data

- Tokenization: characters, including whitespace and punctuation.
- Vocabulary: 84 characters in the current corpus preparation.
- Training examples: contiguous blocks of 64 characters; no padding tokens.
- Dataset size: roughly 5.3 million characters.

Use a held-out contiguous tail of the corpus for validation so that validation
blocks do not overlap training blocks.

## Architecture

The denoiser is a non-causal Transformer: every character position can attend
to every other position in its 64-character block. This is appropriate for
diffusion, since all positions are denoised together.

| Component | Initial setting |
| --- | --- |
| Sequence length | 64 |
| Gaussian/Transformer width | 256 |
| Transformer layers | 6 |
| Attention heads | 8 |
| Feed-forward width | 1,024 |
| Positional embedding | learned, length 64 |
| Timestep embedding | sinusoidal projection followed by an MLP |
| Dropout | 0.1 |
| Diffusion steps during training | 200 |
| Sampling | DDIM, initially 50 reverse steps |

This is about five million denoiser parameters, depending on the exact
Transformer implementation. It is enough for a first character-level
Diffusion-LM run; do not increase it until the point-embedding and Gaussian
codebook variants both train and sample correctly.

## Gaussian codebook

For each vocabulary token \(w\), learn

\[
q_\phi(x_0 \mid w) =
\mathcal N\u2060(\mu_w, \operatorname{diag}(\sigma_w^2)).
\]

The codebook has two trainable tensors of shape `[vocab_size, 256]`:

- `mu`: component centers.
- `raw_std`: converted to a positive standard deviation with
  `softplus(raw_std) + min_std`.

For a token block `w` with shape `[batch, length]`, sample the clean diffusion
state using the reparameterization trick:

```python
x0 = mu[w] + std[w] * torch.randn_like(mu[w])
```

Then use the standard variance-preserving forward transition:

\[
x_t = \sqrt{\bar\alpha_t}x_0 +
      \sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

The Transformer receives `x_t`, positional embeddings, and a timestep
embedding, and predicts \(\hat{x}_0\) directly.

## Rounding

Use a linear rounding head with its vocabulary weights tied to the codebook
means, as in the authors' end-to-end Diffusion-LM implementation. Train this
decoder on sampled clean states \(x_0\), then use the same decoder to round
the denoiser's \(\hat{x}_0\) at sampling time.

At sample time, choose the highest-logit token at each position. Clamping the
predicted \(\hat{x}_0\) to the selected token's mean before a reverse step is
an optional Diffusion-LM decoding ablation.

## Losses

For a normal training batch:

\[
\mathcal L_\text{diffusion} =
\operatorname{MSE}(\hat{x}_0, x_0)
\]

\[
\mathcal L_\text{rounding} =
-\log p_\theta(w\mid x_0).
\]

The third loss operates on the codebook as a whole, not separately on each
token. Draw token IDs from \(\pi\), sample their Gaussian cells, and compare
those samples with \(\mathcal N(0,I)\):

\[
z_i = \mu_{w_i} + \sigma_{w_i}\odot\epsilon_i,
\quad w_i\sim\pi,
\]

\[
\mathcal L_\text{prior} =
D\left(\sum_w\pi_w\mathcal N(\mu_w,\Sigma_w),\mathcal N(0,I)\right).
\]

The initial implementation uses sliced Wasserstein distance as `D`; it is
implemented in `gaussian_codebook_loss.py`. Moment matching is available as a
cheap diagnostic or auxiliary loss, but is weaker because it only matches
mean and covariance.

\[
\mathcal L =
\mathcal L_\text{diffusion} +
\mathcal L_\text{rounding} +
\lambda_\text{prior}\mathcal L_\text{prior}.
\]

Start with `pi` equal to empirical character frequencies, because it matches
the distribution the denoiser sees in training. Uniform `pi` is a separate
codebook-coverage ablation.

## Essential comparisons

1. Point codebook: `x0 = embedding[w] + fixed_small_noise`.
2. Gaussian codebook, diffusion and rounding losses only.
3. Gaussian codebook plus aggregate-prior loss.

Log sample quality, rounding accuracy across timesteps, denoising MSE, prior
loss, aggregate sample mean/covariance, per-token standard-deviation norms,
and validation loss. The third comparison isolates whether matching the
aggregate codebook to the diffusion prior helps beyond merely adding learned
noise.
