"""Gaussian diffusion LM with a separate noise level at each character position.

Run ``python3 coherent_gauss_dlm_token_time.py --steps 3000`` from this directory.
This is a bidirectional denoiser, not a next-character language model.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from old.coherent_gauss_dlm import (
    BATCH_SIZE, BLOCK_SIZE, DIM, DIFFUSION_STEPS, FF_DIM, GRAD_CLIP_NORM,
    HOLDOUT, LEARNING_RATE, MMD_BANDWIDTHS, MODEL_DIM, NUM_HEADS, NUM_LAYERS,
    PRIOR_WEIGHT, SEED, WEIGHT_DECAY, ContextualGaussianDLM, data, mmd_loss,
    schedule,
)


OUTPUT_DIR = Path('checkpoints/coherent_gauss_dlm_token_time')
SAMPLE_STEPS = 100


class TokenTimeGaussianDLM(ContextualGaussianDLM):
    def forward(self, x_t, timesteps):
        if timesteps.ndim != 2 or timesteps.shape != x_t.shape[:2]:
            raise ValueError('timesteps must have one value per character')
        pos = torch.arange(x_t.size(1), device=x_t.device)
        angles = timesteps[..., None] * self.time_frequencies
        time_embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        hidden = self.input_projection(x_t) + self.position(pos)[None] + time_embedding
        return self.token_head(self.final_norm(self.transformer(hidden)))


def draw_timesteps(batch_size, device):
    """Mix old synchronized examples with independently corrupted positions."""
    t = torch.randint(1, DIFFUSION_STEPS + 1, (batch_size, BLOCK_SIZE), device=device)
    # Exact low-noise anchors and almost-pure-noise targets occur in training.
    choices = torch.rand((batch_size, BLOCK_SIZE), device=device)
    t = torch.where(choices < 0.15, 1, t)
    t = torch.where(choices > 0.85, DIFFUSION_STEPS, t)
    synchronized = torch.rand((batch_size,), device=device) < 0.2
    global_t = torch.randint(1, DIFFUSION_STEPS + 1, (batch_size, 1), device=device)
    return torch.where(synchronized[:, None], global_t, t)


def corrupt(x0, timesteps, alpha_bar, noise):
    a = alpha_bar[timesteps][..., None]
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


def posterior_x0_mean(x_t, probabilities, alpha, embedding):
    """E[x0 | xt] for the predicted diagonal-Gaussian token mixture."""
    a = alpha[..., None, None]
    means = embedding.mu[None, None]
    variances = embedding.std.square()[None, None]
    denominator = a * variances + (1 - a)
    gain = a.sqrt() * variances / denominator
    intercept = (1 - a) * means / denominator
    return (probabilities[..., None] * (intercept + gain * x_t[:, :, None])).sum(2)


@torch.no_grad()
def evaluate(model, blocks, frequencies, alpha_bar, device, count=64, seed=1234):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    chosen = blocks[torch.randperm(len(blocks), generator=generator)[:count]].to(device)

    def noise(shape):
        return torch.randn(shape, generator=generator).to(device)

    x0 = model.embedding.mu[chosen] + model.embedding.std[chosen] * noise((*chosen.shape, DIM))
    target = torch.zeros(chosen.shape, dtype=torch.bool, device=device)
    target[:, 3::8] = True
    results = {}
    for t in (250, 500, 750):
        # Each noise bucket has its own RNG, so adding a metric cannot change
        # the later buckets or checkpoint selection.
        generator = torch.Generator().manual_seed(seed * 10 + t)
        # The target's own noisy vector is held fixed across all three conditions.
        steps = torch.where(target, t, 1)
        x_t = corrupt(x0, steps, alpha_bar, noise(x0.shape))
        logits = model(x_t, steps)
        original = F.cross_entropy(logits[target], chosen[target]).item()
        shuffled = x_t.clone()
        shuffled[:, ~target[0]] = x_t[torch.randperm(len(chosen), generator=generator)][:, ~target[0]]
        shuffled_logits = model(shuffled, steps)
        shuffled_ce = F.cross_entropy(shuffled_logits[target], chosen[target]).item()
        a = alpha_bar[t]
        variance = a * model.embedding.std.square() + 1 - a
        delta = x_t[target][:, None] - a.sqrt() * model.embedding.mu
        independent = -.5 * (delta.square() / variance + variance.log()).sum(-1) + frequencies.log()
        independent_ce = F.cross_entropy(independent, chosen[target]).item()
        global_steps = torch.full_like(steps, t)
        global_x_t = corrupt(x0, global_steps, alpha_bar, noise(x0.shape))
        global_ce = F.cross_entropy(model(global_x_t, global_steps)[target], chosen[target]).item()
        global_shuffled = global_x_t.clone()
        global_shuffled[:, ~target[0]] = global_x_t[torch.randperm(len(chosen), generator=generator)][:, ~target[0]]
        global_shuffled_ce = F.cross_entropy(model(global_shuffled, global_steps)[target], chosen[target]).item()
        mixed_steps = torch.randint(1, DIFFUSION_STEPS + 1, chosen.shape, generator=generator).to(device)
        mixed_steps[target] = t
        mixed_x_t = corrupt(x0, mixed_steps, alpha_bar, noise(x0.shape))
        mixed_x_t[target] = x_t[target]
        mixed_ce = F.cross_entropy(model(mixed_x_t, mixed_steps)[target], chosen[target]).item()
        mixed_shuffled = mixed_x_t.clone()
        mixed_shuffled_steps = mixed_steps.clone()
        permutation = torch.randperm(len(chosen), generator=generator)
        mixed_shuffled[:, ~target[0]] = mixed_x_t[permutation][:, ~target[0]]
        mixed_shuffled_steps[:, ~target[0]] = mixed_steps[permutation][:, ~target[0]]
        mixed_shuffled_ce = F.cross_entropy(
            model(mixed_shuffled, mixed_shuffled_steps)[target], chosen[target]
        ).item()
        results[str(t)] = {
            'readable_neighbors_ce': original, 'shuffled_neighbors_ce': shuffled_ce,
            'independent_ce': independent_ce, 'all_positions_same_t_ce': global_ce,
            'all_positions_same_t_shuffled_ce': global_shuffled_ce,
            'mixed_neighbors_ce': mixed_ce, 'mixed_shuffled_neighbors_ce': mixed_shuffled_ce,
            'target_accuracy': (logits[target].argmax(-1) == chosen[target]).float().mean().item(),
        }
    model.train()
    return results


@torch.no_grad()
def sample(model, alpha_bar, chars, device, seed=1, batch_size=2, steps=SAMPLE_STEPS):
    """Positionwise DDIM: every character starts at T and denoises at its own rate."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)

    def noise(shape):
        return torch.randn(shape, generator=generator).to(device)

    x_t = noise((batch_size, BLOCK_SIZE, DIM))
    # Different powers create early and late denoising positions; no causal mask.
    powers = 0.35 + 4.65 * torch.rand((batch_size, BLOCK_SIZE), generator=generator).to(device)
    for i in range(steps):
        progress, next_progress = i / steps, (i + 1) / steps
        current = (DIFFUSION_STEPS * (1 - progress) ** powers).round().long()
        next_time = (DIFFUSION_STEPS * (1 - next_progress) ** powers).round().long()
        logits = model(x_t, current)
        a_t = alpha_bar[current]
        a_next = alpha_bar[next_time]
        x0_hat = posterior_x0_mean(x_t, logits.softmax(-1), a_t, model.embedding)
        eps = (x_t - a_t.sqrt()[..., None] * x0_hat) / (1 - a_t).sqrt()[..., None].clamp_min(1e-8)
        updated = a_next.sqrt()[..., None] * x0_hat + (1 - a_next).sqrt()[..., None] * eps
        x_t = torch.where((current > next_time)[..., None], updated, x_t)
    final_logits = model(x_t, torch.zeros((batch_size, BLOCK_SIZE), device=device, dtype=torch.long))
    model.train()
    return [''.join(chars[i] for i in row) for row in final_logits.argmax(-1).tolist()]


def save(path, model, optimizer, scheduler, step, best_score, chars, train_paths):
    torch.save({
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'step': step, 'best_score': best_score,
        'chars': chars, 'train_paths': train_paths, 'holdout': HOLDOUT,
        'config': {
            'dim': DIM, 'model_dim': MODEL_DIM, 'layers': NUM_LAYERS,
            'heads': NUM_HEADS, 'ff_dim': FF_DIM, 'block_size': BLOCK_SIZE,
            'batch_size': BATCH_SIZE, 'prior_weight': PRIOR_WEIGHT,
            'bandwidths': MMD_BANDWIDTHS, 'learning_rate': LEARNING_RATE,
            'objective': 'per_character_noise_token_ce_plus_analytic_mmd',
            'timestep_draw': '20% synchronous; otherwise independent uniform plus 15% t=1 and 15% t=T',
            'sampler': 'positionwise posterior-mean DDIM, eta=0, random per-position time powers',
        },
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    train_blocks, val_blocks, frequencies, chars, train_paths = data()
    model = TokenTimeGaussianDLM(len(chars)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=3e-5)
    alpha_bar = schedule(device)
    frequencies = frequencies.to(device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latest = OUTPUT_DIR / 'latest.pt'
    start, best_score = 0, math.inf
    if args.resume:
        checkpoint = torch.load(latest, map_location=device, weights_only=True)
        assert checkpoint['chars'] == chars and checkpoint['train_paths'] == train_paths
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start, best_score = checkpoint['step'], checkpoint['best_score']
    print(f'device={device} parameters={sum(p.numel() for p in model.parameters()):,} '
          f'train_blocks={len(train_blocks)} val_blocks={len(val_blocks)} resume_step={start}', flush=True)
    history = OUTPUT_DIR / 'history.jsonl'
    for step in range(start + 1, args.steps + 1):
        started = time.monotonic()
        ids = train_blocks[torch.randint(len(train_blocks), (BATCH_SIZE,))].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x0 = model.embedding.sample(ids)
        timesteps = draw_timesteps(BATCH_SIZE, device)
        x_t = corrupt(x0, timesteps, alpha_bar, torch.randn_like(x0))
        logits = model(x_t, timesteps)
        ce = F.cross_entropy(logits.flatten(0, 1), ids.flatten())
        prior = PRIOR_WEIGHT * mmd_loss(ids, model.embedding)
        loss = ce + prior
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f'Non-finite loss at step {step}')
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % 200 == 0 or step % 500 == 0 or step == args.steps:
            record = {'step': step, 'train_ce': ce.item(), 'weighted_prior': prior.item(),
                      'grad_norm': grad_norm.item(), 'lr': scheduler.get_last_lr()[0],
                      'step_seconds': time.monotonic() - started}
            if step % 500 == 0 or step == args.steps:
                record['validation'] = evaluate(model, val_blocks, frequencies, alpha_bar, device)
                score = record['validation']['500']['readable_neighbors_ce']
                if score < best_score:
                    best_score = score
                    save(OUTPUT_DIR / 'best.pt', model, optimizer, scheduler,
                         step, best_score, chars, train_paths)
                record['best_500_ce'] = best_score
                if step % 1000 == 0 or step == args.steps:
                    record['samples'] = sample(model, alpha_bar, chars, device)
                save(latest, model, optimizer, scheduler, step, best_score, chars, train_paths)
            with history.open('a') as file:
                file.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
