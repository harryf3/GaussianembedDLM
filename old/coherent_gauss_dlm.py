"""Train a Gaussian-embedding diffusion LM with a contextual token posterior.

Run `python3 coherent_gauss_dlm.py --steps 10000` from the project root.
Checkpoints and a JSONL history are kept under `checkpoints/coherent_gauss_dlm/`.
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


SEED = 7
DIM, MODEL_DIM, NUM_LAYERS, NUM_HEADS, FF_DIM = 16, 128, 6, 4, 512
BLOCK_SIZE, BATCH_SIZE = 64, 64
DIFFUSION_STEPS, BETA_START, BETA_END = 1000, 1e-4, 0.02
PRIOR_WEIGHT, PRIOR_SAMPLES = 10.0, 1024
MMD_BANDWIDTHS = tuple(math.sqrt(DIM) * x for x in (0.25, 0.5, 1, 2, 4))
LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP_NORM = 3e-4, 1e-2, 1.0
HOLDOUT = ('macbeth_TXT_FolgerShakespeare.txt', 'the-tempest_TXT_FolgerShakespeare.txt')
OUTPUT_DIR = Path('checkpoints/coherent_gauss_dlm')


def clean_text(path):
    raw = path.read_text(encoding='utf-8').replace('\r\n', '\n').replace('\r', '\n')
    return ''.join(raw.splitlines(keepends=True)[7:]).lstrip('\n')


def data():
    paths = sorted(Path('shakespeare-dataset-main/text').glob('*.txt'))
    train_paths = [p for p in paths if p.name not in HOLDOUT]
    val_paths = [p for p in paths if p.name in HOLDOUT]
    assert len(paths) == 42 and len(val_paths) == len(HOLDOUT)
    train_text = '\n\n'.join(map(clean_text, train_paths))
    val_text = '\n\n'.join(map(clean_text, val_paths))
    chars = sorted(set(train_text))
    unseen = set(val_text) - set(chars)
    if unseen:
        raise ValueError(f'Holdout has characters missing from training: {unseen}')
    stoi = {c: i for i, c in enumerate(chars)}
    train_ids = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
    val_ids = torch.tensor([stoi[c] for c in val_text], dtype=torch.long)
    train_blocks = train_ids[:len(train_ids) // BLOCK_SIZE * BLOCK_SIZE].view(-1, BLOCK_SIZE)
    val_blocks = val_ids[:len(val_ids) // BLOCK_SIZE * BLOCK_SIZE].view(-1, BLOCK_SIZE)
    probabilities = torch.bincount(train_ids, minlength=len(chars)).float() / len(train_ids)
    return train_blocks, val_blocks, probabilities, chars, [p.name for p in train_paths]


class GaussianEmbedding(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        token_var = 0.05
        raw_std = math.log(math.expm1(math.sqrt(token_var)))
        self.mu = nn.Parameter(torch.randn(vocab_size, DIM) * math.sqrt(1 - token_var))
        self.raw_std = nn.Parameter(torch.full((vocab_size, DIM), raw_std))

    @property
    def std(self):
        return F.softplus(self.raw_std)

    def sample(self, ids):
        mean = self.mu[ids]
        return mean + self.std[ids] * torch.randn_like(mean)


class ContextualGaussianDLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = GaussianEmbedding(vocab_size)
        self.input_projection = nn.Linear(DIM, MODEL_DIM)
        self.position = nn.Embedding(BLOCK_SIZE, MODEL_DIM)
        frequencies = torch.exp(-math.log(10_000) * torch.arange(MODEL_DIM // 2) / (MODEL_DIM // 2))
        self.register_buffer('time_frequencies', frequencies)
        layer = nn.TransformerEncoderLayer(MODEL_DIM, NUM_HEADS, FF_DIM, 0.1,
                                           batch_first=True, norm_first=False, activation='gelu')
        self.transformer = nn.TransformerEncoder(layer, NUM_LAYERS)
        self.final_norm = nn.LayerNorm(MODEL_DIM)
        self.token_head = nn.Linear(MODEL_DIM, vocab_size)

    def forward(self, x_t, timesteps):
        pos = torch.arange(x_t.size(1), device=x_t.device)
        angles = timesteps[:, None] * self.time_frequencies
        time_embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        hidden = self.input_projection(x_t) + self.position(pos)[None] + time_embedding[:, None]
        return self.token_head(self.final_norm(self.transformer(hidden)))


def gaussian_kernel(means, variances, other_means, other_variances):
    variance_sum = variances[:, None, :] + other_variances[None, :, :]
    squared_difference = (means[:, None, :] - other_means[None, :, :]).square()
    kernel = means.new_zeros((len(means), len(other_means)))
    for bandwidth in MMD_BANDWIDTHS:
        h2 = bandwidth ** 2
        log_kernel = -0.5 * (
            torch.log1p(variance_sum / h2) + squared_difference / (h2 + variance_sum)
        ).sum(-1)
        kernel = kernel + log_kernel.exp()
    return kernel


def mmd_loss(ids, embedding):
    values = ids.flatten()
    selected = values[torch.randperm(len(values), device=values.device)[:PRIOR_SAMPLES]]
    components, counts = selected.unique(return_counts=True)
    means = embedding.mu[components]
    variances = embedding.std[components].square()
    weights = counts.to(means.dtype) / len(selected)
    zero = means.new_zeros((1, DIM))
    one = torch.ones_like(zero)
    kxx = gaussian_kernel(means, variances, means, variances)
    kxy = gaussian_kernel(means, variances, zero, one).squeeze(-1)
    kyy = gaussian_kernel(zero, one, zero, one).squeeze()
    return weights @ kxx @ weights + kyy - 2 * (weights * kxy).sum()


def posterior_x0_mean(x_t, probabilities, alpha, embedding):
    means = embedding.mu
    variances = embedding.std.square()
    denominator = alpha * variances + (1 - alpha)
    gain = alpha.sqrt() * variances / denominator
    intercept = (1 - alpha) * means / denominator
    return probabilities @ intercept + x_t * (probabilities @ gain)


def schedule(device):
    betas = torch.linspace(BETA_START, BETA_END, DIFFUSION_STEPS, device=device)
    return torch.cat((torch.ones(1, device=device), torch.cumprod(1 - betas, 0)))


@torch.no_grad()
def evaluate(model, blocks, probabilities, alpha_bar, device):
    model.eval()
    generator = torch.Generator(device='cpu').manual_seed(1234)
    chosen = blocks[torch.randperm(len(blocks), generator=generator)[:64]].to(device)
    def noise(shape):
        return torch.randn(shape, generator=generator).to(device)
    x0 = model.embedding.mu[chosen] + model.embedding.std[chosen] * noise((*chosen.shape, DIM))
    results = {}
    for t in (100, 250, 500, 750, 1000):
        a = alpha_bar[t]
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise(x0.shape)
        steps = torch.full((len(chosen),), t, device=device, dtype=torch.long)
        logits = model(x_t, steps)
        ce = F.cross_entropy(logits.flatten(0, 1), chosen.flatten()).item()
        accuracy = (logits.argmax(-1) == chosen).float().mean().item()
        v = a * model.embedding.std.square() + 1 - a
        delta = x_t[:, :, None, :] - a.sqrt() * model.embedding.mu
        baseline = -.5 * (delta.square() / v + v.log()).sum(-1) + probabilities.log()
        baseline_ce = F.cross_entropy(baseline.flatten(0, 1), chosen.flatten()).item()
        results[str(t)] = {'ce': ce, 'accuracy': accuracy, 'independent_ce': baseline_ce}
    model.train()
    return results


@torch.no_grad()
def sample(model, alpha_bar, chars, device, eta=0.1, reverse_steps=250, seed=1, batch_size=2):
    model.eval()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    def noise(shape):
        return torch.randn(shape, generator=generator).to(device)
    x_t = noise((batch_size, BLOCK_SIZE, DIM))
    last_logits = None
    times = torch.linspace(DIFFUSION_STEPS, 0, reverse_steps + 1).long().tolist()
    for current, next_time in zip(times[:-1], times[1:]):
        t = torch.full((batch_size,), current, device=device, dtype=torch.long)
        last_logits = model(x_t, t)
        a_t, a_next = alpha_bar[current], alpha_bar[next_time]
        x0_hat = posterior_x0_mean(x_t, last_logits.softmax(-1), a_t, model.embedding)
        eps = (x_t - a_t.sqrt() * x0_hat) / (1 - a_t).sqrt().clamp_min(1e-8)
        sigma_squared = eta**2 * ((1 - a_next) / (1 - a_t)) * (1 - a_t / a_next)
        direction = (1 - a_next - sigma_squared).clamp_min(0).sqrt()
        x_t = a_next.sqrt() * x0_hat + direction * eps
        if eta > 0 and next_time > 0:
            x_t += sigma_squared.clamp_min(0).sqrt() * noise(x_t.shape)
    model.train()
    return [''.join(chars[i] for i in row) for row in last_logits.argmax(-1).tolist()]


def save(path, model, optimizer, scheduler, step, best_score, chars, train_paths):
    torch.save({
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'step': step, 'best_score': best_score,
        'chars': chars, 'train_paths': train_paths, 'holdout': HOLDOUT,
        'config': {'dim': DIM, 'model_dim': MODEL_DIM, 'layers': NUM_LAYERS,
                   'heads': NUM_HEADS, 'ff_dim': FF_DIM, 'block_size': BLOCK_SIZE,
                   'batch_size': BATCH_SIZE, 'prior_weight': PRIOR_WEIGHT,
                   'bandwidths': MMD_BANDWIDTHS, 'learning_rate': LEARNING_RATE,
                   'objective': 'direct_contextual_token_ce_plus_analytic_mmd',
                   'sampler': 'posterior-mean DDIM from contextual token probabilities'},
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    train_blocks, val_blocks, probabilities, chars, train_paths = data()
    model = ContextualGaussianDLM(len(chars)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=3e-5)
    alpha_bar = schedule(device)
    probabilities = probabilities.to(device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latest = OUTPUT_DIR / 'latest.pt'
    start, best_score = 0, float('inf')
    if args.resume:
        checkpoint = torch.load(latest, map_location=device, weights_only=True)
        assert checkpoint['chars'] == chars and checkpoint['train_paths'] == train_paths
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start, best_score = checkpoint['step'], checkpoint['best_score']
    print(f'device={device} parameters={sum(p.numel() for p in model.parameters()):,} '
          f'train_blocks={len(train_blocks)} val_blocks={len(val_blocks)} '
          f'resume_step={start}', flush=True)
    history = OUTPUT_DIR / 'history.jsonl'
    for step in range(start + 1, args.steps + 1):
        start_time = time.monotonic()
        indices = torch.randint(len(train_blocks), (BATCH_SIZE,))
        ids = train_blocks[indices].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x0 = model.embedding.sample(ids)
        t = torch.randint(1, DIFFUSION_STEPS + 1, (BATCH_SIZE,), device=device)
        a = alpha_bar[t, None, None]
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn_like(x0)
        logits = model(x_t, t)
        ce = F.cross_entropy(logits.flatten(0, 1), ids.flatten())
        prior = PRIOR_WEIGHT * mmd_loss(ids, model.embedding)
        loss = ce + prior
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f'Non-finite loss at step {step}')
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM,
                                            error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % 200 == 0 or step % 500 == 0 or step == args.steps:
            record = {'step': step, 'train_ce': ce.item(), 'weighted_prior': prior.item(),
                      'grad_norm': grad_norm.item(), 'lr': scheduler.get_last_lr()[0],
                      'step_seconds': time.monotonic() - start_time}
            if step % 500 == 0 or step == args.steps:
                record['validation'] = evaluate(model, val_blocks, probabilities, alpha_bar, device)
                score = (record['validation']['250']['ce'] + record['validation']['500']['ce']) / 2
                if score < best_score:
                    best_score = score
                    save(OUTPUT_DIR / 'best.pt', model, optimizer, scheduler,
                         step, best_score, chars, train_paths)
                record['best_mid_noise_ce'] = best_score
                if step % 1000 == 0 or step == args.steps:
                    record['samples'] = sample(model, alpha_bar, chars, device)
                save(latest, model, optimizer, scheduler, step, best_score, chars, train_paths)
            with history.open('a') as file:
                file.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
