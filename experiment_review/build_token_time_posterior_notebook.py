"""Build the standalone per-character-time posterior notebook from local sources."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'old/gauss_embed_dlm_mmd_scaled.ipynb'
DESTINATION = ROOT / 'old/gauss_embed_dlm_mmd_token_time_posterior.ipynb'


def code(source):
    return nbformat.v4.new_code_cell(source.strip() + '\n')


def main():
    template = nbformat.read(SOURCE, as_version=4)
    model_source = template.cells[2].source
    model_source = model_source.replace(
        '        pos = torch.arange(x_t.size(1), device=x_t.device)',
        "        if timesteps.shape != x_t.shape[:2]:\n"
        "            raise ValueError('One timestep is required for each character')\n"
        '        pos = torch.arange(x_t.size(1), device=x_t.device)',
    ).replace(
        'angles = timesteps[:, None] * self.time_frequencies',
        'angles = timesteps[..., None] * self.time_frequencies',
    ).replace(
        'time_embedding.unsqueeze(1)', 'time_embedding',
    )
    prior_source = template.cells[3].source
    prior_source = prior_source[
        prior_source.index('def gaussian_kernel('):prior_source.index('def loss_terms(')
    ]

    cells = []
    cells.append(code('''
import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# Posterior-vector architecture + analytic MMD + one Gaussian-noise level per character.
SEED = 7
DATA_DIR = Path('shakespeare-dataset-main/text')
HOLDOUT = ('macbeth_TXT_FolgerShakespeare.txt', 'the-tempest_TXT_FolgerShakespeare.txt')
BLOCK_SIZE, BATCH_SIZE, NUM_WORKERS = 64, 64, 0
DIM, MODEL_DIM, NUM_LAYERS, NUM_HEADS, FF_DIM, DROPOUT = 16, 128, 6, 4, 512, 0.1
DIFFUSION_STEPS, BETA_START, BETA_END = 1000, 1e-4, 0.02
LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP_NORM = 3e-4, 1e-2, 1.0
TRAIN_STEPS, LOG_EVERY, EVAL_EVERY, SAMPLE_EVERY = 3000, 200, 500, 1000
CE_WEIGHT, PRIOR_WEIGHT, PRIOR_SAMPLES = 1.0, 10.0, 1024
MMD_BANDWIDTHS = tuple(math.sqrt(DIM) * x for x in (0.25, 0.5, 1, 2, 4))
SAMPLE_BATCH_SIZE, SAMPLE_REVERSE_STEPS = 2, 100
DDIM_ETA, SAMPLE_SEED = 0.0, 1
CHECKPOINT_PATH = Path('checkpoints/gauss_embed_dlm_mmd_token_time_posterior.pt')

device = torch.device('mps' if torch.backends.mps.is_available()
                      else 'cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
random.seed(SEED)
print(f'Using {device}')
'''))
    cells.append(code('''
def clean_folger_text(raw):
    lines = raw.replace('\\r\\n', '\\n').replace('\\r', '\\n').splitlines(keepends=True)
    return ''.join(lines[7:]).lstrip('\\n')

paths = sorted(DATA_DIR.glob('*.txt'))
train_paths = [path for path in paths if path.name not in HOLDOUT]
holdout_paths = [path for path in paths if path.name in HOLDOUT]
if len(paths) != 42 or len(holdout_paths) != len(HOLDOUT):
    raise ValueError(f'Expected 42 works including both held-out works; found {len(paths)}')
train_text = '\\n\\n'.join(clean_folger_text(p.read_text(encoding='utf-8')) for p in train_paths)
holdout_text = '\\n\\n'.join(clean_folger_text(p.read_text(encoding='utf-8')) for p in holdout_paths)
chars = sorted(set(train_text))
unseen = set(holdout_text) - set(chars)
if unseen:
    raise ValueError(f'Held-out characters missing from training vocabulary: {unseen}')
stoi = {char: index for index, char in enumerate(chars)}
itos = dict(enumerate(chars))
tokens = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
holdout_tokens = torch.tensor([stoi[c] for c in holdout_text], dtype=torch.long)
token_probabilities = (torch.bincount(tokens, minlength=len(chars)) / len(tokens)).to(device)

class CharacterBlocks(Dataset):
    def __init__(self, values): self.values = values
    def __len__(self): return len(self.values) // BLOCK_SIZE
    def __getitem__(self, index):
        start = index * BLOCK_SIZE
        return self.values[start:start + BLOCK_SIZE]

train_dataset = CharacterBlocks(tokens)
holdout_dataset = CharacterBlocks(holdout_tokens)
loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                    drop_last=True, num_workers=NUM_WORKERS)
eval_generator = torch.Generator().manual_seed(1234)
eval_indices = torch.randperm(len(holdout_dataset), generator=eval_generator)[:64].tolist()
eval_blocks = torch.stack([holdout_dataset[i] for i in eval_indices])
print(f'Train={len(train_paths)} works, {len(tokens):,} characters | '
      f'held out={len(holdout_paths)} works, {len(holdout_tokens):,} characters | '
      f'vocab={len(chars)} | batch={tuple(next(iter(loader)).shape)}')
'''))
    cells.append(code(model_source))
    cells.append(code('''
betas = torch.linspace(BETA_START, BETA_END, DIFFUSION_STEPS, device=device)
alpha_bar = torch.cat((torch.ones(1, device=device), torch.cumprod(1 - betas, 0)))

def draw_timesteps(batch_size):
    # 20% synchronized examples; otherwise positions receive independent noise.
    t = torch.randint(1, DIFFUSION_STEPS + 1, (batch_size, BLOCK_SIZE), device=device)
    choices = torch.rand((batch_size, BLOCK_SIZE), device=device)
    t = torch.where(choices < 0.15, 1, t)
    t = torch.where(choices > 0.85, DIFFUSION_STEPS, t)
    synchronized = torch.rand((batch_size,), device=device) < 0.2
    global_t = torch.randint(1, DIFFUSION_STEPS + 1, (batch_size, 1), device=device)
    return torch.where(synchronized[:, None], global_t, t)

def q_sample(x0, timesteps, noise=None):
    alpha = alpha_bar[timesteps][..., None]
    if noise is None: noise = torch.randn_like(x0)
    return alpha.sqrt() * x0 + (1 - alpha).sqrt() * noise

''' + prior_source + '''
def loss_terms(ids):
    x0 = model.embedding.sample(ids)
    timesteps = draw_timesteps(ids.size(0))
    predicted_x0 = model(q_sample(x0, timesteps), timesteps)
    token_ce = F.nll_loss(model.rounding_log_probs(predicted_x0).flatten(0, 1),
                          ids.flatten())
    weighted_prior = PRIOR_WEIGHT * mmd_loss(ids)
    return CE_WEIGHT * token_ce + weighted_prior, token_ce, weighted_prior

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10_000, eta_min=3e-5)
completed_steps = 0

OBJECTIVE = 'per_character_noise_vector_ce_plus_analytic_mmd_prior'
CONFIG_NAMES = (
    'SEED', 'DATA_DIR', 'HOLDOUT', 'BLOCK_SIZE', 'BATCH_SIZE', 'NUM_WORKERS',
    'DIM', 'MODEL_DIM', 'NUM_LAYERS', 'NUM_HEADS', 'FF_DIM', 'DROPOUT',
    'DIFFUSION_STEPS', 'BETA_START', 'BETA_END', 'LEARNING_RATE',
    'WEIGHT_DECAY', 'GRAD_CLIP_NORM', 'TRAIN_STEPS', 'LOG_EVERY', 'EVAL_EVERY',
    'SAMPLE_EVERY', 'CE_WEIGHT', 'PRIOR_WEIGHT', 'PRIOR_SAMPLES',
    'MMD_BANDWIDTHS', 'SAMPLE_BATCH_SIZE', 'SAMPLE_REVERSE_STEPS',
    'DDIM_ETA', 'SAMPLE_SEED',
)
run_config = {name.lower(): str(globals()[name]) if isinstance(globals()[name], Path)
              else globals()[name] for name in CONFIG_NAMES}
run_config.update({
    'objective': OBJECTIVE,
    'source_notebook': 'old/gauss_embed_dlm_wass_ce_posterior.ipynb',
    'scaled_prior_source': 'old/gauss_embed_dlm_mmd_scaled.ipynb',
    'per_character_time_source': 'coherent_gauss_dlm_token_time.py',
    'experiment_notebook': 'old/gauss_embed_dlm_mmd_token_time_posterior.ipynb',
    'initialization': 'fresh; not compatible with the separate-token-head checkpoint',
    'architecture': 'bidirectional vector x0 head, Gaussian log-density token decoder',
    'timestep_sampling': '20% one uniform timestep per block; otherwise independent per position with 15% t=1 and 15% t=1000',
    'ce_timestep_weighting': 'uniform over all characters',
    'prior_estimator': 'analytic biased Gaussian-mixture MMD squared; diagonal included',
    'prior_component_selection': 'up to 1024 minibatch positions without replacement',
    'prior_component_weighting': 'selected-position frequencies, repeated IDs coalesced',
    'prior_reference': 'analytic standard normal N(0,I); no reference samples',
    'sampler': 'positionwise posterior-mean DDIM with random time powers 0.35..5.0',
    'final_decoding': 'Gaussian log_probs argmax on final continuous state',
    'data_split': '40 training works; Macbeth and The Tempest held out',
    'train_files': [str(p) for p in train_paths],
    'holdout_files': [str(p) for p in holdout_paths],
    'data_cleaning': 'normalize newlines, remove first 7 lines per work, join with two newlines',
    'parameters': sum(p.numel() for p in model.parameters()),
    'device': str(device), 'torch_version': str(torch.__version__),
})
'''))
    cells.append(code('''
def step(batch):
    global completed_steps
    ids = batch.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    values = loss_terms(ids)
    if not torch.isfinite(values[0]).item():
        raise FloatingPointError('Non-finite loss; optimizer was not updated')
    values[0].backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM,
                                         error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    completed_steps += 1
    return *values, grad_norm

def save_checkpoint(path=None):
    path = CHECKPOINT_PATH if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    run_config.update(ddim_eta=sampling_eta(None),
                      sample_reverse_steps=SAMPLE_REVERSE_STEPS, sample_seed=SAMPLE_SEED)
    torch.save({
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'chars': chars,
        'train_files': [str(p) for p in train_paths],
        'holdout_files': [str(p) for p in holdout_paths],
        'config': run_config, 'objective': OBJECTIVE,
        'completed_steps': completed_steps,
    }, path)

def load_checkpoint(path=None):
    global completed_steps
    path = CHECKPOINT_PATH if path is None else Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint['chars'] != chars or checkpoint['train_files'] != [str(p) for p in train_paths]:
        raise ValueError('Checkpoint vocabulary or training files do not match')
    if checkpoint['objective'] != OBJECTIVE:
        raise ValueError('Checkpoint objective does not match')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    completed_steps = checkpoint['completed_steps']
    print(f'Resumed {path} at step {completed_steps}')

def train(steps=TRAIN_STEPS):
    iterator = iter(loader)
    running = torch.zeros(4)
    window_steps = 0
    for _ in range(steps):
        try: batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        values = step(batch)
        running += torch.tensor([v.item() for v in values])
        window_steps += 1
        if completed_steps % LOG_EVERY == 0 or completed_steps % EVAL_EVERY == 0 or _ == steps - 1:
            mean = running / window_steps
            geometry = embedding_diagnostics()
            print(f'step {completed_steps} | total={mean[0]:.4f} | token_ce={mean[1]:.4f} | '
                  f'weighted_prior={mean[2]:.4f} | grad_norm={mean[3]:.4f} | '
                  f'lr={scheduler.get_last_lr()[0]:.6f} | '
                  + ' | '.join(f'{k}={v:.4f}' for k, v in geometry.items()))
            running.zero_()
            window_steps = 0
        if completed_steps % EVAL_EVERY == 0 or _ == steps - 1:
            metrics = validation_diagnostics()
            row = metrics[500]
            print(f'Held-out t=500 | clean_neighbors_ce={row["readable_ce"]:.4f} | '
                  f'shuffled={row["readable_shuffled_ce"]:.4f} | '
                  f'mixed_neighbors_ce={row["mixed_ce"]:.4f} | '
                  f'all_t500_ce={row["uniform_ce"]:.4f} | independent={row["independent_ce"]:.4f}')
            save_checkpoint()
        if completed_steps % SAMPLE_EVERY == 0:
            print(f'Samples after {completed_steps} optimizer steps:')
            blocks, norms = sample(return_norms=True)
            for row in norms:
                print(f't_mean={row["t_mean"]:.1f} -> {row["next_t_mean"]:.1f} | '
                      f'raw_norm²/d={row["raw_norm"]:.4f} | '
                      f'used_x0_norm²/d={row["used_x0_norm"]:.4f} | '
                      f'state_norm²/d={row["state_norm"]:.4f}')
            for row in blocks.cpu().tolist():
                print(repr(''.join(itos[i] for i in row)))
'''))
    cells.append(code('''
@torch.no_grad()
def posterior_x0_mean(x_t, probabilities, alpha):
    # Exact Gaussian conditional mean for each predicted character component.
    a = alpha[..., None, None]
    means = model.embedding.mu[None, None]
    variances = model.embedding.std.square()[None, None]
    denominator = a * variances + (1 - a)
    gain = a.sqrt() * variances / denominator
    intercept = (1 - a) * means / denominator
    return (probabilities[..., None] *
            (intercept + gain * x_t[:, :, None])).sum(2)

def sampling_eta(eta):
    eta = float(DDIM_ETA if eta is None else eta)
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError('eta must be finite and in [0, 1]')
    return eta

def sampling_noise(shape, reference, generator):
    return torch.randn(shape, generator=generator, device='cpu',
                       dtype=reference.dtype).to(reference.device)

def position_times(number, reverse_steps, powers):
    progress = number / reverse_steps
    return (DIFFUSION_STEPS * (1 - progress) ** powers).round().long()

@torch.no_grad()
def reverse_step(x_t, current, next_time, *, eta, generator):
    raw_prediction = model(x_t, current)
    probabilities = model.rounding_log_probs(raw_prediction).exp()
    a_t, a_next = alpha_bar[current], alpha_bar[next_time]
    x0_hat = posterior_x0_mean(x_t, probabilities, a_t)
    variance_t = (1 - a_t).clamp_min(1e-8)
    eps_hat = (x_t - a_t.sqrt()[..., None] * x0_hat) / variance_t.sqrt()[..., None]
    sigma_squared = eta**2 * ((1 - a_next) / variance_t) * (1 - a_t / a_next)
    direction = (1 - a_next - sigma_squared).clamp_min(0).sqrt()
    next_state = a_next.sqrt()[..., None] * x0_hat + direction[..., None] * eps_hat
    active = current > next_time
    if eta > 0:
        sigma = sigma_squared.clamp_min(0).sqrt()
        next_state = next_state + (sigma * (next_time > 0))[..., None] * sampling_noise(
            x_t.shape, x_t, generator)
    return torch.where(active[..., None], next_state, x_t), raw_prediction, x0_hat

@torch.no_grad()
def sample(batch_size=SAMPLE_BATCH_SIZE, reverse_steps=SAMPLE_REVERSE_STEPS,
           return_norms=False, seed=SAMPLE_SEED, eta=None):
    eta = sampling_eta(eta)
    if not 1 <= reverse_steps <= DIFFUSION_STEPS or batch_size < 1:
        raise ValueError('reverse_steps must be in 1..DIFFUSION_STEPS and batch_size positive')
    was_training = model.training
    model.eval()
    try:
        generator = torch.Generator(device='cpu').manual_seed(seed)
        means = model.embedding.mu
        x_t = sampling_noise((batch_size, BLOCK_SIZE, DIM), means, generator)
        powers = 0.35 + 4.65 * torch.rand((batch_size, BLOCK_SIZE), generator=generator).to(device)
        trace_steps = {1, max(1, reverse_steps // 2), max(1, reverse_steps * 9 // 10), reverse_steps}
        norms = []
        for number in range(reverse_steps):
            current = position_times(number, reverse_steps, powers)
            next_time = position_times(number + 1, reverse_steps, powers)
            x_t, raw, x0_hat = reverse_step(x_t, current, next_time,
                                             eta=eta, generator=generator)
            if return_norms and number + 1 in trace_steps:
                norms.append({'t_mean': current.float().mean().item(),
                              'next_t_mean': next_time.float().mean().item(),
                              'raw_norm': raw.square().mean().item(),
                              'used_x0_norm': x0_hat.square().mean().item(),
                              'state_norm': x_t.square().mean().item()})
        if not torch.isfinite(x_t).all().item():
            raise FloatingPointError('Non-finite final reverse state')
        result = model.round_to_tokens(x_t)
        return (result, norms) if return_norms else result
    finally:
        model.train(was_training)
'''))
    cells.append(code('''
@torch.no_grad()
def validation_diagnostics(blocks=eval_blocks, seed=1234,
                           timesteps=(250, 500, 750)):
    # Held-out targets at positions 3, 11, ...; preserve each target across
    # readable, mixed, and uniform-neighbor comparisons.
    was_training = model.training
    model.eval()
    try:
        ids = blocks.to(device)
        generator = torch.Generator().manual_seed(seed)
        def draw_noise(shape):
            return torch.randn(shape, generator=generator).to(device)
        x0 = model.embedding.mu[ids] + model.embedding.std[ids] * draw_noise((*ids.shape, DIM))
        target = torch.zeros(ids.shape, dtype=torch.bool, device=device)
        target[:, 3::8] = True
        results = {}
        for timestep in timesteps:
            generator = torch.Generator().manual_seed(seed * 10 + timestep)
            t = torch.where(target, timestep, 1)
            x_t = q_sample(x0, t, draw_noise(x0.shape))
            readable = model.rounding_log_probs(model(x_t, t))
            readable_ce = F.nll_loss(readable[target], ids[target]).item()
            perm = torch.randperm(len(ids), generator=generator).to(device)
            shuffled = x_t.clone()
            shuffled[:, ~target[0]] = x_t[perm][:, ~target[0]]
            readable_shuffled = model.rounding_log_probs(model(shuffled, t))

            alpha = alpha_bar[timestep]
            variance = alpha * model.embedding.std.square() + 1 - alpha
            delta = x_t[target][:, None] - alpha.sqrt() * model.embedding.mu
            independent = -.5 * (delta.square() / variance + variance.log()).sum(-1)
            independent = independent + token_probabilities.log()

            uniform_t = torch.full_like(t, timestep)
            uniform_x = q_sample(x0, uniform_t, draw_noise(x0.shape))
            uniform_ce = F.nll_loss(model.rounding_log_probs(model(uniform_x, uniform_t))[target],
                                    ids[target]).item()
            perm = torch.randperm(len(ids), generator=generator).to(device)
            uniform_shuffled = uniform_x.clone()
            uniform_shuffled[:, ~target[0]] = uniform_x[perm][:, ~target[0]]

            mixed_t = torch.randint(1, DIFFUSION_STEPS + 1, ids.shape,
                                    generator=generator).to(device)
            mixed_t[target] = timestep
            mixed_x = q_sample(x0, mixed_t, draw_noise(x0.shape))
            mixed_x[target] = x_t[target]
            mixed_ce = F.nll_loss(model.rounding_log_probs(model(mixed_x, mixed_t))[target],
                                  ids[target]).item()
            perm = torch.randperm(len(ids), generator=generator).to(device)
            mixed_shuffled_x = mixed_x.clone()
            mixed_shuffled_t = mixed_t.clone()
            mixed_shuffled_x[:, ~target[0]] = mixed_x[perm][:, ~target[0]]
            mixed_shuffled_t[:, ~target[0]] = mixed_t[perm][:, ~target[0]]

            results[timestep] = {
                'readable_ce': readable_ce,
                'readable_shuffled_ce': F.nll_loss(readable_shuffled[target], ids[target]).item(),
                'mixed_ce': mixed_ce,
                'mixed_shuffled_ce': F.nll_loss(
                    model.rounding_log_probs(model(mixed_shuffled_x, mixed_shuffled_t))[target],
                    ids[target]).item(),
                'uniform_ce': uniform_ce,
                'uniform_shuffled_ce': F.nll_loss(
                    model.rounding_log_probs(model(uniform_shuffled, uniform_t))[target],
                    ids[target]).item(),
                'independent_ce': F.cross_entropy(independent, ids[target]).item(),
                'readable_accuracy': (readable[target].argmax(-1) == ids[target]).float().mean().item(),
            }
        return results
    finally:
        model.train(was_training)
'''))
    cells.append(code('''
# Run this cell to train from scratch. To resume instead, load_checkpoint()
# before this cell, then call train(additional_steps) explicitly.
train()
'''))
    cells.append(code('''
print(f'Positionwise posterior-mean DDIM | eta={sampling_eta(None)} | seed={SAMPLE_SEED}')
blocks, norms = sample(return_norms=True)
for row in norms:
    print(f't_mean={row["t_mean"]:.1f} -> {row["next_t_mean"]:.1f} | '
          f'raw_norm²/d={row["raw_norm"]:.4f} | '
          f'used_x0_norm²/d={row["used_x0_norm"]:.4f} | '
          f'state_norm²/d={row["state_norm"]:.4f}')
for row in blocks.cpu().tolist():
    print(repr(''.join(itos[i] for i in row)))
'''))
    cells.append(code(template.cells[8].source))
    cells.append(code('''
@torch.no_grad()
def reconstruction_diagnostics(blocks=eval_blocks,
                               timesteps=(1, 10, 100, 250, 500, 750, 1000), seed=1234):
    was_training = model.training
    model.eval()
    try:
        ids = blocks.to(device)
        generator = torch.Generator().manual_seed(seed)
        def draw_noise(shape):
            return torch.randn(shape, generator=generator).to(device)
        x0 = model.embedding.mu[ids] + model.embedding.std[ids] * draw_noise((*ids.shape, DIM))
        clean_accuracy = (model.round_to_tokens(x0) == ids).float().mean().item()
        results = []
        for timestep in timesteps:
            t = torch.full(ids.shape, timestep, device=device, dtype=torch.long)
            x_t = q_sample(x0, t, draw_noise(x0.shape))
            prediction = model(x_t, t)
            log_probs = model.rounding_log_probs(prediction)
            x0_hat = posterior_x0_mean(x_t, log_probs.exp(), alpha_bar[t])
            results.append({
                't': timestep,
                'token_ce': F.nll_loss(log_probs.flatten(0, 1), ids.flatten()).item(),
                'token_accuracy': (log_probs.argmax(-1) == ids).float().mean().item(),
                'raw_norm': prediction.square().mean().item(),
                'used_x0_norm': x0_hat.square().mean().item(),
            })
        return clean_accuracy, results
    finally:
        model.train(was_training)

@torch.no_grad()
def trace_reverse(reverse_steps=50, eta=None, seed=SAMPLE_SEED):
    eta = sampling_eta(eta)
    if not 1 <= reverse_steps <= DIFFUSION_STEPS:
        raise ValueError('reverse_steps must be in 1..DIFFUSION_STEPS')
    was_training = model.training
    model.eval()
    try:
        generator = torch.Generator(device='cpu').manual_seed(seed)
        x_t = sampling_noise((SAMPLE_BATCH_SIZE, BLOCK_SIZE, DIM), model.embedding.mu, generator)
        powers = 0.35 + 4.65 * torch.rand((SAMPLE_BATCH_SIZE, BLOCK_SIZE),
                                         generator=generator).to(device)
        inspect_steps = {1, max(1, reverse_steps // 2),
                         max(1, reverse_steps * 9 // 10), reverse_steps}
        print(f'Positionwise posterior-mean DDIM trace | eta={eta} | seed={seed}')
        for number in range(reverse_steps):
            current = position_times(number, reverse_steps, powers)
            next_time = position_times(number + 1, reverse_steps, powers)
            x_t, prediction, x0_hat = reverse_step(x_t, current, next_time,
                                                    eta=eta, generator=generator)
            if not torch.isfinite(x_t).all().item():
                raise FloatingPointError(f'Non-finite reverse state at step {number + 1}')
            if number + 1 in inspect_steps:
                raw_text = [''.join(itos[i] for i in row) for row in
                            model.round_to_tokens(prediction).cpu().tolist()]
                used_text = [''.join(itos[i] for i in row) for row in
                             model.round_to_tokens(x0_hat).cpu().tolist()]
                print(f'step {number + 1} | t_mean={current.float().mean():.1f} '
                      f'-> {next_time.float().mean():.1f} | '
                      f'raw_norm²/d={prediction.square().mean():.4f} | '
                      f'used_x0_norm²/d={x0_hat.square().mean():.4f} | '
                      f'state_norm²/d={x_t.square().mean():.4f}')
                print(f'  raw decode: {raw_text[0]!r}')
                print(f'  used x0:    {used_text[0]!r}')
        return model.round_to_tokens(x_t)
    finally:
        model.train(was_training)

clean_accuracy, reconstruction = reconstruction_diagnostics()
print(f'Clean embedding token accuracy (diagnostic only): {clean_accuracy:.1%}')
for row in reconstruction:
    print(f't={row["t"]:4d} | token_ce={row["token_ce"]:.4f} | '
          f'token_accuracy={row["token_accuracy"]:.1%} | '
          f'raw_norm²/d={row["raw_norm"]:.4f} | '
          f'used_x0_norm²/d={row["used_x0_norm"]:.4f}')
print('Embedding geometry | ' + ' | '.join(
    f'{name}={value:.4f}' for name, value in embedding_diagnostics().items()))
print('Held-out context checks:', validation_diagnostics())
trace_tokens = trace_reverse()
'''))
    cells.append(code('''
save_checkpoint()
print(f'Saved {CHECKPOINT_PATH.resolve()} after {completed_steps} completed optimizer steps')
'''))

    notebook = nbformat.v4.new_notebook(
        cells=cells, metadata=template.metadata.copy())
    nbformat.validate(notebook)
    nbformat.write(notebook, DESTINATION)
    print(f'Wrote {DESTINATION}')


if __name__ == '__main__':
    main()
