"""Causal character-language-model control for the Gaussian diffusion experiments."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from old.coherent_gauss_dlm import HOLDOUT, clean_text


SEED = 7
BLOCK_SIZE, BATCH_SIZE = 64, 64
WIDTH, LAYERS, HEADS, FF_DIM = 128, 4, 4, 512
OUTPUT_DIR = Path('checkpoints/coherent_char_baseline')


def corpus():
    paths = sorted(Path('shakespeare-dataset-main/text').glob('*.txt'))
    train = '\n\n'.join(clean_text(p) for p in paths if p.name not in HOLDOUT)
    validation = '\n\n'.join(clean_text(p) for p in paths if p.name in HOLDOUT)
    chars = sorted(set(train))
    assert set(validation) <= set(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    train_ids = torch.tensor([stoi[c] for c in train], dtype=torch.long)
    val_ids = torch.tensor([stoi[c] for c in validation], dtype=torch.long)
    return train_ids, val_ids, chars


class CausalCharLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, WIDTH)
        self.position = nn.Embedding(BLOCK_SIZE, WIDTH)
        layer = nn.TransformerEncoderLayer(WIDTH, HEADS, FF_DIM, 0.1,
                                           batch_first=True, norm_first=False, activation='gelu')
        self.transformer = nn.TransformerEncoder(layer, LAYERS)
        self.norm = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, vocab_size)

    def forward(self, ids):
        length = ids.size(1)
        positions = torch.arange(length, device=ids.device)
        causal_mask = torch.ones((length, length), device=ids.device, dtype=torch.bool).triu(1)
        hidden = self.embedding(ids) + self.position(positions)[None]
        return self.head(self.norm(self.transformer(hidden, mask=causal_mask)))


def batches(values, count, generator, device):
    starts = torch.randint(len(values) - BLOCK_SIZE, (count,), generator=generator)
    offsets = torch.arange(BLOCK_SIZE + 1)
    return values[starts[:, None] + offsets[None]].to(device)


@torch.no_grad()
def evaluate(model, validation, device):
    model.eval()
    generator = torch.Generator(device='cpu').manual_seed(1234)
    batch = batches(validation, 128, generator, device)
    logits = model(batch[:, :-1])
    loss = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten()).item()
    model.train()
    return loss


@torch.no_grad()
def sample(model, chars, device, seed=1, prompt='\n', length=256,
           temperature=0.8, top_k=20, context_size=BLOCK_SIZE):
    model.eval()
    stoi = {c: i for i, c in enumerate(chars)}
    tokens = [stoi[c] for c in prompt]
    generator = torch.Generator(device='cpu').manual_seed(seed)
    for _ in range(length):
        context = torch.tensor(tokens[-context_size:], device=device)[None]
        logits = model(context)[0, -1] / temperature
        if top_k:
            threshold = logits.topk(top_k).values[-1]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        probability = logits.softmax(-1).cpu()
        tokens.append(torch.multinomial(probability, 1, generator=generator).item())
    model.train()
    return ''.join(chars[i] for i in tokens)


def save(path, model, optimizer, scheduler, step, best_ce, chars):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'step': step, 'best_ce': best_ce,
                'chars': chars, 'holdout': HOLDOUT,
                'config': {'block_size': BLOCK_SIZE, 'batch_size': BATCH_SIZE,
                           'width': WIDTH, 'layers': LAYERS, 'heads': HEADS,
                           'ff_dim': FF_DIM, 'objective': 'causal_next_character_ce'}}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    train, validation, chars = corpus()
    model = CausalCharLM(len(chars)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=3e-5)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start, best_ce = 0, math.inf
    if args.resume:
        checkpoint = torch.load(OUTPUT_DIR / 'latest.pt', map_location=device, weights_only=True)
        assert checkpoint['chars'] == chars
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start, best_ce = checkpoint['step'], checkpoint['best_ce']
    print(f'device={device} parameters={sum(p.numel() for p in model.parameters()):,} '
          f'train_characters={len(train)} holdout_characters={len(validation)} '
          f'resume_step={start}', flush=True)
    generator = torch.Generator(device='cpu').manual_seed(SEED + start)
    history = OUTPUT_DIR / 'history.jsonl'
    for step in range(start + 1, args.steps + 1):
        began = time.monotonic()
        batch = batches(train, BATCH_SIZE, generator, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten())
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f'Non-finite CE at step {step}')
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % 200 == 0 or step % 500 == 0 or step == args.steps:
            row = {'step': step, 'train_ce': loss.item(), 'grad_norm': grad_norm.item(),
                   'lr': scheduler.get_last_lr()[0], 'step_seconds': time.monotonic() - began}
            if step % 500 == 0 or step == args.steps:
                row['validation_ce'] = evaluate(model, validation, device)
                if row['validation_ce'] < best_ce:
                    best_ce = row['validation_ce']
                    save(OUTPUT_DIR / 'best.pt', model, optimizer, scheduler, step, best_ce, chars)
                row['best_validation_ce'] = best_ce
                if step % 1000 == 0 or step == args.steps:
                    row['sample'] = sample(model, chars, device)
                save(OUTPUT_DIR / 'latest.pt', model, optimizer, scheduler, step, best_ce, chars)
            with history.open('a') as file:
                file.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
