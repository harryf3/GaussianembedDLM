"""Causal character model with the Gaussian codebook and analytic MMD prior."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from old.coherent_char_baseline import BLOCK_SIZE, BATCH_SIZE, HEADS, LAYERS, WIDTH, FF_DIM
from old.coherent_char_baseline import batches, corpus, evaluate, sample
from old.coherent_gauss_dlm import DIM, PRIOR_WEIGHT, MMD_BANDWIDTHS, GaussianEmbedding, mmd_loss


SEED = 7
OUTPUT_DIR = Path('checkpoints/coherent_gaussian_causal_lm')


class GaussianCausalCharLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = GaussianEmbedding(vocab_size)
        self.input_projection = nn.Linear(DIM, WIDTH)
        self.position = nn.Embedding(BLOCK_SIZE, WIDTH)
        layer = nn.TransformerEncoderLayer(WIDTH, HEADS, FF_DIM, 0.1,
                                           batch_first=True, norm_first=False, activation='gelu')
        self.transformer = nn.TransformerEncoder(layer, LAYERS)
        self.norm = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, vocab_size)

    def forward(self, ids):
        length = ids.size(1)
        pos = torch.arange(length, device=ids.device)
        mask = torch.ones((length, length), device=ids.device, dtype=torch.bool).triu(1)
        embeddings = self.embedding.sample(ids) if self.training else self.embedding.mu[ids]
        hidden = self.input_projection(embeddings) + self.position(pos)[None]
        return self.head(self.norm(self.transformer(hidden, mask=mask)))


def save(path, model, optimizer, scheduler, step, best_ce, chars):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'step': step, 'best_ce': best_ce,
                'chars': chars,
                'config': {'block_size': BLOCK_SIZE, 'batch_size': BATCH_SIZE,
                           'dim': DIM, 'model_dim': WIDTH, 'layers': LAYERS,
                           'heads': HEADS, 'ff_dim': FF_DIM, 'prior_weight': PRIOR_WEIGHT,
                           'bandwidths': MMD_BANDWIDTHS,
                           'objective': 'causal_next_character_ce_plus_analytic_mmd'}}, path)


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
    model = GaussianCausalCharLM(len(chars)).to(device)
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
        ce = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten())
        prior = PRIOR_WEIGHT * mmd_loss(batch[:, :-1], model.embedding)
        loss = ce + prior
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f'Non-finite loss at step {step}')
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % 200 == 0 or step % 500 == 0 or step == args.steps:
            row = {'step': step, 'train_ce': ce.item(), 'weighted_prior': prior.item(),
                   'grad_norm': grad_norm.item(), 'lr': scheduler.get_last_lr()[0],
                   'step_seconds': time.monotonic() - began}
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
