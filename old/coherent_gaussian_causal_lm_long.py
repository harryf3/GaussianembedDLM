"""Fine-tune the best Gaussian causal LM from 64 to 128 characters of context."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from old.coherent_char_baseline import WIDTH, corpus
from old.coherent_gauss_dlm import DIM, PRIOR_WEIGHT, mmd_loss
from old.coherent_gaussian_causal_lm import GaussianCausalCharLM


CONTEXT, BATCH_SIZE = 128, 32
OUTPUT_DIR = Path('checkpoints/coherent_gaussian_causal_lm_long')
SOURCE = Path('checkpoints/coherent_gaussian_causal_lm/best.pt')


class LongGaussianCausalLM(GaussianCausalCharLM):
    def __init__(self, vocab_size):
        super().__init__(vocab_size)
        self.position = nn.Embedding(CONTEXT, WIDTH)


def batches(values, count, generator):
    starts = torch.randint(len(values) - CONTEXT, (count,), generator=generator)
    offsets = torch.arange(CONTEXT + 1)
    return values[starts[:, None] + offsets[None]]


@torch.no_grad()
def evaluate(model, source_model, validation):
    model.eval()
    source_model.eval()
    generator = torch.Generator().manual_seed(1234)
    total_new, total_old, count = 0.0, 0.0, 0
    for _ in range(2):
        ids = batches(validation, 32, generator)
        targets = ids[:, 65:]
        new_logits = model(ids[:, :-1])[:, 64:]
        old_logits = source_model(ids[:, 64:128])
        total_new += F.cross_entropy(new_logits.flatten(0, 1), targets.flatten(),
                                     reduction='sum').item()
        total_old += F.cross_entropy(old_logits.flatten(0, 1), targets.flatten(),
                                     reduction='sum').item()
        count += targets.numel()
    model.train()
    return {'long_context_ce': total_new / count,
            'original_context_ce': total_old / count}


@torch.no_grad()
def sample(model, chars, seed=1, length=256, temperature=0.65, top_k=12):
    model.eval()
    tokens = [chars.index('\n')]
    generator = torch.Generator().manual_seed(seed)
    for _ in range(length):
        context = torch.tensor(tokens[-CONTEXT:])[None]
        logits = model(context)[0, -1] / temperature
        threshold = logits.topk(top_k).values[-1]
        probabilities = logits.masked_fill(logits < threshold, -torch.inf).softmax(-1)
        tokens.append(torch.multinomial(probabilities, 1, generator=generator).item())
    model.train()
    return ''.join(chars[i] for i in tokens)


def save(path, model, optimizer, scheduler, step, best_ce, chars):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'step': step,
                'source_step': 10000, 'best_late_ce': best_ce, 'chars': chars,
                'config': {'context': CONTEXT, 'batch_size': BATCH_SIZE,
                           'dim': DIM, 'model_dim': WIDTH, 'prior_weight': PRIOR_WEIGHT,
                           'objective': 'causal_next_character_ce_plus_analytic_mmd',
                           'source_checkpoint': str(SOURCE)}}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(17)
    random.seed(17)
    train, validation, chars = corpus()
    source = torch.load(SOURCE, map_location='cpu', weights_only=True)
    assert source['chars'] == chars and source['step'] == 10000
    source_model = GaussianCausalCharLM(len(chars))
    source_model.load_state_dict(source['model'])
    model = LongGaussianCausalLM(len(chars))
    state = source['model'].copy()
    old_position = state.pop('position.weight')
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert missing == ['position.weight'] and unexpected == []
    with torch.no_grad():
        model.position.weight[:64].copy_(old_position)
        model.position.weight[64:].copy_(old_position)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.steps, eta_min=2e-5)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(17)
    best_ce = float('inf')
    print(f'parameters={sum(p.numel() for p in model.parameters()):,} '
          f'train_characters={len(train)} holdout_characters={len(validation)}', flush=True)
    for step in range(1, args.steps + 1):
        began = time.monotonic()
        ids = batches(train, BATCH_SIZE, generator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids[:, :-1])
        ce = F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten())
        prior = PRIOR_WEIGHT * mmd_loss(ids[:, :-1], model.embedding)
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
                row['validation'] = evaluate(model, source_model, validation)
                if row['validation']['long_context_ce'] < best_ce:
                    best_ce = row['validation']['long_context_ce']
                    save(OUTPUT_DIR / 'best.pt', model, optimizer, scheduler,
                         step, best_ce, chars)
                row['best_late_ce'] = best_ce
                if step % 1000 == 0 or step == args.steps:
                    row['sample'] = sample(model, chars)
                save(OUTPUT_DIR / 'latest.pt', model, optimizer, scheduler,
                     step, best_ce, chars)
            with (OUTPUT_DIR / 'history.jsonl').open('a') as file:
                file.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
