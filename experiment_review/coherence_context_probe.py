"""Measure whether the contextual Gaussian DLM uses neighboring characters.

The focal noisy vector and target stay fixed; other positions receive vectors
from different validation blocks. This evaluates four positions per block.
"""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from old.coherent_gauss_dlm import ContextualGaussianDLM, DIM, data, schedule


def main():
    torch.set_num_threads(2)
    _, blocks, probabilities, chars, _ = data()
    checkpoint = torch.load('checkpoints/coherent_gauss_dlm/latest.pt',
                            map_location='cpu', weights_only=True)
    assert checkpoint['chars'] == chars
    model = ContextualGaussianDLM(len(chars))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    generator = torch.Generator().manual_seed(1234)
    ids = blocks[torch.randperm(len(blocks), generator=generator)[:64]]
    positions = (8, 24, 40, 56)
    with torch.no_grad():
        x0 = model.embedding.mu[ids] + model.embedding.std[ids] * torch.randn(
            (*ids.shape, DIM), generator=generator)
        result = {'checkpoint_step': checkpoint['step'], 'positions': positions,
                  'heldout_works': checkpoint['holdout'], 'timesteps': {}}
        for t in (250, 500):
            a = schedule('cpu')[t]
            x_t = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn(
                x0.shape, generator=generator)
            steps = torch.full((len(ids),), t, dtype=torch.long)
            original = model(x_t, steps)
            shuffled = torch.empty_like(x_t)
            for column in range(ids.size(1)):
                shuffled[:, column] = x_t[torch.randperm(len(ids), generator=generator), column]
            intact_logits, changed_logits, labels = [], [], []
            for column in positions:
                intervention = shuffled.clone()
                intervention[:, column] = x_t[:, column]
                scrambled = model(intervention, steps)
                intact_logits.append(original[:, column])
                changed_logits.append(scrambled[:, column])
                labels.append(ids[:, column])
            labels = torch.cat(labels)
            intact = torch.cat(intact_logits)
            changed = torch.cat(changed_logits)
            result['timesteps'][str(t)] = {
                'original_ce': F.cross_entropy(intact, labels).item(),
                'shuffled_neighbor_ce': F.cross_entropy(changed, labels).item(),
                'original_accuracy': (intact.argmax(-1) == labels).float().mean().item(),
                'shuffled_neighbor_accuracy': (changed.argmax(-1) == labels).float().mean().item(),
            }
    output = Path('experiment_review/coherence_context_probe.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
