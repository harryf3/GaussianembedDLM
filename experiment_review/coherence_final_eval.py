"""Evaluate best causal checkpoints on fresh held-out windows and fixed samples."""

import json
import math
import re
from pathlib import Path

import torch
from torch.nn import functional as F

from old.coherent_char_baseline import CausalCharLM, batches, corpus, sample
from old.coherent_gaussian_causal_lm import GaussianCausalCharLM
from old.coherent_gauss_dlm import HOLDOUT, clean_text, data, gaussian_kernel


@torch.no_grad()
def validation_ce(model, validation):
    model.eval()
    generator = torch.Generator().manual_seed(4321)
    total_loss = 0.0
    total_tokens = 0
    for _ in range(8):
        batch = batches(validation, 64, generator, 'cpu')
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        total_loss += F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                      reduction='sum').item()
        total_tokens += targets.numel()
    return total_loss / total_tokens


def text_metrics(samples, train_text):
    known_words = set(re.findall(r"[A-Za-z']{3,}", train_text.lower()))
    words = [word.lower() for passage in samples
             for word in re.findall(r"[A-Za-z']{3,}", passage)]
    pieces = [passage[i:i + 4] for passage in samples for i in range(len(passage) - 3)]
    long_matches = sum(window in train_text for passage in samples
                       for window in (passage[i:i + 64]
                                      for i in range(len(passage) - 63)))
    return {
        'known_word_fraction': sum(word in known_words for word in words) / len(words),
        'distinct_fourgram_fraction': len(set(pieces)) / len(pieces),
        'exact_training_64gram_windows': long_matches,
        'generated_64gram_windows': sum(max(0, len(s) - 63) for s in samples),
    }


@torch.no_grad()
def gaussian_geometry(model, probabilities):
    embedding = model.embedding
    means, std = embedding.mu, embedding.std
    variances = std.square()
    weighted_mean = probabilities @ means
    centered = means - weighted_mean
    covariance = (probabilities[:, None] * centered).T @ centered
    covariance += torch.diag(probabilities @ variances)
    zero = means.new_zeros((1, means.size(-1)))
    one = torch.ones_like(zero)
    mmd_squared = (probabilities @ gaussian_kernel(means, variances, means, variances)
                   @ probabilities + gaussian_kernel(zero, one, zero, one).squeeze()
                   - 2 * (probabilities * gaussian_kernel(means, variances, zero, one)
                          .squeeze(-1)).sum())
    eig = torch.linalg.eigvalsh(covariance)
    return {'weighted_mmd_squared': mmd_squared.item(),
            'within_variance': (probabilities[:, None] * variances).sum(-1).sum().item()
            / means.size(-1),
            'covariance_rms_error': (torch.linalg.matrix_norm(covariance - torch.eye(
                means.size(-1))) / math.sqrt(means.size(-1))).item(),
            'covariance_eigen_min': eig.min().item(),
            'covariance_eigen_max': eig.max().item()}


def main():
    torch.set_num_threads(2)
    _, validation, chars = corpus()
    _, _, probabilities, gaussian_chars, _ = data()
    assert gaussian_chars == chars
    train_text = '\n\n'.join(clean_text(p) for p in sorted(
        Path('shakespeare-dataset-main/text').glob('*.txt')) if p.name not in HOLDOUT)
    report = {'holdout': HOLDOUT, 'validation_windows': 512,
              'tokens_per_window': 64, 'sample_temperature': 0.65,
              'sample_top_k': 12, 'samples_per_model': 3, 'models': {}}
    for name, constructor, location in (
        ('point', CausalCharLM, 'checkpoints/coherent_char_baseline/best.pt'),
        ('gaussian', GaussianCausalCharLM, 'checkpoints/coherent_gaussian_causal_lm/best.pt'),
    ):
        checkpoint = torch.load(location, map_location='cpu', weights_only=True)
        assert checkpoint['chars'] == chars
        model = constructor(len(chars))
        model.load_state_dict(checkpoint['model'])
        generated = [sample(model, chars, 'cpu', seed=seed, length=256,
                            temperature=0.65, top_k=12) for seed in (1, 2, 3)]
        report['models'][name] = {
            'best_step': checkpoint['step'],
            'validation_ce': validation_ce(model, validation),
            'samples': generated,
            **text_metrics(generated, train_text),
        }
        if name == 'gaussian':
            report['models'][name]['geometry'] = gaussian_geometry(model, probabilities)
    output = Path('experiment_review/coherence_final_metrics.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
