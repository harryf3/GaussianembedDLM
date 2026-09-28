"""Reevaluate saved per-character-noise Gaussian DLM checkpoints."""

import json
from pathlib import Path

import torch

from old.coherent_gauss_dlm import data, schedule
from old.coherent_gauss_dlm_token_time import TokenTimeGaussianDLM, evaluate, sample


def main():
    torch.set_num_threads(4)
    _, validation, frequencies, chars, train_paths = data()
    device = torch.device('cpu')
    alpha_bar = schedule(device)
    output = {}
    root = Path('checkpoints/coherent_gauss_dlm_token_time')
    for name in ('best', 'latest'):
        checkpoint = torch.load(root / f'{name}.pt', map_location=device, weights_only=True)
        assert checkpoint['chars'] == chars and checkpoint['train_paths'] == train_paths
        model = TokenTimeGaussianDLM(len(chars)).to(device)
        model.load_state_dict(checkpoint['model'])
        output[name] = {
            'step': checkpoint['step'],
            'heldout': evaluate(model, validation, frequencies, alpha_bar, device),
            'samples': sample(model, alpha_bar, chars, device),
        }
        if name == 'latest':
            output[name]['larger_heldout_probes'] = {
                str(seed): evaluate(model, validation, frequencies, alpha_bar, device,
                                    count=256, seed=seed)
                for seed in (1234, 2234, 3234)
            }
    path = Path('experiment_review/token_time_metrics.json')
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
