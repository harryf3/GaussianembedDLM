"""Generate character text from the best held-out causal checkpoint."""

import argparse

import torch

from old.coherent_char_baseline import CausalCharLM, sample
from old.coherent_gaussian_causal_lm import GaussianCausalCharLM
from old.coherent_gaussian_causal_lm_long import LongGaussianCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('gaussian', 'point', 'gaussian-long'), default='gaussian')
    parser.add_argument('--prompt', default='\n')
    parser.add_argument('--length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.65)
    parser.add_argument('--top-k', type=int, default=12)
    args = parser.parse_args()
    if not args.prompt:
        parser.error('prompt must contain at least one character')
    if args.length < 1 or args.temperature <= 0 or args.top_k < 1:
        parser.error('length, temperature, and top-k must be positive')
    torch.set_num_threads(2)
    if args.model == 'gaussian':
        path = 'checkpoints/coherent_gaussian_causal_lm/best.pt'
        constructor = GaussianCausalCharLM
        context_size = 64
    elif args.model == 'gaussian-long':
        path = 'checkpoints/coherent_gaussian_causal_lm_long/best.pt'
        constructor = LongGaussianCausalLM
        context_size = 128
    else:
        path = 'checkpoints/coherent_char_baseline/best.pt'
        constructor = CausalCharLM
        context_size = 64
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    chars = checkpoint['chars']
    unknown = set(args.prompt) - set(chars)
    if unknown:
        parser.error(f'prompt contains unknown characters: {sorted(unknown)!r}')
    model = constructor(len(chars))
    model.load_state_dict(checkpoint['model'])
    text = sample(model, chars, 'cpu', seed=args.seed, prompt=args.prompt,
                  length=args.length, temperature=args.temperature,
                  top_k=min(args.top_k, len(chars)), context_size=context_size)
    print(text)


if __name__ == '__main__':
    main()
