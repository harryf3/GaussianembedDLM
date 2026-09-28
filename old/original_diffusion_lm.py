#!/usr/bin/env python3
"""Train the point-embedding (original) Diffusion-LM on character data.

Example:
    python original_diffusion_lm.py --epochs 10 --batch-size 512 --lr 3e-4 \
        --output checkpoints/original_diffusion_lm.pt
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from old.model import ModelConfig, PointEmbeddingDiffusionLM


class CharacterBlocks(Dataset[Tensor]):
    """Fixed-size, optionally overlapping blocks of encoded characters."""

    def __init__(self, tokens: Tensor, block_size: int, stride: int) -> None:
        if len(tokens) < block_size:
            raise ValueError("the corpus is shorter than --block-size")
        self.tokens = tokens
        self.block_size = block_size
        self.starts = range(0, len(tokens) - block_size + 1, stride)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> Tensor:
        start = self.starts[index]
        return self.tokens[start : start + self.block_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("shakespeare-dataset-main/text"),
                        help="directory containing UTF-8 .txt corpus files")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/original_diffusion_lm.pt"),
                        help="path for the final checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=None,
                        help="characters between blocks (default: block size)")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--x0-noise-std", type=float, default=None,
                        help="override initial clean-state noise; default follows the schedule")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if requested == "auto":
        requested = "mps" if available["mps"] else "cuda" if available["cuda"] else "cpu"
    if not available[requested]:
        raise ValueError(f"requested device '{requested}' is not available")
    return torch.device(requested)


def load_corpus(data_dir: Path) -> tuple[Tensor, list[str]]:
    paths = sorted(data_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"no .txt files found in {data_dir}")
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    return torch.tensor([stoi[char] for char in text], dtype=torch.long), chars


def validate_args(args: argparse.Namespace) -> None:
    positive = ("epochs", "batch_size", "block_size", "dim", "layers", "heads", "ff_dim",
                "diffusion_steps", "log_every")
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.stride is not None and args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.dim % args.heads:
        raise ValueError("--dim must be divisible by --heads")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0:
        raise ValueError("--lr must be positive; --weight-decay and --grad-clip cannot be negative")
    if args.x0_noise_std is not None and args.x0_noise_std < 0:
        raise ValueError("--x0-noise-std cannot be negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokens, vocabulary = load_corpus(args.data_dir)
    stride = args.stride or args.block_size
    dataset = CharacterBlocks(tokens, args.block_size, stride)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    config = ModelConfig(vocab_size=len(vocabulary), sequence_length=args.block_size,
                         dim=args.dim, layers=args.layers, heads=args.heads,
                         ff_dim=args.ff_dim, dropout=args.dropout,
                         diffusion_steps=args.diffusion_steps)
    model = PointEmbeddingDiffusionLM(config, x0_noise_std=args.x0_noise_std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    parameters = sum(parameter.numel() for parameter in model.parameters())

    print("Training started", flush=True)
    print(
        f"epochs={args.epochs} | lr={args.lr:g} | batch_size={args.batch_size} | "
        f"blocks={len(dataset):,} | block_size={args.block_size} | dim={args.dim} | "
        f"layers={args.layers} | heads={args.heads} | diffusion_steps={args.diffusion_steps} | "
        f"device={device} | parameters={parameters:,}",
        flush=True,
    )
    print(f"corpus={args.data_dir} | characters={len(tokens):,} | vocabulary={len(vocabulary)}", flush=True)

    final_metrics: dict[str, float] = {}
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = torch.zeros(4)
        epoch_started = time.perf_counter()
        for step, batch in enumerate(loader, start=1):
            batch = batch.to(device, non_blocking=device.type == "cuda")
            terms = model.training_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            terms.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip else torch.tensor(0.0)
            optimizer.step()

            values = torch.tensor([
                terms.total.item(), terms.diffusion.item(), terms.rounding.item(),
                terms.aggregate_prior.item(),
            ])
            totals += values
            if step % args.log_every == 0 or step == len(loader):
                running = totals / step
                print(
                    f"epoch {epoch:03d}/{args.epochs:03d} step {step:04d}/{len(loader):04d} | "
                    f"loss={values[0]:.4f} running={running[0]:.4f} | "
                    f"mse={running[1]:.4f} ce={running[2]:.4f} prior={running[3]:.6f} | "
                    f"grad_norm={float(grad_norm):.3f}",
                    flush=True,
                )
        means = totals / len(loader)
        final_metrics = {"loss": means[0].item(), "diffusion": means[1].item(),
                         "rounding": means[2].item(), "prior": means[3].item()}
        print(f"epoch {epoch:03d} complete | loss={final_metrics['loss']:.4f} | "
              f"mse={final_metrics['diffusion']:.4f} ce={final_metrics['rounding']:.4f} "
              f"prior={final_metrics['prior']:.6f} | seconds={time.perf_counter() - epoch_started:.1f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(config), "vocabulary": vocabulary, "stoi": {char: i for i, char in enumerate(vocabulary)},
        "training_args": vars(args), "epochs_completed": args.epochs, "final_metrics": final_metrics,
    }
    torch.save(checkpoint, args.output)
    print(f"Training complete | seconds={time.perf_counter() - started:.1f}", flush=True)
    print(f"Finished checkpoint: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
