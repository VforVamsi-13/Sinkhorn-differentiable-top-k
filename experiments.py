"""Reproducible, bounded training and evaluation helpers."""
from __future__ import annotations
import argparse
import random
import numpy as np
import torch
from torch import nn
from data import cifar10_loaders
from model import PatchConfig, PatchSelectionClassifier

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def evaluate(model: nn.Module, loader, device: torch.device, max_batches=None):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for index, (images, labels) in enumerate(loader):
            if max_batches is not None and index >= max_batches: break
            logits, _ = model(images.to(device))
            correct += (logits.argmax(-1).cpu() == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)

def train(args: argparse.Namespace):
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, test_loader = cifar10_loaders(data_root=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers, download=args.download)
    model = PatchSelectionClassifier(PatchConfig(patch_size=args.patch_size), k=args.k,
        selection=args.selection, temperature=args.temperature,
        sinkhorn_iters=args.sinkhorn_iters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        model.train()
        for index, (images, labels) in enumerate(train_loader):
            if args.max_train_batches is not None and index >= args.max_train_batches: break
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(images.to(device))[0], labels.to(device)).backward()
            optimizer.step()
    return {"device": str(device), "test_accuracy": evaluate(
        model, test_loader, device, args.max_test_batches)}

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data"); parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2); parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=100); parser.add_argument("--max-test-batches", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=8); parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--selection", choices=("soft", "hard", "random"), default="soft")
    parser.add_argument("--temperature", type=float, default=0.1); parser.add_argument("--sinkhorn-iters", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser

if __name__ == "__main__":
    print(train(build_parser().parse_args()))
