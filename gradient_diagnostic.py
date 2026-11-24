
"""Compare scorer gradient flow across patch-selection methods."""

from pathlib import Path
import csv
import copy
import math

import numpy as np
import torch
import matplotlib.pyplot as plt

from data import cifar10_split_loaders
from model import PatchConfig, PatchSelectionClassifier


METHODS = [
    "soft",
    "soft_fixed",
    "hard",
    "random",
    "attention",
    "uniform",
    "straight_through",
]


def gradient_diagnostic():
    seed = 2026
    learning_rate = 1e-3
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Device: {device}")

    # Obtain one fixed CIFAR-10 batch.
    train_loader, _, _ = cifar10_split_loaders(
        data_root="data",
        batch_size=128,
        num_workers=0,
        download=False,
        val_fraction=0.1,
        split_seed=2026,
        loader_seed=seed,
    )

    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    # Create one common model initialization.
    base_model = PatchSelectionClassifier(
        PatchConfig(patch_size=8),
        k=4,
        selection="soft",
        temperature=0.1,
        sinkhorn_iters=50,
    )

    initial_state = copy.deepcopy(base_model.state_dict())

    results = []

    for method in METHODS:
        print(f"\nTesting: {method}")

        # Every method starts with identical parameters.
        model = PatchSelectionClassifier(
            PatchConfig(patch_size=8),
            k=4,
            selection=method,
            temperature=0.1,
            sinkhorn_iters=50,
        ).to(device)

        model.load_state_dict(initial_state)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
        )

        loss_fn = torch.nn.CrossEntropyLoss()

        # Save scorer parameters before the optimizer update.
        scorer_before = [
            p.detach().clone()
            for p in model.scorer.parameters()
        ]

        optimizer.zero_grad(set_to_none=True)

        logits, weights = model(images)
        loss = loss_fn(logits, labels)

        loss.backward()

        # Measure gradients specifically in the scorer.
        gradients = [
            p.grad
            for p in model.scorer.parameters()
            if p.grad is not None
        ]

        grad_available = len(gradients) > 0

        if grad_available:
            grad_norm = math.sqrt(
                sum(
                    g.detach().pow(2).sum().item()
                    for g in gradients
                )
            )
        else:
            grad_norm = 0.0

        # Apply one optimizer step.
        optimizer.step()

        # Measure actual scorer parameter change.
        update_norm = math.sqrt(
            sum(
                (p.detach() - before).pow(2).sum().item()
                for p, before in zip(
                    model.scorer.parameters(),
                    scorer_before,
                )
            )
        )

        result = {
            "method": method,
            "loss": loss.item(),
            "scorer_grad_available": grad_available,
            "scorer_gradient_tensors": len(gradients),
            "scorer_grad_norm": grad_norm,
            "scorer_update_norm": update_norm,
            "mean_selection_weight": weights.mean().item(),
            "selection_mass": weights.sum(dim=-1).mean().item(),
        }

        results.append(result)

        print(
            f"Loss: {result['loss']:.4f} | "
            f"Gradient available: {grad_available} | "
            f"Gradient norm: {grad_norm:.8e} | "
            f"Scorer update: {update_norm:.8e}"
        )

    # Save numerical results.
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "gradient_diagnostics.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    # Plot scorer gradient norms.
    fig, ax = plt.subplots(figsize=(11, 6))

    values = [r["scorer_grad_norm"] for r in results]

    bars = ax.bar(METHODS, values)

    for bar, result in zip(bars, results):
        if not result["scorer_grad_available"]:
            ax.annotate(
                "No gradient",
                (bar.get_x() + bar.get_width() / 2, 0),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                rotation=90,
                fontsize=9,
            )

    ax.set(
        xlabel="Selection method",
        ylabel="Scorer gradient L2 norm",
        title="Gradient flow to the patch-scoring network",
    )

    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    plot_path = output_dir / "scorer_gradient_norms.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nDiagnostic completed.")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    gradient_diagnostic()
