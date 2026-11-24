
"""Reproducible, bounded training and evaluation helpers."""
from __future__ import annotations
import csv
from pathlib import Path
import argparse
import random
import time

import numpy as np
import torch
from torch import nn

from data import cifar10_split_loaders
from model import PatchConfig, PatchSelectionClassifier


SELECTION_METHODS = (
    "soft",
    "soft_fixed",
    "hard",
    "random",
    "attention",
    "uniform",
    "straight_through",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    return torch.device(device_name)


def run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer=None,
    max_batches=None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for index, (images, labels) in enumerate(loader):
            if max_batches is not None and max_batches > 0 and index >= max_batches:

                break

            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits, _ = model(images)
            loss = loss_fn(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    if total == 0:
        raise RuntimeError("Data loader produced no samples.")

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "samples": total,
    }


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    loss_fn=None,
    max_batches=None,
):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    return run_epoch(
        model, loader, device, loss_fn,
        optimizer=None, max_batches=max_batches,
    )



def save_patch_heatmaps(model, loader, device, output_path, n_images=4):
    """Save report-quality patch selection visualizations."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    model.eval()

    images, _ = next(iter(loader))
    images = images[:n_images].to(device)

    with torch.no_grad():
        _, weights = model(images)

    images = images.cpu()
    weights = weights.cpu()

    patch_size = model.config.patch_size
    grid_size = model.config.grid_size

    # Convert model inputs back to displayable RGB images.
    images = (images.permute(0, 2, 3, 1).numpy() * 0.5) + 0.5
    images = np.clip(images, 0, 1)

    weights = weights.reshape(-1, grid_size, grid_size).numpy()

    n = len(images)

    fig, axes = plt.subplots(
        n, 3,
        figsize=(12, 4 * n),
        squeeze=False,
    )

    # Use a common scale across images in this figure.
    vmax = max(float(weights.max()), 1e-8)

    for i in range(n):
        image = images[i]
        patch_map = weights[i]

        # Expand the coarse patch map to image resolution.
        expanded_map = np.repeat(
            np.repeat(patch_map, patch_size, axis=0),
            patch_size,
            axis=1,
        )

        # Panel 1: Original image.
        axes[i, 0].imshow(image, interpolation="nearest")
        axes[i, 0].set_title("Original image", fontsize=12)

        # Panel 2: Actual 4x4 selection weights.
        heat = axes[i, 1].imshow(
            patch_map,
            cmap="inferno",
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
        )
        axes[i, 1].set_title("Patch-selection weights", fontsize=12)
        axes[i, 1].set_xticks(range(grid_size))
        axes[i, 1].set_yticks(range(grid_size))
        axes[i, 1].set_xticks(
            np.arange(-0.5, grid_size, 1), minor=True
        )
        axes[i, 1].set_yticks(
            np.arange(-0.5, grid_size, 1), minor=True
        )
        axes[i, 1].grid(
            which="minor",
            color="white",
            linewidth=1.5,
        )
        axes[i, 1].tick_params(which="minor", bottom=False, left=False)

        # Display the actual numerical weight in each patch.
        for row in range(grid_size):
            for col in range(grid_size):
                value = patch_map[row, col]
                axes[i, 1].text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < vmax * 0.65 else "black",
                    fontsize=8,
                    fontweight="bold",
                )

        # Panel 3: Selection overlay.
        axes[i, 2].imshow(image, interpolation="nearest")
        overlay = axes[i, 2].imshow(
            expanded_map,
            cmap="inferno",
            alpha=0.35,
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
        )
        axes[i, 2].set_title("Selection overlay", fontsize=12)

        # Keep axes clean.
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    # Shared colour bar.
    fig.colorbar(
        heat,
        ax=axes[:, 1],
        fraction=0.046,
        pad=0.04,
        label="Selection weight",
    )

    fig.colorbar(
        overlay,
        ax=axes[:, 2],
        fraction=0.046,
        pad=0.04,
        label="Selection weight",
    )

    fig.suptitle(
        f"Learned patch selection (k={model.k}, "
        f"patch size={patch_size}x{patch_size})",
        fontsize=15,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Saved heatmap visualization: {output_path}")



def train(args: argparse.Namespace):
    seed_everything(args.seed)
    device = resolve_device(args.device)

    train_loader, val_loader, test_loader = cifar10_split_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        loader_seed=args.seed,
    )

    model = PatchSelectionClassifier(
        PatchConfig(patch_size=args.patch_size),
        k=args.k,
        selection=args.selection,
        temperature=args.temperature,
        sinkhorn_iters=args.sinkhorn_iters,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )
    loss_fn = nn.CrossEntropyLoss()

    history = []
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model, train_loader, device, loss_fn,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )

        val_metrics = evaluate(
            model, val_loader, device, loss_fn,
            max_batches=args.max_val_batches,
        )

        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(record)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train loss: {record['train_loss']:.4f} | "
            f"Train acc: {record['train_accuracy']:.4f} | "
            f"Val loss: {record['val_loss']:.4f} | "
            f"Val acc: {record['val_accuracy']:.4f}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    training_time = time.perf_counter() - start_time

    test_metrics = evaluate(
        model, test_loader, device, loss_fn,
        max_batches=args.max_test_batches,
    )

    if getattr(args, "save_heatmaps", False):
        heatmap_path = (
            Path(args.output_dir)
            / "plots"
            / "heatmaps"
            / f"{args.selection}.png"
        )
        save_patch_heatmaps(
            model,
            test_loader,
            device,
            heatmap_path,
        )
        print(f"Saved patch heatmaps: {heatmap_path}")    
    return {
        "method": args.selection,
        "seed": args.seed,
        "device": str(device),
        "training_time": training_time,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "history": history,
    }


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def save_results(raw_rows, history_rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "raw_results.csv",
        raw_rows,
        [
            "method",
            "seed",
            "device",
            "training_time",
            "test_loss",
            "test_accuracy",
        ],
    )

    write_csv(
        output_dir / "epoch_history.csv",
        history_rows,
        [
            "method",
            "seed",
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ],
    )

    summary_rows = []

    methods = list(dict.fromkeys(row["method"] for row in raw_rows))

    for method in methods:
        runs = [row for row in raw_rows if row["method"] == method]

        final_val = []
        for run in runs:
            matching = [
                row for row in history_rows
                if row["method"] == method
                and row["seed"] == run["seed"]
            ]
            if matching:
                final_val.append(
                    max(matching, key=lambda row: row["epoch"])["val_accuracy"]
                )

        accuracy_mean, accuracy_std = mean_std(
            [row["test_accuracy"] for row in runs]
        )
        loss_mean, loss_std = mean_std(
            [row["test_loss"] for row in runs]
        )
        time_mean, time_std = mean_std(
            [row["training_time"] for row in runs]
        )
        val_mean, val_std = mean_std(final_val)

        summary_rows.append({
            "method": method,
            "runs": len(runs),
            "test_accuracy_mean": accuracy_mean,
            "test_accuracy_std": accuracy_std,
            "test_loss_mean": loss_mean,
            "test_loss_std": loss_std,
            "val_accuracy_mean": val_mean,
            "val_accuracy_std": val_std,
            "training_time_mean": time_mean,
            "training_time_std": time_std,
        })

    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "method",
            "runs",
            "test_accuracy_mean",
            "test_accuracy_std",
            "test_loss_mean",
            "test_loss_std",
            "val_accuracy_mean",
            "val_accuracy_std",
            "training_time_mean",
            "training_time_std",
        ],
    )

    return summary_rows


def plot_results(raw_rows, history_rows, output_dir):
    """Generate report-ready comparison plots from experiment results."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(dict.fromkeys(row["method"] for row in raw_rows))
    epochs = sorted(set(row["epoch"] for row in history_rows))

    # 1. Validation accuracy: mean +/- standard deviation.
    fig, ax = plt.subplots(figsize=(11, 6))

    for method in methods:
        means, stds = [], []

        for epoch in epochs:
            values = [
                row["val_accuracy"] for row in history_rows
                if row["method"] == method and row["epoch"] == epoch
            ]
            mean, std = mean_std(values)
            means.append(mean)
            stds.append(std)

        means = np.asarray(means)
        stds = np.asarray(stds)

        ax.plot(epochs, means, marker="o", label=method)
        ax.fill_between(
            epochs, means - stds, means + stds, alpha=0.15
        )

    ax.set(
        xlabel="Epoch",
        ylabel="Validation accuracy",
        title="Validation accuracy by selection method",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_accuracy.png", dpi=300)
    plt.close(fig)

    # 2. Training and validation loss curves.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for method in methods:
        train_means, train_stds = [], []
        val_means, val_stds = [], []

        for epoch in epochs:
            train_values = [
                row["train_loss"] for row in history_rows
                if row["method"] == method and row["epoch"] == epoch
            ]
            val_values = [
                row["val_loss"] for row in history_rows
                if row["method"] == method and row["epoch"] == epoch
            ]

            train_mean, train_std = mean_std(train_values)
            val_mean, val_std = mean_std(val_values)

            train_means.append(train_mean)
            train_stds.append(train_std)
            val_means.append(val_mean)
            val_stds.append(val_std)

        train_means = np.asarray(train_means)
        train_stds = np.asarray(train_stds)
        val_means = np.asarray(val_means)
        val_stds = np.asarray(val_stds)

        axes[0].plot(epochs, train_means, marker="o", label=method)
        axes[0].fill_between(
            epochs,
            train_means - train_stds,
            train_means + train_stds,
            alpha=0.15,
        )

        axes[1].plot(epochs, val_means, marker="o", label=method)
        axes[1].fill_between(
            epochs,
            val_means - val_stds,
            val_means + val_stds,
            alpha=0.15,
        )

    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=300)
    plt.close(fig)

    # 3. Training time comparison.
    fig, ax = plt.subplots(figsize=(11, 6))

    time_means, time_stds = [], []

    for method in methods:
        values = [
            row["training_time"] for row in raw_rows
            if row["method"] == method
        ]
        mean, std = mean_std(values)
        time_means.append(mean)
        time_stds.append(std)

    ax.bar(methods, time_means, yerr=time_stds, capsize=4)
    ax.set(
        ylabel="Training time (seconds)",
        title="Training time by selection method",
    )
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "training_time.png", dpi=300)
    plt.close(fig)

    # 4. Test accuracy versus training time.
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        runs = [r for r in raw_rows if r["method"] == method]

        mean_time = mean_std([r["training_time"] for r in runs])[0]
        mean_accuracy = mean_std([r["test_accuracy"] for r in runs])[0]

        ax.scatter(mean_time, mean_accuracy, s=90)
        ax.annotate(
            method,
            (mean_time, mean_accuracy),
            xytext=(6, 6),
            textcoords="offset points",
        )

    ax.set(
        xlabel="Mean training time (seconds)",
        ylabel="Mean test accuracy",
        title="Accuracy versus training time",
    )
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_time.png", dpi=300)
    plt.close(fig)

    print(f"All four comparison plots saved to {output_dir}")


def run_experiments(args):
    methods = (
        list(SELECTION_METHODS)
        if args.all_methods
        else [args.selection]
    )

    raw_rows = []
    history_rows = []

    total_runs = len(methods) * len(args.seeds)
    run_number = 0

    for method in methods:
        for seed in args.seeds:
            run_number += 1
            print(
                f"\n{'=' * 60}\n"
                f"Run {run_number}/{total_runs}: "
                f"method={method}, seed={seed}\n"
                f"{'=' * 60}"
            )

            run_args = argparse.Namespace(**vars(args))
            run_args.selection = method
            run_args.seed = seed

            # Generate one heatmap per method, using the first seed.
            run_args.save_heatmaps = (
                args.save_heatmaps and seed == args.seeds[0]
            )

            result = train(run_args)

            raw_rows.append({
                key: value
                for key, value in result.items()
                if key != "history"
            })

            for epoch_result in result["history"]:
                history_rows.append({
                    "method": method,
                    "seed": seed,
                    **epoch_result,
                })

            # Save incrementally so completed runs are not lost.
            save_results(raw_rows, history_rows, args.output_dir)

    summary_rows = save_results(
        raw_rows,
        history_rows,
        args.output_dir,
    )

    plot_results(raw_rows, history_rows, args.output_dir)

    print("\nFinal summary:")
    for row in summary_rows:
        print(
            f"{row['method']:18s} "
            f"test accuracy = {row['test_accuracy_mean']:.4f} "
            f"+/- {row['test_accuracy_std']:.4f} | "
            f"time = {row['training_time_mean']:.1f}s"
        )

    print(f"\nAll results saved under: {args.output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=100)
    parser.add_argument("--max-val-batches", type=int, default=20)
    parser.add_argument("--max-test-batches", type=int, default=20)

    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=2026)

    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--selection", choices=SELECTION_METHODS, default="soft"
    )

    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Run all seven selection methods.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 17, 27],
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--save-heatmaps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_experiments(args)