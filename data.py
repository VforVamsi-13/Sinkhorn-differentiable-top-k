"""Optional CIFAR-10 loading with Kaggle-friendly defaults."""
from pathlib import Path

def cifar10_loaders(*, data_root: str | Path = "data", batch_size: int = 128,
                    num_workers: int = 2, download: bool = True):
    try:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("CIFAR-10 loading requires torchvision") from exc
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    root = Path(data_root).expanduser()
    common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
    return (DataLoader(datasets.CIFAR10(root, train=True, transform=transform,
                                        download=download), shuffle=True, **common),
            DataLoader(datasets.CIFAR10(root, train=False, transform=transform,
                                        download=download), **common))

def cifar10_split_loaders(
    *,
    data_root: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = True,
    val_fraction: float = 0.1,
    split_seed: int = 2026,
    loader_seed: int = 7,
):
    """
    Return reproducible training, validation and test loaders.

    The training dataset is split into training and validation
    subsets using a fixed split_seed.

    All subsets use the same deterministic transformation.
    """
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    root = Path(data_root).expanduser()

    full_train = datasets.CIFAR10(
        root,
        train=True,
        transform=transform,
        download=download,
    )

    test_dataset = datasets.CIFAR10(
        root,
        train=False,
        transform=transform,
        download=download,
    )

    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size

    split_generator = torch.Generator().manual_seed(split_seed)

    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=split_generator,
    )

    loader_generator = torch.Generator().manual_seed(loader_seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
        **common,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common,
    )

    return train_loader, val_loader, test_loader
