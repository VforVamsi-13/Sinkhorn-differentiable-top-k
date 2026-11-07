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
