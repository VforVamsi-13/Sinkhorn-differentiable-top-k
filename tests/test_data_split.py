
import pytest
import torch

from data import cifar10_split_loaders


@pytest.fixture(scope="module")
def loaders():
    return cifar10_split_loaders(
        batch_size=128,
        num_workers=0,
        download=True,
        split_seed=2026,
    )


def test_cifar10_split_sizes(loaders):
    train_loader, val_loader, test_loader = loaders

    assert len(train_loader.dataset) == 45000
    assert len(val_loader.dataset) == 5000
    assert len(test_loader.dataset) == 10000


def test_cifar10_split_is_reproducible(loaders):
    train_loader, val_loader, _ = loaders

    train_indices = train_loader.dataset.indices
    val_indices = val_loader.dataset.indices

    generator = torch.Generator().manual_seed(2026)

    expected_train, expected_val = torch.utils.data.random_split(
        train_loader.dataset.dataset,
        [45000, 5000],
        generator=generator,
    )

    assert train_indices == expected_train.indices
    assert val_indices == expected_val.indices
