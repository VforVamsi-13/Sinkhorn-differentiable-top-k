"""Small patch-selection image classifier used by the experiments."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from topk import select_weights

@dataclass(frozen=True)
class PatchConfig:
    image_size: int = 32
    patch_size: int = 8
    channels: int = 3
    @property
    def grid_size(self) -> int:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        return self.image_size // self.patch_size
    @property
    def n_patches(self) -> int:
        return self.grid_size * self.grid_size

def image_to_patches(images: torch.Tensor, config: PatchConfig) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if images.shape[1] != config.channels or images.shape[2:] != (
        config.image_size, config.image_size
    ):
        raise ValueError("images do not match PatchConfig")
    p, g = config.patch_size, config.grid_size
    return images.reshape(-1, config.channels, g, p, g, p).permute(
        0, 2, 4, 1, 3, 5
    ).reshape(images.shape[0], config.n_patches, config.channels, p, p)

class PatchSelectionClassifier(nn.Module):
    def __init__(self, config: PatchConfig = PatchConfig(), *, k: int = 4,
                 n_classes: int = 10, selection: str = "soft",
                 temperature: float = 0.1, sinkhorn_iters: int = 50) -> None:
        super().__init__()
        if not 1 <= k <= config.n_patches:
            raise ValueError("k must be between one and the number of patches")
        self.config, self.k, self.selection = config, k, selection
        self.temperature, self.sinkhorn_iters = temperature, sinkhorn_iters
        feature_dim = 64
        self.scorer = nn.Sequential(nn.Conv2d(config.channels, 16, 3, padding=1),
                                    nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten(), nn.Linear(16, 1))
        self.encoder = nn.Sequential(nn.Conv2d(config.channels, 32, 3, padding=1),
                                     nn.ReLU(), nn.Conv2d(32, feature_dim, 3, padding=1),
                                     nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(),
                                        nn.Linear(64, n_classes))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = image_to_patches(images, self.config)
        batch, n_patches = patches.shape[:2]
        flat = patches.reshape(batch * n_patches, *patches.shape[2:])
        scores = self.scorer(flat).reshape(batch, n_patches)
        weights = select_weights(scores, self.k, mode=self.selection,
                                 temperature=self.temperature,
                                 n_iters=self.sinkhorn_iters)
        features = self.encoder(flat).reshape(batch, n_patches, -1)
        pooled = (features * weights.unsqueeze(-1)).sum(dim=1) / self.k
        return self.classifier(pooled), weights
