import pytest

torch = pytest.importorskip("torch")

from model import PatchConfig, PatchSelectionClassifier, image_to_patches
from topk import hard_topk, soft_topk


def test_soft_topk_mass_and_gradient():
    scores = torch.tensor([[2.0, 1.0, 0.0, -1.0]], requires_grad=True)
    weights = soft_topk(scores, 2, temperature=0.5, n_iters=30)
    assert torch.allclose(weights.sum(-1), torch.tensor([2.0]), atol=1e-4)
    weights.sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_hard_topk_is_binary():
    weights = hard_topk(torch.tensor([[0.1, 0.9, 0.2]]), 1)
    assert torch.equal(weights, torch.tensor([[0.0, 1.0, 0.0]]))


def test_patch_model_shapes_and_score_gradients():
    config = PatchConfig()
    images = torch.randn(2, 3, 32, 32)
    patches = image_to_patches(images, config)
    assert patches.shape == (2, 16, 3, 8, 8)
    model = PatchSelectionClassifier(config, k=4)
    logits, weights = model(images)
    assert logits.shape == (2, 10)
    assert weights.shape == (2, 16)
    logits.sum().backward()
    assert model.scorer[0].weight.grad is not None
