
import torch
import torch.nn.functional as F

from model import PatchSelectionClassifier


def test_classification_loss_reaches_patch_scorer():
    torch.manual_seed(42)

    model = PatchSelectionClassifier(
        k=4,
        n_classes=10,
        selection="soft",
        temperature=0.1,
        sinkhorn_iters=100,
    )

    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 2, 5, 8])

    # Forward pass
    logits, weights = model(images)

    # Check output dimensions
    assert logits.shape == (4, 10)
    assert weights.shape == (4, 16)

    # Check selection constraints
    assert torch.all(weights >= 0)
    assert torch.all(weights <= 1.001)

    assert torch.allclose(
        weights.sum(dim=1),
        torch.full((4,), 4.0),
        atol=1e-3,
    )

    # Classification loss
    loss = F.cross_entropy(logits, labels)

    # Backpropagation
    loss.backward()

    # Check gradients in the scorer
    scorer_grads = [
        p.grad for p in model.scorer.parameters()
        if p.grad is not None
    ]

    assert len(scorer_grads) > 0

    assert all(
        torch.isfinite(grad).all()
        for grad in scorer_grads
    )

    scorer_grad_norm = sum(
        grad.norm().item()
        for grad in scorer_grads
    )

    assert scorer_grad_norm > 1e-8, (
        "Classification loss did not produce a meaningful "
        "gradient in the patch scorer."
    )

    # Check gradients in the encoder
    encoder_grads = [
        p.grad for p in model.encoder.parameters()
        if p.grad is not None
    ]

    assert len(encoder_grads) > 0

    assert any(
        grad.norm().item() > 0
        for grad in encoder_grads
    )

    # Check gradients in the classifier
    classifier_grads = [
        p.grad for p in model.classifier.parameters()
        if p.grad is not None
    ]

    assert len(classifier_grads) > 0

    assert any(
        grad.norm().item() > 0
        for grad in classifier_grads
    )
