# Sinkhorn Differentiable Top-k

This repository contains a small PyTorch implementation of Sinkhorn-based
differentiable top-k patch selection, plus hard and random-selection baselines.
The NumPy `sinkhorn` function remains available for matrix-scaling experiments.

## Install and test

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

PyTorch-dependent tests are skipped when PyTorch is unavailable.

## Run the components separately

Run these commands from the repository folder. In a Kaggle notebook, use the
same code in separate cells.

### 1. NumPy Sinkhorn matrix scaling

```bash
python -c "import numpy as np; from sinkhorn import sinkhorn; A=np.array([[1.,2.],[3.,4.]]); B,it=sinkhorn(A); print(B); print('iterations:', it)"
```

This returns a matrix whose row and column sums are approximately one.

### 2. PyTorch Sinkhorn normalization

```bash
python -c "import torch; from sinkhorn import sinkhorn_torch; X=torch.randn(2,4,6); Y=sinkhorn_torch(X, n_iters=50); print(Y.shape); print('row sums:', Y.sum(-1)); print('column sums:', Y.sum(-2))"
```

For a tensor with shape `[..., rows, columns]`, each row has mass `1` and
each column has mass `rows / columns`.

### 3. Differentiable soft top-k

```bash
python -c "import torch; from topk import soft_topk; scores=torch.tensor([[2.,1.,0.,-1.]], requires_grad=True); weights=soft_topk(scores, k=2, temperature=.1); print(weights); print('selected mass:', weights.sum(-1)); weights.sum().backward(); print('gradient:', scores.grad)"
```

`weights` is differentiable and has total mass `k`. Lower `temperature`
typically makes the selection sharper.

### 4. Hard and random baselines

```bash
python -c "import torch; from topk import hard_topk, random_selection; scores=torch.tensor([[2.,1.,0.,-1.]]); print('hard:', hard_topk(scores, 2)); print('random:', random_selection(scores, 2))"
```

You can also use `select_weights(scores, k, mode="soft"|"hard"|"random")`.

### 5. Patch-selection model without CIFAR-10

```python
import torch
from model import PatchSelectionClassifier

model = PatchSelectionClassifier(k=4, selection="soft")
images = torch.randn(2, 3, 32, 32)
logits, patch_weights = model(images)
print(logits.shape)        # torch.Size([2, 10])
print(patch_weights.shape) # torch.Size([2, 16])
```

Use `selection="hard"` or `selection="random"` to run the corresponding
baseline. The soft mode allows gradients to reach the patch scorer.

## Bounded CIFAR-10 run

```bash
python experiments.py --data-root /kaggle/working/cifar10 \
  --max-train-batches 100 --max-test-batches 20 --device auto
```

`--device auto` selects CUDA when available. Reduce batch size, epochs, or batch
limits for a smoke run; use `--no-download` when the data is already cached.
