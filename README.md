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
python -c "import torch; from sinkhorn import sinkhorn_torch; X=abs(torch.randn(2,4,6)); Y=sinkhorn_torch(X, n_iters=50); print(Y.shape); print('row sums:', Y.sum(-1)); print('column sums:', Y.sum(-2))"
```

OR in IDE
```bash
>>> from sinkhorn import sinkhorn_torch
>>> import numpy as np
>>> import torch
>>> X = torch.abs(torch.randn(4, 4))
>>> Y = sinkhorn_torch((X),n_iters=200);
>>> X
tensor([[1.5850, 1.0709, 1.5036, 0.4964],
        [1.4543, 0.8335, 0.2934, 0.4903],
        [0.8935, 0.7083, 1.5385, 0.2009],
        [0.0239, 0.2276, 1.1634, 1.6116]])
>>> Y
tensor([[0.3148, 0.2782, 0.2686, 0.1384],
        [0.3873, 0.3076, 0.1123, 0.1928],
        [0.2152, 0.2643, 0.3798, 0.1406],
        [0.0827, 0.1498, 0.2392, 0.5282]])
>>> print('row sums:', Y.sum(-1))
row sums: tensor([1., 1., 1., 1.])
>>> print('col sums:', Y.sum(-2))
col sums: tensor([1., 1., 1., 1.])
>>> 
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

# Kaggle 


Error: API error (attempt 1): UNAVAILABLE (code 503): No capacity available for model gemini-3.8-flash-low on the server (Error ID: 7f723af6-2412-41d7-a3f2-541a6f8cd691-44-2010)

Here is a step-by-step guide on how to set up, test, and run this repository in a **Kaggle Notebook**.

---

### Step 1: Open a Kaggle Notebook and Enable Settings

1. Go to [kaggle.com](https://www.kaggle.com) $\rightarrow$ **Create** $\rightarrow$ **New Notebook**.
2. On the right-hand panel:
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100** (or keep CPU for basic tests).
   - **Internet**: Toggle **Internet ON** (required to clone GitHub repo / download CIFAR-10).

---

### Step 2: Get the Code into Kaggle

You have two easy options:

#### Option A: Clone via Git (Recommended if uploaded to GitHub)
In the first code cell:
```bash
!git clone https://github.com/<your-username>/Sinkhorn-differentiable-top-k.git
%cd Sinkhorn-differentiable-top-k
```

#### Option B: Upload as a Kaggle Dataset / Folder
1. Zip your local repo folder (excluding `.git` and `.mvc_env`).
2. In the right panel of Kaggle notebook, click **Add Input** $\rightarrow$ **Upload Dataset** $\rightarrow$ upload your zip.
3. In a code cell, copy or navigate to the directory:
```python
import sys
import shutil
# Copy to working directory so you can write outputs
!cp -r /kaggle/input/<dataset-name> /kaggle/working/repo
%cd /kaggle/working/repo
```

---

### Step 3: Run the Basic Unit Tests

Create a new cell and run the test suite:

```bash
!python -m unittest discover -s tests -p "*.py"
```

---

### Step 4: Test Components Interactively (Cell-by-Cell)

#### 1. Test PyTorch Sinkhorn & Soft Top-k Gradients
```python
import torch
from topk import soft_topk, hard_topk, random_selection

# Test score tensor with autograd enabled
scores = torch.tensor([[3.0, 1.0, 0.5, -2.0]], requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")

# Run soft top-k
weights = soft_topk(scores, k=2, temperature=0.1)
print("Soft top-k weights:", weights)
print("Total mass (k=2):", weights.sum().item())

# Backprop to verify gradient path
weights.sum().backward()
print("Gradients on scores:", scores.grad)

# Baselines
print("Hard top-2 mask:", hard_topk(scores, k=2))
print("Random top-2 mask:", random_selection(scores, k=2))
```

#### 2. Test Patch Classifier on Dummy Data
```python
from model import PatchSelectionClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PatchSelectionClassifier(k=4, selection="soft").to(device)

dummy_images = torch.randn(4, 3, 32, 32, device=device)
logits, patch_weights = model(dummy_images)

print("Logits shape:", logits.shape)          # Expected: [4, 10]
print("Patch weights shape:", patch_weights.shape)  # Expected: [4, 16]
print("Selected patch mass:", patch_weights.sum(dim=-1))  # Expected: [4., 4., 4., 4.]
```

---

### Step 5: Run Experiments on CIFAR-10

In Kaggle, always set `--data-root /kaggle/working/cifar10` or a directory under `/kaggle/working/` so torchvision has permission to download and unpack the dataset.

#### Quick Bounded Test Run (Checks pipeline with GPU in ~10 seconds)
```bash
!python experiments.py \
  --data-root /kaggle/working/cifar10 \
  --max-train-batches 10 \
  --max-test-batches 5 \
  --epochs 1 \
  --selection soft \
  --device auto
```

#### Full 1-Epoch Comparison (Soft vs. Hard vs. Random)
Run these cells to compare the three selection modes on the GPU:

```bash
# 1. Soft Top-k (Differentiable)
!python experiments.py --data-root /kaggle/working/cifar10 --epochs 1 --selection soft --device cuda

# 2. Hard Top-k (Discrete baseline)
!python experiments.py --data-root /kaggle/working/cifar10 --epochs 1 --selection hard --device cuda

# 3. Random Selection (No-scoring baseline)
!python experiments.py --data-root /kaggle/working/cifar10 --epochs 1 --selection random --device cuda
```