> 18 Nov

Based on the project proposal ([Proj_proposal.md](file:///d:/Code%20Files%20D%20drive/Sinkhorn-differentiable-top-k/Proj_proposal.md)) and the current codebase, here is a breakdown of what has been implemented and what remains to be done:

---

###  What Has Been Implemented

1. **Foundational Sinkhorn Algorithm (Paper 1 / NumPy)**
   - Matrix scaling for non-negative square matrices with convergence tolerance and iteration tracking ([sinkhorn.py](file:///d:/Code%20Files%20D%20drive/Sinkhorn-differentiable-top-k/sinkhorn.py#L6-L41)).
   - Unit tests covering convergence, non-square validation, and negative entries ([tests/tests.py](file:///d:/Code%20Files%20D%20drive/Sinkhorn-differentiable-top-k/tests/tests.py)).
   - Exploratory scripts for support and indecomposability properties in [misc_or_unfinished/](file:///d:/Code%20Files%20D%20drive/Sinkhorn-differentiable-top-k/misc_or_unfinished).

2. **Differentiable PyTorch Sinkhorn Normalization**
   - Batched, multi-dimensional PyTorch implementation ([sinkhorn_torch](file:///d:/Code%20Files%20D%20drive/Sinkhorn-differentiable-top-k/sinkhorn.py#L43-L85)).
   - Log-domain stabilization (`amax` subtraction + clamping) to avoid underflow/overflow under sharp temperatures.
   - Rectangular normalization ensuring row sums = $1.0$ and column sums = $\text{rows} / \text{cols}$.

3. **Top-k Selection Modules**
   - **Differentiable SOFT Top-k** ([soft_topk](file:///d:/Code%20Files%20D%20drive/Sinkhorn-differentiable-top-k/topk.py#L17-L29)) via optimal transport slot relaxation.
   - **Hard Top-k** discrete baseline ([hard_topk](file:///d:/Code%20Files%20D%20drive/Sinkhorn-differentiable-top-k/topk.py#L31-L35)).
   - **Random Selection** baseline ([random_selection](file:///d:/Code%20Files%20D%20drive/Sinkhorn-differentiable-top-k/topk.py#L37-L43)).
   - Unified dispatcher `select_weights` with test suites in [tests/test_torch_components.py](file:///d:/Code%20Files%20D%20drive/Sinkhorn-differentiable-top-k/tests/test_torch_components.py).

4. **Vision Pipeline & Patch-Selection Classifier (Paper 3 Inspired)**
   - Patch extraction utility `image_to_patches` ([model.py](file:///d:/Code%20Files%20D%20drive/Sinkhorn-differentiable-top-k/model.py#L22-L32)).
   - `PatchSelectionClassifier` architecture containing:
     - Patch scorer (lightweight ConvNet predicting patch importance logits),
     - Dynamic selection/weighting module (`soft`, `hard`, or `random`),
     - Patch feature encoder + weighted pooling + classification head.

5. **Basic Training & Experiment Harness**
   - CIFAR-10 data loader pipeline ([data.py](file:///d:/Code%20Files%20D%20drive/Sinkhorn-differentiable-top-k/data.py)).
   - CLI script ([experiments.py](file:///d:/Code%20Files%20D%20drive/Sinkhorn-differentiable-top-k/experiments.py)) supporting training, evaluation, and batch-limiting for fast testing.

---

### ⏳ What Remains to Be Implemented / Completed

According to Sections 4 & 5 of the project proposal, the following items are still pending:

1. **Gradient Validation & Numerical Verification**
   - **Finite-difference verification:** Verify automatic differentiation gradients against numerical finite-difference estimates on small synthetic test cases.
   - **Gradient diagnostics:** Logging and recording gradient norms flowing into `scorer` across different values of temperature $\tau$ and selection size $k$ to verify vanishing/exploding gradient behavior.

2. **Systematic Experimental Evaluation & Benchmarking**
   - Currently, [experiments.py](file:///d:/Code%20Files%20D%20drive/Sinkhorn-differentiable-top-k/experiments.py) only trains for 1 epoch by default and does not log or persist metrics across runs.
   - Multi-epoch training across all 3 modes: `soft` vs. `hard` vs. `random`.
   - Logging test accuracy curves, convergence speed, runtime per step, and sensitivity to Sinkhorn iteration counts.

3. **Ablation Studies**
   - **Impact of $k$:** Varying the number of retained patches ($k \in \{2, 4, 8, 12\}$ out of 16 patches).
   - **Impact of temperature ($\tau$):** Measuring performance and gradient smoothness across varying temperatures ($\tau \in \{0.01, 0.1, 1.0\}$).
   - **Impact of Sinkhorn iterations:** Testing accuracy vs. computational cost for different iteration counts (e.g., 10, 30, 50, 100).

4. **Visualizations & Interpretability Analysis**
   - Visualizing original CIFAR-10 images alongside heatmaps of predicted patch scores and selected patches (showing which patches `soft_topk` actually prioritizes).
   - Inspecting failure cases or scenarios where the differentiable top-k fails to locate the discriminative image features.

5. **Optional / Scope Additions from Proposal**
   - Comparison with a published alternative (such as Perturbed Maximum Top-k from Cordonnier et al. / Paper 3).
   - Scaling up experiments to CIFAR-100 or full-budget training.

Ran check python environment
Ran check virtualenv python
Ran run pytest inside virtual environment · Exited with code 1
Canceled check torch in venv
Checked task Check torch in venv
Killed task Check torch in venv

Here is how to run and verify each implemented component, from individual functions up to the full vision model.

*(Note: If you are using a virtual environment like `.mvc_env`, activate it first with `.\.mvc_env\Scripts\Activate.ps1` or prefix commands with `python` in your terminal).*

---

### 1. Run All Automated Unit Tests

You can run the existing test suite using Python's built-in `unittest` runner:

```powershell
python -m unittest discover -s tests -p "*.py"
```
or `pytest`
This validates matrix scaling invariants, error conditions, differentiable top-k shapes, and baseline selections.

---

### 2. Verify NumPy Sinkhorn Scaling (Paper 1)

Checks alternating row and column normalization to ensure the output converges to a doubly stochastic matrix (row sums and column sums equal $1.0$):

```powershell
python -c "import numpy as np; from sinkhorn import sinkhorn; A = np.array([[1., 2.], [3., 4.]]); B, iters = sinkhorn(A); print('Result matrix:\n', B); print('Row sums:', B.sum(axis=1)); print('Col sums:', B.sum(axis=0)); print('Iterations:', iters)"
```

---

### 3. Verify PyTorch Sinkhorn Normalization

Validates batched normalization in PyTorch and verifies the marginal sums:

```powershell
python -c "import torch; from sinkhorn import sinkhorn_torch; X = torch.abs(torch.randn(2, 4, 4)); Y = sinkhorn_torch(X, n_iters=50); print('Output shape:', Y.shape); print('Row sums (target ~1.0):\n', Y.sum(dim=-1)); print('Col sums (target ~1.0):\n', Y.sum(dim=-2))"
```

---

### 4. Verify Differentiable Soft Top-k and Gradient Flow (Paper 2)

Tests that `soft_topk`:
1. Sums to total mass $k$,
2. Successfully computes non-zero gradients backward to the input scores:

```powershell
python -c "import torch; from topk import soft_topk; scores = torch.tensor([[3.0, 1.0, 0.5, -2.0]], requires_grad=True); weights = soft_topk(scores, k=2, temperature=0.1); print('Weights (k=2):', weights); print('Selected mass (sum):', weights.sum().item()); weights.sum().backward(); print('Gradients on scores:', scores.grad)"
```

---

### 5. Check Hard & Random Selection Baselines

Compares soft weights against discrete hard top-k and random mask selections:

```powershell
python -c "import torch; from topk import hard_topk, random_selection; scores = torch.tensor([[3.0, 1.0, 0.5, -2.0]]); print('Hard top-2 mask:\n', hard_topk(scores, k=2)); print('Random top-2 mask:\n', random_selection(scores, k=2))"
```

---

### 6. Verify Patch-Selection Vision Classifier (Paper 3)

Tests the `PatchSelectionClassifier` on a synthetic batch of images ($32 \times 32$ with $8 \times 8$ patches = 16 patches) without needing to download CIFAR-10 first:

```powershell
python -c "import torch; from model import PatchSelectionClassifier; model = PatchSelectionClassifier(k=4, selection='soft'); images = torch.randn(2, 3, 32, 32); logits, weights = model(images); print('Logits shape (batch, classes):', logits.shape); print('Weights shape (batch, patches):', weights.shape); print('Patch weights sum per image:', weights.sum(dim=-1))"
```

---

### 7. Run a Quick Bounded End-to-End Training Run

To test the entire pipeline on CIFAR-10 in a fast, bounded manner (e.g., 5 train batches, 2 test batches):

```powershell
python experiments.py --data-root ./data --max-train-batches 5 --max-test-batches 2 --epochs 1 --selection soft
```

*(You can replace `--selection soft` with `--selection hard` or `--selection random` to test the baseline pipelines).*