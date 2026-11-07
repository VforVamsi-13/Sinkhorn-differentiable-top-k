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

## Bounded CIFAR-10 run

```bash
python experiments.py --data-root /kaggle/working/cifar10 \
  --max-train-batches 100 --max-test-batches 20 --device auto
```

`--device auto` selects CUDA when available. Reduce batch size, epochs, or batch
limits for a smoke run; use `--no-download` when the data is already cached.
