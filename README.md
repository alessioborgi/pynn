# Neural Sheaf Diffusion – PyTorch Implementation

A clean PyTorch / PyG implementation of **Neural Sheaf Diffusion (NSD)** with three restriction-map variants and a benchmark suite across 14 node-classification datasets.

**Copyright © 2026, _Sheaf Neural Networks as Message Passing_.**
Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwhite, Mario Severino,
Emanuele Mule, Dario Loi, Francesco Restuccia, Fabrizio Silvestri, and Pietro Liò.

## Quick Start

```bash
git clone https://github.com/<you>/pytorch-SheafNeuralNetworks.git
cd pytorch-SheafNeuralNetworks
uv sync && source .venv/bin/activate

# 10-fold CV on Cora with the best-known config (data downloaded automatically)
python -m exp.run --preset cora
```

Use the library directly:

```python
from sheaf_mpnn.nsd import NSDModel, NSDVariant

model = NSDModel(
    in_channels=1433, out_channels=7,
    d=4, hidden_dim=16, num_layers=2,
    variant=NSDVariant.GENERAL, alpha=1.0,
)
logits = model(x, edge_index)  # → [N, 7]
```

Use the layers directly when you want to build your own architecture:

```python
from torch import nn

from sheaf_mpnn.nsd import GeneralNSDConv

d, hidden_dim = 4, 16
encoder = nn.Linear(1433, d * hidden_dim)
layer = GeneralNSDConv(
    d=d,
    in_channels=hidden_dim,
    hidden_dim=hidden_dim,
    context_dim=d * hidden_dim,
    alpha=1.0,
)

x_stalk = encoder(x).view(-1, d, hidden_dim)       # [N, d, hidden_dim]
x_feat = x_stalk.reshape(x_stalk.size(0), -1)      # [N, d * hidden_dim]
x_stalk = layer(x_feat, x_stalk, edge_index)       # [N, d, hidden_dim]
```

## Installation

```bash
uv sync                   # core dependencies
uv sync --extra wandb     # + Weights & Biases / Optuna-WandB
```

**Requirements:** Python ≥ 3.13, PyTorch ≥ 2.4, PyTorch Geometric ≥ 2.5, Lightning ≥ 2.3.

## Datasets

All 14 datasets download automatically into `exp/data/` on first use.

| Dataset | Nodes | Edges | Features | Classes | Metric | Split |
|---------|------:|------:|--------:|--------:|:------:|-------|
| `cora` | 2 708 | 10 556 | 1 433 | 7 | Acc | Geom-GCN |
| `citeseer` | 3 327 | 9 104 | 3 703 | 6 | Acc | Geom-GCN |
| `chameleon` | 2 277 | 36 101 | 2 325 | 5 | Acc | Geom-GCN |
| `chameleon_filtered` | 890 | 8 854 | 2 325 | 5 | Acc | Geom-GCN filtered |
| `squirrel` | 5 201 | 217 073 | 2 089 | 5 | Acc | Geom-GCN |
| `squirrel_filtered` | 2 223 | 47 138 | 2 089 | 5 | Acc | Geom-GCN filtered |
| `cornell` | 183 | 298 | 1 703 | 5 | Acc | Geom-GCN |
| `texas` | 183 | 325 | 1 703 | 5 | Acc | Geom-GCN |
| `film` | 7 600 | 30 019 | 932 | 5 | Acc | Geom-GCN |
| `amazon_ratings` | 24 492 | 186 100 | 300 | 5 | Acc | Platonov |
| `minesweeper` | 10 000 | 39 402 | 7 | 2 | ROC-AUC | Platonov |
| `questions` | 48 921 | 153 540 | 301 | 2 | ROC-AUC | Platonov |
| `roman_empire` | 22 662 | 32 927 | 300 | 18 | Acc | Platonov |
| `tolokers` | 11 758 | 519 000 | 10 | 2 | ROC-AUC | Platonov |

## Model Variants

All variants share the same `encoder → NSD layers → decoder` architecture; only the restriction-map parameterisation differs.

| Variant | Flag | Params / edge | Notes |
|---------|------|:-------------:|-------|
| Diagonal | `--model.variant diagonal` | O(d) | Lightweight baseline |
| General | `--model.variant general` | O(d²) | Most expressive |
| Orthogonal | `--model.variant orthogonal` | O(d(d−1)/2) | Numerically stable via Cayley transform |

## Running Experiments

### Presets

Every dataset has a built-in preset. Any field can still be overridden:

```bash
python -m exp.run --preset cora
python -m exp.run --preset texas --model.variant orthogonal --model.d 5
```

Run `python -m exp.run --help` for the full list of flags.

### Weights & Biases

```bash
python -m exp.run --preset cora \
    --wandb.enabled --wandb.entity your_entity --wandb.project nsd-cora
```

### Hyperparameter Sweeps

```bash
python -m exp.sweep --preset cora --n-trials 100

# Distributed sweep via shared storage
python -m exp.sweep --preset cora --study-name nsd-cora \
    --storage sqlite:///nsd_sweep.db --n-trials 50
```

## Running Tests

```bash
uv run pytest              # full suite
uvx ruff check .           # lint
uvx ruff format --check .  # formatting
```

## Citation

```bibtex
@article{bodnar2022neural,
  title   = {Neural Sheaf Diffusion: A Topological Perspective on
             Heterophily and Oversmoothing in GNNs},
  author  = {Bodnar, Cristian and Di Giovanni, Francesco and Chamberlain,
             Benjamin Paul and Lio, Pietro and Bronstein, Michael M.},
  journal = {Advances in Neural Information Processing Systems (NeurIPS)},
  year    = {2022}
}

@inproceedings{pei2020geomgcn,
  title     = {Geom-{GCN}: Geometric Graph Convolutional Networks},
  author    = {Pei, Hongbin and Wei, Bingzhe and Chang, Kevin Chen-Chuan
               and Lei, Yu and Yang, Bo},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020}
}

@inproceedings{platonov2023a,
  title     = {A Critical Look at the State of Graph Learning Benchmarks},
  author    = {Platonov, Oleg and Kuznedelev, Denis and Diskin, Michael
               and Babenko, Artem and Prokhorenkova, Liudmila},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}
```
