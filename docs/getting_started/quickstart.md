# Quickstart

## Run a preset

10-fold cross-validation on Cora with the best-known hyperparameters:

```bash
python -m exp.run --preset cora
```

Override individual fields on top of a preset:

```bash
python -m exp.run --preset cora --model.num-layers 4 --optim.lr 5e-3
```

Fully manual configuration (no preset):

```bash
python -m exp.run \
    --dataset.name cora \
    --model.variant general \
    --model.d 4 \
    --model.num-layers 2
```

See all flags with `python -m exp.run --help`.

## Hyperparameter sweep

```bash
python -m exp.sweep --preset cora --n-trials 100
```

Distributed sweeps share an SQLite study:

```bash
python -m exp.sweep --preset cora \
    --storage sqlite:///studies/cora.db \
    --study-name cora-v1
```

## Use the layers directly

```python
import torch
from torch_geometric.data import Data
from sheaf_mpnn.nsd import DiagonalNSDConv

x = torch.randn(10, 4 * 16)  # N=10, stalk dim d=4, channels=16
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

conv = DiagonalNSDConv(in_channels=16, out_channels=16, d=4)
h = conv(x, edge_index)
```
