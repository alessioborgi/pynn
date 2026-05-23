# Datasets and splits

14 node-classification datasets are downloaded on demand into
`src/exp/data/` (PyG cache) and `exp/splits/` (`.npz` for Geom-GCN,
fetched on first request by `gen_splits.py`). Three split strategies
are selected automatically per dataset:

::::{grid} 1
:gutter: 2

:::{grid-item-card} Geom-GCN 48/32/20
**Datasets:** cora, citeseer, chameleon, squirrel, cornell, texas, film
(alias `fil`).
Canonical Pei et al. (2020) splits. Filename
`{dataset}_split_0.6_0.2_{fold}.npz` is historical and does *not*
reflect actual ratios.
:::

:::{grid-item-card} Geom-GCN filtered ≈48/32/20
**Datasets:** chameleon_filtered, squirrel_filtered.
Splits embedded in raw `.npz` from yandex-research after
duplicate-node removal.
:::

:::{grid-item-card} Platonov 50/25/25
**Datasets:** amazon_ratings, minesweeper, questions, roman_empire,`
tolokers.
Shipped with PyG `HeterophilousGraphDataset`.
:::
::::

Metric is `acc` everywhere except **minesweeper**, **tolokers**, and
**questions**, which use ROC-AUC (one-vs-rest, scikit-learn).

## Pre-fetching splits

```bash
python -m exp.gen_splits                          # all datasets
python -m exp.gen_splits --datasets cora citeseer # subset
```
