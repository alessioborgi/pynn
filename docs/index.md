# sheaf_mpnn

```{toctree}
:caption: Getting started
:maxdepth: 2

getting_started/installation
getting_started/quickstart
```

```{toctree}
:caption: Guides
:maxdepth: 2

guides/experiments
guides/datasets
```

```{toctree}
:caption: API reference
:maxdepth: 2

api/sheaf_mpnn
api/exp
```

```{toctree}
:caption: Project
:maxdepth: 1

changelog
```

## What is this?

`sheaf_mpnn` is a PyTorch / PyTorch Geometric implementation of
**Neural Sheaf Diffusion** message-passing layers — the model family
introduced by Bodnar et al. (2022) — together with an experiment
framework (`exp`) built on PyTorch Lightning, `tyro`, and Optuna for
running 10-fold cross-validation and hyperparameter sweeps on standard
node-classification benchmarks.

The library focuses on three NSD restriction-map variants:

::::{grid} 3
:gutter: 2

:::{grid-item-card} Diagonal
{py:class}`sheaf_mpnn.nsd.DiagonalNSDConv` — `O(d)` parameters per edge.
:::

:::{grid-item-card} General
{py:class}`sheaf_mpnn.nsd.GeneralNSDConv` — full `d × d` maps.
:::

:::{grid-item-card} Orthogonal
{py:class}`sheaf_mpnn.nsd.OrthogonalNSDConv` — `O(d)`-constrained via
Cayley transform of `d(d−1)/2` scalars.
:::
::::

See the [API reference](api/sheaf_mpnn) for full details.

## Indices

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
