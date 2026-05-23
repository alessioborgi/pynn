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

## Neural Sheaf Diffusion

A cellular sheaf assigns a stalk $\mathbb{R}^d$ to each node and a restriction map $\rho_{v \leftarrow e} : \mathbb{R}^d \to \mathbb{R}^d$ to each edge endpoint. The **sheaf Laplacian** is

$$L_F = B^\top \left(\bigoplus_{e \in \mathcal{E}} \begin{bmatrix} \rho_{u \leftarrow e}^\top \rho_{u \leftarrow e} & -\rho_{u \leftarrow e}^\top \rho_{v \leftarrow e} \\ -\rho_{v \leftarrow e}^\top \rho_{u \leftarrow e} & \rho_{v \leftarrow e}^\top \rho_{v \leftarrow e} \end{bmatrix} \right) B$$

where $B$ is the node-edge incidence matrix. The standard graph Laplacian is the special case where every restriction map is the identity. By learning $\rho_{v \leftarrow e}$ per edge, NSD can encode heterophilic structure that scalar-Laplacian methods cannot represent.

The NSD diffusion update is $\mathbf{H}^{(t+1)} = \sigma\!\left(\left(I - \sigma_\Delta(X;\theta)\, \tilde{L}_F\right)\mathbf{H}^{(t)} W\right)$, where $\tilde{L}_F$ is the normalised sheaf Laplacian and $\sigma_\Delta$ predicts the per-edge restriction maps from node features.

`sheaf_mpnn` implements this model family (Bodnar et al., 2022) as PyTorch Geometric message-passing layers, paired with an experiment runner (`exp`) using PyTorch Lightning, `tyro`, and Optuna for 10-fold CV and hyperparameter sweeps on standard node-classification benchmarks.

::::{grid} 3
:gutter: 2

:::{grid-item-card} Diagonal
{py:class}`sheaf_mpnn.nsd.DiagonalNSDConv`

Restriction maps are diagonal $d \times d$ matrices, $d$ parameters per edge endpoint. Lowest capacity; fastest training.
:::

:::{grid-item-card} General
{py:class}`sheaf_mpnn.nsd.GeneralNSDConv`

Unrestricted $d \times d$ maps, $d^2$ parameters per edge endpoint. Maximum expressivity; can represent any linear relationship between stalks.
:::

:::{grid-item-card} Orthogonal
{py:class}`sheaf_mpnn.nsd.OrthogonalNSDConv`

Orthogonal maps parameterised via Cayley transform of $\tfrac{d(d-1)}{2}$ scalars, $O(d)$ parameters, norm-preserving diffusion.
:::
::::

See the [API reference](api/sheaf_mpnn) for full details.

## Indices

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
