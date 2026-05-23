# Experiments

`exp.run` orchestrates **10-fold cross-validation**: a fresh model and
datamodule are instantiated per fold with a deterministic per-fold seed,
trained to convergence, and evaluated on the held-out test split. Final
performance is reported as

$$
\mu \pm \sigma, \quad \mu = \frac{1}{K}\sum_{k=1}^{K} s_k, \quad
\sigma = \sqrt{\frac{1}{K}\sum_{k=1}^{K}(s_k - \mu)^2}
$$

where $K = 10$ and $s_k$ is the test score (accuracy or ROC-AUC) on fold $k$.

## Stopping strategy

Each fold uses Lightning `EarlyStopping` monitoring either `val_loss`
(default) or `val_<metric>` depending on `--optim.stop-strategy`.
`ModelCheckpoint` saves the epoch with the best monitored value, and
`Trainer.test` is called on that checkpoint — not on the final epoch
weights. This prevents over-optimistic results from early stopping
collateral: the model never sees the test split during training or
validation.

## Configuration surface

Every field in {py:mod}`exp.config` is exposed as a CLI flag, grouped by
nested dataclass:

* `--dataset.*` — dataset name, root path for downloads, split override.
* `--model.*` — `variant` (model family), `d` (stalk dimension), `hidden_dim`,
  `num_layers`, and architecture-specific flags.
* `--reg.*` — input dropout, intermediate dropout, weight decay.
* `--optim.*` — optimizer choice, learning rate, LR scheduler, and
  `stop-strategy` (`loss` or `metric`).
* `--cv.*` — number of folds, global RNG seed.
* `--hardware.*` — accelerator (`cpu`/`gpu`/`auto`), floating-point
  precision, dataloader workers.
* `--wandb.*` — project, entity, run tags; requires `--extra wandb`.

## Presets

{py:mod}`exp.registries.presets` ships one entry per dataset in the `PRESETS` dict,
storing the hyperparameters found by the sweep. Selecting one with
`--preset <name>` injects it as the tyro default; any field can be
overridden on the same command line:

```bash
python -m exp.run --preset cora --model.hidden_dim 128
```

## Concrete example: full run with WandB logging

```bash
python -m exp.run \
    --preset cora \
    --wandb.project my-project \
    --wandb.entity my-team \
    --extra wandb
```

## Sweeps

{py:mod}`exp.sweeps.sweep` runs an Optuna study with `MedianPruner`. At each
reporting step $t$, the pruner computes the median intermediate value
$\tilde{v}(t)$ over all *completed* trials. A running trial is pruned if
its value falls below that median:

$$
\text{prune trial } i \text{ at step } t
\iff v_i(t) < \tilde{v}(t)
$$

This discards underperforming hyperparameter configurations early,
concentrating budget on promising regions of the search space.

Sweeps can be parallelised across machines by pointing every worker at the
same SQLite study file:

```bash
# Machine 1 (creates the study)
python -m exp.sweep --dataset cora --study-name cora_sweep \
    --storage sqlite:///sweeps/cora.db --n-trials 50

# Machine 2 … N (join the same study)
python -m exp.sweep --dataset cora --study-name cora_sweep \
    --storage sqlite:///sweeps/cora.db --n-trials 50
```

Optuna handles concurrent writes to the SQLite backend with file locking;
for larger parallel sweeps a PostgreSQL or MySQL backend is more robust.
