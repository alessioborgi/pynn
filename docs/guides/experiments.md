# Experiments

`exp.run` orchestrates 10-fold cross-validation: per-fold seeding, fresh
model and datamodule per fold, Lightning `EarlyStopping` and
`ModelCheckpoint` (best monitored by `val_loss` or `val_<metric>`
depending on `--optim.stop-strategy`), then `Trainer.test` on the best
checkpoint. Results are aggregated as mean ± std over folds.

## Configuration surface

Every field in {py:mod}`exp.config` is exposed as a CLI flag, grouped by
nested dataclass:

* `--dataset.*` — dataset selection, root path, etc.
* `--model.*` — `variant`, `d`, `hidden_dim`, `num_layers`, …
* `--reg.*` — input dropout, dropout, weight decay, …
* `--optim.*` — optimizer, lr, scheduler, stop strategy.
* `--cv.*` — fold count, seed.
* `--hardware.*` — accelerator, precision, num workers.
* `--wandb.*` — project, entity, tags (requires `--extra wandb`).

## Presets

{py:mod}`exp.presets` ships one entry per dataset in the `PRESETS` dict.
Selecting one with `--preset <name>` injects it as the tyro default; any
field can then be overridden on the same command line.

## Sweeps

{py:mod}`exp.sweep` runs an Optuna study with `MedianPruner`. Sweeps
can be parallelised across machines by pointing every worker at the same
SQLite study.
