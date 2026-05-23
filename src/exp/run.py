# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

"""Neural Sheaf Diffusion - 10-fold cross-validation experiment runner.

Usage
-----
    # Run with a preset (populates all defaults from best-known config):
    python -m exp.run --preset cora
    python -m exp.run --preset texas --model.num_layers 3  # override one field

    # Fully manual (no preset):
    python -m exp.run --dataset.name cora --model.variant general --model.stalk_dim 4

    # With Weights & Biases:
    python -m exp.run --preset cora --wandb.enabled --wandb.entity <you>
"""

from __future__ import annotations

import dataclasses
import logging
import random
import sys
import tempfile
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.pytorch.loggers import WandbLogger as _WandbLogger

import numpy as np
import torch
import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities.model_summary import ModelSummary
from tqdm import tqdm

from exp.config import Config
from exp.data import DatasetInfo, SheafDataModule
from exp.module import SheafLightningModule
from exp.registries.presets import preset_registry
from sheaf_mpnn.utils import setup_torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_config() -> Config:
    """Handle ``--preset <name>`` before handing the rest to tyro.

    The preset is stripped from ``sys.argv`` and used to set the ``default``
    argument of ``tyro.cli``, so every field can still be overridden.
    """
    argv = sys.argv[1:]
    preset_name: str | None = None
    clean: list[str] = []

    # Walk through the raw CLI arguments and remove only the custom --preset flag.
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--preset":
            # Support "--preset name" by reading the following argument as the preset.
            if i + 1 < len(argv):
                preset_name = argv[i + 1]
                i += 2
            else:
                i += 1
        elif a.startswith("--preset="):
            # Support "--preset=name" as a compact equivalent form.
            preset_name = a.split("=", 1)[1]
            i += 1
        else:
            # Keep every normal Config override for tyro to parse later.
            clean.append(a)
            i += 1

    # Give tyro the cleaned arguments, using the preset as its default config.
    sys.argv = [sys.argv[0]] + clean
    if preset_name is not None:
        if preset_name not in preset_registry:
            known = sorted(preset_registry.list_keys())
            raise SystemExit(
                f"Unknown preset {preset_name!r}. "
                f"Run with --help to see available presets.\n"
                f"Known: {', '.join(known)}"
            )
        default: Config | None = preset_registry.get(preset_name)
    else:
        default = None
    result = tyro.cli(Config, default=default)
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------


def _make_logger(
    cfg: Config, info: DatasetInfo, fold: int, run_name: str
) -> _WandbLogger | bool:
    if not cfg.wandb.enabled:
        return False
    from lightning.pytorch.loggers import WandbLogger  # lazy - wandb is optional

    return WandbLogger(
        project=cfg.wandb.project or f"nsd-{info.name}",
        entity=cfg.wandb.entity,
        name=f"{run_name}-fold{fold}",
        group=run_name,
        config=dataclasses.asdict(cfg),
    )


def _model_label(cfg: Config) -> str:
    """Short identifier for logging and W&B run names."""
    return f"{cfg.model.type}-{cfg.model.variant}"


def _run_fold(
    cfg: Config,
    info: DatasetInfo,
    fold: int,
    monitor: str,
    ckpt_mode: str,
) -> float:
    """Train and evaluate one cross-validation fold; return the test metric."""
    run_name = (
        f"{_model_label(cfg)}-d{cfg.model.stalk_dim}"
        f"-h{cfg.model.hidden_dim}-L{cfg.model.num_layers}"
    )
    # Build the fold-specific data module and a fresh model for this split.
    dm = SheafDataModule(cfg.dataset.name, root=cfg.dataset.root, fold=fold)
    module = SheafLightningModule(cfg, info)
    logger = _make_logger(cfg, info, fold, run_name)

    # Store checkpoints in a temporary directory; only the best fold checkpoint
    # is needed.
    with tempfile.TemporaryDirectory() as ckpt_dir:
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=monitor,
            mode=ckpt_mode,
            save_top_k=1,
            filename="best",
        )

        # Train with early stopping on the chosen validation metric and hardware target.
        trainer = Trainer(
            max_epochs=cfg.optim.epochs,
            callbacks=[
                EarlyStopping(
                    monitor=monitor,
                    patience=cfg.optim.early_stopping,
                    mode=ckpt_mode,
                ),
                ckpt_cb,
            ],
            logger=logger,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=[cfg.hardware.cuda] if torch.cuda.is_available() else "auto",
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
        )

        # Evaluate the best validation checkpoint on the held-out test split.
        trainer.fit(module, dm)
        [test_res] = trainer.test(module, dm, ckpt_path="best", verbose=False)

    # Return the scalar test metric for aggregation across folds.
    return float(test_res.get(f"test_{info.metric}", 0.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _silence_third_party() -> None:
    # Suppress Lightning's per-fold INFO chatter (GPU/TPU/LOCAL_RANK/checkpoint lines).
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    # Suppress numpy VisibleDeprecationWarning from torch_geometric's planetoid loader.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Suppress Lightning's num_workers, batch_size, and _pytree warnings.
    warnings.filterwarnings("ignore", category=UserWarning, module="lightning.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.*")
    # _pytree.py emits a non-UserWarning deprecation; match by message text instead.
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")


def main() -> None:
    """Entry point: parse config, log setup info, run CV, and report results."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _silence_third_party()
    cfg = _parse_config()
    setup_torch(precision="high", seed=cfg.cv.seed)

    # Load dataset metadata once; these properties do not depend on the fold.
    dm_meta = SheafDataModule(cfg.dataset.name, root=cfg.dataset.root, fold=0)
    dm_meta.setup()
    info = dm_meta.info
    n_folds = min(cfg.cv.n_folds, info.num_splits)

    # Compute graph-level stats once (cheap after setup()).
    n_train, n_val, n_test = dm_meta.split_sizes
    avg_deg = (dm_meta.num_edges * 2) / dm_meta.num_nodes

    sep = "-" * 60
    log.info(sep)
    log.info(
        "Dataset  %s N=%s  E=%s  avg_deg=%.1f",
        f"{info.name:<16}",
        f"{dm_meta.num_nodes:,}",
        f"{dm_meta.num_edges:,}",
        avg_deg,
    )
    log.info(
        "         F=%d  C=%d  homophily=%.3f  metric=%s",
        info.num_features,
        info.num_classes,
        dm_meta.homophily,
        info.metric,
    )
    log.info(
        "Split    train=%s  val=%s  test=%s  (%d-fold CV)",
        f"{n_train:,}",
        f"{n_val:,}",
        f"{n_test:,}",
        n_folds,
    )
    log.info(
        "Model    %s-%s  d=%d  hidden=%d  layers=%d  alpha=%s  dropout=%s",
        cfg.model.type.upper(),
        cfg.model.variant,
        cfg.model.stalk_dim,
        cfg.model.hidden_dim,
        cfg.model.num_layers,
        cfg.model.alpha,
        cfg.reg.dropout,
    )
    # setup_torch already printed GPU/CPU info
    log.info(sep)
    log.info("%s", ModelSummary(SheafLightningModule(cfg, info), max_depth=1))
    log.info(sep)

    monitor = "val_loss" if cfg.optim.stop_strategy == "loss" else f"val_{info.metric}"
    ckpt_mode = "min" if cfg.optim.stop_strategy == "loss" else "max"

    results: list[float] = []

    # Run each cross-validation split with a fold-specific seed for reproducibility.
    for fold in tqdm(range(n_folds), desc="CV folds", unit="fold"):
        _set_seed(cfg.cv.seed + fold)
        test_metric = _run_fold(cfg, info, fold, monitor, ckpt_mode)
        results.append(test_metric)
        log.info("  fold %2d | test %s: %.4f", fold, info.metric, test_metric)

        # Optional quick-fail guard for clearly broken accuracy runs.
        if fold == 0 and info.metric == "acc" and test_metric < cfg.cv.min_acc:
            log.warning(
                "  fold 0 test %s=%.4f < min_acc=%.4f; aborting run.",
                info.metric,
                test_metric,
                cfg.cv.min_acc,
            )
            break

    # Aggregate fold metrics and present accuracy as a percentage.
    arr = np.array(results)
    scale = 100.0 if info.metric == "acc" else 1.0
    sep = "-" * 60
    log.info("\n%s", sep)
    log.info("%s on %s [%d folds]", _model_label(cfg), info.name, len(results))
    log.info(
        "Test  %s: %.2f +/- %.2f%s",
        info.metric,
        arr.mean() * scale,
        arr.std() * scale,
        "%" if info.metric == "acc" else "",
    )
    log.info(sep)


if __name__ == "__main__":
    main()
