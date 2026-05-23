# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

"""PyTorch Lightning module wrapping Sheaf models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from sklearn.metrics import roc_auc_score

from exp.config import Config
from exp.data import DatasetInfo
from exp.registries.models import model_registry


class SheafLightningModule(LightningModule):
    """Wraps Sheaf models with Lightning training / evaluation logic."""

    def __init__(self, cfg: Config, info: DatasetInfo) -> None:
        super().__init__()
        self.cfg = cfg
        self.info = info
        try:
            self.model: Any = model_registry.build(
                str(cfg.model.type),
                info.num_features,
                info.num_classes,
                cfg.model,
                cfg.reg,
            )
        except KeyError as exc:
            raise ValueError(f"Unknown model type: {cfg.model.type!r}") from exc

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        data = batch
        logits = self.model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        self._eval_step(batch, "val_mask", "val")

    def test_step(self, batch, batch_idx) -> None:
        self._eval_step(batch, "test_mask", "test")

    # ------------------------------------------------------------------
    # Shared evaluation
    # ------------------------------------------------------------------

    def _eval_step(self, data, mask_attr: str, prefix: str) -> None:
        # Evaluation uses the model's standard forward pass, without training dropout.
        logits = self.model(data.x, data.edge_index)

        # If the model diverges numerically, log a sentinel loss/metric and
        # skip scoring.
        # Keep validation metrics visible in the progress bar; test metrics
        # are logged only.
        prog = prefix == "val"

        if not torch.isfinite(logits).all():
            bad = 0.0 if self.info.metric == "acc" else 0.5
            self.log(
                f"{prefix}_loss",
                torch.tensor(float("inf")),
                on_step=False,
                on_epoch=True,
                prog_bar=prog,
            )
            self.log(
                f"{prefix}_{self.info.metric}",
                torch.tensor(bad),
                on_step=False,
                on_epoch=True,
                prog_bar=prog,
            )
            return

        # Select the requested split mask, then compute loss and metric on
        # that split only.
        mask = getattr(data, mask_attr)
        loss = F.cross_entropy(logits[mask], data.y[mask])
        metric = self._compute_metric(logits, data.y, mask)

        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=prog)
        self.log(
            f"{prefix}_{self.info.metric}",
            metric,
            on_step=False,
            on_epoch=True,
            prog_bar=prog,
        )

    def _compute_metric(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        # Accuracy is the fraction of correct class predictions on the masked nodes.
        if self.info.metric == "acc":
            pred = logits[mask].argmax(dim=-1)
            return float(pred.eq(labels[mask]).sum().item()) / int(mask.sum().item())

        # AUC is computed on CPU probabilities for sklearn compatibility.
        probs = F.softmax(logits[mask], dim=-1).detach().cpu()
        y_true = labels[mask].detach().cpu().numpy()

        # roc_auc_score is undefined when the split contains only one class.
        if np.unique(y_true).size < 2:
            return 0.5

        # Binary AUC uses the positive-class probability; multiclass uses one-vs-rest.
        if probs.size(1) == 2:
            return float(roc_auc_score(y_true, probs[:, 1].numpy()))
        return float(
            roc_auc_score(
                y_true,
                probs.numpy(),
                multi_class="ovr",
                labels=np.arange(probs.size(1)),
            )
        )

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):  # noqa: ANN201
        return torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
        )
