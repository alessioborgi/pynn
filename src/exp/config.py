# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

"""Typed configuration dataclasses for the NSD benchmark runner.

Consumed by ``tyro.cli`` in ``exp/run.py``.  Per-dataset preset defaults live
in ``exp/presets.py`` and are injected via ``tyro.cli(Config, default=...)``,
so every field remains overridable from the command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Literal


class ModelType(StrEnum):
    NSD = auto()


class ModelVariant(StrEnum):
    DIAGONAL = auto()
    GENERAL = auto()
    ORTHOGONAL = auto()
    GENERAL_ATTENTION = auto()
    ORTHOGONAL_ATTENTION = auto()


@dataclass
class DatasetConfig:
    """Dataset identity and storage location."""

    name: str = "cora"
    root: str = "exp/data"


@dataclass
class ModelConfig:
    """Sheaf architecture hyperparameters."""

    type: ModelType = ModelType.NSD
    variant: Literal[
        "diagonal",
        "general",
        "orthogonal",
        "low_rank",
        "general_attention",
        "orthogonal_attention",
    ] = "general"
    stalk_dim: int = 4
    hidden_dim: int = 16
    num_layers: int = 2
    alpha: float = 1.0
    rank: int = 1
    orth_strategy: Literal["cayley", "fasth"] = "cayley"


@dataclass
class RegConfig:
    """Regularisation hyperparameters."""

    input_dropout: float = 0.0
    dropout: float = 0.0


@dataclass
class OptimConfig:
    """Optimisation schedule."""

    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 1000
    early_stopping: int = 200
    stop_strategy: Literal["loss", "acc"] = "loss"


@dataclass
class CVConfig:
    """Cross-validation setup."""

    n_folds: int = 10
    seed: int = 42
    min_acc: float = 0.0


@dataclass
class HardwareConfig:
    """Hardware selection."""

    cuda: int = 0


@dataclass
class WandBConfig:
    """Weights & Biases integration."""

    enabled: bool = False
    entity: str | None = None
    project: str | None = None


@dataclass
class Config:
    """Root configuration for a benchmark run."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reg: RegConfig = field(default_factory=RegConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
