# Installation

`sheaf_mpnn` targets **Python ≥ 3.13** and uses [uv](https://github.com/astral-sh/uv)
for dependency management.

## Base environment

```bash
git clone https://github.com/alessioborgi/pytorch-SheafNeuralNetworks
cd pytorch-SheafNeuralNetworks
uv sync
```

This installs both `sheaf_mpnn` (the core library) and `exp` (the
experiment runner), so `python -m exp.run …` works out of the box once
the venv is activated.

## Optional extras

| Extra        | Command                              | Provides                                  |
|--------------|--------------------------------------|-------------------------------------------|
| `wandb`      | `uv sync --extra wandb`              | W&B logger + Optuna–W&B integration       |
| `dev` group  | `uv sync --all-extras --dev`         | tests, ruff, mypy, pre-commit             |
| `docs` group | `uv sync --group docs`               | Sphinx, Furo, MyST, autodoc extensions    |

The `docs` group is what CI uses to build this site — see
[the docs CI workflow](https://github.com/alessioborgi/pytorch-SheafNeuralNetworks/blob/main/.github/workflows/docs.yml).
