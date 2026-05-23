# Installation

`sheaf_mpnn` requires **Python >= 3.13** and uses [uv](https://github.com/astral-sh/uv) for dependency management.

## Base environment

```bash
git clone https://github.com/alessioborgi/pytorch-SheafNeuralNetworks
cd pytorch-SheafNeuralNetworks
uv sync
```

This installs both `sheaf_mpnn` (the core library) and `exp` (the experiment runner). Activate the venv, then verify:

```bash
python -c "import sheaf_mpnn; print(sheaf_mpnn.__version__)"
```

## Optional extras

| Extra        | Command                              | Provides                                              |
|--------------|--------------------------------------|-------------------------------------------------------|
| `wandb`      | `uv sync --extra wandb`              | W&B logger + Optuna-W&B integration                   |
| `dev` group  | `uv sync --all-extras --dev`         | tests, ruff, mypy, pre-commit                         |
| `docs` group | `uv sync --group docs`               | Sphinx, pydata-sphinx-theme, MyST, autodoc extensions |

The `docs` group is what CI uses to build this site; see
[the docs CI workflow](https://github.com/alessioborgi/pytorch-SheafNeuralNetworks/blob/main/.github/workflows/docs.yml).
