# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

"""Generate or download pre-computed 60/20/20 train/val/test splits
for datasets that use NPZ file splits.

Two sources are supported:

canonical (default)
    Downloads the official Geom-GCN (Pei et al. 2020) splits from GitHub.
    These are the same splits used in the original NSD paper and most
    heterophily benchmarks. 10 folds per dataset.

generate
    Creates 10 stratified splits locally using StratifiedShuffleSplit.
    Useful as a fallback when offline or for datasets not covered by
    the canonical repository.

Files are saved to exp/splits/ as:
    exp/splits/{name}_split_0.6_0.2_{fold}.npz

Each .npz contains three 1-D boolean arrays of length N:
    train_mask, val_mask, test_mask

Usage
-----
    # Download canonical Geom-GCN splits (default):
    python -m exp.gen_splits
    python -m exp.gen_splits --datasets cora citeseer texas

    # Generate local stratified splits instead:
    python -m exp.gen_splits --source generate

    # Custom data root or output directory:
    python -m exp.gen_splits --root /data/pyg --splits-dir /data/splits

    # Different number of folds (default: 10):
    python -m exp.gen_splits --folds 5

    # Re-download/re-generate even if files already exist:
    python -m exp.gen_splits --overwrite
"""

from __future__ import annotations

import argparse
import os
import urllib.request

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from exp.data import load_dataset
from exp.registries import dataset_registry


def _npz_split_datasets() -> frozenset[str]:
    return frozenset(
        name
        for name in dataset_registry.list_keys()
        if dataset_registry.get(name).split_type == "npz_file"
    )


_DEFAULT_SPLITS_DIR = os.path.join(os.path.dirname(__file__), "splits")
_N_FOLDS = 10
_TRAIN_RATIO = 0.6
_VAL_RATIO = 0.2  # test = 0.2 follows automatically

# Canonical Geom-GCN splits repository (Pei et al. 2020, ICLR).
# Each file is a .npz with train_mask / val_mask / test_mask boolean arrays.
_GEOM_GCN_BASE = "https://github.com/graphdml-uiuc-jlu/geom-gcn/raw/master/splits"


def _canonical_url(name: str, fold: int) -> str:
    return f"{_GEOM_GCN_BASE}/{name}_split_0.6_0.2_{fold}.npz"


def download_canonical_splits(
    name: str,
    splits_dir: str = _DEFAULT_SPLITS_DIR,
    n_folds: int = _N_FOLDS,
    overwrite: bool = False,
) -> None:
    """Download the official Geom-GCN splits for *name* from GitHub.
    Skips any fold whose file already exists unless *overwrite* is True.
    """
    print(f"[{name}] Downloading canonical Geom-GCN splits...")

    # Ensure the local split cache exists before writing per-fold .npz files.
    os.makedirs(splits_dir, exist_ok=True)

    for fold in range(n_folds):
        # Keep the filename convention expected by exp/splits.py.
        out_path = os.path.join(splits_dir, f"{name}_split_0.6_0.2_{fold}.npz")

        # Avoid re-downloading existing folds unless the caller explicitly asks.
        if os.path.exists(out_path) and not overwrite:
            print(f"  fold {fold:2d}: already exists, skipping.")
            continue

        url = _canonical_url(name, fold)
        try:
            urllib.request.urlretrieve(url, out_path)

            # Load the saved file immediately to verify the masks and report sizes.
            arr = np.load(out_path)
            train_n = int(arr["train_mask"].sum())
            val_n = int(arr["val_mask"].sum())
            test_n = int(arr["test_mask"].sum())
            print(
                f"  fold {fold:2d}: train={train_n:4d}  val={val_n:4d}  "
                f"test={test_n:4d}  <- {url}"
            )
        except Exception as exc:
            # Remove partial file if the download failed.
            if os.path.exists(out_path):
                os.remove(out_path)
            raise RuntimeError(
                f"Failed to download {url}: {exc}\n"
                "Tip: run with --source generate to create splits locally."
            ) from exc

    print(f"[{name}] Done.\n")


def generate_splits(
    name: str,
    root: str = "exp/data",
    splits_dir: str = _DEFAULT_SPLITS_DIR,
    n_folds: int = _N_FOLDS,
    overwrite: bool = False,
) -> None:
    """Generate and save n_folds stratified 60/20/20 splits for *name*.
    Skips any fold whose file already exists unless *overwrite* is True.
    """
    print(f"[{name}] Loading dataset...")

    # Load labels once; only labels are needed to create stratified masks.
    data, info = load_dataset(name, root=root)
    assert isinstance(data.y, torch.Tensor)
    labels = data.y.numpy()
    n = int(labels.shape[0])

    # Ensure the output directory exists before writing any fold files.
    os.makedirs(splits_dir, exist_ok=True)

    for fold in range(n_folds):
        # Use the same filename convention as the downloaded canonical splits.
        out_path = os.path.join(splits_dir, f"{name}_split_0.6_0.2_{fold}.npz")

        # Preserve existing splits unless the caller requests regeneration.
        if os.path.exists(out_path) and not overwrite:
            print(f"  fold {fold:2d}: already exists, skipping.")
            continue

        # Step 1: 60 % train vs 40 % rest (stratified by class label).
        sss1 = StratifiedShuffleSplit(
            n_splits=1, train_size=_TRAIN_RATIO, random_state=fold
        )
        train_idx, rest_idx = next(sss1.split(np.zeros(n), labels))

        # Step 2: split the remaining 40 % equally into val (20 %) and test (20 %).
        val_of_rest = _VAL_RATIO / (1.0 - _TRAIN_RATIO)  # 0.5
        sss2 = StratifiedShuffleSplit(
            n_splits=1, train_size=val_of_rest, random_state=fold
        )
        val_local, test_local = next(
            sss2.split(np.zeros(len(rest_idx)), labels[rest_idx])
        )
        val_idx = rest_idx[val_local]
        test_idx = rest_idx[test_local]

        # Convert index splits into boolean masks expected by exp/splits.py.
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # Save masks under the exact keys consumed by _apply_npz_split.
        np.savez(
            out_path, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
        )
        print(
            f"  fold {fold:2d}: train={train_mask.sum():4d}  "
            f"val={val_mask.sum():4d}  test={test_mask.sum():4d}  -> {out_path}"
        )

    print(f"[{name}] Done.\n")


def main() -> None:
    # Small CLI for either downloading official splits or generating local ones.
    parser = argparse.ArgumentParser(
        description="Download or generate 60/20/20 splits for NPZ-split datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(_npz_split_datasets()),
        help="Datasets to process (default: all datasets with split_type='npz_file').",
    )
    parser.add_argument(
        "--source",
        choices=["canonical", "generate"],
        default="canonical",
        help=(
            "canonical: download official Geom-GCN splits from GitHub (default). "
            "generate: create stratified splits locally."
        ),
    )
    parser.add_argument(
        "--root",
        default="exp/data",
        help="PyG data root directory (default: exp/data). Only used with \
        --source generate.",
    )
    parser.add_argument(
        "--splits-dir",
        default=_DEFAULT_SPLITS_DIR,
        help=f"Output directory for .npz files (default: {_DEFAULT_SPLITS_DIR}).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=_N_FOLDS,
        help=f"Number of folds to process (default: {_N_FOLDS}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download/re-generate split files that already exist.",
    )
    args = parser.parse_args()

    # Create the destination once so individual dataset handlers can assume it exists.
    os.makedirs(args.splits_dir, exist_ok=True)

    for name in args.datasets:
        # Only datasets configured for .npz splits should be processed by this script.
        if name not in _npz_split_datasets():
            print(f"Warning: '{name}' does not use NPZ splits -- skipping.")
            continue

        # Choose between official Geom-GCN splits and locally generated stratified
        # splits.
        if args.source == "canonical":
            download_canonical_splits(
                name,
                splits_dir=args.splits_dir,
                n_folds=args.folds,
                overwrite=args.overwrite,
            )
        else:
            generate_splits(
                name,
                root=args.root,
                splits_dir=args.splits_dir,
                n_folds=args.folds,
                overwrite=args.overwrite,
            )

    print("All requested splits processed.")


if __name__ == "__main__":
    main()
