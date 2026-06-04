#!/usr/bin/env python
"""Train a monomer stability adapter (SVM / MLP / CatBoost) from a task CSV.

A thin, path-parameterized wrapper over ``train_adapter_monomer`` so training is
reproducible without editing the example ``run_train_*_monomer.py`` scripts.

Example
-------
    python scripts/train_monomer.py \
        --model svm \
        --train-csv data/cdna/cdna_task_train.csv \
        --features-dir data/cdna/features \
        --npy-dir data/cdna/npy
"""

import argparse
import os
import sys

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Ensure the repo root (with train_adapter_monomer.py) is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_adapter_monomer import train_adapter_monomer  # noqa: E402


def build_base_model(name):
    if name == "svm":
        return Pipeline([("scaler", StandardScaler()), ("svm", SVR(cache_size=2000))])
    if name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=500,
                                 early_stopping=True, random_state=42)),
        ])
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=False)
    raise ValueError(f"unknown model {name!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["svm", "mlp", "catboost"], required=True)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--features-dir", required=True,
                   help="Folder with per-mutation ProteinTask .pkl features.")
    p.add_argument("--npy-dir", default="data/cdna/npy",
                   help="Where to cache/load the .npy feature matrices.")
    p.add_argument("--features", nargs="+", default=["pair", "lddt_logits", "plddt"])
    args = p.parse_args()

    npy_dir = os.path.join(args.npy_dir, "+".join(args.features))
    os.makedirs(npy_dir, exist_ok=True)

    train_adapter_monomer(
        base_model=build_base_model(args.model),
        adapter_name=args.model,
        train_dataset_path=args.train_csv,
        input_features=args.features,
        concat_features=True,
        train_dataset_npy_folder=npy_dir,
        protein_tasks_folder=args.features_dir,
        train_mut_types=["ss"],
        protein_aggregation="mutpos",
        multi_aggregation="sum",
    )


if __name__ == "__main__":
    main()
