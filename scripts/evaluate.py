#!/usr/bin/env python
"""Evaluate a trained monomer adapter on a benchmark test set.

Loads the per-mutation features produced by ``aftoolkit task``, runs the
adapter, and reports the same metrics the paper uses (Spearman / Pearson R,
RMSE, ROC-AUC, MCC).

Example
-------
    python scripts/evaluate.py \
        --model data/models/monomer/pair+lddt_logits+plddt/trained_svm_*.pkl \
        --test-csv data/prostata/s669_task.csv \
        --features-dir data/prostata/features_n4
"""

import argparse
import glob
import pickle

from aftoolkit.models.utils import calculate_metrics

# Subset of metrics the paper reports for stability tasks.
_KEYS = ["SpearmanR", "PearsonR", "RMSE", "ROC AUC", "MCC"]


def evaluate(model_path, test_csv, features_dir):
    import pandas as pd  # local import keeps --help torch-free-ish

    with open(model_path, "rb") as f:
        adapter = pickle.load(f)
    df = pd.read_csv(test_csv, index_col=0)
    X, y = adapter.create_npy_dataset(df, features_dir)
    y = y.ravel()
    y_pred = adapter.base_model.predict(X).ravel()
    metrics = calculate_metrics(y_pred, y)
    return metrics, len(df)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Trained adapter .pkl (glob allowed).")
    p.add_argument("--test-csv", required=True)
    p.add_argument("--features-dir", required=True)
    args = p.parse_args()

    model_path = sorted(glob.glob(args.model))[0]
    metrics, n = evaluate(model_path, args.test_csv, args.features_dir)
    print(f"\n{args.test_csv}  (n={n}, model={model_path.split('/')[-1]})")
    for k in _KEYS:
        if k in metrics:
            print(f"  {k:<10} {metrics[k]:.3f}")


if __name__ == "__main__":
    main()
