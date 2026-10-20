#!/usr/bin/env python
"""Benchmark AlphaFold-inference *formats* by their confidence signal.

For each inference format (AFToolkit "mask", vanilla AF2, AF2Rank "gap") we take
the per-residue pLDDT at the mutated site from the wildtype and mutant
``ProteinTask`` features and use the change ``delta_pLDDT = pLDDT_mt - pLDDT_wt``
as a single, adapter-free predictor of the experimental ddG. The format whose
confidence change correlates best with ddG carries the most mutation signal.

This mirrors the paper's "task (1)" comparison (confidence vs. structure), but
scored against ddG and without training any adapter.

Example
-------
    python scripts/bench_confidence_formats.py --test-csv data/cdna/bench_test.csv \\
        --format AFToolkit=data/cdna/features_n4 \\
        --format vanilla=data/cdna/feat_vanilla \\
        --format AF2Rank=data/cdna/feat_af2rank
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _site_plddt(features: dict) -> float:
    """pLDDT at the (single) observed position of a ProteinTask feature dict."""
    return float(np.asarray(next(iter(features.values()))["plddt"]).reshape(-1)[0])


def collect(test_csv: str, features_dir: str):
    """Return (delta_pLDDT array, ddG array) over mutations with features present."""
    df = pd.read_csv(test_csv, index_col=0)
    d_plddt, ddg = [], []
    for _, row in df.iterrows():
        path = os.path.join(features_dir, f"{row['id']}.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            pt = pickle.load(f)
        wt = pt.wildtype.features
        mt = pt.mutant.features
        if not wt or not mt:
            continue
        d_plddt.append(_site_plddt(mt) - _site_plddt(wt))
        ddg.append(float(row["ddg"]))
    return np.array(d_plddt), np.array(ddg)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--format", action="append", default=[], metavar="NAME=DIR",
                   help="Format label and its features dir (repeatable).")
    args = p.parse_args()

    print(f"{'format':<12}{'n':>5}{'Spearman':>10}{'Pearson':>9}   (delta_pLDDT vs ddG)")
    for spec in args.format:
        name, _, fdir = spec.partition("=")
        dpl, ddg = collect(args.test_csv, fdir)
        if len(dpl) < 3:
            print(f"{name:<12}{len(dpl):>5}   (insufficient data)")
            continue
        rs = spearmanr(dpl, ddg).statistic
        rp = pearsonr(dpl, ddg).statistic
        print(f"{name:<12}{len(dpl):>5}{rs:>10.3f}{rp:>9.3f}")


if __name__ == "__main__":
    main()
