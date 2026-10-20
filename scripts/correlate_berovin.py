#!/usr/bin/env python
"""Correlate AFToolkit predictions with the berovin experimental measurements.

Experimental values are Table 1 of Burakova, Ivanisenko et al., *Life* 2024,
14, 1508 (doi:10.3390/life14111508): yield of active protein (%), specific
bioluminescence L_specific, Ca2+-independent luminescence L_Ca-free, and the
apparent dissociation constant K_D. None of these is a ddG, so we report
rank (Spearman) and linear (Pearson) correlation of our predicted quantities
(adapter ddG, mean delta-pLDDT at the mutated sites) against each.

Usage:
    python scripts/correlate_berovin.py --reports-dir results/berovin_table1
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

# Experimental measurements, keyed by the frozenset of point mutations.
# (yield_pct, L_specific x1e9, L_Ca_free x1e-8, K_D_uM or None)
EXPERIMENT = {
    frozenset({"K90A"}):                       (10, 23.0, 17, None),
    frozenset({"K90M"}):                       (14, 0.09, 300, 2.61),
    frozenset({"N107S"}):                      (13, 1.4, 71, None),
    frozenset({"K90A", "N107S"}):              (8, 1.3, 480, 1.30),
    frozenset({"K90M", "N107S"}):              (10, 6.0, 58, None),
    frozenset({"K90A", "W103F", "N107S"}):     (8, 0.54, 37, None),
    frozenset({"K90M", "W103F", "N107S"}):     (47, 7.0, 8.1, 6.62),
    frozenset({"K90A", "W103F", "N107V"}):     (7, 1.1, 77, None),
}


def load_predictions(reports_dir):
    """Return {frozenset(mutations): {svm, ensemble, mean_dplddt}}."""
    out = {}
    for path in glob.glob(os.path.join(reports_dir, "*.json")):
        d = json.load(open(path))
        key = frozenset(d["input"]["mutations"])
        svm = [v for k, v in d["predicted_ddg"]["per_model"].items() if "svm" in k]
        deltas = list(d["confidence"]["change"]["delta_per_site_plddt"].values())
        out[key] = {
            "svm": svm[0] if svm else np.nan,
            "ensemble": d["predicted_ddg"]["ensemble_mean"],
            "mean_dplddt": float(np.mean(deltas)) if deltas else np.nan,
        }
    return out


def report(name, pred, exp):
    pred, exp = np.asarray(pred), np.asarray(exp)
    rs = spearmanr(pred, exp).statistic
    rp = pearsonr(pred, exp).statistic
    print(f"  {name:<28}n={len(pred):<3} Spearman={rs:+.3f}  Pearson={rp:+.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports-dir", default="results/berovin_table1")
    args = p.parse_args()

    preds = load_predictions(args.reports_dir)
    keys = [k for k in EXPERIMENT if k in preds]
    print(f"Matched {len(keys)} mutants with both predictions and experiment.\n")

    svm = [preds[k]["svm"] for k in keys]
    ens = [preds[k]["ensemble"] for k in keys]
    dpl = [preds[k]["mean_dplddt"] for k in keys]
    yld = [EXPERIMENT[k][0] for k in keys]
    lspec = [np.log10(EXPERIMENT[k][1]) for k in keys]   # spans orders of magnitude
    lcaf = [np.log10(EXPERIMENT[k][2]) for k in keys]

    for exp_name, exp_vals in [("Yield active protein %", yld),
                               ("log10 L_specific", lspec),
                               ("log10 L_Ca-free", lcaf)]:
        print(f"vs {exp_name}:")
        report("predicted ddG (SVM)", svm, exp_vals)
        report("predicted ddG (ensemble)", ens, exp_vals)
        report("mean delta-pLDDT@site", dpl, exp_vals)
        print()

    # K_D (free-energy-like) for the few mutants that have it.
    kd_keys = [k for k in keys if EXPERIMENT[k][3] is not None]
    if len(kd_keys) >= 3:
        print(f"vs K_D (n={len(kd_keys)}):")
        report("predicted ddG (SVM)", [preds[k]["svm"] for k in kd_keys],
               [EXPERIMENT[k][3] for k in kd_keys])
        report("mean delta-pLDDT@site", [preds[k]["mean_dplddt"] for k in kd_keys],
               [EXPERIMENT[k][3] for k in kd_keys])


if __name__ == "__main__":
    main()
