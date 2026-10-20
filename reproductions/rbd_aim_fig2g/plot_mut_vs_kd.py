#!/usr/bin/env python
"""Scatter: number of RBD mutations vs K_D change (delta_log_kd, escape magnitude),
plotted two ways on the same axes:
  * mutations counted relative to the per-antibody REFERENCE (highest-affinity) variant
  * mutations counted relative to WT (Wuhan)
Each point is one (reference -> escape variant) comparison. A trend line + Pearson R
is drawn for each series.

In : data/rbdaim/escape_pairs.csv, data/rbdaim/kd_absolute_long.csv
Out: reproductions/rbd_aim_fig2g/mut_vs_kd.png
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mut_vs_kd.png")


def fit(ax, x, y, color, label):
    x = np.asarray(x, float); y = np.asarray(y, float)
    r, p = pearsonr(x, y) if len(x) > 2 else (float("nan"), float("nan"))
    ax.scatter(x, y, s=42, alpha=0.7, color=color,
               edgecolor="k", linewidth=0.4, label=f"{label}  (R={r:+.2f}, n={len(x)})")
    if len(x) > 2:
        b, a = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, a + b * xs, color=color, lw=1.6, ls="--")
    return r


def main():
    pairs = pd.read_csv(os.path.join(ROOT, "data/rbdaim/escape_pairs.csv"))
    long = pd.read_csv(os.path.join(ROOT, "data/rbdaim/kd_absolute_long.csv"))
    # variant's mutation count vs WT (Wuhan)
    wt = long[["antibody", "strain", "n_mut"]].rename(columns={"n_mut": "n_mut_vs_WT"})
    df = pairs.merge(wt, left_on=["antibody", "var_strain"], right_on=["antibody", "strain"], how="left")
    # drop cross-species / unaligned (n_mut == -1, n_escape_mut == -1)
    df = df[(df.n_escape_mut >= 0) & (df.n_mut_vs_WT >= 0)].copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    fit(ax, df.n_escape_mut, df.delta_log_kd, "#1f77b4", "mutations vs REFERENCE")
    fit(ax, df.n_mut_vs_WT, df.delta_log_kd, "#ff7f0e", "mutations vs WT (Wuhan)")
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_xlabel("number of RBD mutations")
    ax.set_ylabel(r"$\Delta\log_{10}K_D$  (escape variant vs reference)")
    ax.set_title(f"RBD mutation count vs affinity loss  (n={len(df)} escape pairs)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}  ({len(df)} pairs)")


if __name__ == "__main__":
    main()
