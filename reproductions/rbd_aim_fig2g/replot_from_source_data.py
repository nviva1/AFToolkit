#!/usr/bin/env python
"""Reproduce RBD-AIM Fig 2g from the journal Source Data (exact re-plot).

Reads the 'Fig 2g' sheet of the Nature Communications Source Data file
(Delta log10 K_D vs Delta iPAE), reproduces the scatter + linear fit + 95% CI,
and reports the Pearson correlation -- both on all points and on the
affinity-changing subset (Delta log10 K_D != 0), the latter matching the
paper's reported R = 0.52.

The Source Data file is the journal's supplementary material (not redistributed
here). Download "Source Data" from the article page (doi:10.1038/s41467-026-70553-6)
-- it is ``41467_2026_70553_MOESM9_ESM.xlsx`` -- and pass its path.

Usage:
    python replot_from_source_data.py --source-data ~/Downloads/41467_2026_70553_MOESM9_ESM.xlsx
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr


def load_fig2g(path):
    raw = pd.read_excel(path, sheet_name="Fig 2g", header=None, skiprows=2)
    x = pd.to_numeric(raw[0], errors="coerce")
    y = pd.to_numeric(raw[1], errors="coerce")
    m = x.notna() & y.notna()
    return x[m].to_numpy(), y[m].to_numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-data",
                   default=os.path.expanduser("~/Downloads/41467_2026_70553_MOESM9_ESM.xlsx"),
                   help="Path to the Nature Comms Source Data .xlsx (MOESM9).")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "fig2g_reproduced.png"))
    args = p.parse_args()

    if not os.path.exists(args.source_data):
        raise SystemExit(
            f"Source Data not found at {args.source_data}\n"
            "Download 'Source Data' (MOESM9) from the article and pass --source-data."
        )

    x, y = load_fig2g(args.source_data)
    sub = x != 0  # points with an actual affinity change (as the paper reports)

    r_all, _ = pearsonr(x, y)
    r_sub, _ = pearsonr(x[sub], y[sub])
    rs_sub, _ = spearmanr(x[sub], y[sub])
    print(f"all points        : n={len(x):<3} Pearson={r_all:+.3f}")
    print(f"affinity-changing : n={int(sub.sum()):<3} Pearson={r_sub:+.3f}  "
          f"Spearman={rs_sub:+.3f}   (paper Fig 2g: |R| = 0.52)")

    lr = linregress(x, y)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.scatter(x, y, s=20, color="#2a9d8f", alpha=.8, edgecolor="none")
    xs = np.linspace(x.min(), x.max(), 100)
    fit = lr.slope * xs + lr.intercept
    n = len(x)
    se = np.sqrt(np.sum((y - (lr.slope * x + lr.intercept)) ** 2) / (n - 2))
    ci = 1.96 * se * np.sqrt(1 / n + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.plot(xs, fit, color="#2a9d8f")
    ax.fill_between(xs, fit - ci, fit + ci, color="#2a9d8f", alpha=.2)
    ax.set_xlabel("delta log10 K_D")
    ax.set_ylabel("delta iPAE")
    ax.set_title(f"RBD-AIM Fig 2g (reproduced; subset R={r_sub:.2f})")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
