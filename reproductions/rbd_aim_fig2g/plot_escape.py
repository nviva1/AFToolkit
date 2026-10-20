#!/usr/bin/env python
"""ΔiPAE vs ΔlogK_D for whatever escape variants are computed so far.
Grid: rows = {chains, mutantpos}, cols = {c0, c1, c2}. One point per escape variant,
coloured by antibody. Pearson R per panel. Reads the (possibly partial) results CSV.

Out: reproductions/rbd_aim_fig2g/escape_dipae.png
"""
import os, re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data/rbdaim/escape_ipae_results.csv")
OUT = os.path.join(HERE, "escape_dipae_" + re.sub(r"\.csv$", "", os.path.basename(SRC)).replace("escape_ipae_results", "").strip("_") + ".png").replace("__", "_").replace("_.png", ".png")
if os.path.basename(SRC) == "escape_ipae_results.csv":
    OUT = os.path.join(HERE, "escape_dipae.png")


def main():
    d = pd.read_csv(SRC)
    abs_ = sorted(d.antibody.unique())
    cmap = plt.get_cmap("tab20")
    color = {a: cmap(i % 20) for i, a in enumerate(abs_)}
    metrics = [("dCHAINS", "ΔiPAE chains (RBD×Ab)"), ("dMUTPOS", "ΔiPAE mutant-pos×Ab")]
    cyc = sorted(int(m.group(1)) for c in d.columns
                 for m in [re.match(r"dCHAINS_c(\d+)$", c)] if m)

    fig, axes = plt.subplots(2, len(cyc), figsize=(5 * len(cyc), 9), sharex=True, squeeze=False)
    for ri, (mkey, mlab) in enumerate(metrics):
        for ci, k in enumerate(cyc):
            ax = axes[ri, ci]
            col = f"{mkey}_c{k}"
            v = d[d[col].notna()]
            for a in abs_:
                s = v[v.antibody == a]
                if len(s):
                    ax.scatter(s.dlogKD, s[col], s=45, color=color[a],
                               edgecolor="k", linewidth=0.3, label=a)
            if len(v) > 2:
                r, _ = pearsonr(v.dlogKD, v[col])
                b, a0 = np.polyfit(v.dlogKD, v[col], 1)
                xs = np.array([v.dlogKD.min(), v.dlogKD.max()])
                ax.plot(xs, a0 + b * xs, "k--", lw=1.2)
                ax.set_title(f"{mlab}  c{k}   (R={r:+.2f}, n={len(v)})", fontsize=10)
            else:
                ax.set_title(f"{mlab}  c{k}", fontsize=10)
            ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
            ax.grid(alpha=0.25)
            if ci == 0:
                ax.set_ylabel(r"$\Delta$iPAE (variant - reference)")
            if ri == 1:
                ax.set_xlabel(r"$\Delta\log_{10}K_D$ (escape)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7, frameon=False,
               title="antibody", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"Fig 2g escape: ΔiPAE vs ΔlogK_D  ({len(d)} variants, "
                 f"{d.antibody.nunique()} antibodies so far)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.9, 0.97])
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}  ({len(d)} variants, {d.antibody.nunique()} antibodies)")


if __name__ == "__main__":
    main()
