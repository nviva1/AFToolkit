#!/usr/bin/env python
"""Plot the WT-anchored escape results (signed ΔlogK_D). 2 rows (chains, mutant-pos)
x 3 cols (c0,c1,c2). mutant-pos only shows antibodies whose old reference was WT."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(ROOT, "data/rbdaim/escape_ipae_wt.csv"))
OUT = os.path.join(HERE, "escape_dipae_wt.png")

abs_ = sorted(d.antibody.unique()); cmap = plt.get_cmap("tab20")
color = {a: cmap(i % 20) for i, a in enumerate(abs_)}
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
for ri, mkey in enumerate(["dCHAINS_WT", "dMUTPOS_WT"]):
    for ci, k in enumerate(range(3)):
        ax = axes[ri, ci]; col = f"{mkey}_c{k}"; v = d[d[col].notna()]
        for a in abs_:
            s = v[v.antibody == a]
            if len(s):
                ax.scatter(s.dlogKD_WT, s[col], s=45, color=color[a], edgecolor="k", linewidth=0.3, label=a)
        if len(v) > 2:
            r, _ = pearsonr(v.dlogKD_WT, v[col])
            b, a0 = np.polyfit(v.dlogKD_WT, v[col], 1)
            xs = np.array([v.dlogKD_WT.min(), v.dlogKD_WT.max()])
            ax.plot(xs, a0 + b * xs, "k--", lw=1.2)
            ax.set_title(f"{'chains' if ri==0 else 'mutant-pos'}  c{k}  (R={r:+.2f}, n={len(v)})", fontsize=10)
        ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6); ax.grid(alpha=0.25)
        if ci == 0: ax.set_ylabel(r"$\Delta$iPAE (variant - WT)")
        if ri == 1: ax.set_xlabel(r"$\Delta\log_{10}K_D$ vs WT (signed)")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="center right", fontsize=7, frameon=False, title="antibody", bbox_to_anchor=(1.0, 0.5))
fig.suptitle(f"Fig 2g escape, WT-anchored (signed): {len(d)} points, {d.antibody.nunique()} antibodies (partial)", fontsize=12)
fig.tight_layout(rect=[0, 0, 0.9, 0.97]); fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
