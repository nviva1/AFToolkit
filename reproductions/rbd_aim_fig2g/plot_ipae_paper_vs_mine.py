#!/usr/bin/env python
"""Scatter: paper delta-iPAE (Fig 2g source data) vs AFToolkit-recomputed delta-iPAE.

Winning protocol: cycle-1 readout, with the 6 variant-crystal points taken from
P2 (WT-superpose + thread) and the rest from P1 (mutate crystal). The paper's
source data has no Ab/strain labels, so points are matched by delta-log10[KD].
The paper's sign convention is opposite ours (escape -> negative delta-iPAE for
them), so our values are negated to compare on the same axis.
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from aftoolkit.scoring import interface_pae

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PAEDIR = os.path.join(HERE, "recomputed/pae")
safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")
aslist = lambda v: str(np.atleast_1d(v).tolist()[0]).split(",")


def p1_dipae(ab, strain, cyc=1):
    def ip(npz):
        z = np.load(npz, allow_pickle=True)
        return interface_pae(z[f"pae_cycle{cyc}"], z["labels"], aslist(z["antigen"]),
                             aslist(z["antibody"]), coords=z["coords"], cutoff=12.0)
    ref = os.path.join(PAEDIR, f"{ab}__WTref.npz")
    p = os.path.join(PAEDIR, f"{ab}__{safe(strain)}.npz")
    if not (os.path.exists(ref) and os.path.exists(p)):
        return None
    return ip(p) - ip(ref)


def main():
    aff = pd.read_csv(os.path.join(ROOT, "data/rbdaim/fig2g_affinity.csv"))
    p2 = pd.read_csv(os.path.join(HERE, "protocol_P2_variants.csv"))
    variant_abs = set(p2.ab.unique())

    # my best delta-iPAE per affinity-changing point (cycle 1; P2 for variant crystals)
    mine = []
    for _, r in aff.iterrows():
        if abs(r.delta_log_kd) < 1e-6:
            continue
        d1 = p1_dipae(r.ab, r.strain, cyc=1)
        if d1 is None:
            continue
        m = p2[(p2.ab == r.ab) & (p2.strain == r.strain)]
        best = float(m.d_c1.iloc[0]) if (r.ab in variant_abs and len(m)) else d1
        mine.append({"ab": r.ab, "strain": r.strain, "dlogkd": r.delta_log_kd, "mine": best})
    mine = pd.DataFrame(mine)

    # paper Fig 2g source data
    pp = pd.read_excel(os.path.join(os.path.expanduser("~"),
                       "Downloads/41467_2026_70553_MOESM9_ESM.xlsx"),
                       sheet_name="Fig 2g", header=1)
    pp = pp.dropna(how="all"); pp.columns = ["dlogkd", "dipae"]
    pp = pp.apply(pd.to_numeric, errors="coerce").dropna()
    pp = pp[pp.dlogkd.abs() > 1e-6].reset_index(drop=True)   # affinity-changing only

    # match each of my points to the nearest unused paper point by delta-log KD
    used = set(); rows = []
    for _, r in mine.iterrows():
        diffs = (pp.dlogkd - r.dlogkd).abs()
        for idx in diffs.sort_values().index:
            if idx in used:
                continue
            if diffs[idx] < 0.03:                # same experimental KD -> exact-ish
                used.add(idx)
                rows.append({**r.to_dict(), "dipae_paper": pp.dipae[idx],
                             "dlogkd_paper": pp.dlogkd[idx]})
            break
    M = pd.DataFrame(rows)
    M["mine_papersign"] = -M["mine"]             # flip to paper convention

    r_all = pearsonr(M.dipae_paper, M.mine_papersign)[0]
    print(f"matched points: {len(M)} / {len(mine)}")
    print(M[["ab", "strain", "dlogkd", "dipae_paper", "mine", "mine_papersign"]].round(3).to_string(index=False))
    print(f"\nPearson(paper delta-iPAE, my delta-iPAE [paper sign]) = {r_all:+.3f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(M.dipae_paper, M.mine_papersign, c=M.dlogkd, cmap="viridis",
                    s=70, edgecolor="k", linewidth=0.5, zorder=3)
    lim = [min(M.dipae_paper.min(), M.mine_papersign.min()) - 0.2,
           max(M.dipae_paper.max(), M.mine_papersign.max()) + 0.2]
    ax.plot(lim, lim, "--", color="grey", lw=1, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("paper $\\Delta$iPAE (Fig 2g source data)")
    ax.set_ylabel("AFToolkit $\\Delta$iPAE (cycle 1, paper sign)")
    ax.set_title(f"iPAE: paper vs AFToolkit  (n={len(M)}, Pearson={r_all:+.2f})")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("$\\Delta$log$_{10}$K$_D$")
    ax.legend(loc="upper left"); fig.tight_layout()
    out = os.path.join(HERE, "ipae_paper_vs_mine.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
