#!/usr/bin/env python
"""Compare P1 (mutate crystal) vs P2 (WT-superpose + thread) on the global Fig 2g
correlation, by splicing P2's variant-crystal points over P1's.

P1 cycle-1 delta-iPAE comes from the saved per-cycle PAE npz; P2 from
protocol_P2_variants.csv. We report the pooled Pearson for:
  - all-P1
  - P1 with the variant-crystal points replaced by P2
to see whether clean WT-normalisation of variant crystals lifts the correlation.
"""
import os, re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from aftoolkit.scoring import interface_pae

HERE = os.path.dirname(os.path.abspath(__file__))
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
    aff = pd.read_csv(os.path.join(os.path.dirname(HERE), "..", "data/rbdaim/fig2g_affinity.csv"))
    p2 = pd.read_csv(os.path.join(HERE, "protocol_P2_variants.csv"))
    variant_abs = set(p2.ab.unique())

    rows = []
    for _, r in aff.iterrows():
        if abs(r.delta_log_kd) < 1e-6:
            continue
        d1 = p1_dipae(r.ab, r.strain, cyc=1)
        if d1 is None:
            continue
        m = p2[(p2.ab == r.ab) & (p2.strain == r.strain)]
        d2 = float(m.d_c1.iloc[0]) if len(m) else None
        rows.append({"ab": r.ab, "strain": r.strain, "dlogkd": r.delta_log_kd,
                     "p1": d1, "p2": d2, "variant": r.ab in variant_abs})
    df = pd.DataFrame(rows)

    print("Variant-crystal points: P1 vs P2 (cycle 1 delta-iPAE)")
    print(df[df.variant][["ab", "strain", "dlogkd", "p1", "p2"]].round(3).to_string(index=False))

    allp1 = df.p1
    spliced = df.apply(lambda x: x.p2 if (x.variant and pd.notna(x.p2)) else x.p1, axis=1)
    print(f"\nPooled Pearson (n={len(df)}), cycle 1:")
    print(f"  all-P1 (mutate crystal everywhere) : {pearsonr(df.dlogkd, allp1)[0]:+.3f}")
    print(f"  P1 + P2-spliced variant crystals   : {pearsonr(df.dlogkd, spliced)[0]:+.3f}")
    print(f"  Spearman all-P1   : {spearmanr(df.dlogkd, allp1)[0]:+.3f}")
    print(f"  Spearman spliced  : {spearmanr(df.dlogkd, spliced)[0]:+.3f}")


if __name__ == "__main__":
    main()
