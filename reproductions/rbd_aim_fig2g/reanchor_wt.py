#!/usr/bin/env python
"""Re-anchor the escape results to WT (Wuhan) instead of the best-affinity variant.

Absolute per-state iPAEs are reference-independent, so the CHAINS metric re-anchors
by pure subtraction:  ΔiPAE = iPAE(state) - iPAE(WT),  ΔlogKD = log10(kd/kd_WT).
This makes ΔlogKD SIGNED (escape > 0, gain-of-affinity < 0) and matches Fig 2g.

The MUTANT-POS metric's residue set is reference-dependent: where the old reference
was already WT the stored dMUTPOS is correct (copied); where it was a variant
(510A5, CR3022) the set moved -> flagged NaN (needs a recompute, not a subtraction).

In : data/rbdaim/escape_ipae_results.csv
Out: data/rbdaim/escape_ipae_wt.csv
"""
import math, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data/rbdaim/escape_ipae_results.csv")
OUT = os.path.join(ROOT, "data/rbdaim/escape_ipae_wt.csv")
NCYC = 3


def main():
    d = pd.read_csv(SRC)
    rows = []
    for ab, g in d.groupby("antibody"):
        ref_strain = g.iloc[0].ref_strain
        # WT absolute iPAE (chains, per cycle) + WT kd
        wt_rows = g[g.variant == "WT"]
        if len(wt_rows):                              # WT was one of the variants
            wr = wt_rows.iloc[0]
            wt_chains = {k: wr[f"var_chains_c{k}"] for k in range(NCYC)}
            wt_kd = wr.var_kd_nM
        else:                                         # WT was the (old) reference
            a = g.iloc[0]
            wt_chains = {k: a[f"ref_chains_c{k}"] for k in range(NCYC)}
            wt_kd = a.ref_kd_nM

        # every measured STATE = each variant row, plus the old reference if it isn't WT
        states = [(r.variant, r.var_kd_nM,
                   {k: r[f"var_chains_c{k}"] for k in range(NCYC)},
                   {k: (r[f"dMUTPOS_c{k}"] if ref_strain == "WT" else np.nan) for k in range(NCYC)})
                  for _, r in g.iterrows()]
        if ref_strain != "WT":
            a = g.iloc[0]
            states.append((ref_strain, a.ref_kd_nM,
                           {k: a[f"ref_chains_c{k}"] for k in range(NCYC)},
                           {k: np.nan for k in range(NCYC)}))   # ref-vs-WT mutpos unknown

        for vstrain, vkd, vch, vmut in states:
            if vstrain == "WT":
                continue                              # WT is the origin
            rec = {"antibody": ab, "ref_strain": "WT", "variant": vstrain,
                   "var_kd_nM": vkd, "wt_kd_nM": wt_kd,
                   "dlogKD_WT": round(math.log10(vkd / wt_kd), 4),
                   "mutpos_ok": ref_strain == "WT"}
            for k in range(NCYC):
                rec[f"dCHAINS_WT_c{k}"] = round(vch[k] - wt_chains[k], 3)
                rec[f"dMUTPOS_WT_c{k}"] = (round(vmut[k], 3) if vmut[k] == vmut[k] else np.nan)
            rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    neg = (out.dlogKD_WT < 0).sum()
    print(f"wrote {OUT}  ({len(out)} points, {out.antibody.nunique()} antibodies)")
    print(f"  dlogKD_WT range {out.dlogKD_WT.min():+.2f} .. {out.dlogKD_WT.max():+.2f}  "
          f"(now SIGNED: {neg} gain-of-affinity points < 0)")
    print(f"  mutpos valid (old ref was WT): {int(out.mutpos_ok.sum())}/{len(out)}  "
          f"-- pending recompute: {sorted(set(out[~out.mutpos_ok].antibody))}")
    from scipy.stats import pearsonr
    for k in range(NCYC):
        v = out[out[f"dCHAINS_WT_c{k}"].notna()]
        m = out[out[f"dMUTPOS_WT_c{k}"].notna()]
        rc = pearsonr(v.dlogKD_WT, v[f"dCHAINS_WT_c{k}"])[0]
        rm = pearsonr(m.dlogKD_WT, m[f"dMUTPOS_WT_c{k}"])[0] if len(m) > 2 else float("nan")
        print(f"  c{k}: chains R={rc:+.3f} (n={len(v)})   mutpos R={rm:+.3f} (n={len(m)})")


if __name__ == "__main__":
    main()
