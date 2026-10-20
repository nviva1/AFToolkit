#!/usr/bin/env python
"""Test a masking mode on a fixed SUBSET of (antibody, variant) points spanning the
full ΔlogK_D range, before committing to the whole panel.

Reads _subset10.json (list of {antibody, variant}). For each involved antibody,
builds the reference + the selected variants with the chosen prep, n_recycle=2,
and reports dCHAINS/dMUTPOS per cycle. Out: escape_ipae_subset_{MODE}.csv

Usage: python run_subset.py [MODE]   MODE in {bb, cb}   (default cb)
"""
import ast, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_load as M
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
MODE = sys.argv[1] if len(sys.argv) > 1 else "cb"
NCYC = 3; GAP = 25
OUT = os.path.join(ROOT, f"data/rbdaim/escape_ipae_subset_{MODE}.csv")


def run_state(wr, df, rbd):
    wr.evaluate(df)
    c = wr.get_of_protein(); c = c[c.atom_name == "CA"].reset_index(drop=True)
    labels = np.array(c["chain_id_original"].tolist())
    resnums = np.array(c["residue_number_original"].astype(int).tolist())
    return labels, resnums, [np.asarray(cy["predicted_aligned_error"]) for cy in wr.of_output_per_cycle]


def bmean(p, r, c):
    return float(p[np.ix_(r, c)].mean()) if len(r) and len(c) else float("nan")


def main():
    sub = pd.DataFrame(json.load(open(os.path.join(HERE, "_subset10.json"))))
    pairs = pd.read_csv(os.path.join(ROOT, "data/rbdaim/escape_pairs.csv"))
    PREP = M.prep_state_cb if MODE == "cb" else M.prep_state
    from aftoolkit.folding import OpenFoldWrapper
    wr = OpenFoldWrapper(device="mps", inference_n_recycle=2,
                         always_use_template=False, return_all_cycles=True)
    wr.init_model()
    print(f"=== subset test, mask_mode={MODE} ({len(sub)} points / {sub.antibody.nunique()} antibodies) ===", flush=True)

    rows = []
    for ab, sg in sub.groupby("antibody"):
        g = pairs[pairs.antibody == ab]; row0 = g.iloc[0]
        chids = ast.literal_eval(row0.chids); rbd = row0.rbd_chain; chains = chids
        pdb = row0.pdb_path; apply_ref = row0.apply_to_xray_for_ref
        mask_pos = M.positions_in(apply_ref)
        for _, r in g.iterrows():
            mask_pos |= M.positions_in(r.apply_to_xray_for_variant)
        mutpos = sorted(mask_pos)
        rl, rn, rp = run_state(wr, PREP(pdb, chains, rbd, apply_ref, mask_pos, GAP), rbd)
        ab_cols_r = np.where(rl != rbd)[0]; rbd_rows_r = np.where(rl == rbd)[0]
        mrr = np.where((rl == rbd) & np.isin(rn, mutpos))[0]
        for vstrain in sg.variant:
            r = g[g.var_strain == vstrain].iloc[0]
            vl, vn, vp = run_state(wr, PREP(pdb, chains, rbd, r.apply_to_xray_for_variant, mask_pos, GAP), rbd)
            ab_cols_v = np.where(vl != rbd)[0]; rbd_rows_v = np.where(vl == rbd)[0]
            mrv = np.where((vl == rbd) & np.isin(vn, mutpos))[0]
            rec = {"antibody": ab, "variant": vstrain, "dlogKD": r.delta_log_kd}
            for k in range(NCYC):
                rec[f"dCHAINS_c{k}"] = round(bmean(vp[k], rbd_rows_v, ab_cols_v) - bmean(rp[k], rbd_rows_r, ab_cols_r), 3)
                rec[f"dMUTPOS_c{k}"] = round(bmean(vp[k], mrv, ab_cols_v) - bmean(rp[k], mrr, ab_cols_r), 3)
            rows.append(rec)
            print(f"  {ab:<14}{vstrain:<26} dlogKD={r.delta_log_kd:+.2f} "
                  f"dCHAINS_c2={rec['dCHAINS_c2']:+.3f} dMUTPOS_c2={rec['dMUTPOS_c2']:+.3f}", flush=True)

    out = pd.DataFrame(rows); out.to_csv(OUT, index=False)
    from scipy.stats import pearsonr
    print(f"\nwrote {OUT}")
    for k in range(NCYC):
        rc = pearsonr(out.dlogKD, out[f"dCHAINS_c{k}"])[0]
        rm = pearsonr(out.dlogKD, out[f"dMUTPOS_c{k}"])[0]
        print(f"  c{k}: chains R={rc:+.3f}   mutpos R={rm:+.3f}")


if __name__ == "__main__":
    main()
