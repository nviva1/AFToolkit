#!/usr/bin/env python
"""CORRECT per-pair no-leak: for each variant, mask ONLY positions(ref-vs-crystal)
UNION positions(variant-vs-crystal), and build+infer BOTH the reference and the
variant with THAT pair-specific mask (the reference is recomputed per variant, so
its iPAE differs per pair). Compare to the (wrong) shared-union baseline.

Usage: python run_pairwise.py [ANTIBODY]   (default EY6A)
Out: data/rbdaim/pairwise_<ANTIBODY>.csv
"""
import ast, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_load as M
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AB = sys.argv[1] if len(sys.argv) > 1 else "EY6A"
GAP = 25


def state(wr, df, rbd):
    wr.evaluate(df)
    c = wr.get_of_protein(); c = c[c.atom_name == "CA"].reset_index(drop=True)
    lab = np.array(c["chain_id_original"].tolist())
    rn = np.array(c["residue_number_original"].astype(int).tolist())
    paes = [np.asarray(cy["predicted_aligned_error"]) for cy in wr.of_output_per_cycle]
    return lab, rn, paes


def main():
    pairs = pd.read_csv(os.path.join(ROOT, "data/rbdaim/escape_pairs.csv"))
    g = pairs[pairs.antibody == AB]; row0 = g.iloc[0]
    chids = ast.literal_eval(row0.chids); rbd = row0.rbd_chain; chains = chids
    pdb = row0.pdb_path; apply_ref = row0.apply_to_xray_for_ref
    ref_pos = M.positions_in(apply_ref)
    from aftoolkit.folding import OpenFoldWrapper
    wr = OpenFoldWrapper(device="mps", inference_n_recycle=3,
                         always_use_template=False, return_all_cycles=True)
    wr.init_model()
    print(f"=== {AB} PER-PAIR no-leak (fresh reference per variant) ===", flush=True)

    rows = []
    for _, r in g.iterrows():
        if str(r.escape_mutations).startswith("cross-species"):
            continue
        var_pos = M.positions_in(r.apply_to_xray_for_variant)
        pair_mask = ref_pos | var_pos                       # PAIR-specific mask
        esc = M.positions_in(r.escape_mutations)            # positions differing ref<->variant
        # build & infer BOTH with the pair mask (reference recomputed for THIS pair)
        rl, rn, rp = state(wr, M.prep_state(pdb, chains, rbd, apply_ref, pair_mask, GAP), rbd)
        vl, vn, vp = state(wr, M.prep_state(pdb, chains, rbd, r.apply_to_xray_for_variant, pair_mask, GAP), rbd)
        Ar = np.where(rl != rbd)[0]; Rr = np.where(rl == rbd)[0]
        Av = np.where(vl != rbd)[0]; Rv = np.where(vl == rbd)[0]
        Rr_e = np.where((rl == rbd) & np.isin(rn, list(esc)))[0]
        Rv_e = np.where((vl == rbd) & np.isin(vn, list(esc)))[0]
        rec = {"variant": r.var_strain, "dlogKD": r.delta_log_kd, "n_mask": len(pair_mask), "n_esc": len(esc)}
        for k in range(4):
            bm = lambda P, R, C: float(P[np.ix_(R, C)].mean()) if len(R) and len(C) else np.nan
            rec[f"ref_mutpos_c{k}"] = round(bm(rp[k], Rr_e, Ar), 3)
            rec[f"var_mutpos_c{k}"] = round(bm(vp[k], Rv_e, Av), 3)
            rec[f"dMUTPOS_c{k}"] = round(bm(vp[k], Rv_e, Av) - bm(rp[k], Rr_e, Ar), 3)
            rec[f"ref_chains_c{k}"] = round(bm(rp[k], Rr, Ar), 3)
            rec[f"var_chains_c{k}"] = round(bm(vp[k], Rv, Av), 3)
            rec[f"dCHAINS_c{k}"] = round(bm(vp[k], Rv, Av) - bm(rp[k], Rr, Ar), 3)
        rows.append(rec)
        print(f"  {r.var_strain:<26} mask={len(pair_mask):2d}  ref_chains_c1={rec['ref_chains_c1']:.3f} "
              f"var_chains_c1={rec['var_chains_c1']:.3f}  dCHAINS_c1={rec['dCHAINS_c1']:+.3f}  dMUTPOS_c1={rec['dMUTPOS_c1']:+.3f}", flush=True)

    out = pd.DataFrame(rows); out.to_csv(os.path.join(ROOT, f"data/rbdaim/pairwise_{AB}.csv"), index=False)
    from scipy.stats import pearsonr, spearmanr
    print(f"\nPER-PAIR (fresh ref) correlations:")
    for k in range(4):
        for m in ("dCHAINS", "dMUTPOS"):
            v = out[out[f"{m}_c{k}"].notna()]
            if len(v) > 2:
                print(f"  {m}_c{k}: Rp={pearsonr(v.dlogKD, v[f'{m}_c{k}'])[0]:+.3f} Rs={spearmanr(v.dlogKD, v[f'{m}_c{k}'])[0]:+.3f}")
    print("\nNote: ref_chains_c1 should now VARY across variants (different pair masks).")
    print("  distinct ref_chains_c1 values:", sorted(out.ref_chains_c1.unique()))


if __name__ == "__main__":
    main()
