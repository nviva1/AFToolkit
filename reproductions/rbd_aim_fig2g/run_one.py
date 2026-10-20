#!/usr/bin/env python
"""Run the Fig 2g escape protocol for ONE antibody, reporting BOTH iPAE flavors,
PER CYCLE.

For each point (same crystal, masked-template inference, gap=25, n_recycle=2):
  iPAE_chains    = mean PAE over (all RBD residues) x (all antibody residues)
  iPAE_mutantpos = mean PAE over (this variant's mutated RBD residues) x (antibody)
ΔiPAE_* = variant - reference, computed for every recycle (c0, c1, c2).

Loading uses no-leak masking: positions mutated-vs-crystal in ANY endpoint are
reduced to backbone-only in EVERY build.

Usage: python run_one.py [ANTIBODY]   (default Windsor_C98C7)
"""
import ast, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_load as M
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AB = sys.argv[1] if len(sys.argv) > 1 else "Windsor_C98C7"
GAP = 25


def run_state(wr, df, rbd_chain):
    """Returns (labels, resnums, [pae_c0, pae_c1, ...]) for the CA-ordered output."""
    wr.evaluate(df)
    c = wr.get_of_protein(); c = c[c.atom_name == "CA"].reset_index(drop=True)
    labels = np.array(c["chain_id_original"].tolist())
    resnums = np.array(c["residue_number_original"].astype(int).tolist())
    paes = [np.asarray(cy["predicted_aligned_error"]) for cy in wr.of_output_per_cycle]
    return labels, resnums, paes


def bmean(pae, rows, cols):
    return float(pae[np.ix_(rows, cols)].mean())


def main():
    pairs = pd.read_csv(os.path.join(ROOT, "data/rbdaim/escape_pairs.csv"))
    g = pairs[pairs.antibody == AB]
    if g.empty:
        sys.exit(f"{AB}: no escape pairs")
    row0 = g.iloc[0]
    chids = ast.literal_eval(row0.chids); rbd = row0.rbd_chain; chains = chids
    pdb = row0.pdb_path; ref_strain = row0.ref_strain; apply_ref = row0.apply_to_xray_for_ref

    mask_pos = M.positions_in(apply_ref)
    for _, r in g.iterrows():
        mask_pos |= M.positions_in(r.apply_to_xray_for_variant)
    print(f"[{AB}] {row0.pdbid}  H/L/RBD={chids}  reference={ref_strain}  ({len(g)} escape variants)")
    print(f"  no-leak backbone-only positions (n={len(mask_pos)}): {sorted(mask_pos)}\n")

    from aftoolkit.folding import OpenFoldWrapper
    wr = OpenFoldWrapper(device="mps", inference_n_recycle=2,
                         always_use_template=False, return_all_cycles=True)
    wr.init_model()

    # reference once
    rl, rn, rp = run_state(wr, M.prep_state(pdb, chains, rbd, apply_ref, mask_pos, GAP), rbd)
    ncyc = len(rp)
    ab_cols_r = np.where(rl != rbd)[0]; rbd_rows_r = np.where(rl == rbd)[0]

    rows = []
    for _, r in g.iterrows():
        mutpos = M.positions_in(r.escape_mutations)
        vl, vn, vp = run_state(wr, M.prep_state(pdb, chains, rbd, r.apply_to_xray_for_variant, mask_pos, GAP), rbd)
        ab_cols_v = np.where(vl != rbd)[0]; rbd_rows_v = np.where(vl == rbd)[0]
        mut_rows_r = np.where((rl == rbd) & np.isin(rn, list(mutpos)))[0]
        mut_rows_v = np.where((vl == rbd) & np.isin(vn, list(mutpos)))[0]
        rec = {"variant": r.var_strain, "dlogKD": r.delta_log_kd, "nmut": len(mutpos)}
        for k in range(ncyc):
            rec[f"dCHAINS_c{k}"] = round(bmean(vp[k], rbd_rows_v, ab_cols_v)
                                        - bmean(rp[k], rbd_rows_r, ab_cols_r), 3)
            rec[f"dMUTPOS_c{k}"] = round(bmean(vp[k], mut_rows_v, ab_cols_v)
                                        - bmean(rp[k], mut_rows_r, ab_cols_r), 3) if len(mutpos) else float("nan")
        rows.append(rec)

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))

    if len(out) >= 3:
        from scipy.stats import pearsonr
        print("\nPearson(ΔlogKD, ΔiPAE) per cycle:")
        for k in range(ncyc):
            rc = pearsonr(out.dlogKD, out[f"dCHAINS_c{k}"])[0]
            valid = out[f"dMUTPOS_c{k}"].notna()
            rm = pearsonr(out.dlogKD[valid], out[f"dMUTPOS_c{k}"][valid])[0] if valid.sum() >= 3 else float("nan")
            print(f"  c{k}: chains={rc:+.3f}   mutantpos={rm:+.3f}")


if __name__ == "__main__":
    main()
