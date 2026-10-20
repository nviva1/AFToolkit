#!/usr/bin/env python
"""On the 10 range-spanning points, extract MANY iPAE definitions from a single
inference per state and see which Δ-metric best tracks ΔlogK_D:

  rbd2ab   : mean pae[RBD rows, Ab cols]        (current default, one direction)
  ab2rbd   : mean pae[Ab rows, RBD cols]        (other direction)
  sym      : mean of the two directions
  iface    : rbd2ab restricted to interface-contact residues (CA < 12 A)
  pmin     : 10th-percentile of pae[RBD, Ab]    (best-aligned pairs)
each also in a mutant-position variant (RBD rows = mutated residues).

bb masking, n_recycle=2 (cycles c1,c2 reported). Out: escape_paevars_subset.csv
"""
import ast, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from biopandas.pdb import PandasPdb
from abnumber import Chain
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_load as M, native_crop as N
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
GAP = 25; IFACE_A = 12.0


def cdr_set(xray_pdb, H, L):
    """{(chain, resnum)} of the antibody CDR loops (CDR1/2/3, IMGT) via AbNumber."""
    df = PandasPdb().read_pdb(xray_pdb).df["ATOM"]; df = df[df.alt_loc.isin(["", "A"])]
    s = set()
    for ch in [H, L]:
        seq, resns = N.chain_seq(df, ch)
        for c in Chain.multiple_domains(seq, scheme="imgt"):
            for cdr in (c.cdr1_seq, c.cdr2_seq, c.cdr3_seq):
                i = seq.find(cdr)
                if i >= 0:
                    s.update((ch, r) for r in resns[i:i + len(cdr)])
    return s


def state(wr, df, rbd):
    wr.evaluate(df)
    c = wr.get_of_protein(); c = c[c.atom_name == "CA"].reset_index(drop=True)
    lab = np.array(c["chain_id_original"].tolist())
    rn = np.array(c["residue_number_original"].astype(int).tolist())
    xyz = c[["x_coord", "y_coord", "z_coord"]].to_numpy()
    paes = [np.asarray(cy["predicted_aligned_error"]) for cy in wr.of_output_per_cycle]
    return lab, rn, xyz, paes


def metrics(pae, lab, rn, xyz, rbd, mutpos, cdrset):
    R = np.where(lab == rbd)[0]; A = np.where(lab != rbd)[0]
    Rm = np.where((lab == rbd) & np.isin(rn, mutpos))[0]
    Acdr = np.array([i for i in A if (lab[i], int(rn[i])) in cdrset], dtype=int)
    # interface: residues within IFACE_A (CA) of the other side
    d = np.linalg.norm(xyz[R][:, None] - xyz[A][None], axis=-1)
    Ri = R[d.min(1) < IFACE_A]; Ai = A[d.min(0) < IFACE_A]
    blk = lambda r, c: float(pae[np.ix_(r, c)].mean()) if len(r) and len(c) else np.nan
    out = {
        "rbd2ab": blk(R, A), "ab2rbd": blk(A, R), "sym": 0.5 * (blk(R, A) + blk(A, R)),
        "iface": blk(Ri, Ai),
        "pmin": float(np.percentile(pae[np.ix_(R, A)], 10)) if len(R) and len(A) else np.nan,
        "rbd2cdr": blk(R, Acdr),                # whole RBD x CDR loops
        "mut_rbd2ab": blk(Rm, A),
        "mut_cdr": blk(Rm, Acdr),               # mutated RBD x CDR loops
        "mut_iface": blk(Rm[np.isin(Rm, Ri)] if len(Rm) else Rm, Ai),
    }
    return out


def main():
    sub = pd.DataFrame(json.load(open(os.path.join(HERE, "_subset10.json"))))
    pairs = pd.read_csv(os.path.join(ROOT, "data/rbdaim/escape_pairs.csv"))
    from aftoolkit.folding import OpenFoldWrapper
    wr = OpenFoldWrapper(device="mps", inference_n_recycle=2,
                         always_use_template=False, return_all_cycles=True)
    wr.init_model()
    print("=== PAE-variant scan on 10 points (cycles c1,c2) ===", flush=True)
    rows = []
    for ab, sg in sub.groupby("antibody"):
        g = pairs[pairs.antibody == ab]; row0 = g.iloc[0]
        chids = ast.literal_eval(row0.chids); rbd = row0.rbd_chain
        cdrset = cdr_set(row0.pdb_path, chids[0], chids[1])
        mask_pos = M.positions_in(row0.apply_to_xray_for_ref)
        for _, r in g.iterrows():
            mask_pos |= M.positions_in(r.apply_to_xray_for_variant)
        mutpos = sorted(mask_pos)
        rl, rn, rx, rp = state(wr, M.prep_state(row0.pdb_path, chids, rbd, row0.apply_to_xray_for_ref, mask_pos, GAP), rbd)
        refm = {k: metrics(rp[k], rl, rn, rx, rbd, mutpos, cdrset) for k in (1, 2)}
        for vstrain in sg.variant:
            r = g[g.var_strain == vstrain].iloc[0]
            vl, vn, vx, vp = state(wr, M.prep_state(row0.pdb_path, chids, rbd, r.apply_to_xray_for_variant, mask_pos, GAP), rbd)
            rec = {"antibody": ab, "variant": vstrain, "dlogKD": r.delta_log_kd}
            for k in (1, 2):
                vm = metrics(vp[k], vl, vn, vx, rbd, mutpos, cdrset)
                for key in refm[k]:
                    rec[f"{key}_c{k}"] = round(vm[key] - refm[k][key], 3)
            rows.append(rec); print(f"  {ab:<14}{vstrain:<24} dlogKD={r.delta_log_kd:+.2f}", flush=True)
    out = pd.DataFrame(rows); out.to_csv(os.path.join(ROOT, "data/rbdaim/escape_paevars_subset.csv"), index=False)
    from scipy.stats import pearsonr
    print("\nPearson(ΔlogKD, Δmetric) over 10 points:")
    keys = ["rbd2ab", "ab2rbd", "sym", "iface", "pmin", "rbd2cdr", "mut_rbd2ab", "mut_cdr", "mut_iface"]
    for k in (1, 2):
        print(f" cycle c{k}:")
        for key in keys:
            col = f"{key}_c{k}"; v = out[out[col].notna()]
            print(f"    {key:<11} R={pearsonr(v.dlogKD, v[col])[0]:+.3f}  (n={len(v)})")


if __name__ == "__main__":
    main()
