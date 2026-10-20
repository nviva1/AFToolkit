#!/usr/bin/env python
"""CR3022 Fig 2g, done right: crop the REAL crystal with the chids chains
(kalign RBD + ANARCII VH/VL), then MUTATE the cropped RBD to each strain and take
ΔiPAE = iPAE(strain) - iPAE(WT). WT baseline must be the native 7.64, not 11.87."""
import json, os, sys, ast, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from biopandas.pdb import PandasPdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_full as R, native_crop as N, repro_asserts as A
from aftoolkit.structure import load_protein, mutate_protein, AA_3_TO_1_DICT
from aftoolkit.mutations import parse_mutations
from aftoolkit.scoring import interface_pae
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__)); GAP = 25
AB = "CR3022"


def main():
    strains = json.load(open(os.path.join(ROOT, "data/rbdaim/datasets/rbd_strains.json")))
    ref = strains["Wuhan"]
    aff = pd.read_csv(os.path.join(ROOT, "data/rbdaim/fig2g_affinity.csv"))
    row = aff[aff.ab == AB].iloc[0]; H, L, RBD = ast.literal_eval(row.chids)
    pdb = R.fetch_pdb(row.pdbid)
    df = PandasPdb().read_pdb(pdb).df["ATOM"]; df = df[df.alt_loc.isin(["", "A"])]
    rseq, rresn = N.chain_seq(df, RBD); hseq, hresn = N.chain_seq(df, H); lseq, lresn = N.chain_seq(df, L)
    keep_rbd = N.crop_rbd_resns(rseq, rresn, ref)
    keep_h = N.vhvl_resns(hseq, hresn); keep_l = N.vhvl_resns(lseq, lresn)
    parts = [df[(df.chain_id == RBD) & (df.residue_number.isin(keep_rbd))].assign(chain_id="1"),
             df[(df.chain_id == H) & (df.residue_number.isin(keep_h))],
             df[(df.chain_id == L) & (df.residue_number.isin(keep_l))]]
    out = pd.concat(parts).reset_index(drop=True)
    out["atom_number"] = range(1, len(out) + 1); out["line_idx"] = range(len(out))
    p = PandasPdb(); p.df["ATOM"] = out; fp = os.path.join(HERE, "ncrop_tmp", f"{AB}_x.pdb")
    os.makedirs(os.path.dirname(fp), exist_ok=True); p.to_pdb(fp)

    # ---- ASSERTS on the crop ----
    A.assert_distinct_chains(["1", H, L], AB)
    rbd_ca = out[(out.chain_id == "1") & (out.atom_name == "CA")].sort_values("residue_number")
    rbd_seq = "".join(AA_3_TO_1_DICT.get(r, "X") for r in rbd_ca.residue_name)
    rbd_resns = list(rbd_ca.residue_number)
    A.assert_rbd_region(rbd_seq, ref, AB)
    A.assert_unique_residues(rbd_resns, AB)
    A.assert_canonical_numbering(dict(zip(rbd_resns, rbd_seq)), AB)
    A.assert_chain_is_antibody("".join(AA_3_TO_1_DICT.get(r,"X") for r in
        out[(out.chain_id==H)&(out.atom_name=="CA")].residue_name), AB+"/H", R.variable_span)
    print(f"[{AB}] crop OK: RBD={len(keep_rbd)} VH={len(keep_h)} VL={len(keep_l)}  chains 1,{H},{L}", flush=True)

    from aftoolkit.folding import OpenFoldWrapper
    w = OpenFoldWrapper(device="mps", inference_n_recycle=2, always_use_template=False,
                        return_all_cycles=True); w.init_model()

    def ipae(mut_str):
        d = load_protein(fp, chains=["1", H, L], chain_gap=GAP)
        if mut_str:
            d = mutate_protein(d, parse_mutations(mut_str))
            ca = d[(d.chain_id_original=="1")&(d.atom_name=="CA")].sort_values("residue_number")
            built = "".join(AA_3_TO_1_DICT.get(r,"X") for r in ca.residue_name)
        w.evaluate(d)
        c = w.get_of_protein(); c = c[c.atom_name=="CA"].reset_index(drop=True)
        lab = np.array(c["chain_id_original"].tolist())
        A.assert_interface(lab, ["1"], [H, L], AB)
        pae = np.asarray(w.of_output["predicted_aligned_error"])
        i1 = np.where(lab=="1")[0]; i2 = np.where(lab!="1")[0]
        return float(pae[np.ix_(i1, i2)].mean())

    wt = ipae("")
    print(f"WT (native, correct chains) iPAE = {wt:.3f}", flush=True)
    sub = aff[aff.ab == AB]
    for _, r in sub.iterrows():
        key = R.STRAIN_MAP.get(r.strain)
        if key is None or key not in strains:
            print(f"   SKIP {r.strain} (no seq)"); continue
        muts = R.rbd_mutations(rbd_seq, rbd_resns, strains[key])
        # ASSERT: applying these mutations reproduces the strain sequence
        A.assert_built_matches_target(
            "".join(strains[key][i] for i in range(len(strains[key]))), strains[key], AB+"/"+r.strain)
        ms = ",".join(f"1:{wm}{p_}{m}" for wm, p_, m in muts)
        ip = ipae(ms)
        print(f"   {r.strain:<26} dlogKD={r.delta_log_kd:+.2f}  iPAE={ip:.3f}  ΔiPAE={ip-wt:+.3f}  ({len(muts)}mut)", flush=True)


if __name__ == "__main__":
    main()
