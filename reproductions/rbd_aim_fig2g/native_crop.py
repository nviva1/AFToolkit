#!/usr/bin/env python
"""Native X-ray test: fetch the antibody's real crystal, crop RBD by kalign-aligning
to the real Wuhan RBD, crop VH/VL with AbNumber, using the chain ids from the
deposited names (chids = [heavy, light, RBD]). NO threading/mutation. Compute iPAE
and see if the native cropped complex is similar to the paper."""
import json, os, subprocess, sys, warnings, ast
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from biopandas.pdb import PandasPdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_full as R
from aftoolkit.structure import load_protein, AA_3_TO_1_DICT
from aftoolkit.scoring import interface_pae
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__)); GAP = 25
SYS = ["TAU-2303", "Hastie_1H2", "COVOX-222"]   # 3 well-modelled systems (low RMS)


def kalign(s1, s2):
    p = subprocess.Popen(["kalign"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    out, _ = p.communicate(input=f">a\n{s1}\n>b\n{s2}\n")
    al = []
    for r in out.split("\n"):
        if not r: continue
        if r[0] == ">": al.append([]); continue
        if al: al[-1].append(r.rstrip())
    return "".join(al[0]), "".join(al[1])


def chain_seq(df, ch):
    g = df[(df.chain_id == ch) & (df.atom_name == "CA")].sort_values("residue_number")
    return "".join(AA_3_TO_1_DICT.get(r, "X") for r in g.residue_name), list(g.residue_number)


def vhvl_resns(seq, resns):
    # AbNumber requires HMMER-anarci (absent); use ANARCII variable-domain span (same numbering family)
    sp = R.variable_span(seq)                  # (start, end) indices of the variable domain
    return set(resns[sp[0]:sp[1] + 1])


def crop_rbd_resns(seq, resns, ref):
    a, b = kalign(seq, ref)                    # align chain seq to Wuhan RBD
    keep, si = [], 0
    for x, y in zip(a, b):
        if x != "-":
            if y != "-": keep.append(resns[si])   # residue aligns to the real RBD
            si += 1
    return set(keep)


def main():
    strains = json.load(open(os.path.join(ROOT, "data/rbdaim/datasets/rbd_strains.json")))
    ref = strains["Wuhan"]
    aff = pd.read_csv(os.path.join(ROOT, "data/rbdaim/fig2g_affinity.csv"))
    rp = pd.read_csv(os.path.join(HERE, "recomputed_paper/fig2g_recomputed_full.csv"))
    from aftoolkit.folding import OpenFoldWrapper
    w = OpenFoldWrapper(device="mps", inference_n_recycle=2, always_use_template=False,
                        return_all_cycles=True); w.init_model()
    tdir = os.path.join(HERE, "ncrop_tmp"); os.makedirs(tdir, exist_ok=True)

    for ab in SYS:
      try:
        row = aff[aff.ab == ab].iloc[0]; pdbid = row.pdbid
        H, L, RBD = ast.literal_eval(row.chids)
        pdb = R.fetch_pdb(pdbid)
        df = PandasPdb().read_pdb(pdb).df["ATOM"]; df = df[df.alt_loc.isin(["", "A"])]
        rseq, rresn = chain_seq(df, RBD); hseq, hresn = chain_seq(df, H); lseq, lresn = chain_seq(df, L)
        keep_rbd = crop_rbd_resns(rseq, rresn, ref)
        keep_h = vhvl_resns(hseq, hresn); keep_l = vhvl_resns(lseq, lresn)
        parts = [df[(df.chain_id == RBD) & (df.residue_number.isin(keep_rbd))].assign(chain_id="1"),
                 df[(df.chain_id == H) & (df.residue_number.isin(keep_h))],
                 df[(df.chain_id == L) & (df.residue_number.isin(keep_l))]]
        out = pd.concat(parts).reset_index(drop=True)
        out["atom_number"] = range(1, len(out) + 1); out["line_idx"] = range(len(out))
        p = PandasPdb(); p.df["ATOM"] = out; fp = os.path.join(tdir, f"{ab}_native.pdb"); p.to_pdb(fp)
        d = load_protein(fp, chains=["1", H, L], chain_gap=GAP)
        w.evaluate(d)
        ca = w.get_of_protein(); ca = ca[ca.atom_name == "CA"].reset_index(drop=True)
        lab = np.array(ca["chain_id_original"].tolist())
        pae = np.asarray(w.of_output["predicted_aligned_error"])
        i1 = np.where(lab == "1")[0]; i2 = np.where(lab != "1")[0]
        ipae = float(pae[np.ix_(i1, i2)].mean())
        wt_mutate = float(rp[rp.ab == ab].ipae_ref.iloc[0]) if (rp.ab == ab).any() else None
        print(f"[{ab}] {pdbid} chains H={H} L={L} RBD={RBD}  "
              f"RBD crop={len(keep_rbd)} VH={len(keep_h)} VL={len(keep_l)}", flush=True)
        print(f"   NATIVE cropped iPAE = {ipae:.3f}   |  crystal+mutate WT ref = {wt_mutate}", flush=True)
      except Exception as exc:
        print(f"[{ab}] FAILED: {exc}", flush=True)


if __name__ == "__main__":
    main()
