#!/usr/bin/env python
"""Alternative Fig 2g protocol probe: take the antibody's own crystal complex,
thread each strain's identities onto the RBD, reduce the WHOLE RBD to [N,CA,C,O,CB]
(backbone+CB, no sidechains), run AFToolkit masked-template inference, and return
iPAE at EVERY recycle. Test whether some cycle matches the paper's Fig 2g ΔiPAE
(hypothesis: Fig 2f and 2g used different protocols)."""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_full as R
from aftoolkit.structure import load_protein, AA_1_TO_3_DICT
from aftoolkit.scoring import interface_pae

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
BBCB = ["N", "CA", "C", "O", "CB"]
GAP = 25
# antibodies + strains with reliable (unique-ish) paper Fig2g anchors
POINTS = {
    "EH3":           ["alpha", "beta", "delta", "omicron (BA1, B.1.1.529)", "Iota (B.1.526)", "Epsilon (B.1.429)"],
    "Windsor_G32A4": ["alpha", "gamma", "delta"],
}


def cb_df(trimmed, rbd_chain, ab_chains, struct_seq, struct_resns, target_seq):
    df = load_protein(trimmed, chains=ab_chains + [rbd_chain], chain_gap=GAP)
    rbd = df["chain_id_original"] == rbd_chain
    # thread strain identities onto the RBD
    for w, p, m in R.rbd_mutations(struct_seq, struct_resns, target_seq):
        if m in AA_1_TO_3_DICT:
            df.loc[rbd & (df["residue_number_original"] == p), "residue_name"] = AA_1_TO_3_DICT[m]
    # reduce ALL RBD residues to [N,CA,C,O,CB]; keep Fab full
    drop = rbd & (~df["atom_name"].isin(BBCB))
    gly = rbd & (df["residue_name"] == "GLY") & (df["atom_name"] == "CB")
    return df[~(drop | gly)].copy()


def main():
    strains = json.load(open(os.path.join(ROOT, "data/rbdaim/datasets/rbd_strains.json")))
    rbd_ref = strains["Wuhan"]
    aff = pd.read_csv(os.path.join(ROOT, "data/rbdaim/fig2g_affinity.csv"))
    from aftoolkit.folding import OpenFoldWrapper
    wrapper = OpenFoldWrapper(device="mps", inference_n_recycle=2,
                              always_use_template=False, return_all_cycles=True)
    wrapper.init_model()
    tdir = os.path.join(HERE, "cb_tmp"); os.makedirs(tdir, exist_ok=True)

    def ipae_cycles(df, rbd_chain, ab_chains):
        wrapper.evaluate(df)
        ca = wrapper.get_of_protein(); ca = ca[ca.atom_name == "CA"].reset_index(drop=True)
        labels = ca["chain_id_original"].tolist()
        coords = ca[["x_coord", "y_coord", "z_coord"]].to_numpy()
        out = {}
        for c in (0, 1, 2):
            pae = wrapper.of_output_per_cycle[c]["predicted_aligned_error"]
            out[c] = interface_pae(pae, labels, [rbd_chain], ab_chains)   # all-pairs
        return out

    rows = []
    for ab, slist in POINTS.items():
        try:
            pdbid = aff[aff.ab == ab].pdbid.iloc[0]
            pdb = R.fetch_pdb(pdbid)
            rbd_chain, rbd_resnums, ab_chains = R.select_complex(pdb, rbd_ref)
            ab_var = {c: R.variable_span(R.chain_sequences(pdb)[c][0]) for c in ab_chains}
            trimmed = R.trim_complex(pdb, rbd_chain, rbd_resnums, ab_chains, ab_var,
                                     os.path.join(tdir, f"{ab}_fv.pdb"))
            sseq, sresn = R.chain_sequences(trimmed)[rbd_chain]
            wt = ipae_cycles(cb_df(trimmed, rbd_chain, ab_chains, sseq, sresn, rbd_ref), rbd_chain, ab_chains)
            print(f"[{ab}] {pdbid} WT iPAE/cyc: c0={wt[0]:.2f} c1={wt[1]:.2f} c2={wt[2]:.2f}", flush=True)
            for s in slist:
                key = R.STRAIN_MAP.get(s)
                if key is None or key not in strains:
                    continue
                ip = ipae_cycles(cb_df(trimmed, rbd_chain, ab_chains, sseq, sresn, strains[key]),
                                 rbd_chain, ab_chains)
                dk = float(aff[(aff.ab == ab) & (aff.strain == s)].delta_log_kd.iloc[0])
                rows.append({"ab": ab, "strain": s.split()[0], "dlogkd": round(dk, 3),
                             **{f"d_c{c}": round(ip[c] - wt[c], 3) for c in (0, 1, 2)}})
                print(f"   {s:<26} dlogKD={dk:+.2f}  dC0={rows[-1]['d_c0']:+.3f} "
                      f"dC1={rows[-1]['d_c1']:+.3f} dC2={rows[-1]['d_c2']:+.3f}", flush=True)
        except Exception as exc:
            print(f"[{ab}] FAILED: {exc}", flush=True)
    out = pd.DataFrame(rows); out.to_csv(os.path.join(HERE, "fig2g_cb.csv"), index=False)
    # compare each cycle to paper Fig2g by dlogKD
    pp = pd.read_excel(os.path.join(os.path.expanduser("~"),
         "Downloads/41467_2026_70553_MOESM9_ESM.xlsx"), sheet_name="Fig 2g", header=1).dropna(how="all")
    pp.columns = ["dlogkd", "dipae"]; pp = pp.apply(pd.to_numeric, errors="coerce").dropna()
    print("\n=== per-cycle ΔiPAE vs paper (matched by dlogKD, paper sign) ===")
    for c in (0, 1, 2):
        used = set(); diffs = []
        for _, r in out.iterrows():
            near = (pp.dlogkd - r.dlogkd).abs(); idx = next((i for i in near.sort_values().index if i not in used and near[i] < 0.03), None)
            if idx is not None:
                used.add(idx); diffs.append(abs(-r[f"d_c{c}"] - pp.dipae[idx]))
        if diffs:
            print(f"  cycle{c}: matched {len(diffs)}, mean|diff vs paper|={np.mean(diffs):.3f}")


if __name__ == "__main__":
    main()
