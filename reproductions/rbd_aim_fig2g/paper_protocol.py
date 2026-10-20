#!/usr/bin/env python
"""Corrected protocol, matching the RBD-AIM inference code (RBDAIM/inference.py).

KEY FIX: the RBD is ALWAYS the single WT ref.pdb, with the strain sequence
THREADED onto it (full sidechain at conserved positions, [N,C,CA,O] backbone at
mutated positions = mutate_protein). The crystal is used ONLY for the Fab (VH/VL).
Never the crystal RBD -> no variant-crystal artifact.

  RBD  = ref.pdb (chain '1')  + strain point-mutations vs Wuhan
  Fab  = crystal heavy+light (VH/VL trimmed)
  gap=25, masked template, 3 recycles (n_recycle=2), iPAE = all-pairs RBD->Ab.
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_full as R
import protocol_test as PT
from aftoolkit.structure import load_protein, mutate_protein, save_to_pdb, AA_3_TO_1_DICT
from aftoolkit.mutations import parse_mutations
from aftoolkit.scoring import interface_pae

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_PDB = os.path.join(ROOT, "data/rbdaim/datasets/ref.pdb")
GAP = 25
RBD_CH = "1"

# (antibody, strain, paper delta-iPAE); last one is a variant-crystal antibody
POINTS = [
    ("J08",           "omicron (BA1, B.1.1.529)", -2.29),
    ("Hastie_1H2",    "omicron (BA1, B.1.1.529)", -1.67),
    ("EH3",           "omicron (BA1, B.1.1.529)", -1.45),
    ("COVOX-278",     "delta",                    -0.86),
    ("Windsor_G32A4", "gamma",                    -0.81),
    ("510A5",         "omicron (BA1, B.1.1.529)",  None),   # variant crystal: artifact should vanish
]


def build_combined(crystal_pdb, rbd_chain, rbd_resnums, ab_chains, ab_var, out_pdb):
    """ref.pdb RBD (chain '1'), SUPERPOSED onto the crystal RBD so it docks with the
    crystal Fab, + crystal Fab VH/VL -> one PDB for load_protein. All ref atoms kept
    (mutate_protein trims to [N,C,CA,O] only at mutated positions)."""
    cr = PandasPdb().read_pdb(crystal_pdb).df["ATOM"]
    cr = cr[cr.alt_loc.isin(["", "A"])]
    # crystal RBD CA (the frame the Fab is docked against)
    cr_rbd = cr[(cr.chain_id == rbd_chain) & (cr.residue_number.isin(set(rbd_resnums)))]
    cr_ca = cr_rbd[cr_rbd.atom_name == "CA"].sort_values("residue_number")
    cr_seq = "".join(AA_3_TO_1_DICT.get(r, "X") for r in cr_ca.residue_name)
    cr_xyz = cr_ca[["x_coord", "y_coord", "z_coord"]].to_numpy()
    # ref.pdb, superposed onto the crystal RBD
    ref = PandasPdb().read_pdb(REF_PDB).df["ATOM"]
    ref = ref[ref.alt_loc.isin(["", "A"])].copy()
    ref_ca = ref[ref.atom_name == "CA"].sort_values("residue_number")
    ref_seq = "".join(AA_3_TO_1_DICT.get(r, "X") for r in ref_ca.residue_name)
    ref_xyz = ref_ca[["x_coord", "y_coord", "z_coord"]].to_numpy()
    P, Q = PT.matched_ca(ref_seq, ref_xyz, cr_seq, cr_xyz)
    rot, t = PT.kabsch(P, Q)
    xyz = ref[["x_coord", "y_coord", "z_coord"]].to_numpy()
    ref[["x_coord", "y_coord", "z_coord"]] = (xyz @ rot.T) + t
    ref["chain_id"] = RBD_CH
    fab = []
    for c in ab_chains:
        cc = cr[cr.chain_id == c]
        span = ab_var.get(c)
        if span is not None:
            order = cc[cc.atom_name == "CA"].sort_values("residue_number")
            keep = set(order["residue_number"].iloc[span[0]:span[1] + 1])
            cc = cc[cc.residue_number.isin(keep)]
        fab.append(cc)
    out = pd.concat([ref] + fab).reset_index(drop=True)
    out["atom_number"] = range(1, len(out) + 1); out["line_idx"] = range(len(out))
    p = PandasPdb(); p.df["ATOM"] = out; p.to_pdb(out_pdb)
    return out_pdb


def main():
    strains = json.load(open(os.path.join(ROOT, "data/rbdaim/datasets/rbd_strains.json")))
    rbd_ref = strains["Wuhan"]
    ref_resnums = list(range(334, 334 + len(rbd_ref)))      # ref.pdb numbering 334..527
    aff = pd.read_csv(os.path.join(ROOT, "data/rbdaim/fig2g_affinity.csv"))

    from aftoolkit.folding import OpenFoldWrapper
    wrapper = OpenFoldWrapper(device="mps", inference_n_recycle=2,
                              always_use_template=False, return_all_cycles=True)
    wrapper.init_model()
    tdir = os.path.join(os.path.dirname(__file__), "paper_tmp"); os.makedirs(tdir, exist_ok=True)

    def ipae(combined_pdb, ab_chains, mut_str):
        df = load_protein(combined_pdb, chains=[RBD_CH] + ab_chains, chain_gap=GAP)
        if mut_str:
            df = mutate_protein(df, parse_mutations(mut_str))
        wrapper.evaluate(df)
        ca = wrapper.get_of_protein(); ca = ca[ca.atom_name == "CA"].reset_index(drop=True)
        labels = ca["chain_id_original"].tolist()
        pae = wrapper.of_output["predicted_aligned_error"]           # final cycle
        return interface_pae(pae, labels, [RBD_CH], ab_chains)       # all-pairs

    rows = []
    for ab, strain, paper_d in POINTS:
        try:
            pdbid = aff[aff.ab == ab].pdbid.iloc[0]
            pdb = R.fetch_pdb(pdbid)
            rbd_chain, rbd_resnums, ab_chains = R.select_complex(pdb, rbd_ref)
            seqs = R.chain_sequences(pdb)
            ab_var = {c: R.variable_span(seqs[c][0]) for c in ab_chains}
            combined = build_combined(pdb, rbd_chain, rbd_resnums, ab_chains, ab_var,
                                      os.path.join(tdir, f"{ab}_ref_fab.pdb"))
            # mutations on the WT ref RBD -> strain
            muts = R.rbd_mutations(rbd_ref, ref_resnums, strains[R.STRAIN_MAP[strain]])
            mut_str = ",".join(f"{RBD_CH}:{w}{p}{m}" for w, p, m in muts)
            iw = ipae(combined, ab_chains, "")            # WT ref RBD + Fab
            im = ipae(combined, ab_chains, mut_str)       # strain-threaded RBD + Fab
            d = -(im - iw)                                # paper sign
            rows.append({"ab": ab, "strain": strain.split()[0], "paper_dipae": paper_d,
                         "mine_dipae": round(d, 3), "wt": round(iw, 2), "mt": round(im, 2),
                         "n_mut": len(muts)})
            ps = f"{paper_d:+.2f}" if paper_d is not None else "  ? "
            print(f"[{ab:<13} {strain.split()[0]:<8}] paper={ps}  mine={d:+.3f}  "
                  f"(WT {iw:.2f} -> mut {im:.2f}, {len(muts)}mut)", flush=True)
        except Exception as exc:
            print(f"[{ab}] FAILED: {exc}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(os.path.dirname(__file__), "paper_protocol.csv"), index=False)
    print("\n" + out.to_string(index=False))
    anchored = out[out.paper_dipae.notna()]
    if len(anchored) >= 3:
        from scipy.stats import pearsonr
        print(f"\nvs paper (anchored {len(anchored)}): Pearson={pearsonr(anchored.paper_dipae, anchored.mine_dipae)[0]:+.3f}  "
              f"MAE={(anchored.mine_dipae-anchored.paper_dipae).abs().mean():.3f}")


if __name__ == "__main__":
    main()
