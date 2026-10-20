#!/usr/bin/env python
"""EXACT RBD-AIM reproduction, mirroring rbd-aim/reward_test.py::model_rbd.

  RBD  = datasets/ref.pdb (chain '1'), strain sequence threaded on
         (mutate_protein: full sidechain at conserved, [N,C,CA,O] at mutated).
  Fab  = datasets/pdb_clean_ab_only/<pdbid>_H_L.pdb, used AS-IS (pre-aligned to the
         RBD frame -> NO superposition, just concatenated, exactly like the paper).
  concat with +25 residue gap, masked template, n_recycle=2, iPAE = all-pairs RBD->Ab.
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_full as R
from aftoolkit.structure import load_protein, mutate_protein
from aftoolkit.mutations import parse_mutations
from aftoolkit.scoring import interface_pae

DS = "/Users/nikitaivanisenko/projects/AFToolkit/rbd-aim/extra_downloads/figures/datasets"
REF_PDB = os.path.join(DS, "ref.pdb")
TPL_DIR = os.path.join(DS, "pdb_clean_ab_only")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GAP, RBD_CH = 25, "1"

# (antibody, crystal pdbid, strain, paper delta-iPAE)
POINTS = [
    ("J08",           "7S6K", "omicron (BA1, B.1.1.529)", -2.29),
    ("Hastie_1H2",    "8F0H", "omicron (BA1, B.1.1.529)", -1.67),
    ("EH3",           "7UL1", "omicron (BA1, B.1.1.529)", -1.45),
    ("COVOX-278",     "7OR9", "delta",                    -0.86),
    ("Windsor_G32A4", "7SWN", "gamma",                    -0.81),
]


def build(pdbid, out_pdb):
    """ref.pdb RBD (chain '1') + the pre-aligned Fab template, concatenated as-is."""
    tpl_path = os.path.join(TPL_DIR, f"{pdbid}_H_L.pdb")
    ref = PandasPdb().read_pdb(REF_PDB).df["ATOM"]
    ref = ref[ref.alt_loc.isin(["", "A"])].copy(); ref["chain_id"] = RBD_CH
    tpl = PandasPdb().read_pdb(tpl_path).df["ATOM"]
    tpl = tpl[tpl.alt_loc.isin(["", "A"])].copy()
    fab_chains = [c for c in tpl.chain_id.unique() if c != RBD_CH]
    out = pd.concat([ref, tpl]).reset_index(drop=True)
    out["atom_number"] = range(1, len(out) + 1); out["line_idx"] = range(len(out))
    p = PandasPdb(); p.df["ATOM"] = out; p.to_pdb(out_pdb)
    return out_pdb, fab_chains


def main():
    strains = json.load(open(os.path.join(ROOT, "data/rbdaim/datasets/rbd_strains.json")))
    rbd_ref = strains["Wuhan"]
    ref_resnums = list(range(334, 334 + len(rbd_ref)))

    from aftoolkit.folding import OpenFoldWrapper
    wrapper = OpenFoldWrapper(device="mps", inference_n_recycle=2,
                              always_use_template=False, return_all_cycles=True)
    wrapper.init_model()
    tdir = os.path.join(os.path.dirname(__file__), "exact_tmp"); os.makedirs(tdir, exist_ok=True)

    def ipae(combined_pdb, fab_chains, mut_str):
        df = load_protein(combined_pdb, chains=[RBD_CH] + fab_chains, chain_gap=GAP)
        if mut_str:
            df = mutate_protein(df, parse_mutations(mut_str))
        wrapper.evaluate(df)
        ca = wrapper.get_of_protein(); ca = ca[ca.atom_name == "CA"].reset_index(drop=True)
        labels = np.array(ca["chain_id_original"].tolist())
        pae = wrapper.of_output["predicted_aligned_error"]
        # EXACT paper iPAE: np.average(pae[RBD_rows][:, Ab_cols]) -- ONE direction.
        ids1 = np.where(labels == RBD_CH)[0]
        ids2 = np.where(labels != RBD_CH)[0]
        return float(np.asarray(pae)[np.ix_(ids1, ids2)].mean())

    rows = []
    for ab, pdbid, strain, paper_d in POINTS:
        try:
            combined, fab_chains = build(pdbid, os.path.join(tdir, f"{ab}_exact.pdb"))
            muts = R.rbd_mutations(rbd_ref, ref_resnums, strains[R.STRAIN_MAP[strain]])
            mut_str = ",".join(f"{RBD_CH}:{w}{p}{m}" for w, p, m in muts)
            iw = ipae(combined, fab_chains, "")
            im = ipae(combined, fab_chains, mut_str)
            d = -(im - iw)
            rows.append({"ab": ab, "strain": strain.split()[0], "paper_dipae": paper_d,
                         "mine_dipae": round(d, 3), "wt": round(iw, 2), "mt": round(im, 2), "n_mut": len(muts)})
            print(f"[{ab:<13} {strain.split()[0]:<8}] tpl={pdbid}_{'_'.join(fab_chains)}  "
                  f"paper={paper_d:+.2f}  mine={d:+.3f}  (WT {iw:.2f}->mut {im:.2f}, {len(muts)}mut)", flush=True)
        except Exception as exc:
            print(f"[{ab}] FAILED: {exc}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(os.path.dirname(__file__), "paper_exact.csv"), index=False)
    print("\n" + out.to_string(index=False))
    if len(out) >= 3:
        from scipy.stats import pearsonr
        print(f"\nvs paper (n={len(out)}): Pearson={pearsonr(out.paper_dipae, out.mine_dipae)[0]:+.3f}  "
              f"MAE={(out.mine_dipae-out.paper_dipae).abs().mean():.3f}")


if __name__ == "__main__":
    main()
