#!/usr/bin/env python
"""Simplest baseline: run 5 deposited Ab-RBD crystal structures through the paper's
inference protocol AS-IS -- no mutations, no domain swapping/threading.

Paper settings (doi:10.1038/s41467-026-70553-6):
  * merged input = heavy + light + RBD (RBD domain + Fab VH/VL)
  * inter-chain gap = 25 residues
  * masked template (template in 1st cycle only), 3 rounds of recycling
  * model_2_ptm
  * iPAE = AVERAGE PAE between antibody and RBD (all cross-pairs) -- the paper's
    definition. We also report the contact-restricted (12 A) variant for context.
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_full as R
from aftoolkit.structure import load_protein
from aftoolkit.scoring import interface_pae

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAIN_GAP = 25

# 5 crystals spanning WT-RBD and variant-RBD complexes
FIVE = ["EH3", "510A5", "CR3022", "Hastie_1H2", "EY6A"]


def main():
    strains = json.load(open(os.path.join(ROOT, "data/rbdaim/datasets/rbd_strains.json")))
    rbd_ref = strains["Wuhan"]
    aff = pd.read_csv(os.path.join(ROOT, "data/rbdaim/fig2g_affinity.csv"))

    from aftoolkit.folding import OpenFoldWrapper
    wrapper = OpenFoldWrapper(device="mps", inference_n_recycle=2,   # 3 rounds (cycles 0,1,2)
                              always_use_template=False,              # masked template
                              return_all_cycles=True)
    wrapper.init_model()

    tdir = os.path.join(os.path.dirname(__file__), "native_tmp"); os.makedirs(tdir, exist_ok=True)
    rows = []
    for ab in FIVE:
        pdbid = aff[aff.ab == ab].pdbid.iloc[0]
        pdb = R.fetch_pdb(pdbid)
        rbd_chain, rbd_resnums, ab_chains = R.select_complex(pdb, rbd_ref)
        seqs = R.chain_sequences(pdb)
        ab_var = {c: R.variable_span(seqs[c][0]) for c in ab_chains}
        trimmed = os.path.join(tdir, f"{ab}_native.pdb")
        R.trim_complex(pdb, rbd_chain, rbd_resnums, ab_chains, ab_var, trimmed)

        df = load_protein(trimmed, chains=ab_chains + [rbd_chain], chain_gap=CHAIN_GAP)
        wrapper.evaluate(df)                                          # NO mutation, NO threading
        ca = wrapper.get_of_protein(); ca = ca[ca.atom_name == "CA"].reset_index(drop=True)
        labels = ca["chain_id_original"].tolist()
        coords = ca[["x_coord", "y_coord", "z_coord"]].to_numpy()

        rec = {"ab": ab, "pdbid": pdbid, "rbd_chain": rbd_chain, "ab_chains": ",".join(ab_chains)}
        for c in (0, 1, 2):
            pae = wrapper.of_output_per_cycle[c]["predicted_aligned_error"]
            rec[f"ipae_allpairs_c{c}"] = round(interface_pae(pae, labels, [rbd_chain], ab_chains), 3)
            rec[f"ipae_contact_c{c}"] = round(interface_pae(pae, labels, [rbd_chain], ab_chains,
                                                            coords=coords, cutoff=12.0), 3)
        rows.append(rec)
        print(f"[{ab:<11} {pdbid}]  iPAE(all-pairs) c0/1/2 = "
              f"{rec['ipae_allpairs_c0']}/{rec['ipae_allpairs_c1']}/{rec['ipae_allpairs_c2']}   "
              f"contact = {rec['ipae_contact_c0']}/{rec['ipae_contact_c1']}/{rec['ipae_contact_c2']}",
              flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(os.path.dirname(__file__), "native_xray_ipae.csv"), index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nwrote native_xray_ipae.csv")


if __name__ == "__main__":
    main()
