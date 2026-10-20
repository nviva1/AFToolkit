#!/usr/bin/env python
"""Is the CB-threading approach valid, or are individual point-mutations better?

For 5 (antibody, variant) points -- all with WT crystal RBDs and a known paper
Fig 2g delta-iPAE -- build a WT model AND a mutant model TWO ways, then compare
each approach's delta-iPAE to the paper's:

  * CB        : one clean WT RBD (ref.pdb) superposed, reduced to [N,CA,C,O,CB],
                strain sequence THREADED on (WT-thread = reference).
  * individual: the crystal RBD POINT-MUTATED residue-by-residue to the variant
                (crystal as-is = WT reference, since these crystals are WT).

Paper settings throughout: chain_gap=25, masked template, 3 recycles (read final
cycle), iPAE = AVERAGE PAE over all antibody<->RBD pairs (the paper's definition).
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_full as R
import protocol_test as PT
from aftoolkit.structure import load_protein, mutate_protein
from aftoolkit.mutations import parse_mutations
from aftoolkit.scoring import interface_pae

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GAP = 25
READ_CYCLE = 2                      # 3 rounds -> final

# 5 distinct WT-crystal antibodies, each matched to a paper delta-iPAE anchor
POINTS = [
    ("J08",           "omicron (BA1, B.1.1.529)", -2.29),
    ("Hastie_1H2",    "omicron (BA1, B.1.1.529)", -1.67),
    ("EH3",           "omicron (BA1, B.1.1.529)", -1.45),
    ("COVOX-278",     "delta",                    -0.86),
    ("Windsor_G32A4", "gamma",                    -0.81),
]


def main():
    strains = json.load(open(os.path.join(ROOT, "data/rbdaim/datasets/rbd_strains.json")))
    rbd_ref = strains["Wuhan"]
    aff = pd.read_csv(os.path.join(ROOT, "data/rbdaim/fig2g_affinity.csv"))

    from aftoolkit.folding import OpenFoldWrapper
    wrapper = OpenFoldWrapper(device="mps", inference_n_recycle=2,
                              always_use_template=False, return_all_cycles=True)
    wrapper.init_model()
    tdir = os.path.join(os.path.dirname(__file__), "cbvi_tmp"); os.makedirs(tdir, exist_ok=True)

    def ipae(df, rbd_chain, ab_chains):
        wrapper.evaluate(df)
        ca = wrapper.get_of_protein(); ca = ca[ca.atom_name == "CA"].reset_index(drop=True)
        labels = ca["chain_id_original"].tolist()
        pae = wrapper.of_output_per_cycle[READ_CYCLE]["predicted_aligned_error"]
        return interface_pae(pae, labels, [rbd_chain], ab_chains)   # all-pairs (paper)

    rows = []
    for ab, strain, paper_d in POINTS:
        try:
            pdbid = aff[aff.ab == ab].pdbid.iloc[0]
            pdb = R.fetch_pdb(pdbid)
            rbd_chain, rbd_resnums, ab_chains = R.select_complex(pdb, rbd_ref)
            seqs = R.chain_sequences(pdb)
            ab_var = {c: R.variable_span(seqs[c][0]) for c in ab_chains}
            variant_seq = strains[R.STRAIN_MAP[strain]]
            # Sanity filter: skip only true strain indels (strain vs Wuhan ref),
            # not crystal density gaps.
            ok, why, _, ident = R.rbd_sanity_check(rbd_ref, variant_seq)
            if not ok:
                print(f"[{ab}] SKIP {strain}: {why} (identity={ident:.3f})", flush=True)
                continue

            # ---- CB approach (thread onto clean WT backbone) ----
            tpl, wt_seq, wt_resns = PT.build_wt_template(
                pdb, rbd_chain, rbd_resnums, ab_chains, ab_var,
                os.path.join(tdir, f"{ab}_wt_tpl.pdb"))
            cb_wt = ipae(PT.thread(tpl, rbd_chain, wt_seq, wt_resns, rbd_ref, chain_gap=GAP),
                         rbd_chain, ab_chains)
            cb_mt = ipae(PT.thread(tpl, rbd_chain, wt_seq, wt_resns, variant_seq, chain_gap=GAP),
                         rbd_chain, ab_chains)

            # ---- individual point-mutation approach (on the crystal) ----
            trimmed = R.trim_complex(pdb, rbd_chain, rbd_resnums, ab_chains, ab_var,
                                     os.path.join(tdir, f"{ab}_fv.pdb"))
            cseq, cresn = R.chain_sequences(trimmed)[rbd_chain]
            def mut_df(target_seq):
                df = load_protein(trimmed, chains=ab_chains + [rbd_chain], chain_gap=GAP)
                muts = R.rbd_mutations(cseq, cresn, target_seq)
                if muts:
                    ms = ",".join(f"{rbd_chain}:{w}{p}{m}" for w, p, m in muts)
                    df = mutate_protein(df, parse_mutations(ms))
                return df, len(muts)
            df_iw, n_w = mut_df(rbd_ref)        # crystal -> WT (empty for WT crystals)
            iv_wt = ipae(df_iw, rbd_chain, ab_chains)
            df_im, n_m = mut_df(variant_seq)    # crystal -> variant
            iv_mt = ipae(df_im, rbd_chain, ab_chains)

            rec = {"ab": ab, "strain": strain.split()[0], "paper_dipae": paper_d,
                   "CB_dipae": round(-(cb_mt - cb_wt), 3),      # paper sign convention
                   "indiv_dipae": round(-(iv_mt - iv_wt), 3),
                   "CB_wt": round(cb_wt, 2), "CB_mt": round(cb_mt, 2),
                   "iv_wt": round(iv_wt, 2), "iv_mt": round(iv_mt, 2), "n_mut": n_m}
            rows.append(rec)
            print(f"[{ab:<13} {strain.split()[0]:<8}] paper={paper_d:+.2f}  "
                  f"CB={rec['CB_dipae']:+.3f}  indiv={rec['indiv_dipae']:+.3f}  "
                  f"(CB {cb_wt:.1f}->{cb_mt:.1f} | iv {iv_wt:.1f}->{iv_mt:.1f}, {n_m}mut)", flush=True)
        except Exception as exc:
            print(f"[{ab}] FAILED: {exc}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(os.path.dirname(__file__), "cb_vs_individual.csv"), index=False)
    print("\n" + out.to_string(index=False))
    if len(out) >= 3:
        from scipy.stats import pearsonr
        for k in ["CB_dipae", "indiv_dipae"]:
            r = pearsonr(out.paper_dipae, out[k])[0]
            mae = (out[k] - out.paper_dipae).abs().mean()
            print(f"\n{k:<12}: Pearson vs paper = {r:+.3f}   MAE = {mae:.3f}")


if __name__ == "__main__":
    main()
