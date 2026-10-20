#!/usr/bin/env python
"""Which recycle count best reproduces the paper's iPAE?

Individual point-mutation approach (the validated one), paper settings
(gap=25, masked template, all-pairs antibody<->RBD iPAE), but run with 5 rounds
of recycling and the all-pairs delta-iPAE read at EVERY cycle (0..4). Compares
each cycle's delta-iPAE to the paper's Fig 2g anchors -> tells us if 3 rounds
(the paper's stated value) is really optimal, or if 2 / 4 / 5 is better.
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_full as R
from aftoolkit.structure import load_protein, mutate_protein
from aftoolkit.mutations import parse_mutations
from aftoolkit.scoring import interface_pae

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GAP, N_RECYCLE = 25, 4                      # 5 rounds -> cycles 0..4
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
    cycles = list(range(N_RECYCLE + 1))

    from aftoolkit.folding import OpenFoldWrapper
    wrapper = OpenFoldWrapper(device="mps", inference_n_recycle=N_RECYCLE,
                              always_use_template=False, return_all_cycles=True)
    wrapper.init_model()
    tdir = os.path.join(os.path.dirname(__file__), "cbvi_tmp"); os.makedirs(tdir, exist_ok=True)

    def ipae_all_cycles(df, rbd_chain, ab_chains):
        wrapper.evaluate(df)
        ca = wrapper.get_of_protein(); ca = ca[ca.atom_name == "CA"].reset_index(drop=True)
        labels = ca["chain_id_original"].tolist()
        return {c: interface_pae(wrapper.of_output_per_cycle[c]["predicted_aligned_error"],
                                 labels, [rbd_chain], ab_chains) for c in cycles}

    rows = []
    for ab, strain, paper_d in POINTS:
        try:
            pdbid = aff[aff.ab == ab].pdbid.iloc[0]
            pdb = R.fetch_pdb(pdbid)
            rbd_chain, rbd_resnums, ab_chains = R.select_complex(pdb, rbd_ref)
            seqs = R.chain_sequences(pdb)
            ab_var = {c: R.variable_span(seqs[c][0]) for c in ab_chains}
            trimmed = R.trim_complex(pdb, rbd_chain, rbd_resnums, ab_chains, ab_var,
                                     os.path.join(tdir, f"{ab}_fv.pdb"))
            cseq, cresn = R.chain_sequences(trimmed)[rbd_chain]
            ok, why, _, ident = R.rbd_sanity_check(rbd_ref, strains[R.STRAIN_MAP[strain]])
            if not ok:
                print(f"[{ab}] SKIP {strain}: {why} (identity={ident:.3f})", flush=True)
                continue
            def mut_df(target_seq):
                df = load_protein(trimmed, chains=ab_chains + [rbd_chain], chain_gap=GAP)
                muts = R.rbd_mutations(cseq, cresn, target_seq)
                if muts:
                    df = mutate_protein(df, parse_mutations(
                        ",".join(f"{rbd_chain}:{w}{p}{m}" for w, p, m in muts)))
                return df
            wt = ipae_all_cycles(mut_df(rbd_ref), rbd_chain, ab_chains)
            mt = ipae_all_cycles(mut_df(strains[R.STRAIN_MAP[strain]]), rbd_chain, ab_chains)
            rec = {"ab": ab, "strain": strain.split()[0], "paper_dipae": paper_d}
            for c in cycles:
                rec[f"d_c{c}"] = round(-(mt[c] - wt[c]), 3)         # paper sign
            rows.append(rec)
            print(f"[{ab:<13} {rec['strain']:<8}] paper={paper_d:+.2f}  "
                  + " ".join(f"c{c}={rec[f'd_c{c}']:+.2f}" for c in cycles), flush=True)
        except Exception as exc:
            print(f"[{ab}] FAILED: {exc}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(os.path.dirname(__file__), "recycle_sweep.csv"), index=False)
    print("\n" + out.to_string(index=False))
    from scipy.stats import pearsonr
    print(f"\n{'cycle(=rounds-1)':<18}{'Pearson':>9}{'MAE':>8}")
    for c in cycles:
        r = pearsonr(out.paper_dipae, out[f"d_c{c}"])[0]
        mae = (out[f"d_c{c}"] - out.paper_dipae).abs().mean()
        tag = "  <- paper says 3 rounds (c2)" if c == 2 else ""
        print(f"  cycle {c} ({c+1} rounds){r:>+8.3f}{mae:>8.3f}{tag}")


if __name__ == "__main__":
    main()
