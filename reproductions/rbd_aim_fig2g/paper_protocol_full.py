#!/usr/bin/env python
"""Full Fig 2g recompute under the PAPER-FAITHFUL protocol (RBDAIM/inference.py):

  RBD  = ref.pdb (chain '1'), SUPERPOSED onto the crystal RBD, strain sequence
         THREADED on (mutate_protein: full sidechain at conserved, [N,C,CA,O] at
         mutated).  -- never the crystal RBD.
  Fab  = crystal heavy+light (VH/VL trimmed).
  gap = 25, masked template, 3 recycles (n_recycle=2), iPAE = all-pairs RBD->Ab.

Indel strains (need insertion vs Wuhan, e.g. JN.1) are skipped. Incremental,
crash-safe, resumable CSV.
"""
import csv, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_full as R
import paper_protocol as PP                      # build_combined, REF_PDB, GAP, RBD_CH
from aftoolkit.structure import load_protein, mutate_protein
from aftoolkit.mutations import parse_mutations
from aftoolkit.scoring import interface_pae

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GAP, RBD_CH = PP.GAP, PP.RBD_CH
OUT = os.path.join(os.path.dirname(__file__), "paper_protocol_full.csv")


def main():
    strains = json.load(open(os.path.join(ROOT, "data/rbdaim/datasets/rbd_strains.json")))
    rbd_ref = strains["Wuhan"]
    ref_resnums = list(range(334, 334 + len(rbd_ref)))
    aff = pd.read_csv(os.path.join(ROOT, "data/rbdaim/fig2g_affinity.csv"))
    aff = aff[aff.delta_log_kd.abs() > 1e-6]

    fields = ["ab", "pdbid", "strain", "delta_log_kd", "ipae", "ipae_ref", "delta_ipae", "n_mut"]
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT); done = set(zip(prev.ab, prev.strain))
        print(f"resuming: {len(done)} rows done", flush=True)
    fh = open(OUT, "a", newline=""); w = csv.DictWriter(fh, fieldnames=fields)
    if os.path.getsize(OUT) == 0 if os.path.exists(OUT) else True:
        if not done:
            w.writeheader(); fh.flush()
    def emit(r): w.writerow(r); fh.flush(); os.fsync(fh.fileno())

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
        return interface_pae(wrapper.of_output["predicted_aligned_error"], labels, [RBD_CH], ab_chains)

    for ab, g in aff.groupby("ab"):
        todo = [r for _, r in g.iterrows() if (ab, r.strain) not in done]
        if not todo:
            continue
        try:
            pdbid = g.pdbid.iloc[0]
            pdb = R.fetch_pdb(pdbid)
            rbd_chain, rbd_resnums, ab_chains = R.select_complex(pdb, rbd_ref)
            ab_var = {c: R.variable_span(R.chain_sequences(pdb)[c][0]) for c in ab_chains}
            combined = PP.build_combined(pdb, rbd_chain, rbd_resnums, ab_chains, ab_var,
                                         os.path.join(tdir, f"{ab}_ref_fab.pdb"))
            ipae_ref = ipae(combined, ab_chains, "")          # ref.pdb WT + Fab
            print(f"[{ab}/{pdbid}] Ab={ab_chains}  iPAE(WT ref)={ipae_ref:.3f}", flush=True)
            for r in todo:
                key = R.STRAIN_MAP.get(r.strain)
                if key is None or key not in strains:
                    continue
                del_resnums, reproducible = R.strain_plan(rbd_ref, strains[key])
                if not reproducible:
                    print(f"    SKIP {r.strain:<22} (insertion/complex indel)", flush=True)
                    continue
                muts = R.rbd_mutations(rbd_ref, ref_resnums, strains[key])
                mut_str = ",".join(f"{RBD_CH}:{w_}{p}{m}" for w_, p, m in muts)
                ip = ipae(combined, ab_chains, mut_str)
                emit({"ab": ab, "pdbid": pdbid, "strain": r.strain, "delta_log_kd": r.delta_log_kd,
                      "ipae": round(ip, 4), "ipae_ref": round(ipae_ref, 4),
                      "delta_ipae": round(ip - ipae_ref, 4), "n_mut": len(muts)})
                print(f"    {r.strain:<24} dlogKD={r.delta_log_kd:+.2f} iPAE={ip:.2f} "
                      f"diPAE={ip-ipae_ref:+.3f} ({len(muts)}mut)", flush=True)
        except Exception as exc:
            print(f"[{ab}] FAILED: {exc}", flush=True)
    fh.close()

    out = pd.read_csv(OUT)
    if len(out) >= 3:
        from scipy.stats import pearsonr, spearmanr
        print(f"\nn={len(out)}  HEADLINE diPAE vs dlogKD: Pearson="
              f"{pearsonr(out.delta_log_kd, out.delta_ipae)[0]:+.3f} "
              f"Spearman={spearmanr(out.delta_log_kd, out.delta_ipae)[0]:+.3f}  (paper |R|=0.52, Sp 0.63)")


if __name__ == "__main__":
    main()
