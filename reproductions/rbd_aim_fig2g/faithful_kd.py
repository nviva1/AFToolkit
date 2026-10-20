#!/usr/bin/env python
"""Validated faithful pipeline WITH log-KD: for one antibody, thread each RBD strain
onto ref.pdb (their homology_model) on its NN-selected Fab template, score
one-direction iPAE, and test delta-iPAE vs experimental delta-log10[KD].
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_faithful as PF                      # their load_protein + homology_model builder
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    job = json.load(open("/tmp/cr3022_kd.json"))
    from aftoolkit.structure import load_protein
    from aftoolkit.folding import OpenFoldWrapper
    wrapper = OpenFoldWrapper(device="mps", inference_n_recycle=2,
                              always_use_template=False, return_all_cycles=True)
    wrapper.init_model()
    tdir = os.path.join(HERE, "faithful_tmp"); os.makedirs(tdir, exist_ok=True)

    def ipae(rbd_seq, tag):
        out, chains = PF.build(job["tpl"], job["H"], job["L"], rbd_seq,
                               os.path.join(tdir, f"{job['ab']}_{tag}.pdb"))
        df = load_protein(out, chains=chains, chain_gap=25)
        wrapper.evaluate(df)
        ca = wrapper.get_of_protein(); ca = ca[ca.atom_name == "CA"].reset_index(drop=True)
        labels = np.array(ca["chain_id_original"].tolist())
        pae = np.asarray(wrapper.of_output["predicted_aligned_error"])
        ids1 = np.where(labels == "1")[0]; ids2 = np.where(labels != "1")[0]
        return float(pae[np.ix_(ids1, ids2)].mean())

    wt = ipae(job["wuhan"], "WT")
    print(f"[{job['ab']}] tpl={job['tpl']}  WT iPAE={wt:.3f}", flush=True)
    rows = []
    for s in job["strains"]:
        ip = ipae(s["seq"], s["strain"].split()[0].replace("(", "").replace(")", ""))
        rows.append({"strain": s["strain"], "dlogkd": s["dlogkd"], "ipae": round(ip, 3),
                     "delta_ipae": round(ip - wt, 3)})
        print(f"   {s['strain']:<26} dlogKD={s['dlogkd']:+.2f}  iPAE={ip:.3f}  "
              f"diPAE={ip-wt:+.3f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "faithful_kd_cr3022.csv"), index=False)
    nz = out[out.dlogkd.abs() > 1e-6]
    if len(nz) >= 3:
        print(f"\ndelta-iPAE vs delta-logKD (n={len(nz)}): "
              f"Pearson={pearsonr(nz.dlogkd, nz.delta_ipae)[0]:+.3f}  "
              f"Spearman={spearmanr(nz.dlogkd, nz.delta_ipae)[0]:+.3f}")


if __name__ == "__main__":
    main()
