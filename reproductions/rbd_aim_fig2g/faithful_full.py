#!/usr/bin/env python
"""FULL Fig 2g reproduction on the VALIDATED faithful pipeline:
RBD = ref.pdb threaded with the strain sequence (their homology_model, kalign),
Fab = the antibody's NN-selected template (their load_protein), gap=25,
masked template, n_recycle=2, one-direction iPAE = mean(pae[RBD][:,Ab]).
delta-iPAE = iPAE(strain) - iPAE(WT). Incremental, crash-safe, resumable.
"""
import csv, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_faithful as PF
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "faithful_full.csv")


def main():
    job = json.load(open("/tmp/full_job.json"))
    wuhan = job["wuhan"]
    from aftoolkit.structure import load_protein
    from aftoolkit.folding import OpenFoldWrapper
    wrapper = OpenFoldWrapper(device="mps", inference_n_recycle=2,
                              always_use_template=False, return_all_cycles=True)
    wrapper.init_model()
    tdir = os.path.join(HERE, "faithful_tmp"); os.makedirs(tdir, exist_ok=True)

    fields = ["ab", "tpl", "strain", "dlogkd", "ipae", "ipae_ref", "delta_ipae"]
    done, ref_cache = set(), {}
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.ab, prev.strain)); ref_cache = dict(zip(prev.ab, prev.ipae_ref))
        print(f"resuming: {len(done)} rows done", flush=True)
    fh = open(OUT, "a", newline=""); w = csv.DictWriter(fh, fieldnames=fields)
    if os.path.getsize(OUT) == 0:
        w.writeheader(); fh.flush()
    def emit(r): w.writerow(r); fh.flush(); os.fsync(fh.fileno())

    def ipae(tpl, H, L, rbd_seq, tag):
        out, chains = PF.build(tpl, H, L, rbd_seq, os.path.join(tdir, f"{tag}.pdb"))
        df = load_protein(out, chains=chains, chain_gap=25)
        wrapper.evaluate(df)
        ca = wrapper.get_of_protein(); ca = ca[ca.atom_name == "CA"].reset_index(drop=True)
        labels = np.array(ca["chain_id_original"].tolist())
        pae = np.asarray(wrapper.of_output["predicted_aligned_error"])
        i1 = np.where(labels == "1")[0]; i2 = np.where(labels != "1")[0]
        return float(pae[np.ix_(i1, i2)].mean())

    for j in job["jobs"]:
        ab = j["ab"]
        todo = [s for s in j["strains"] if (ab, s["strain"]) not in done]
        if not todo:
            continue
        try:
            if ab in ref_cache and not pd.isna(ref_cache[ab]):
                wt = float(ref_cache[ab])
            else:
                wt = ipae(j["tpl"], j["H"], j["L"], wuhan, f"{ab}_WT")
                ref_cache[ab] = wt
            print(f"[{ab}] tpl={j['tpl']} WT iPAE={wt:.3f}", flush=True)
            for s in todo:
                ip = ipae(j["tpl"], j["H"], j["L"], s["seq"],
                          f"{ab}_{s['strain'].split()[0].replace('(','').replace(')','')}")
                emit({"ab": ab, "tpl": j["tpl"], "strain": s["strain"], "dlogkd": s["dlogkd"],
                      "ipae": round(ip, 4), "ipae_ref": round(wt, 4),
                      "delta_ipae": round(ip - wt, 4)})
                print(f"   {s['strain']:<26} dlogKD={s['dlogkd']:+.2f} iPAE={ip:.3f} "
                      f"diPAE={ip-wt:+.3f}", flush=True)
        except Exception as exc:
            print(f"[{ab}] FAILED: {exc}", flush=True)
    fh.close()

    out = pd.read_csv(OUT); nz = out[out.dlogkd.abs() > 1e-6]
    if len(nz) >= 3:
        from scipy.stats import pearsonr, spearmanr
        print(f"\n=== FULL Fig 2g (faithful pipeline), n={len(nz)} ===")
        print(f"delta-iPAE vs delta-logKD: Pearson={pearsonr(nz.dlogkd, nz.delta_ipae)[0]:+.3f} "
              f"Spearman={spearmanr(nz.dlogkd, nz.delta_ipae)[0]:+.3f}  (paper |R|=0.52, Sp 0.63)")


if __name__ == "__main__":
    main()
