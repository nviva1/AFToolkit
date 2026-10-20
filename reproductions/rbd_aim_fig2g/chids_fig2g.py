#!/usr/bin/env python
"""Full Fig 2g, CORRECTED: chids-based chains (from deposited names) + kalign-RBD
crop + ANARCII VH/VL crop, then MUTATE the cropped RBD to each strain. Heavy asserts
(repro_asserts) guard chain/numbering/sequence at every step. Emits a live table:
iPAE_wt | iPAE_mt | diPAE_mine | diPAE_paper. Crash-safe / resumable."""
import csv, json, os, sys, ast, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from biopandas.pdb import PandasPdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_full as R, native_crop as N, repro_asserts as A
from aftoolkit.structure import load_protein, mutate_protein, AA_3_TO_1_DICT
from aftoolkit.mutations import parse_mutations
from aftoolkit.scoring import interface_pae
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__)); GAP = 25
OUT = os.path.join(HERE, "chids_fig2g.csv")


def main():
    strains = json.load(open(os.path.join(ROOT, "data/rbdaim/datasets/rbd_strains.json")))
    ref = strains["Wuhan"]
    aff = pd.read_csv(os.path.join(ROOT, "data/rbdaim/fig2g_affinity.csv"))
    aff = aff[aff.delta_log_kd.abs() > 1e-6]
    pp = pd.read_excel(os.path.expanduser("~/Downloads/41467_2026_70553_MOESM9_ESM.xlsx"),
                       sheet_name="Fig 2g", header=1).dropna(how="all")
    pp.columns = ["dlogkd", "dipae"]; pp = pp.apply(pd.to_numeric, errors="coerce").dropna()
    def paper_d(dk):
        near = (pp.dlogkd - abs(dk)).abs()
        return round(float(pp.dipae[near.idxmin()]), 3) if near.min() < 0.03 else None

    fields = ["antibody", "Xray", "strain", "dlogKD", "iPAE_wt", "iPAE_mt", "diPAE_mine", "diPAE_paper", "mut"]
    done = set()
    if os.path.exists(OUT):
        done = set(zip(pd.read_csv(OUT).antibody, pd.read_csv(OUT).strain))
    fh = open(OUT, "a", newline=""); w = csv.DictWriter(fh, fieldnames=fields)
    if os.path.getsize(OUT) == 0: w.writeheader(); fh.flush()
    def emit(r): w.writerow(r); fh.flush(); os.fsync(fh.fileno())

    from aftoolkit.folding import OpenFoldWrapper
    wr = OpenFoldWrapper(device="mps", inference_n_recycle=2, always_use_template=False,
                         return_all_cycles=True); wr.init_model()
    tdir = os.path.join(HERE, "chids_tmp"); os.makedirs(tdir, exist_ok=True)
    print(f"{'antibody':<14}{'Xray':>6}{'strain':<14}{'dlogKD':>8}{'iPAE_wt':>9}{'iPAE_mt':>9}{'diPAE_mine':>11}{'diPAE_paper':>12}", flush=True)

    for ab, g in aff.groupby("antibody" if "antibody" in aff.columns else "ab"):
        todo = [r for _, r in g.iterrows() if (ab, r.strain) not in done]
        if not todo: continue
        try:
            H, L, RBD = ast.literal_eval(g.chids.iloc[0]); pdbid = g.pdbid.iloc[0]
            df = PandasPdb().read_pdb(R.fetch_pdb(pdbid)).df["ATOM"]; df = df[df.alt_loc.isin(["", "A"])]
            rseq, rresn = N.chain_seq(df, RBD); hseq, hresn = N.chain_seq(df, H); lseq, lresn = N.chain_seq(df, L)
            kr = N.crop_rbd_resns(rseq, rresn, ref); kh = N.vhvl_resns(hseq, hresn); kl = N.vhvl_resns(lseq, lresn)
            parts = [df[(df.chain_id == RBD) & (df.residue_number.isin(kr))].assign(chain_id="1"),
                     df[(df.chain_id == H) & (df.residue_number.isin(kh))],
                     df[(df.chain_id == L) & (df.residue_number.isin(kl))]]
            out = pd.concat(parts).reset_index(drop=True)
            out["atom_number"] = range(1, len(out) + 1); out["line_idx"] = range(len(out))
            fp = os.path.join(tdir, f"{ab}.pdb"); p = PandasPdb(); p.df["ATOM"] = out; p.to_pdb(fp)
            ca = out[(out.chain_id == "1") & (out.atom_name == "CA")].sort_values("residue_number")
            cseq = "".join(AA_3_TO_1_DICT.get(r, "X") for r in ca.residue_name); cresn = list(ca.residue_number)
            # ---- ASSERTS ----
            A.assert_distinct_chains(["1", H, L], ab)
            A.assert_rbd_region(cseq, ref, ab); A.assert_unique_residues(cresn, ab)
            A.assert_canonical_numbering(dict(zip(cresn, cseq)), ab)

            def ipae(mut_str, tgt=None):
                d = load_protein(fp, chains=["1", H, L], chain_gap=GAP)
                if mut_str:
                    d = mutate_protein(d, parse_mutations(mut_str))
                    bca = d[(d.chain_id_original == "1") & (d.atom_name == "CA")].sort_values("residue_number")
                    A.assert_built_matches_target("".join(AA_3_TO_1_DICT.get(r,"X") for r in bca.residue_name), tgt, ab)
                wr.evaluate(d)
                c = wr.get_of_protein(); c = c[c.atom_name == "CA"].reset_index(drop=True)
                lab = np.array(c["chain_id_original"].tolist()); A.assert_interface(lab, ["1"], [H, L], ab)
                pae = np.asarray(wr.of_output["predicted_aligned_error"])
                return float(pae[np.ix_(np.where(lab == "1")[0], np.where(lab != "1")[0])].mean())

            wt = ipae("")
            for r in todo:
                key = R.STRAIN_MAP.get(r.strain)
                if key is None or key not in strains:
                    print(f"{ab:<14}{pdbid:>6}{r.strain.split()[0]:<14}{'SKIP no-seq':>20}", flush=True); continue
                muts = R.rbd_mutations(cseq, cresn, strains[key])
                ms = ",".join(f"1:{wm}{p_}{m}" for wm, p_, m in muts)
                mt = ipae(ms, strains[key]); pv = paper_d(r.delta_log_kd)
                row = {"antibody": ab, "Xray": pdbid, "strain": r.strain, "dlogKD": round(r.delta_log_kd, 3),
                       "iPAE_wt": round(wt, 3), "iPAE_mt": round(mt, 3), "diPAE_mine": round(-(mt - wt), 3),
                       "diPAE_paper": pv, "mut": len(muts)}
                emit(row)
                print(f"{ab:<14}{pdbid:>6}{r.strain.split()[0]:<14}{r.delta_log_kd:>8.2f}{wt:>9.2f}{mt:>9.2f}"
                      f"{-(mt-wt):>11.3f}{(pv if pv is not None else float('nan')):>12}", flush=True)
        except Exception as exc:
            print(f"{ab:<14} FAILED: {exc}", flush=True)
    fh.close()
    o = pd.read_csv(OUT)
    if len(o) >= 4:
        from scipy.stats import pearsonr, spearmanr
        esc = o[o.dlogKD >= 0]
        print(f"\nescape n={len(esc)}: Pearson={pearsonr(esc.dlogKD, esc.diPAE_mine)[0]:+.3f} "
              f"Spearman={spearmanr(esc.dlogKD, esc.diPAE_mine)[0]:+.3f}", flush=True)


if __name__ == "__main__":
    main()
