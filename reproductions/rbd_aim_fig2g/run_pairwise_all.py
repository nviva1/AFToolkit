#!/usr/bin/env python
"""CORRECT per-pair no-leak over the WHOLE panel. For each (antibody, variant):
mask = positions(ref-vs-crystal) UNION positions(variant-vs-crystal) -- specific to
that pair -- and build+infer BOTH the reference and the variant fresh with that mask
(reference recomputed per variant). ΔiPAE = variant - its own pair-masked reference.

Crash-safe / resumable. Out: data/rbdaim/pairwise_all.csv
"""
import ast, csv, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_load as M
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data/rbdaim/pairwise_all.csv")
GAP = 25; NCYC = 4


def state(wr, df, rbd):
    wr.evaluate(df)
    c = wr.get_of_protein(); c = c[c.atom_name == "CA"].reset_index(drop=True)
    lab = np.array(c["chain_id_original"].tolist())
    rn = np.array(c["residue_number_original"].astype(int).tolist())
    return lab, rn, [np.asarray(cy["predicted_aligned_error"]) for cy in wr.of_output_per_cycle]


def fields():
    f = ["antibody", "variant", "dlogKD", "n_mask", "n_esc"]
    for k in range(NCYC):
        f += [f"ref_chains_c{k}", f"var_chains_c{k}", f"dCHAINS_c{k}",
              f"ref_mutpos_c{k}", f"var_mutpos_c{k}", f"dMUTPOS_c{k}"]
    return f


def main():
    pairs = pd.read_csv(os.path.join(ROOT, "data/rbdaim/escape_pairs.csv"))
    F = fields(); done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        prev = pd.read_csv(OUT); done = set(zip(prev.antibody, prev.variant))
    fh = open(OUT, "a", newline=""); w = csv.DictWriter(fh, fieldnames=F)
    if os.path.getsize(OUT) == 0:
        w.writeheader(); fh.flush()

    from aftoolkit.folding import OpenFoldWrapper
    wr = OpenFoldWrapper(device="mps", inference_n_recycle=3,
                         always_use_template=False, return_all_cycles=True)
    wr.init_model()
    abs_ = list(pairs.antibody.unique())
    bm = lambda P, R, C: float(P[np.ix_(R, C)].mean()) if len(R) and len(C) else np.nan

    for ai, ab in enumerate(abs_, 1):
        g = pairs[pairs.antibody == ab]; row0 = g.iloc[0]
        todo = [r for _, r in g.iterrows()
                if (ab, r.var_strain) not in done
                and not str(r.escape_mutations).startswith("cross-species")]
        if not todo:
            print(f"[{ai}/{len(abs_)}] {ab}: nothing to do", flush=True); continue
        try:
            chids = ast.literal_eval(row0.chids); rbd = row0.rbd_chain; chains = chids
            pdb = row0.pdb_path; apply_ref = row0.apply_to_xray_for_ref
            ref_pos = M.positions_in(apply_ref)
            print(f"[{ai}/{len(abs_)}] {ab} ({row0.pdbid}) ref={row0.ref_strain}: {len(todo)} variants", flush=True)
            for r in todo:
                var_pos = M.positions_in(r.apply_to_xray_for_variant)
                pair_mask = ref_pos | var_pos
                esc = M.positions_in(r.escape_mutations)
                rl, rn, rp = state(wr, M.prep_state(pdb, chains, rbd, apply_ref, pair_mask, GAP), rbd)
                vl, vn, vp = state(wr, M.prep_state(pdb, chains, rbd, r.apply_to_xray_for_variant, pair_mask, GAP), rbd)
                Ar = np.where(rl != rbd)[0]; Rr = np.where(rl == rbd)[0]
                Av = np.where(vl != rbd)[0]; Rv = np.where(vl == rbd)[0]
                Rr_e = np.where((rl == rbd) & np.isin(rn, list(esc)))[0]
                Rv_e = np.where((vl == rbd) & np.isin(vn, list(esc)))[0]
                rec = {"antibody": ab, "variant": r.var_strain, "dlogKD": r.delta_log_kd,
                       "n_mask": len(pair_mask), "n_esc": len(esc)}
                for k in range(NCYC):
                    rc = bm(rp[k], Rr, Ar); vc = bm(vp[k], Rv, Av)
                    rm = bm(rp[k], Rr_e, Ar); vm = bm(vp[k], Rv_e, Av)
                    rec[f"ref_chains_c{k}"] = round(rc, 3); rec[f"var_chains_c{k}"] = round(vc, 3)
                    rec[f"dCHAINS_c{k}"] = round(vc - rc, 3)
                    rec[f"ref_mutpos_c{k}"] = round(rm, 3); rec[f"var_mutpos_c{k}"] = round(vm, 3)
                    rec[f"dMUTPOS_c{k}"] = round(vm - rm, 3)
                w.writerow(rec); fh.flush(); os.fsync(fh.fileno())
                print(f"    {r.var_strain:<26} mask={len(pair_mask):2d} dCHAINS_c1={rec['dCHAINS_c1']:+.3f} dMUTPOS_c1={rec['dMUTPOS_c1']:+.3f}", flush=True)
        except Exception as exc:
            print(f"[{ai}/{len(abs_)}] {ab} FAILED: {exc}", flush=True)
    fh.close()

    o = pd.read_csv(OUT)
    if len(o) >= 4:
        from scipy.stats import pearsonr, spearmanr
        print(f"\n=== PER-PAIR pooled, {len(o)} variants / {o.antibody.nunique()} antibodies ===", flush=True)
        for k in range(NCYC):
            for c in (f"dCHAINS_c{k}", f"dMUTPOS_c{k}"):
                v = o[o[c].notna()]
                print(f"  {c}: Rp={pearsonr(v.dlogKD, v[c])[0]:+.3f} Rs={spearmanr(v.dlogKD, v[c])[0]:+.3f} (n={len(v)})", flush=True)


if __name__ == "__main__":
    main()
