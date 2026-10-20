#!/usr/bin/env python
"""Run the Fig 2g escape protocol for ALL multi-point antibodies.

Per antibody: build the reference (highest-affinity) state once and every escape
variant, all on the same crystal with no-leak backbone masking, masked-template
inference (gap=25, n_recycle=2). For each variant report BOTH iPAE flavors PER
CYCLE: chains (whole RBD x Ab) and mutantpos (escape residues x Ab), as absolute
ref/var values and the variant-minus-reference delta.

Crash-safe / resumable: one row per (antibody, variant) is flushed immediately;
re-running skips rows already present.

Out: data/rbdaim/escape_ipae_results.csv
"""
import ast, csv, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from biopandas.pdb import PandasPdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_load as M
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
N_RECYCLE = int(sys.argv[1]) if len(sys.argv) > 1 else 2
MODE = sys.argv[2] if len(sys.argv) > 2 else "bb"   # bb = backbone-only at touched (default); cb = global backbone+CB
GAP = int(sys.argv[3]) if len(sys.argv) > 3 else 25  # inter-chain residue gap in positional encoding
NCYC = N_RECYCLE + 1                        # cycles c0 .. c{N_RECYCLE}
TAG = f"r{N_RECYCLE}" + ("_cb" if MODE == "cb" else "") + (f"_g{GAP}" if GAP != 25 else "")
OUT = os.path.join(ROOT, f"data/rbdaim/escape_ipae_results_{TAG}.csv"
                   if TAG != "r2" else "data/rbdaim/escape_ipae_results.csv")
PDBDIR = os.path.join(HERE, f"pred_pdbs_{TAG}"); os.makedirs(PDBDIR, exist_ok=True)
PAEDIR = os.path.join(HERE, f"pae_mats_{TAG}"); os.makedirs(PAEDIR, exist_ok=True)


def safe(s):
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", str(s)).strip("_")


def run_state(wr, df, rbd, save_path=None, npz_path=None):
    wr.evaluate(df)
    full = wr.get_of_protein()
    if save_path is not None:
        out = full.reset_index(drop=True)
        out["atom_number"] = range(1, len(out) + 1); out["line_idx"] = range(len(out))
        p = PandasPdb(); p.df["ATOM"] = out; p.to_pdb(save_path)
    c = full[full.atom_name == "CA"].reset_index(drop=True)
    labels = np.array(c["chain_id_original"].tolist())
    resnums = np.array(c["residue_number_original"].astype(int).tolist())
    xyz = c[["x_coord", "y_coord", "z_coord"]].to_numpy()
    paes = [np.asarray(cy["predicted_aligned_error"]) for cy in wr.of_output_per_cycle]
    if npz_path is not None:
        # full PAE matrices + geometry -> every metric becomes post-hoc, no more re-runs
        np.savez_compressed(npz_path, pae=np.stack(paes), labels=labels,
                            resnums=resnums, xyz=xyz)
    return labels, resnums, paes


def bmean(pae, rows, cols):
    return float(pae[np.ix_(rows, cols)].mean()) if len(rows) and len(cols) else float("nan")


def fields():
    f = ["antibody", "pdbid", "ref_strain", "variant", "var_kd_nM", "ref_kd_nM",
         "dlogKD", "nmut"]
    for k in range(NCYC):
        f += [f"ref_chains_c{k}", f"var_chains_c{k}", f"dCHAINS_c{k}",
              f"ref_mutpos_c{k}", f"var_mutpos_c{k}", f"dMUTPOS_c{k}"]
    return f


def main():
    pairs = pd.read_csv(os.path.join(ROOT, "data/rbdaim/escape_pairs.csv"))
    F = fields()
    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        prev = pd.read_csv(OUT)
        done = set(zip(prev.antibody, prev.variant))
    fh = open(OUT, "a", newline=""); w = csv.DictWriter(fh, fieldnames=F)
    if os.path.getsize(OUT) == 0:
        w.writeheader(); fh.flush()

    def emit(rec):
        w.writerow(rec); fh.flush(); os.fsync(fh.fileno())

    from aftoolkit.folding import OpenFoldWrapper
    wr = OpenFoldWrapper(device="mps", inference_n_recycle=N_RECYCLE,
                         always_use_template=False, return_all_cycles=True)
    wr.init_model()
    PREP = M.prep_state_cb if MODE == "cb" else M.prep_state
    print(f"=== run {TAG}: n_recycle={N_RECYCLE} ({NCYC} cycles), mask_mode={MODE} "
          f"({'global backbone+CB, touched=backbone' if MODE == 'cb' else 'touched=backbone, rest=full'}), "
          f"PDBs -> {PDBDIR} ===", flush=True)

    abs_ = list(pairs.antibody.unique())
    for ai, ab in enumerate(abs_, 1):
        g = pairs[pairs.antibody == ab]
        todo = [r for _, r in g.iterrows()
                if (ab, r.var_strain) not in done
                and not str(r.escape_mutations).startswith("cross-species")]
        if not todo:
            print(f"[{ai}/{len(abs_)}] {ab}: nothing to do", flush=True); continue
        row0 = g.iloc[0]
        try:
            chids = ast.literal_eval(row0.chids); rbd = row0.rbd_chain; chains = chids
            pdb = row0.pdb_path; apply_ref = row0.apply_to_xray_for_ref
            mask_pos = M.positions_in(apply_ref)
            for _, r in g.iterrows():
                mask_pos |= M.positions_in(r.apply_to_xray_for_variant)
            ref_pdb = os.path.join(PDBDIR, f"{safe(ab)}__REF_{safe(row0.ref_strain)}.pdb")
            rl, rn, rp = run_state(wr, PREP(pdb, chains, rbd, apply_ref, mask_pos, GAP),
                                   rbd, save_path=ref_pdb,
                                   npz_path=os.path.join(PAEDIR, f"{safe(ab)}__REF_{safe(row0.ref_strain)}.npz"))
            ab_cols_r = np.where(rl != rbd)[0]; rbd_rows_r = np.where(rl == rbd)[0]
            print(f"[{ai}/{len(abs_)}] {ab} ({row0.pdbid}) ref={row0.ref_strain}: "
                  f"{len(todo)} variants, mask n={len(mask_pos)}", flush=True)

            for r in todo:
                # localized metric over the FIXED masked positions (antibody-wide,
                # reference-independent) -> re-anchors to any baseline by subtraction
                mutpos = sorted(mask_pos)
                var_pdb = os.path.join(PDBDIR, f"{safe(ab)}__{safe(r.var_strain)}.pdb")
                vl, vn, vp = run_state(wr, PREP(pdb, chains, rbd,
                                       r.apply_to_xray_for_variant, mask_pos, GAP),
                                       rbd, save_path=var_pdb,
                                       npz_path=os.path.join(PAEDIR, f"{safe(ab)}__{safe(r.var_strain)}.npz"))
                ab_cols_v = np.where(vl != rbd)[0]; rbd_rows_v = np.where(vl == rbd)[0]
                mrr = np.where((rl == rbd) & np.isin(rn, mutpos))[0]
                mrv = np.where((vl == rbd) & np.isin(vn, mutpos))[0]
                rec = {"antibody": ab, "pdbid": row0.pdbid, "ref_strain": row0.ref_strain,
                       "variant": r.var_strain, "var_kd_nM": r.var_kd_nM,
                       "ref_kd_nM": r.ref_kd_nM, "dlogKD": r.delta_log_kd, "nmut": len(mutpos)}
                for k in range(NCYC):
                    rc = bmean(rp[k], rbd_rows_r, ab_cols_r); vc = bmean(vp[k], rbd_rows_v, ab_cols_v)
                    rm = bmean(rp[k], mrr, ab_cols_r); vm = bmean(vp[k], mrv, ab_cols_v)
                    rec[f"ref_chains_c{k}"] = round(rc, 3); rec[f"var_chains_c{k}"] = round(vc, 3)
                    rec[f"dCHAINS_c{k}"] = round(vc - rc, 3)
                    rec[f"ref_mutpos_c{k}"] = round(rm, 3); rec[f"var_mutpos_c{k}"] = round(vm, 3)
                    rec[f"dMUTPOS_c{k}"] = round(vm - rm, 3)
                emit(rec)
                print(f"    {r.var_strain:<26} dlogKD={r.delta_log_kd:+.2f} "
                      f"dCHAINS_c2={rec['dCHAINS_c2']:+.3f} dMUTPOS_c2={rec['dMUTPOS_c2']:+.3f}", flush=True)
        except Exception as exc:
            print(f"[{ai}/{len(abs_)}] {ab} FAILED: {exc}", flush=True)
    fh.close()

    # pooled correlations
    o = pd.read_csv(OUT)
    if len(o) >= 4:
        from scipy.stats import pearsonr, spearmanr
        print(f"\n=== pooled over {len(o)} escape variants / {o.antibody.nunique()} antibodies ===", flush=True)
        for k in range(NCYC):
            for col in (f"dCHAINS_c{k}", f"dMUTPOS_c{k}"):
                v = o[o[col].notna()]
                print(f"  {col}: Pearson={pearsonr(v.dlogKD, v[col])[0]:+.3f} "
                      f"Spearman={spearmanr(v.dlogKD, v[col])[0]:+.3f}", flush=True)


if __name__ == "__main__":
    main()
