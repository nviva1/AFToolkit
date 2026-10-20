#!/usr/bin/env python
"""Compute ALL iPAE metric variants from the saved PAE matrices (pae_mats_<TAG>/),
no inference. Every metric is Δ = variant - reference, per cycle.

RBD row sets:   all RBD | union (all-ever-mutated) | per-variant escape positions
Ab  col sets:   all antibody | CDR loops (AbNumber) | interface (8 A) | interface (12 A)
Interface:      RBD rows ALSO restricted to interface (RBD_if x Ab_if), CA-CA < thresh
Reductions:     mean and sum (= mean * n_rows) for the mutated-row metrics.
Interface contacts are defined on the REFERENCE structure (shared indices).

Usage: python analyze_metrics.py [TAG]   (default r2)
Out: data/rbdaim/metrics_<TAG>.csv  + printed Rp/Rs leaderboard.
"""
import ast, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from biopandas.pdb import PandasPdb
from abnumber import Chain
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_load as M, native_crop as N
from scipy.stats import pearsonr, spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "r2"
PAEDIR = os.path.join(HERE, f"pae_mats_{TAG}")


def safe(s):
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", str(s)).strip("_")


def cdr_resnums(xray_pdb, H, L):
    df = PandasPdb().read_pdb(xray_pdb).df["ATOM"]; df = df[df.alt_loc.isin(["", "A"])]
    s = set()
    for ch in dict.fromkeys([H, L]):
        seq, resns = N.chain_seq(df, ch)
        for c in Chain.multiple_domains(seq, scheme="imgt"):
            for cdr in (c.cdr1_seq, c.cdr2_seq, c.cdr3_seq):
                i = seq.find(cdr)
                if i >= 0:
                    s.update((ch, int(r)) for r in resns[i:i + len(cdr)])
    return s


def load(ab, strain, is_ref, ref_strain):
    fn = f"{safe(ab)}__REF_{safe(ref_strain)}.npz" if is_ref else f"{safe(ab)}__{safe(strain)}.npz"
    p = os.path.join(PAEDIR, fn)
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def main():
    pairs = pd.read_csv(os.path.join(ROOT, "data/rbdaim/escape_pairs.csv"))
    rows = []
    for ab, g in pairs.groupby("antibody"):
        row0 = g.iloc[0]; chids = ast.literal_eval(row0.chids); rbd = row0.rbd_chain
        if str(row0.apply_to_xray_for_variant).startswith("cross-species"):
            pass
        refz = load(ab, None, True, row0.ref_strain)
        if refz is None:
            continue
        cdr = cdr_resnums(row0.pdb_path, chids[0], chids[1])
        mask_pos = M.positions_in(row0.apply_to_xray_for_ref)
        for _, r in g.iterrows():
            mask_pos |= M.positions_in(r.apply_to_xray_for_variant)

        lab = refz["labels"]; rn = refz["resnums"].astype(int); xyz = refz["xyz"]
        ncyc = refz["pae"].shape[0]
        R = np.where(lab == rbd)[0]; A = np.where(lab != rbd)[0]
        Acdr = np.array([i for i in A if (lab[i], int(rn[i])) in cdr], int)
        Run = np.array([i for i in R if int(rn[i]) in mask_pos], int)      # union
        # interface on the reference structure (shared indices)
        d = np.linalg.norm(xyz[R][:, None] - xyz[A][None], axis=-1)
        def iface(th):
            return R[d.min(1) < th], A[d.min(0) < th]
        Ri8, Ai8 = iface(8.0); Ri12, Ai12 = iface(12.0)

        for _, r in g.iterrows():
            if str(r.escape_mutations).startswith("cross-species"):
                continue
            vz = load(ab, r.var_strain, False, row0.ref_strain)
            if vz is None:
                continue
            esc = M.positions_in(r.escape_mutations)
            Rvar = np.array([i for i in R if int(rn[i]) in esc], int)
            Rvar_i8 = np.array([i for i in Rvar if i in set(Ri8.tolist())], int)
            rec = {"antibody": ab, "variant": r.var_strain, "dlogKD": r.delta_log_kd,
                   "n_esc": len(esc), "n_union": len(mask_pos)}
            for k in range(ncyc):
                P = vz["pae"][k] - refz["pae"][k]                 # ΔPAE matrix
                blk = lambda ri, ci: float(P[np.ix_(ri, ci)].mean()) if len(ri) and len(ci) else np.nan
                rec[f"chains_c{k}"]     = blk(R, A)
                rec[f"cdr_c{k}"]        = blk(R, Acdr)
                rec[f"munion_ab_c{k}"]  = blk(Run, A)
                rec[f"mvar_ab_c{k}"]    = blk(Rvar, A)
                rec[f"mvar_cdr_c{k}"]   = blk(Rvar, Acdr)
                rec[f"iface8_c{k}"]     = blk(Ri8, Ai8)
                rec[f"iface12_c{k}"]    = blk(Ri12, Ai12)
                rec[f"mvar_if8_c{k}"]   = blk(Rvar_i8, Ai8)
                # sum versions (mean * n_rows) for per-variant mutated sets
                mv = blk(Rvar, A); rec[f"SUM_mvar_ab_c{k}"]  = mv * len(esc) if mv == mv else 0.0
                mc = blk(Rvar, Acdr); rec[f"SUM_mvar_cdr_c{k}"] = mc * len(esc) if mc == mc else 0.0
            rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, f"data/rbdaim/metrics_{TAG}.csv"), index=False)
    print(f"wrote data/rbdaim/metrics_{TAG}.csv  ({len(out)} variants, {out.antibody.nunique()} antibodies)\n")
    ncyc = out["chains_c0"].notna().sum() and (max(int(c.split("_c")[1]) for c in out.columns if c.startswith("chains_c")) + 1)
    metric_keys = ["chains", "cdr", "munion_ab", "mvar_ab", "mvar_cdr", "iface8", "iface12",
                   "mvar_if8", "SUM_mvar_ab", "SUM_mvar_cdr"]
    print("Leaderboard (Rp / Rs vs dlogKD), best cycle per metric:")
    res = []
    for key in metric_keys:
        for k in range(ncyc):
            c = f"{key}_c{k}"
            if c not in out: continue
            v = out[out[c].notna()]
            if len(v) < 4: continue
            res.append((key, k, pearsonr(v.dlogKD, v[c])[0], spearmanr(v.dlogKD, v[c])[0], len(v)))
    best = {}
    for key, k, rp, rs, n in res:
        if key not in best or abs(rp) > abs(best[key][2]):
            best[key] = (key, k, rp, rs, n)
    for key in metric_keys:
        if key in best:
            _, k, rp, rs, n = best[key]
            print(f"  {key:<14} best c{k}:  Rp={rp:+.3f}  Rs={rs:+.3f}  (n={n})")


if __name__ == "__main__":
    main()
