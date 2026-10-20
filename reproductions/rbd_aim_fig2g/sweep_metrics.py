#!/usr/bin/env python
"""Exhaustive post-hoc sweep of iPAE metric definitions from saved PAE matrices.
Grid = {RBD row set} x {Ab col set / interface} x {direction} x {reduction} x {cycle}.
Ranks every candidate by Pearson and Spearman vs ΔlogK_D. No inference.

Row sets:  allRBD | union(all-ever-mutated) | mut(per-variant escape)
Col sets:  allAb | CDR
Interface combos (both sides restricted, heavy-atom < th): iface{8,10,12}, mutiface{8,10,12}
Reductions: mean, sum(=mean*nrows), min, p10  (+ symmetric mean: avg of both block directions)

Usage: python sweep_metrics.py [TAG]      Out: data/rbdaim/sweep_<TAG>.csv
"""
import ast, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from biopandas.pdb import PandasPdb
from abnumber import Chain
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_load as M, native_crop as N
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "r3"
PAEDIR = os.path.join(HERE, f"pae_mats_{TAG}")
THS = [8.0, 10.0, 12.0]


def safe(s): return re.sub(r"[^A-Za-z0-9.+-]+", "_", str(s)).strip("_")


def cdr_set(xray, H, L):
    df = PandasPdb().read_pdb(xray).df["ATOM"]; df = df[df.alt_loc.isin(["", "A"])]
    s = set()
    for ch in dict.fromkeys([H, L]):
        seq, resns = N.chain_seq(df, ch)
        for c in Chain.multiple_domains(seq, scheme="imgt"):
            for cdr in (c.cdr1_seq, c.cdr2_seq, c.cdr3_seq):
                i = seq.find(cdr)
                if i >= 0: s.update((ch, int(r)) for r in resns[i:i + len(cdr)])
    return s


def iface_sets(xray, rbd, ab_chains, th):
    df = PandasPdb().read_pdb(xray).df["ATOM"]; df = df[df.alt_loc.isin(["", "A"])]
    df = df[df.element_symbol != "H"]
    ratoms = {int(r): g[["x_coord", "y_coord", "z_coord"]].to_numpy()
              for r, g in df[df.chain_id == rbd].groupby("residue_number")}
    aatoms = {}
    for ch in ab_chains:
        for r, g in df[df.chain_id == ch].groupby("residue_number"):
            aatoms[(ch, int(r))] = g[["x_coord", "y_coord", "z_coord"]].to_numpy()
    A = np.vstack(list(aatoms.values())); R = np.vstack(list(ratoms.values()))
    rif = {rn for rn, at in ratoms.items() if cdist(at, A).min() < th}
    aif = {k for k, at in aatoms.items() if cdist(at, R).min() < th}
    return rif, aif


def reductions(P, rows, cols):
    """Return dict of scalar reductions of the ΔPAE block rows x cols (+ symmetric mean)."""
    if len(rows) == 0 or len(cols) == 0:
        return {r: np.nan for r in ["mean", "sum", "min", "p10", "sym"]}
    blk = P[np.ix_(rows, cols)]
    rev = P[np.ix_(cols, rows)]
    return {"mean": float(blk.mean()), "sum": float(blk.mean() * len(rows)),
            "min": float(blk.min()), "p10": float(np.percentile(blk, 10)),
            "sym": float(0.5 * (blk.mean() + rev.mean()))}


def main():
    pairs = pd.read_csv(os.path.join(ROOT, "data/rbdaim/escape_pairs.csv"))
    records = []           # one row per (antibody, variant) with all metric scalars
    for ab, g in pairs.groupby("antibody"):
        row0 = g.iloc[0]; chids = ast.literal_eval(row0.chids); rbd = row0.rbd_chain
        ab_ch = [c for c in chids if c != rbd]
        refnpz = os.path.join(PAEDIR, f"{safe(ab)}__REF_{safe(row0.ref_strain)}.npz")
        if not os.path.exists(refnpz): continue
        rz = np.load(refnpz, allow_pickle=True)
        lab = rz["labels"]; rn = rz["resnums"].astype(int); ncyc = rz["pae"].shape[0]
        mask_pos = M.positions_in(row0.apply_to_xray_for_ref)
        for _, r in g.iterrows(): mask_pos |= M.positions_in(r.apply_to_xray_for_variant)
        cdr = cdr_set(row0.pdb_path, chids[0], chids[1])
        ifsets = {th: iface_sets(row0.pdb_path, rbd, ab_ch, th) for th in THS}

        Rall = np.where(lab == rbd)[0]; Aall = np.where(lab != rbd)[0]
        Acdr = np.array([i for i in Aall if (lab[i], int(rn[i])) in cdr], int)
        Run = np.array([i for i in Rall if int(rn[i]) in mask_pos], int)
        Rif = {th: np.array([i for i in Rall if int(rn[i]) in ifsets[th][0]], int) for th in THS}
        Aif = {th: np.array([i for i in Aall if (lab[i], int(rn[i])) in ifsets[th][1]], int) for th in THS}

        for _, r in g.iterrows():
            if str(r.escape_mutations).startswith("cross-species"): continue
            vp = os.path.join(PAEDIR, f"{safe(ab)}__{safe(r.var_strain)}.npz")
            if not os.path.exists(vp): continue
            vz = np.load(vp, allow_pickle=True)["pae"]
            esc = M.positions_in(r.escape_mutations)
            Rmut = np.array([i for i in Rall if int(rn[i]) in esc], int)
            rec = {"antibody": ab, "variant": r.var_strain, "dlogKD": r.delta_log_kd}
            # selections: (name, rows, cols)
            for k in range(ncyc):
                P = vz[k] - rz["pae"][k]
                sels = [("allRBDxAb", Rall, Aall), ("allRBDxCDR", Rall, Acdr),
                        ("unionxAb", Run, Aall), ("unionxCDR", Run, Acdr),
                        ("mutxAb", Rmut, Aall), ("mutxCDR", Rmut, Acdr)]
                for th in THS:
                    Rmif = np.array([i for i in Rmut if i in set(Rif[th].tolist())], int)
                    sels += [(f"iface{int(th)}", Rif[th], Aif[th]),
                             (f"mutiface{int(th)}", Rmif, Aif[th])]
                for nm, R, C in sels:
                    for red, val in reductions(P, R, C).items():
                        rec[f"{nm}|{red}|c{k}"] = val
            records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(ROOT, f"data/rbdaim/sweep_{TAG}.csv"), index=False)
    cols = [c for c in df.columns if "|" in c]
    res = []
    for c in cols:
        v = df[df[c].notna()]
        if len(v) >= 10 and v[c].std() > 1e-9:
            res.append((c, pearsonr(v.dlogKD, v[c])[0], spearmanr(v.dlogKD, v[c])[0], len(v)))
    R = pd.DataFrame(res, columns=["metric", "Rp", "Rs", "n"])
    R["absRp"] = R.Rp.abs(); R["absRs"] = R.Rs.abs()
    R.sort_values("absRp", ascending=False).to_csv(os.path.join(ROOT, f"data/rbdaim/sweep_{TAG}_ranked.csv"), index=False)
    print(f"swept {len(cols)} metric definitions over {len(df)} variants / {df.antibody.nunique()} antibodies\n")
    print("=== TOP 15 by |Pearson| ===")
    print(R.sort_values("absRp", ascending=False).head(15)[["metric", "Rp", "Rs", "n"]].to_string(index=False))
    print("\n=== TOP 15 by |Spearman| (n>=40 only) ===")
    print(R[R.n >= 40].sort_values("absRs", ascending=False).head(15)[["metric", "Rp", "Rs", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
