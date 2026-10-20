#!/usr/bin/env python
"""Interface iPAE at 8 / 10 / 12 A, with a rigorous assert that the interface
residue sets are EXACTLY {residues with a heavy-atom within threshold of the other
chain} -- nothing missed, nothing wrongly included. Interface is taken from the
predicted reference structure (all atoms), shared indices for ref & variant.

Two independent methods must agree (cdist per-residue min  vs  cKDTree ball query),
and every residue's membership is checked against its true min-atom distance.

Also emits a PyMOL script highlighting the interface (red) for one example.
Usage: python interface_metrics.py [TAG] [EXAMPLE_ANTIBODY]
Out: data/rbdaim/interface_metrics_<TAG>.csv  + view_interface_<example>.pml
"""
import ast, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from biopandas.pdb import PandasPdb
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_load as M
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "r3"
EXAMPLE = sys.argv[2] if len(sys.argv) > 2 else "J08"
PAEDIR = os.path.join(HERE, f"pae_mats_{TAG}"); PDBDIR = os.path.join(HERE, f"pred_pdbs_{TAG}")
THRESHOLDS = [8.0, 10.0, 12.0]


def safe(s):
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", str(s)).strip("_")


def residue_atoms(df, chain):
    """{resnum: Nx3 heavy-atom coords} for a chain."""
    g = df[(df.chain_id == chain) & (df.element_symbol != "H")]
    return {int(r): sub[["x_coord", "y_coord", "z_coord"]].to_numpy()
            for r, sub in g.groupby("residue_number")}


def interface_residues(df, rbd_chain, ab_chains, thresh):
    """RBD- and Ab-side interface residue numbers (any heavy-atom pair < thresh)."""
    rbd = residue_atoms(df, rbd_chain)
    ab = {}
    for ch in ab_chains:
        ab.update({(ch, k): v for k, v in residue_atoms(df, ch).items()})
    ab_all = np.vstack(list(ab.values())); rbd_all = np.vstack(list(rbd.values()))
    rbd_if = {rn for rn, at in rbd.items() if cdist(at, ab_all).min() < thresh}
    ab_if = {key for key, at in ab.items() if cdist(at, rbd_all).min() < thresh}
    return rbd, ab, rbd_if, ab_if


def assert_interface(rbd, ab, rbd_if, ab_if, thresh):
    """Cross-check: KDTree must reproduce the same sets, AND every residue's
    membership must match its true min-atom distance (no residue missed/extra)."""
    ab_all = np.vstack(list(ab.values())); rbd_all = np.vstack(list(rbd.values()))
    # independent method: KDTree ball-point
    tree_ab = cKDTree(ab_all); tree_rbd = cKDTree(rbd_all)
    rbd_if_kd = {rn for rn, at in rbd.items() if tree_ab.query_ball_point(at, thresh, return_length=True).sum() > 0}
    ab_if_kd = {key for key, at in ab.items() if tree_rbd.query_ball_point(at, thresh, return_length=True).sum() > 0}
    assert rbd_if == rbd_if_kd, f"RBD interface cdist vs KDTree mismatch: {rbd_if ^ rbd_if_kd}"
    assert ab_if == ab_if_kd, f"Ab interface cdist vs KDTree mismatch: {ab_if ^ ab_if_kd}"
    # boundary check: membership == (true min distance < thresh), no residue missed
    for rn, at in rbd.items():
        dmin = cdist(at, ab_all).min()
        assert (rn in rbd_if) == (dmin < thresh), \
            f"RBD {rn}: in_iface={rn in rbd_if} but dmin={dmin:.3f} vs thresh {thresh}"
    for key, at in ab.items():
        dmin = cdist(at, rbd_all).min()
        assert (key in ab_if) == (dmin < thresh), \
            f"Ab {key}: in_iface={key in ab_if} but dmin={dmin:.3f} vs thresh {thresh}"
    return len(rbd_if), len(ab_if)


def main():
    pairs = pd.read_csv(os.path.join(ROOT, "data/rbdaim/escape_pairs.csv"))
    rows = []; checked = 0
    example_payload = None
    for ab, g in pairs.groupby("antibody"):
        row0 = g.iloc[0]; chids = ast.literal_eval(row0.chids); rbd = row0.rbd_chain
        ab_chains = [c for c in chids if c != rbd]
        refnpz = os.path.join(PAEDIR, f"{safe(ab)}__REF_{safe(row0.ref_strain)}.npz")
        if not os.path.exists(refnpz):
            continue
        # interface from the cropped CRYSTAL: real chains (H/L/RBD), canonical numbering, all heavy atoms
        df = PandasPdb().read_pdb(row0.pdb_path).df["ATOM"]; df = df[df.alt_loc.isin(["", "A"])]
        rz = np.load(refnpz, allow_pickle=True)
        lab = rz["labels"]; rn = rz["resnums"].astype(int); ncyc = rz["pae"].shape[0]
        mask_pos = M.positions_in(row0.apply_to_xray_for_ref)
        for _, r in g.iterrows():
            mask_pos |= M.positions_in(r.apply_to_xray_for_variant)

        iface = {}; iface_sets = {}
        for th in THRESHOLDS:
            rbd_at, ab_at, rbd_if, ab_if = interface_residues(df, rbd, ab_chains, th)
            assert_interface(rbd_at, ab_at, rbd_if, ab_if, th); checked += 1
            # map interface residues -> PAE indices (Ab matched by (chain,resnum) pair)
            Ri = np.array([i for i in range(len(lab)) if lab[i] == rbd and int(rn[i]) in rbd_if], int)
            Ai = np.array([i for i in range(len(lab)) if (lab[i], int(rn[i])) in ab_if], int)
            iface[th] = (Ri, Ai, len(rbd_if), len(ab_if))
            iface_sets[th] = (rbd_if, ab_if)
        if ab == EXAMPLE:
            example_payload = (ab, row0.pdb_path, rbd, ab_chains, iface_sets)

        for _, r in g.iterrows():
            if str(r.escape_mutations).startswith("cross-species"):
                continue
            vz = os.path.join(PAEDIR, f"{safe(ab)}__{safe(r.var_strain)}.npz")
            if not os.path.exists(vz):
                continue
            vp = np.load(vz, allow_pickle=True)["pae"]
            esc = M.positions_in(r.escape_mutations)
            rec = {"antibody": ab, "variant": r.var_strain, "dlogKD": r.delta_log_kd}
            for th in THRESHOLDS:
                Ri, Ai, nr, na = iface[th]
                Rmut_if = np.where((lab == rbd) & np.isin(rn, list(esc)) & np.isin(rn, rn[Ri]))[0]
                for k in range(ncyc):
                    P = vp[k] - rz["pae"][k]
                    if len(Ri) and len(Ai):
                        rec[f"iface{int(th)}_c{k}"] = float(P[np.ix_(Ri, Ai)].mean())
                        # symmetrized: average RBD->Ab and Ab->RBD blocks (interface ids)
                        rec[f"ifaceSYM{int(th)}_c{k}"] = float(0.5 * (P[np.ix_(Ri, Ai)].mean() + P[np.ix_(Ai, Ri)].mean()))
                    else:
                        rec[f"iface{int(th)}_c{k}"] = np.nan; rec[f"ifaceSYM{int(th)}_c{k}"] = np.nan
                    if len(Rmut_if) and len(Ai):
                        rec[f"mut_iface{int(th)}_c{k}"] = float(P[np.ix_(Rmut_if, Ai)].mean())
                        rec[f"mut_ifaceSYM{int(th)}_c{k}"] = float(0.5 * (P[np.ix_(Rmut_if, Ai)].mean() + P[np.ix_(Ai, Rmut_if)].mean()))
                    else:
                        rec[f"mut_iface{int(th)}_c{k}"] = np.nan; rec[f"mut_ifaceSYM{int(th)}_c{k}"] = np.nan
            rows.append(rec)

    out = pd.DataFrame(rows); out.to_csv(os.path.join(ROOT, f"data/rbdaim/interface_metrics_{TAG}.csv"), index=False)
    print(f"ASSERTS PASSED: {checked} interface computations verified (cdist==KDTree, exact boundary)\n")
    print(f"wrote interface_metrics_{TAG}.csv  ({len(out)} variants)\n")
    print("interface iPAE, best cycle (Rp / Rs):")
    for th in THRESHOLDS:
        for key in (f"iface{int(th)}", f"ifaceSYM{int(th)}", f"mut_iface{int(th)}", f"mut_ifaceSYM{int(th)}"):
            best = None
            for k in range(4):
                c = f"{key}_c{k}"
                if c in out:
                    v = out[out[c].notna()]
                    if len(v) >= 4:
                        rp, rs = pearsonr(v.dlogKD, v[c])[0], spearmanr(v.dlogKD, v[c])[0]
                        if best is None or abs(rp) > abs(best[1]):
                            best = (k, rp, rs, len(v))
            if best:
                print(f"  {key:<14} c{best[0]}: Rp={best[1]:+.3f} Rs={best[2]:+.3f} (n={best[3]})")

    # ---- PyMOL example ----
    if example_payload:
        ab, xray, rbd, ab_chains, ths = example_payload
        rbd_if, ab_if = ths[10.0]                       # highlight the 10 A interface
        ab_sel = " or ".join(
            f"(chain {ch} and resi {'+'.join(str(r) for (c, r) in sorted(ab_if) if c == ch)})"
            for ch in ab_chains if any(c == ch for c, _ in ab_if))
        pml = os.path.join(HERE, f"view_interface_{safe(ab)}.pml")
        with open(pml, "w") as fh:
            fh.write(f"# {ab}: 10 A interface (heavy-atom contacts) highlighted RED on the crystal {os.path.basename(xray)}\n")
            fh.write(f"load {os.path.relpath(xray, HERE)}, {safe(ab)}_xray\n")
            fh.write("bg_color white\nhide everything\nshow cartoon\ncolor grey80\n")
            fh.write(f"select iface_rbd, chain {rbd} and resi {'+'.join(map(str, sorted(rbd_if)))}\n")
            fh.write(f"select iface_ab, {ab_sel}\n")
            fh.write("color red, iface_rbd or iface_ab\nshow sticks, (iface_rbd or iface_ab) and not name N+C+O\n")
            fh.write(f"color slate, chain {rbd} and not iface_rbd\n")
            fh.write("set cartoon_transparency, 0.15\norient\nset ray_shadows,0\nset cartoon_fancy_helices,1\n")
            fh.write(f"ray 1500,1100\npng interface_{safe(ab)}.png, dpi=150\n")
        print(f"\nwrote {pml}  (RBD iface={len(rbd_if)} res, Ab iface={len(ab_if)} res @10A)")


if __name__ == "__main__":
    main()
