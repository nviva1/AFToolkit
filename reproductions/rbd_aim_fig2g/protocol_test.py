#!/usr/bin/env python
"""Protocol A/B test: which iPAE protocol best reproduces the paper's escape trend.

Tests, on a clean escape gradient (one antibody x several strains):
  P2 = WT RBD (ref.pdb) superposed onto the crystal RBD, reduced to [N,CA,C,O,CB],
       strain sequence threaded on, masked-template OpenFold.
(P1 = crystal RBD + point-mutation, masked template -- already computed.)

Reports delta-iPAE vs delta-log10-KD per recycling round, to pick the protocol
that reproduces the paper's iPAE<->escape relationship best.
"""
import argparse, csv, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from Bio import Align
from biopandas.pdb import PandasPdb
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_full as R
from aftoolkit.structure import AA_3_TO_1_DICT, AA_1_TO_3_DICT
from aftoolkit.scoring import interface_pae

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WT_RBD_PDB = os.path.join(ROOT, "data/rbdaim/datasets/ref.pdb")
BB_CB = ["N", "CA", "C", "O", "CB"]


def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    rot = Vt.T @ np.diag([1, 1, d]) @ U.T
    return rot, Q.mean(0) - rot @ P.mean(0)


def matched_ca(seq_a, ca_a, seq_b, ca_b):
    """Matched CA coords for residues aligned between two sequences."""
    al = Align.PairwiseAligner(); al.mode = "global"
    al.open_gap_score = -10; al.extend_gap_score = -0.5
    a = al.align(seq_a, seq_b)[0]
    Pa, Pb, i, j = [], [], 0, 0
    for x, y in zip(a[0], a[1]):
        if x != "-" and y != "-":
            Pa.append(ca_a[i]); Pb.append(ca_b[j]); i += 1; j += 1
        elif x != "-":
            i += 1
        elif y != "-":
            j += 1
    return np.array(Pa), np.array(Pb)


def build_wt_template(crystal_pdb, rbd_chain, rbd_resnums, ab_chains, ab_var, out_pdb):
    """Superpose WT RBD (CB-only) onto the crystal RBD; combine with the Fab VH/VL."""
    cr = PandasPdb().read_pdb(crystal_pdb).df["ATOM"]
    cr = cr[cr.alt_loc.isin(["", "A"])]
    # crystal RBD CA (region only)
    cr_rbd = cr[(cr.chain_id == rbd_chain) & (cr.residue_number.isin(set(rbd_resnums)))]
    cr_ca = cr_rbd[cr_rbd.atom_name == "CA"].sort_values("residue_number")
    cr_seq = "".join(AA_3_TO_1_DICT.get(r, "X") for r in cr_ca.residue_name)
    cr_xyz = cr_ca[["x_coord", "y_coord", "z_coord"]].to_numpy()
    # WT RBD
    wt = PandasPdb().read_pdb(WT_RBD_PDB).df["ATOM"]
    wt = wt[wt.alt_loc.isin(["", "A"])]
    wt_ca = wt[wt.atom_name == "CA"].sort_values("residue_number")
    wt_seq = "".join(AA_3_TO_1_DICT.get(r, "X") for r in wt_ca.residue_name)
    wt_xyz = wt_ca[["x_coord", "y_coord", "z_coord"]].to_numpy()
    # superpose WT onto crystal RBD
    P, Q = matched_ca(wt_seq, wt_xyz, cr_seq, cr_xyz)
    rot, t = kabsch(P, Q)
    wt_bbcb = wt[wt.atom_name.isin(BB_CB)].copy()
    xyz = wt_bbcb[["x_coord", "y_coord", "z_coord"]].to_numpy()
    wt_bbcb[["x_coord", "y_coord", "z_coord"]] = (xyz @ rot.T) + t
    wt_bbcb["chain_id"] = rbd_chain                       # relabel to RBD chain
    # Fab VH/VL (full atoms, trimmed to variable domain)
    fab_parts = []
    for c in ab_chains:
        span = ab_var.get(c)
        cc = cr[cr.chain_id == c]
        if span is not None:
            ca_order = cc[cc.atom_name == "CA"].sort_values("residue_number")
            keep = set(ca_order["residue_number"].iloc[span[0]:span[1] + 1])
            cc = cc[cc.residue_number.isin(keep)]
        fab_parts.append(cc)
    out = pd.concat([wt_bbcb] + fab_parts).reset_index(drop=True)
    out["atom_number"] = range(1, len(out) + 1)
    out["line_idx"] = range(len(out))
    p = PandasPdb(); p.df["ATOM"] = out; p.to_pdb(out_pdb)
    return out_pdb, wt_seq, list(wt_ca["residue_number"])


def thread(template_pdb, rbd_chain, wt_seq, wt_resns, target_seq, chain_gap=25):
    """Set RBD residue identities to target strain seq (keep [N,CA,C,O,CB]).

    chain_gap defaults to the RBD-AIM published value (25).
    """
    from aftoolkit.structure import load_protein
    df = load_protein(template_pdb, chains=None, chain_gap=chain_gap)
    al = Align.PairwiseAligner(); al.mode = "global"
    al.open_gap_score = -10; al.extend_gap_score = -0.5
    a = al.align(wt_seq, target_seq)[0]
    newaa = {}
    i = j = 0
    for x, y in zip(a[0], a[1]):
        if x != "-" and y != "-":
            if y in AA_1_TO_3_DICT:
                newaa[wt_resns[i]] = AA_1_TO_3_DICT[y]
            i += 1; j += 1
        elif x != "-":
            i += 1
        elif y != "-":
            j += 1
    m = df["chain_id_original"] == rbd_chain
    for rn, aa3 in newaa.items():
        df.loc[m & (df["residue_number_original"] == rn), "residue_name"] = aa3
    # drop CB where threaded residue is glycine
    gly = m & (df["residue_name"] == "GLY") & (df["atom_name"] == "CB")
    return df[~gly].copy()


def run_antibody(ab, wrapper, strains, rbd_ref, STRAIN_MAP, emit, done, chain_gap=25):
    """P2 (WT-superpose + CB-only thread) for every strain of one antibody."""
    aff = pd.read_csv(os.path.join(ROOT, "data/rbdaim/fig2g_affinity.csv"))
    sub = aff[aff.ab == ab]
    pdbid = sub.pdbid.iloc[0]
    pdb = R.fetch_pdb(pdbid)
    rbd_chain, rbd_resnums, ab_chains = R.select_complex(pdb, rbd_ref)
    seqs = R.chain_sequences(pdb)
    ab_var = {c: R.variable_span(seqs[c][0]) for c in ab_chains}
    tdir = os.path.join(os.path.dirname(__file__), "protocol_tmp")
    os.makedirs(tdir, exist_ok=True)
    tpl, wt_seq, wt_resns = build_wt_template(
        pdb, rbd_chain, rbd_resnums, ab_chains, ab_var,
        os.path.join(tdir, f"{ab}_wt_template.pdb"))

    def ipae_p2(target_seq):
        df = thread(tpl, rbd_chain, wt_seq, wt_resns, target_seq, chain_gap=chain_gap)
        wrapper.evaluate(df)
        ca = wrapper.get_of_protein()
        ca = ca[ca.atom_name == "CA"].reset_index(drop=True)
        labels = ca["chain_id_original"].tolist()
        coords = ca[["x_coord", "y_coord", "z_coord"]].to_numpy()
        return {f"c{c}": interface_pae(
            wrapper.of_output_per_cycle[c]["predicted_aligned_error"], labels,
            [rbd_chain], ab_chains, coords=coords, cutoff=12.0) for c in (0, 1, 2)}

    ref = ipae_p2(rbd_ref)                              # Wuhan WT threaded -> reference
    print(f"[{ab}] {pdbid} WT-ref iPAE/cycle: "
          + " ".join(f"c{c}={ref[f'c{c}']:.2f}" for c in (0, 1, 2)), flush=True)
    for _, r in sub.iterrows():
        if (ab, r.strain) in done:
            continue
        key = STRAIN_MAP.get(r.strain)
        if key is None or key not in strains:
            print(f"[{ab}] skip {r.strain}: no strain seq", flush=True)
            continue
        ip = ipae_p2(strains[key])
        rec = {"ab": ab, "pdbid": pdbid, "strain": r.strain,
               "delta_log_kd": r.delta_log_kd,
               **{f"d_{c}": ip[c] - ref[c] for c in ip}}
        emit(rec)
        print(f"[{ab}] {r.strain:<26} dlogKD={r.delta_log_kd:+.2f} "
              + " ".join(f"{c}={rec['d_'+c]:+.3f}" for c in ("c0", "c1", "c2")), flush=True)


def main():
    ap = argparse.ArgumentParser(description="P2 protocol (WT-superpose + thread) runner.")
    ap.add_argument("--abs", required=True, help="comma-separated antibody ids")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--chain-gap", type=int, default=25,
                    help="Inter-chain residue gap (RBD-AIM published value: 25).")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "protocol_P2.csv"))
    args = ap.parse_args()

    strains = json.load(open(os.path.join(ROOT, "data/rbdaim/datasets/rbd_strains.json")))
    rbd_ref = strains["Wuhan"]

    # resume: skip (ab, strain) already in the CSV
    fields = ["ab", "pdbid", "strain", "delta_log_kd", "d_c0", "d_c1", "d_c2"]
    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = set(zip(prev.ab, prev.strain))
    fh = open(args.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if not done:
        w.writeheader(); fh.flush()

    def emit(rec):
        w.writerow(rec); fh.flush(); os.fsync(fh.fileno())

    from aftoolkit.folding import OpenFoldWrapper
    wrapper = OpenFoldWrapper(device=args.device, inference_n_recycle=2, return_all_cycles=True)
    wrapper.init_model()

    for ab in args.abs.split(","):
        try:
            run_antibody(ab.strip(), wrapper, strains, rbd_ref, R.STRAIN_MAP, emit, done,
                         chain_gap=args.chain_gap)
        except Exception as exc:
            print(f"[{ab}] FAILED: {exc}", flush=True)
    fh.close()
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
