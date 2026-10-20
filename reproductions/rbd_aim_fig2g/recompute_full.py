#!/usr/bin/env python
"""Full Fig 2g recompute: for each antibody, score iPAE of its complex with the
RBD threaded to each measured strain, then correlate Delta iPAE vs Delta log10 K_D.

Inputs (assembled earlier):
  - data/rbdaim/fig2g_affinity.csv     : ab, pdbid, chids, strain, delta_log_kd
  - data/rbdaim/datasets/rbd_strains.json : strain -> RBD sequence (Wuhan = reference)
Antibody complex structures are fetched from RCSB by pdbid.

Threading: the structure's RBD chain residues are aligned to the target strain
RBD sequence; differing positions are applied as point mutations (AFToolkit
multi-mutation), then the complex is scored with aftoolkit.scoring.score_complex.

Usage:
    python recompute_full.py --ab EH3            # one antibody (validation)
    python recompute_full.py                     # all antibodies (long)
"""

import argparse
import json
import os
import urllib.request

import numpy as np
import pandas as pd
from Bio import Align
from biopandas.pdb import PandasPdb

from aftoolkit.structure import load_protein, AA_3_TO_1_DICT, save_to_pdb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CPLX = os.path.join(ROOT, "data/rbdaim/complexes")
TRIM = os.path.join(CPLX, "trimmed")

_ANARCII = None


def _anarcii():
    """Lazily build the (pure-Python, HMMER-free) ANARCII numbering model."""
    global _ANARCII
    if _ANARCII is None:
        from anarcii import Anarcii
        _ANARCII = Anarcii(seq_type="antibody", mode="accuracy", verbose=False)
    return _ANARCII


def variable_span(seq):
    """Return (start, end) indices of the antibody variable domain in ``seq``.

    Uses ANARCII numbering; the numbered region is exactly VH or VL, so this
    drops the constant domain (CH1/CL) -- matching the scFv (VH+VL) the paper
    models. Returns None if the chain is not an antibody variable domain.
    """
    rec = _anarcii().number([seq])["Sequence 1"]
    if rec.get("error") or rec.get("query_start") is None:
        return None
    return int(rec["query_start"]), int(rec["query_end"])

# MOESM4 strain label -> rbd_strains.json key.
STRAIN_MAP = {
    "WT": "Wuhan", "alpha": "20I_(Alpha)", "beta": "20H_(Beta)", "gamma": "20J_(Gamma)",
    "delta": "21A_(Delta)", "lambda C.37": "21G_(Lambda)",
    "omicron (BA1, B.1.1.529)": "21K_(BA.1)", "BA2": "21L_(BA.2)",
    "BA2.12.1": "22C_(BA.2.12.1)", "BA2.75": "22D_(BA.2.75)", "omicron BA5": "22B_(BA.5)",
    "BQ1.1": "22E_(BQ.1.1)", "XBB": "22F(XBB.1)",
    "Epsilon (B.1.429)": "21C_(Epsilon)", "Iota (B.1.526)": "21F_(Iota)",
}


def _valid_pdb(path):
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        return False
    with open(path) as fh:
        head = fh.read(2000)
    return ("ATOM" in head) or head.startswith(("HEADER", "CRYST", "REMARK"))


def fetch_pdb(pdbid):
    os.makedirs(CPLX, exist_ok=True)
    path = os.path.join(CPLX, f"{pdbid}.pdb")
    if not _valid_pdb(path):  # (re)download if missing/partial/non-PDB
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdbid}.pdb", path)
        if not _valid_pdb(path):
            raise ValueError(f"{pdbid}: downloaded file is not valid PDB")
    return path


def trim_complex(pdb, rbd_chain, rbd_resnums, ab_chains, ab_var, out_pdb):
    """Write a PDB: RBD domain (rbd_resnums) + antibody chains trimmed to VH/VL."""
    df = load_protein(pdb)  # keeps chain_id_original / residue_number_original
    keep = (df["chain_id_original"] == rbd_chain) & \
           (df["residue_number_original"].isin(set(rbd_resnums)))
    for c in ab_chains:
        span = ab_var.get(c)
        if span is None:                      # numbering failed -> keep whole chain
            keep |= df["chain_id_original"] == c
            continue
        s, e = span
        ca = df[(df.chain_id_original == c) & (df.atom_name == "CA")].reset_index(drop=True)
        keep_resnums = set(ca["residue_number_original"].iloc[s:e + 1])
        keep |= (df["chain_id_original"] == c) & (df["residue_number_original"].isin(keep_resnums))
    save_to_pdb(df[keep].copy(), out_pdb)
    return out_pdb


def chain_sequences(pdb):
    """Return {chain_id: (sequence, [residue_numbers])} from CA atoms."""
    df = load_protein(pdb)  # collapses to one chain but keeps chain_id_original
    ca = df[df.atom_name == "CA"].reset_index(drop=True)
    out = {}
    for c, g in ca.groupby("chain_id_original", sort=False):
        seq = "".join(AA_3_TO_1_DICT.get(r, "X") for r in g.residue_name)
        out[c] = (seq, list(g.residue_number_original))
    return out


def _chain_table(pdb):
    """Per-chain CA sequence, residue numbers, and CA coordinates."""
    df = load_protein(pdb)
    ca = df[df.atom_name == "CA"]
    out = {}
    for c, g in ca.groupby("chain_id_original", sort=False):
        seq = "".join(AA_3_TO_1_DICT.get(r, "X") for r in g.residue_name)
        out[c] = (seq, list(g.residue_number_original),
                  g[["x_coord", "y_coord", "z_coord"]].to_numpy())
    return out


def select_complex(pdb, rbd_ref):
    """Isolate ONE RBD domain + its cognate Fab (one heavy + one light).

    RBD = the ~190-residue region best matching the Wuhan RBD by identity
    (extracted even if it sits inside a larger spike chain). Antibody = the
    heavy/light variable chains in closest 3D contact with that RBD region.

    Returns (rbd_chain, rbd_resnums, [heavy_chain, light_chain]).
    """
    chains = _chain_table(pdb)
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"

    # --- RBD domain: best identity over a ~150-260 residue window ---
    best = None  # (identity, chain, resnums_of_region, region_center_xyz)
    for c, (seq, resn, xyz) in chains.items():
        if len(seq) < 100:
            continue
        aln = aligner.align(seq, rbd_ref)[0]
        cs, ce = aln.aligned[0][0][0], aln.aligned[0][-1][1]
        region_len = ce - cs
        if not (140 <= region_len <= 280):
            continue
        ident = sum(a == b for a, b in zip(aln[0], aln[1]) if a != "-" and b != "-") / len(rbd_ref)
        if best is None or ident > best[0]:
            best = (ident, c, resn[cs:ce], xyz[cs:ce])
    if best is None:
        raise ValueError("no RBD-like region found")
    _, rbd_chain, rbd_resnums, rbd_xyz = best

    # --- cognate Fab: closest heavy + closest light to the RBD region ---
    heavy = light = None
    h_d = l_d = 1e9
    for c, (seq, resn, xyz) in chains.items():
        if c == rbd_chain or len(seq) < 80:
            continue
        rec = _anarcii().number([seq])["Sequence 1"]
        if rec.get("error") or rec.get("chain_type") not in ("H", "K", "L"):
            continue
        # min CA-CA distance between this chain and the RBD region
        d = np.sqrt(((xyz[:, None, :] - rbd_xyz[None, :, :]) ** 2).sum(-1)).min()
        if rec["chain_type"] == "H":
            if d < h_d:
                heavy, h_d = c, d
        else:
            if d < l_d:
                light, l_d = c, d
    ab = [x for x in (heavy, light) if x is not None]
    if not ab:
        raise ValueError("no antibody variable chains found")
    return rbd_chain, rbd_resnums, ab


def rbd_mutations(struct_seq, struct_resnums, target_seq):
    """Point mutations to turn the structure RBD into the target strain RBD."""
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aln = aligner.align(struct_seq, target_seq)[0]
    muts = []
    si = ti = 0
    a_struct, a_tgt = aln[0], aln[1]
    for cs, ct in zip(a_struct, a_tgt):
        if cs != "-" and ct != "-":
            if cs != ct and ct in AA_3_TO_1_DICT.values():
                muts.append((cs, struct_resnums[si], ct))
            si += 1
            ti += 1
        elif cs != "-":
            si += 1
        elif ct != "-":
            ti += 1
    return muts


def strain_plan(ref_seq, target_seq, ref_start=334):
    """Plan how to build a strain RBD from the canonical Wuhan RBD by DELETIONS +
    SUBSTITUTIONS only (the operations a fixed backbone supports).

    Indels are defined vs Wuhan (NOT the crystal, so a crystal density gap is never
    mistaken for a strain indel). Returns (del_resnums, reproducible):
      * del_resnums: canonical residue numbers to delete from the structure.
      * reproducible: True iff applying those deletions + substitutions to Wuhan
        reconstructs the target EXACTLY. False when the strain needs an insertion
        or a more complex indel (e.g. JN.1/BA.2.86) -- those are skipped rather
        than scored with a wrong sequence.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aln = aligner.align(ref_seq, target_seq)[0]
    del_resnums, built, ri = [], [], 0
    for cr, ct in zip(aln[0], aln[1]):
        if cr != "-" and ct != "-":
            built.append(ct); ri += 1          # match / substitution
        elif cr != "-":                        # Wuhan residue deleted in strain
            del_resnums.append(ref_start + ri); ri += 1
        # ct != "-" (insertion) contributes nothing the backbone can hold
    reproducible = ("".join(built) == target_seq)
    return del_resnums, reproducible


def rbd_sanity_check(struct_seq, target_seq, min_identity=1.0):
    """Verify a strain can be reproduced from the structure RBD by SUBSTITUTIONS only.

    Point-mutation and CB-threading both keep the WT backbone length, so they
    cannot introduce insertions/deletions. This filter aligns the structure RBD
    to the target strain and rejects the point if (a) the alignment contains any
    indel, or (b) the post-substitution identity to the target would fall below
    ``min_identity`` (e.g. residues the structure is missing).

    Returns (ok: bool, reason: str, n_indel: int, identity: float).
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aln = aligner.align(struct_seq, target_seq)[0]
    n_indel = aligned = matchable = 0
    for cs, ct in zip(aln[0], aln[1]):
        if cs != "-" and ct != "-":
            aligned += 1
            matchable += 1               # a substitution can make cs -> ct
        else:
            n_indel += 1                 # gap on either side = indel vs the strain
    identity = matchable / len(target_seq) if target_seq else 0.0
    if n_indel > 0:
        return False, f"{n_indel} indel(s) vs strain (cannot model by substitution)", n_indel, identity
    if identity < min_identity:
        return False, f"post-substitution identity {identity:.3f} < {min_identity}", n_indel, identity
    return True, "ok", n_indel, identity


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ab", default=None, help="Restrict to one antibody (validation).")
    p.add_argument("--top", type=int, default=None,
                   help="Restrict to the N antibodies with the most measured strains.")
    p.add_argument("--device", default="mps")
    p.add_argument("--n-recycle", type=int, default=2)
    p.add_argument("--chain-gap", type=int, default=25,
                   help="Inter-chain residue gap for the merged input "
                        "(RBD-AIM published value: 25; AFToolkit default: 32).")
    p.add_argument("--out", default=os.path.join(HERE, "recomputed"))
    args = p.parse_args()

    strains = json.load(open(os.path.join(ROOT, "data/rbdaim/datasets/rbd_strains.json")))
    rbd_ref = strains["Wuhan"]
    aff = pd.read_csv(os.path.join(ROOT, "data/rbdaim/fig2g_affinity.csv"))
    if args.ab:
        aff = aff[aff["ab"] == args.ab]
    if args.top:
        keep = (aff.groupby("ab")["strain"].nunique()
                .sort_values(ascending=False).head(args.top).index)
        aff = aff[aff["ab"].isin(keep)]
        print(f"Top {args.top} antibodies ({len(aff)} strain points): {list(keep)}")
    os.makedirs(args.out, exist_ok=True)

    from aftoolkit.folding import OpenFoldWrapper
    from aftoolkit.structure import (load_protein, mutate_protein, safe_filename_part,
                                            get_internal_residue_numbering)
    from aftoolkit.mutations import parse_mutations
    from aftoolkit.scoring import interface_pae
    pae_dir = os.path.join(args.out, "pae")
    os.makedirs(pae_dir, exist_ok=True)
    wrapper = OpenFoldWrapper(device=args.device, inference_n_recycle=args.n_recycle,
                              return_all_cycles=True)
    wrapper.init_model()

    def ipae_for(pdb, antigen, antibody, mut_str, tag, del_resnums=()):
        # Build the complex and run ONE inference. chain_gap matches the RBD-AIM
        # published protocol (25; see --chain-gap).
        df = load_protein(pdb, chains=antibody + [antigen], chain_gap=args.chain_gap)
        if del_resnums:
            # Model a strain DELETION: drop the deleted RBD residues, then re-derive
            # the merged-input numbering so the remaining residues stay contiguous.
            keep = ~((df["chain_id_original"] == antigen) &
                     (df["residue_number_original"].isin(set(del_resnums))))
            df = df[keep].copy()
            df["residue_number"] = get_internal_residue_numbering(df, args.chain_gap)
        if mut_str:
            df = mutate_protein(df, parse_mutations(mut_str))
        wrapper.evaluate(df)
        of = wrapper.of_output
        cycles = wrapper.of_output_per_cycle or []
        ca = wrapper.get_of_protein()
        ca = ca[ca.atom_name == "CA"].reset_index(drop=True)
        labels = ca["chain_id_original"].tolist()
        coords = ca[["x_coord", "y_coord", "z_coord"]].to_numpy()
        # Save the full PAE matrix for EVERY recycling round (+ final) for offline
        # recomputation of any interface metric without re-running OpenFold.
        np.savez_compressed(
            os.path.join(pae_dir, f"{safe_filename_part(tag)}.npz"),
            labels=np.array(labels), coords=coords,
            antigen=antigen, antibody=",".join(antibody),
            pae_final=of["predicted_aligned_error"],
            **{f"pae_cycle{i}": c["predicted_aligned_error"] for i, c in enumerate(cycles)},
        )
        # Paper's iPAE: AVERAGE PAE over ALL antibody<->RBD residue pairs
        # (all-pairs, NOT contact-restricted), at the final recycle.
        return interface_pae(of["predicted_aligned_error"], labels,
                             [antigen], antibody)

    # Incremental, crash-safe, resumable output: each row is written + flushed
    # immediately, and an existing CSV is resumed (computed rows are skipped).
    import csv
    results_csv = os.path.join(args.out, "fig2g_recomputed_full.csv")
    fields = ["ab", "pdbid", "strain", "delta_log_kd", "ipae", "ipae_ref",
              "delta_ipae", "n_mut"]
    done, ref_cache = set(), {}
    if os.path.exists(results_csv):
        prev = pd.read_csv(results_csv)
        done = set(zip(prev["ab"], prev["strain"]))
        ref_cache = dict(zip(prev["ab"], prev["ipae_ref"]))
        print(f"resuming: {len(done)} rows already done")
    fh = open(results_csv, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if os.path.getsize(results_csv) == 0:
        writer.writeheader(); fh.flush()

    def emit(row):
        writer.writerow(row); fh.flush(); os.fsync(fh.fileno())

    for ab, g in aff.groupby("ab"):
        pdbid = g["pdbid"].iloc[0]
        todo = [r for _, r in g.iterrows() if (ab, r["strain"]) not in done]
        if not todo:
            continue
        try:
            pdb = fetch_pdb(pdbid)
            rbd_chain, rbd_resnums, ab_chains = select_complex(pdb, rbd_ref)
            seqs = chain_sequences(pdb)
            ab_var = {c: variable_span(seqs[c][0]) for c in ab_chains}
            os.makedirs(TRIM, exist_ok=True)
            pdb = trim_complex(pdb, rbd_chain, rbd_resnums, ab_chains, ab_var,
                               os.path.join(TRIM, f"{pdbid}_fv.pdb"))
            seqs = chain_sequences(pdb)
            struct_seq, struct_resnums = seqs[rbd_chain]
            print(f"[{ab}/{pdbid}] RBD={rbd_chain}({len(rbd_resnums)}) Ab={ab_chains}")

            if ab in ref_cache:
                ipae_ref = float(ref_cache[ab])
            else:
                ref_muts = rbd_mutations(struct_seq, struct_resnums, rbd_ref)
                ref_mut_str = ",".join(f"{rbd_chain}:{w}{p}{m}" for w, p, m in ref_muts)
                ipae_ref = ipae_for(pdb, rbd_chain, ab_chains, ref_mut_str, f"{ab}__WTref")
                ref_cache[ab] = ipae_ref
                print(f"    iPAE(WT ref) = {ipae_ref:.3f}")

            for r in todo:
                key = STRAIN_MAP.get(r["strain"])
                if key is None or key not in strains:
                    continue
                # Strain indels are defined vs the canonical Wuhan RBD (not the
                # crystal). A strain reachable by deletion + substitution is MODELLED
                # (deleted residues removed); one needing an insertion / complex
                # indel (e.g. JN.1) is skipped rather than scored with a wrong seq.
                del_resnums, reproducible = strain_plan(rbd_ref, strains[key])
                if not reproducible:
                    print(f"    SKIP {r['strain']:<22} (needs insertion/complex indel "
                          f"vs Wuhan; cannot build on fixed backbone)")
                    continue
                muts = rbd_mutations(struct_seq, struct_resnums, strains[key])
                mut_str = ",".join(f"{rbd_chain}:{w}{p}{m}" for w, p, m in muts)
                ip = ipae_for(pdb, rbd_chain, ab_chains, mut_str, f"{ab}__{r['strain']}",
                              del_resnums=del_resnums)
                emit({"ab": ab, "pdbid": pdbid, "strain": r["strain"],
                      "delta_log_kd": r["delta_log_kd"], "ipae": round(ip, 4),
                      "ipae_ref": round(ipae_ref, 4), "delta_ipae": round(ip - ipae_ref, 4),
                      "n_mut": len(muts)})
                print(f"    {r['strain']:<26} dlogKD={r['delta_log_kd']:+.2f} "
                      f"iPAE={ip:.3f} diPAE={ip - ipae_ref:+.3f} ({len(muts)} mut)")
        except Exception as exc:
            print(f"[{ab}/{pdbid}] FAILED: {exc}")
    fh.close()

    out = pd.read_csv(results_csv)
    if len(out) >= 3:
        from scipy.stats import pearsonr, spearmanr
        x, y = out["delta_log_kd"].to_numpy(), out["delta_ipae"].to_numpy()
        print(f"\nn={len(out)}  Pearson={pearsonr(x, y)[0]:+.3f}  "
              f"Spearman={spearmanr(x, y)[0]:+.3f}  (paper |R|=0.52)")
    print(f"saved -> {results_csv}")


if __name__ == "__main__":
    main()
