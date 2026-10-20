#!/usr/bin/env python
"""Fetch every antibody-RBD crystal in the Fig 2g set and write a cropped copy to
x_ray_pdbs/ that keeps ONLY:
  * the three corresponding chains  (chids = [heavy, light, RBD]),
  * VH/VL residues of the antibody chains (variable domain found by AbNumber),
  * the RBD residues of the antigen chain (region aligned to the Wuhan ref.pdb).

Chain ids are preserved from the deposition (H, L, RBD). One PDB per antibody:
x_ray_pdbs/<antibody>.pdb. Crash-safe: already-written files are skipped.
"""
import ast, json, os, sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from biopandas.pdb import PandasPdb
from abnumber import Chain

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_full as R
import native_crop as N                       # chain_seq, crop_rbd_resns (kalign)
from aftoolkit.structure import AA_3_TO_1_DICT

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "x_ray_pdbs")
SRC = os.path.join(ROOT, "data/rbdaim/kd_absolute_long.csv")


def variable_resns(seq, resns):
    """Residue numbers of every antibody variable domain on a chain, found by
    AbNumber (IMGT). Handles 1 domain (Fab H or L) and 2 (scFv: VH-linker-VL on
    one chain). Each V-domain is a contiguous substring -> map back by find()."""
    keep = set(); types = []
    for c in Chain.multiple_domains(seq, scheme="imgt"):
        dom = str(c)
        idx = seq.find(dom)
        assert idx >= 0, "AbNumber V-domain not a contiguous substring"
        keep.update(resns[idx:idx + len(dom)]); types.append(c.chain_type)
    return keep, "+".join(types)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ref = json.load(open(os.path.join(ROOT, "data/rbdaim/datasets/rbd_strains.json")))["Wuhan"]
    df_src = pd.read_csv(SRC).drop_duplicates("antibody")
    n_ok = n_skip = n_fail = 0
    print(f"{'antibody':<16}{'pdb':>6}  {'chains(H,L,RBD)':<16}{'V':>5}{'':>3}{'RBD':>5}  type", flush=True)

    for _, r in df_src.iterrows():
        ab = str(r.antibody); safe = ab.replace("/", "_")
        H, L, RBD = ast.literal_eval(r.chids); pdbid = str(r.pdbid)
        # filename: <antibody>_<pdbid>_<H><L><RBD>.pdb  (chains in heavy/light/RBD order)
        fp = os.path.join(OUTDIR, f"{safe}_{pdbid}_{H}{L}{RBD}.pdb")
        if os.path.exists(fp):
            n_skip += 1; continue
        try:
            df = PandasPdb().read_pdb(R.fetch_pdb(pdbid)).df["ATOM"]  # noqa
            df = df[df.alt_loc.isin(["", "A"])]

            rseq, rresn = N.chain_seq(df, RBD)
            keep_rbd = N.crop_rbd_resns(rseq, rresn, ref)
            # antibody chains: dedupe so an scFv (H == L on one chain) is cropped once
            ab_keep = {}; types = []
            for ch in dict.fromkeys([H, L]):
                cseq, cresn = N.chain_seq(df, ch)
                ks, ty = variable_resns(cseq, cresn)
                ab_keep[ch] = ks; types.append(ty)

            parts = [df[(df.chain_id == ch) & (df.residue_number.isin(ks))]
                     for ch, ks in ab_keep.items()]
            parts.append(df[(df.chain_id == RBD) & (df.residue_number.isin(keep_rbd))])
            out = pd.concat(parts).reset_index(drop=True)
            out["atom_number"] = range(1, len(out) + 1)
            out["line_idx"] = range(len(out))
            p = PandasPdb(); p.df["ATOM"] = out; p.to_pdb(fp)
            nv = sum(len(v) for v in ab_keep.values())
            chstr = f"{H},{L},{RBD}" if H != L else f"{H}(scFv),{RBD}"
            print(f"{ab:<16}{pdbid:>6}  {chstr:<16}{nv:>5}{'':>3}{len(keep_rbd):>5}  {'/'.join(types)}", flush=True)
            n_ok += 1
        except Exception as exc:
            print(f"{ab:<16}{'':>6}  FAILED: {exc}", flush=True)
            n_fail += 1

    print(f"\nwrote {n_ok} new, skipped {n_skip} existing, {n_fail} failed -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
