#!/usr/bin/env python
"""Predicted PDBs come out collapsed to one chain 'A' with internal residue
numbering (1..N, gap-spaced). Restore them to the ORIGINAL chain ids (H/L/RBD) and
canonical residue numbers, using the deterministic load_protein mapping of the
cropped crystal -- so they align cleanly with the X-ray input and the RBD positions
can be selected by their real numbers.

Usage: python restore_numbering.py ANTIBODY [PRED_DIR]
"""
import ast, os, sys
import pandas as pd
from biopandas.pdb import PandasPdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aftoolkit.structure import load_protein
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))

AB = sys.argv[1]
PRED_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "pred_pdbs_r5")
OUTDIR = os.path.join(PRED_DIR, "restored"); os.makedirs(OUTDIR, exist_ok=True)


def main():
    pairs = pd.read_csv(os.path.join(ROOT, "data/rbdaim/escape_pairs.csv"))
    row0 = pairs[pairs.antibody == AB].iloc[0]
    chids = ast.literal_eval(row0.chids); xray = row0.pdb_path
    # deterministic internal -> (orig chain, canonical resnum) map for this crop
    d = load_protein(xray, chains=chids, chain_gap=25)
    m = (d.drop_duplicates("residue_number")
          .set_index("residue_number")[["chain_id_original", "residue_number_original"]])
    ch_map = m["chain_id_original"].to_dict()
    rn_map = m["residue_number_original"].astype(int).to_dict()

    preds = [f for f in os.listdir(PRED_DIR)
             if f.endswith(".pdb") and f.split("__")[0] == AB.replace("/", "_")]
    for f in sorted(preds):
        df = PandasPdb().read_pdb(os.path.join(PRED_DIR, f)).df["ATOM"]
        df = df[df.residue_number.isin(ch_map)].copy()
        df["chain_id"] = df.residue_number.map(ch_map)
        df["residue_number"] = df.residue_number.map(rn_map)
        df = df.sort_values(["chain_id", "residue_number", "atom_number"]).reset_index(drop=True)
        df["atom_number"] = range(1, len(df) + 1); df["line_idx"] = range(len(df))
        p = PandasPdb(); p.df["ATOM"] = df
        out = os.path.join(OUTDIR, f); p.to_pdb(out)
        print(f"restored {f}  chains={sorted(df.chain_id.unique())}  "
              f"RBD({chids[2]}) {df[df.chain_id==chids[2]].residue_number.min()}.."
              f"{df[df.chain_id==chids[2]].residue_number.max()}")
    print(f"\nRBD chain = {chids[2]} ; restored -> {OUTDIR}")


if __name__ == "__main__":
    main()
