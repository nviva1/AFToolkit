#!/usr/bin/env python
"""Precompute antibody CDR-loop residue numbers (AbNumber/IMGT) for every escape
antibody, and dump to JSON. Run with the BASE python (where AbNumber's backend is
installed); the .venv model run just loads the JSON.

Out: reproductions/rbd_aim_fig2g/_cdr_resnums.json   {antibody: [[chain, resnum], ...]}
"""
import ast, json, os, sys
import pandas as pd
from biopandas.pdb import PandasPdb
from abnumber import Chain
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import native_crop as N
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))


def cdr_pairs(xray_pdb, H, L):
    df = PandasPdb().read_pdb(xray_pdb).df["ATOM"]; df = df[df.alt_loc.isin(["", "A"])]
    s = []
    for ch in dict.fromkeys([H, L]):
        seq, resns = N.chain_seq(df, ch)
        for c in Chain.multiple_domains(seq, scheme="imgt"):
            for cdr in (c.cdr1_seq, c.cdr2_seq, c.cdr3_seq):
                i = seq.find(cdr)
                if i >= 0:
                    s += [[ch, int(r)] for r in resns[i:i + len(cdr)]]
    return s


def main():
    pairs = pd.read_csv(os.path.join(ROOT, "data/rbdaim/escape_pairs.csv"))
    out = {}
    for ab, g in pairs.groupby("antibody"):
        row0 = g.iloc[0]; chids = ast.literal_eval(row0.chids)
        try:
            out[ab] = cdr_pairs(row0.pdb_path, chids[0], chids[1])
            print(f"{ab:<16} CDR residues: {len(out[ab])}")
        except Exception as e:
            print(f"{ab:<16} FAILED: {e}")
    json.dump(out, open(os.path.join(HERE, "_cdr_resnums.json"), "w"))
    print(f"\nwrote _cdr_resnums.json ({len(out)} antibodies)")


if __name__ == "__main__":
    main()
