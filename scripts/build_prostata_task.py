#!/usr/bin/env python
"""Build an AFToolkit task CSV from a PROSTATA-format dataset CSV.

PROSTATA dataset CSVs (``../PROSTATA/DATASETS/*.csv``) have columns
``wt_seq, mut_seq, ddg, pdb_id, mut_info, pos`` and ship their structures in
``../PROSTATA/PDB/<code>.pdb``. ``pdb_id`` is ``<4-char code><chain>`` (e.g.
``1A43A``) and ``pos`` is the 0-based index of the mutated residue in ``wt_seq``.

Because ``mut_info`` numbering is inconsistent across datasets (sometimes PDB
numbering, sometimes sequence index), we locate the mutated residue by the
``pos``-th CA atom in the loaded structure and read its true PDB residue number.
Rows whose WT identity does not match the structure are skipped and counted.

Output columns match the AFToolkit pipeline:
    id, pdb_id, chain, path, positions, mutations, mut_info, mut_type, ddg, split
"""

import argparse
import os

import pandas as pd

from aftoolkit.structure import load_protein, AA_3_TO_1_DICT


def build(csv, out, pdb_dir="../PROSTATA/PDB", split="train",
          max_mutations=None, seed=0):
    df = pd.read_csv(csv)
    if max_mutations is not None and len(df) > max_mutations:
        df = df.sample(max_mutations, random_state=seed).reset_index(drop=True)

    cache = {}
    rows, skipped = [], 0
    for _, r in df.iterrows():
        pid = str(r["pdb_id"])
        code, chain = pid[:4], pid[4:]
        path = os.path.join(pdb_dir, f"{code}.pdb")
        if not os.path.exists(path):
            skipped += 1
            continue
        key = (code, chain)
        if key not in cache:
            prot = load_protein(path, chains=[chain])
            cache[key] = prot[prot.atom_name == "CA"].reset_index(drop=True)
        ca = cache[key]
        pos = int(r["pos"])
        if pos >= len(ca):
            skipped += 1
            continue
        res = ca.iloc[pos]
        wt = AA_3_TO_1_DICT.get(res["residue_name"])
        mut_info = str(r["mut_info"])
        if wt is None or wt != mut_info[0]:
            skipped += 1
            continue
        resnum = int(res["residue_number_original"])
        mt = mut_info[-1]
        mut = f"{wt}{resnum}{mt}"
        rows.append({
            "id": f"{code}{chain}_{chain}_{mut}",
            "pdb_id": f"{code}{chain}",
            "chain": chain,
            "path": path,
            "positions": f"{chain}:{resnum}",
            "mutations": f"{chain}:{mut}",
            "mut_info": mut,
            "mut_type": "ss",
            "ddg": float(r["ddg"]),
            "split": split,
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    out_df.to_csv(out, index=True)
    print(f"{os.path.basename(csv)}: wrote {len(out_df)} mutations "
          f"({out_df['pdb_id'].nunique()} structures), skipped {skipped} -> {out}")
    return out_df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="PROSTATA dataset CSV.")
    p.add_argument("--pdb-dir", default="../PROSTATA/PDB")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--max-mutations", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    build(args.csv, args.out, pdb_dir=args.pdb_dir, split=args.split,
          max_mutations=args.max_mutations, seed=args.seed)


if __name__ == "__main__":
    main()
