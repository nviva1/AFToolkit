#!/usr/bin/env python
"""Build an AFToolkit task CSV from the MegaScale (Tsuboyama et al. 2023) dataset.

Source data (Hugging Face ``RosettaCommons/MegaScale``):
  * ``dataset3_single``      -- single-mutation ddG labels (parquet)
  * ``AlphaFold_model_PDBs`` -- WT domain structures, PDB content inline (parquet)

This extracts the WT structures to ``--pdb-dir`` and writes a task CSV with the
columns the AFToolkit pipeline needs:

    id, pdb_id, chain, path, positions, mutations, mut_info, mut_type, ddg, split

``id`` matches the feature filename ``run_protein_task`` produces, so the same
CSV drives both feature generation (``run_protein_task -i <csv>``) and adapter
training (``train_adapter_monomer``).

Example
-------
    python scripts/build_cdna_task.py \
        --mutations data/cdna/raw/single_train.parquet \
        --pdbs data/cdna/raw/af_pdbs.parquet \
        --split train --split-value train \
        --pdb-dir data/cdna/pdb --out data/cdna/cdna_task_train.csv \
        --max-domains 3 --max-mutations 200
"""

import argparse
import os

import pandas as pd

CHAIN = "A"  # AlphaFold model PDBs are single-chain, numbered from 1.


def build_task_csv(mutations, pdbs, out, pdb_dir="data/cdna/pdb",
                   split_value="train", split="train", split_col="split_name",
                   max_domains=None, max_mutations=None, seed=0):
    """Extract WT structures and write an AFToolkit task CSV.

    Args:
        mutations: list of dataset3_single parquet paths (or a single path).
        pdbs: AlphaFold_model_PDBs parquet path.
        out: output task CSV path.
        pdb_dir: directory to write WT ``.pdb`` files into.
        split_value: value of ``split_col`` to keep (e.g. train/val/test).
        split: value written to the output ``split`` column.
        max_domains/max_mutations: optional subsetting for tractable compute.
        seed: sampling seed.

    Returns:
        The written ``pandas.DataFrame``.
    """
    if isinstance(mutations, str):
        mutations = [mutations]

    pdb_table = pd.read_parquet(pdbs)
    pdb_by_name = dict(zip(pdb_table["name"], pdb_table["pdb"]))

    mut = pd.concat([pd.read_parquet(f) for f in mutations], ignore_index=True)
    mut = mut[mut[split_col] == split_value].copy()
    # A mutation may recur across split schemes; keep one row per (WT, mutation).
    mut = mut.drop_duplicates(subset=["WT_name", "mut_type"])
    mut = mut[mut["WT_name"].isin(pdb_by_name)]
    mut["ddG_ML"] = pd.to_numeric(mut["ddG_ML"], errors="coerce")
    mut = mut.dropna(subset=["ddG_ML"])

    if max_domains is not None:
        keep = (mut["WT_name"].drop_duplicates()
                .sample(min(max_domains, mut["WT_name"].nunique()), random_state=seed))
        mut = mut[mut["WT_name"].isin(keep)]
    if max_mutations is not None and len(mut) > max_mutations:
        mut = mut.sample(max_mutations, random_state=seed)

    os.makedirs(pdb_dir, exist_ok=True)
    rows = []
    written_pdbs = set()
    for _, r in mut.iterrows():
        wt_name = r["WT_name"]
        cdna_mut = r["mut_type"]              # e.g. "E1Q"
        pos = cdna_mut[1:-1]
        if not pos.isdigit():
            continue                          # skip anything that isn't a clean substitution
        pdb_id = wt_name[:-4] if wt_name.endswith(".pdb") else wt_name
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if wt_name not in written_pdbs:
            with open(pdb_path, "w") as fh:
                fh.write(pdb_by_name[wt_name])
            written_pdbs.add(wt_name)
        rows.append({
            "id": f"{pdb_id}_{CHAIN}_{cdna_mut}",
            "pdb_id": pdb_id,
            "chain": CHAIN,
            "path": pdb_path,
            "positions": f"{CHAIN}:{pos}",
            "mutations": f"{CHAIN}:{cdna_mut}",
            "mut_info": cdna_mut,
            "mut_type": "ss",               # all single substitutions
            "ddg": float(r["ddG_ML"]),
            "split": split,
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # Written WITH a leading index column: run_protein_task reads columns by
    # name (index ignored), while train_adapter_monomer uses index_col=0.
    out_df.to_csv(out, index=True)
    print(f"Wrote {len(out_df)} mutations across {len(written_pdbs)} domains -> {out}")
    return out_df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mutations", nargs="+", required=True,
                   help="One or more dataset3_single parquet files.")
    p.add_argument("--pdbs", required=True, help="AlphaFold_model_PDBs parquet file.")
    p.add_argument("--split-value", default="train",
                   help="Value of the split column to keep (e.g. train/val/test).")
    p.add_argument("--split-col", default="split_name", help="Column holding the split.")
    p.add_argument("--split", default="train", help="Value written to output 'split'.")
    p.add_argument("--pdb-dir", default="data/cdna/pdb", help="Where to write WT .pdb files.")
    p.add_argument("--out", required=True, help="Output task CSV path.")
    p.add_argument("--max-domains", type=int, default=None)
    p.add_argument("--max-mutations", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    build_task_csv(
        mutations=args.mutations, pdbs=args.pdbs, out=args.out, pdb_dir=args.pdb_dir,
        split_value=args.split_value, split=args.split, split_col=args.split_col,
        max_domains=args.max_domains, max_mutations=args.max_mutations, seed=args.seed,
    )


if __name__ == "__main__":
    main()
