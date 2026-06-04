#!/usr/bin/env python
"""Predict the ddG impact of a point mutation with trained adapter(s), and
report AlphaFold confidence scores (pLDDT, pTM, PAE).

Runs OpenFold once on the wildtype and once on the mutant, extracts features at
the mutation site, feeds them to each trained adapter, and prints the predicted
ddG alongside per-structure confidence.

Example
-------
    python scripts/predict.py \
        --pdb ~/Downloads/p53.pdb --chain A --mutation E285K \
        --models data/models/monomer/pair+lddt_logits+plddt/trained_svm_*.pkl \
        --device mps
"""

import argparse
import glob
import os
import pickle

import numpy as np

from aftoolkit import OpenFoldWrapper, ProteinTask, parse_mutations, parse_positions


def _confidence(of_output, mut_idx):
    """Extract global + per-site confidence from a raw OpenFold output dict."""
    plddt = np.asarray(of_output["plddt"]).reshape(-1)
    conf = {
        "mean_pLDDT": float(plddt.mean()),
        "pLDDT_at_site": float(plddt[mut_idx]) if mut_idx is not None else None,
    }
    if "predicted_tm_score" in of_output:
        conf["pTM"] = float(np.asarray(of_output["predicted_tm_score"]).reshape(-1)[0])
    if "predicted_aligned_error" in of_output and mut_idx is not None:
        pae = np.asarray(of_output["predicted_aligned_error"])
        conf["mean_PAE_at_site"] = float(pae[mut_idx].mean())
    return conf


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", required=True)
    p.add_argument("--chain", required=True)
    p.add_argument("--mutation", required=True, help="e.g. E285K (WT<pos>MT)")
    p.add_argument("--models", nargs="+", required=True,
                   help="Trained adapter .pkl files (globs allowed).")
    p.add_argument("--device", default=None, help="cuda:0 / mps / cpu (auto if unset)")
    p.add_argument("--n-recycle", type=int, default=1)
    args = p.parse_args()

    pdb = os.path.expanduser(args.pdb)
    pos = int(args.mutation[1:-1])
    mut_str = f"{args.chain}:{args.mutation}"

    wrapper = OpenFoldWrapper(device=args.device, inference_n_recycle=args.n_recycle)
    wrapper.init_model()

    task = ProteinTask()
    task.set_input_protein_task(pdb, chains=[args.chain])
    task.set_task_mutants(parse_mutations(mut_str))
    task.set_observable_positions(parse_positions(f"{args.chain}:{pos}"))

    # Evaluate WT and MT separately so we can capture each structure's confidence.
    task.evaluate_wildtype_protein(wrapper, store_of_protein=True)
    wt_idx = next(iter(task.get_wildtype_observable_positions().values()), None)
    wt_conf = _confidence(wrapper.of_output, wt_idx)

    task.evaluate_mutate_protein(wrapper, store_of_protein=True)
    mt_idx = next(iter(task.get_mutate_observable_positions().values()), None)
    mt_conf = _confidence(wrapper.of_output, mt_idx)

    print(f"\n=== {os.path.basename(pdb)} {mut_str} ===")
    print("Confidence (wildtype):", {k: round(v, 3) for k, v in wt_conf.items() if v is not None})
    print("Confidence (mutant):  ", {k: round(v, 3) for k, v in mt_conf.items() if v is not None})

    model_paths = [m for pat in args.models for m in sorted(glob.glob(os.path.expanduser(pat)))]
    if not model_paths:
        print("\n(no adapter models matched; reported confidence scores only)")
        return

    print("\nPredicted ddG (kcal/mol; sign per the model's training convention):")
    for mp in model_paths:
        with open(mp, "rb") as f:
            adapter = pickle.load(f)
        ddg = float(np.ravel(adapter.predict(task))[0])
        print(f"  {os.path.basename(mp):<60} {ddg:+.3f}")


if __name__ == "__main__":
    main()
