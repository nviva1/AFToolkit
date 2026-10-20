#!/usr/bin/env python
"""Reproduce RBD-AIM Fig 2g (Delta iPAE vs Delta log10 K_D) from structures.

Given a manifest of antibody-RBD complex structures with their experimental
binding affinities, this recomputes the interface PAE (iPAE) for each complex
with AFToolkit's OpenFold pipeline, forms Delta iPAE relative to each antibody's
reference RBD strain, correlates it with the experimental Delta log10 K_D, and
plots the scatter + regression -- the same relationship as Fig 2g.

Manifest CSV columns:
    ab_id        antibody identifier (groups strains of the same antibody)
    pdb          path to the Ab-RBD complex structure
    antigen      antigen chain id(s), comma-separated
    antibody     antibody chain id(s), comma-separated
    delta_log_kd experimental Delta log10 K_D for this Ab x strain (0 for reference)
    is_reference 1 for the antibody's reference strain, else 0

Example:
    python reproductions/rbd_aim_fig2g/recompute_from_structures.py \\
        --manifest data/rbdaim/fig2g_manifest.csv --device mps --n-recycle 2 \\
        --out reproductions/rbd_aim_fig2g/recomputed
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress

from aftoolkit.scoring import score_complex


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--n-recycle", type=int, default=2)
    p.add_argument("--out", default="results/fig2g_recomputed")
    args = p.parse_args()

    df = pd.read_csv(args.manifest)
    os.makedirs(args.out, exist_ok=True)

    # One model load, reused across all complexes.
    from aftoolkit.folding import OpenFoldWrapper
    wrapper = OpenFoldWrapper(device=args.device, inference_n_recycle=args.n_recycle)
    wrapper.init_model()

    # --- Step 1: run AFToolkit on each input structure -> scores ----------
    rows = []
    for _, r in df.iterrows():
        res = score_complex(
            pdb=r["pdb"], antigen_chains=str(r["antigen"]).split(","),
            antibody_chains=str(r["antibody"]).split(","), wrapper=wrapper,
        )
        rows.append({"ipae": res["ipae"], "mean_plddt": res["mean_plddt"],
                     "ptm": res["ptm"], "n_residues": res["n_residues"]})
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    scores_csv = os.path.join(args.out, "scores.csv")
    df.to_csv(scores_csv, index=False)
    print(f"\nAFToolkit scores for {len(df)} structure(s) -> {scores_csv}")
    print(df[["ipae", "mean_plddt", "ptm", "n_residues"]].to_string(index=False))

    # --- Step 2 (optional): reproduce the Fig 2g correlation --------------
    # Only when the manifest carries experimental affinities + a reference.
    if {"ab_id", "delta_log_kd", "is_reference"}.issubset(df.columns):
        ref = df[df["is_reference"] == 1].set_index("ab_id")["ipae"].to_dict()
        df["delta_ipae"] = [row["ipae"] - ref.get(row["ab_id"], np.nan)
                            for _, row in df.iterrows()]
        sub = df[(df["delta_log_kd"] != 0) & df["delta_ipae"].notna()]
        if len(sub) >= 3:
            x, y = sub["delta_log_kd"].to_numpy(), sub["delta_ipae"].to_numpy()
            r, _ = pearsonr(x, y)
            rs, _ = spearmanr(x, y)
            lr = linregress(x, y)
            print(f"\nFig 2g: n={len(x)}  Pearson={r:+.3f}  Spearman={rs:+.3f}  "
                  f"(paper |R|=0.52)")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4, 3.2))
            ax.scatter(x, y, s=20, color="#2a9d8f", alpha=.8, edgecolor="none")
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, lr.slope * xs + lr.intercept, color="#2a9d8f")
            ax.set_xlabel("delta log10 K_D")
            ax.set_ylabel("delta iPAE (recomputed)")
            ax.set_title(f"Fig 2g recomputed (R={r:.2f})")
            plt.tight_layout()
            out_png = os.path.join(args.out, "fig2g_recomputed.png")
            plt.savefig(out_png, dpi=150)
            print(f"saved -> {out_png}")
    else:
        print("\n(no affinity columns in manifest -> scored structures only; "
              "add ab_id / delta_log_kd / is_reference to reproduce the Fig 2g plot)")


if __name__ == "__main__":
    main()
