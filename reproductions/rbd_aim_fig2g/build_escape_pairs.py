#!/usr/bin/env python
"""Pairwise escape table: ONE row = ONE comparison of (reference variant) vs
(escape variant) for an antibody. Each row carries BOTH K_Ds, their ratio
(delta_log_kd = escape magnitude), the X-ray path, and the mutations needed to
build each endpoint from the crystal -- everything to evaluate one pair.

The reference is the highest-affinity variant per antibody; every other measured
variant becomes one escape-comparison row.

In : data/rbdaim/variant_mutation_table.csv
Out: data/rbdaim/escape_pairs.csv
"""
import math, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data/rbdaim/variant_mutation_table.csv")
OUT = os.path.join(ROOT, "data/rbdaim/escape_pairs.csv")


def escape_mutations(ref_to_variant_padded, chain):
    """Real (self-stripped), chain-prefixed mutations separating reference -> variant."""
    if not isinstance(ref_to_variant_padded, str) or ref_to_variant_padded.startswith("cross-species"):
        return ref_to_variant_padded if isinstance(ref_to_variant_padded, str) else ""
    toks = []
    for t in ref_to_variant_padded.split(";"):
        if t and t[0] != t[-1]:                       # drop self-mutations A123A
            toks.append(f"{chain}:{t}")
    return ",".join(toks)


def main():
    d = pd.read_csv(SRC)
    rows = []
    for ab, g in d.groupby("antibody"):
        ref = g[g.is_reference].iloc[0]
        apply_ref = ref.apply_to_xray_for_ref
        for _, v in g[~g.is_reference].iterrows():
            ref_kd, var_kd = float(ref.kd_nM), float(v.kd_nM)
            rows.append({
                "antibody": ab, "pdbid": v.pdbid, "rbd_chain": v.rbd_chain,
                "chids": v.chids, "pdb_path": v.pdb_path,
                "ref_strain": ref.strain, "ref_kd_nM": ref_kd,
                "var_strain": v.strain, "var_kd_nM": var_kd,
                "delta_log_kd": round(math.log10(var_kd / ref_kd), 4),     # escape magnitude
                "fold_change_kd": round(var_kd / ref_kd, 3),
                "delta_kd_nM": round(var_kd - ref_kd, 4),
                "n_escape_mut": v.n_mut_vs_ref,
                "escape_mutations": escape_mutations(v.mutations_ref_to_variant, v.rbd_chain),
                "xray_construct_fixes": v.xray_construct_fixes,
                "apply_to_xray_for_ref": apply_ref,
                "apply_to_xray_for_variant": v.apply_to_xray_for_variant,
                "aligned_positions": v.aligned_positions,
            })
    out = (pd.DataFrame(rows)
           .sort_values(["antibody", "delta_log_kd"], ascending=[True, False])
           .reset_index(drop=True))
    out.to_csv(OUT, index=False)

    n_xs = int(out.escape_mutations.astype(str).str.startswith("cross-species").sum())
    print(f"wrote {OUT}")
    print(f"  {len(out)} escape-comparison rows across {out.antibody.nunique()} antibodies")
    print(f"  cross-species pairs (unaligned mutations): {n_xs}")
    print(f"  delta_log_kd range: {out.delta_log_kd.min():+.2f} .. {out.delta_log_kd.max():+.2f}")
    print("\nstrongest 8 escape pairs:")
    cols = ["antibody", "ref_strain", "ref_kd_nM", "var_strain", "var_kd_nM",
            "delta_log_kd", "n_escape_mut", "escape_mutations"]
    with pd.option_context("display.max_colwidth", 55, "display.width", 200):
        print(out.sort_values("delta_log_kd", ascending=False).head(8)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
