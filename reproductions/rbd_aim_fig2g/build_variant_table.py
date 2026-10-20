#!/usr/bin/env python
"""Per-antibody variant table for a FAIR cross-variant comparison.

For each antibody:
  * pick the highest-affinity measurement (lowest K_D) as the REFERENCE variant;
  * express every other variant's RBD as mutations RELATIVE TO that reference;
  * take the UNION of all positions that mutate in any variant, and pad every
    variant's mutation string with self-mutations (e.g. A123A) at the positions
    where it keeps the reference residue -- so all variants of one antibody share
    an identical position list and are directly comparable.

Also maps each row to its cropped X-ray PDB in x_ray_pdbs/.
Cross-species antigens (SARS-CoV-1, different length) can't be position-aligned;
they are kept but flagged 'cross-species(unaligned)' and excluded from the union.

Out: data/rbdaim/variant_mutation_table.csv
"""
import ast, math, os, sys
import pandas as pd
from biopandas.pdb import PandasPdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aftoolkit.structure import AA_3_TO_1_DICT
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
PDBDIR = os.path.join(HERE, "x_ray_pdbs")
SRC = os.path.join(ROOT, "data/rbdaim/kd_absolute_long.csv")
OUT = os.path.join(ROOT, "data/rbdaim/variant_mutation_table.csv")
REF_START = 334                      # canonical numbering of RBD index 0


def pdb_path(ab, pdbid, chids):
    H, L, RBD = chids
    fp = os.path.join(PDBDIR, f"{ab.replace('/', '_')}_{pdbid}_{H}{L}{RBD}.pdb")
    return fp if os.path.exists(fp) else "NA(missing)"


def xray_rbd_map(fp, rbd_chain):
    """{canonical_resnum: 1-letter} for the RBD chain actually present in the crystal.
    Residue numbers in the cropped PDB are already canonical (334..527)."""
    df = PandasPdb().read_pdb(fp).df["ATOM"]
    ca = df[(df.chain_id == rbd_chain) & (df.atom_name == "CA")].sort_values("residue_number")
    return {int(n): AA_3_TO_1_DICT.get(r, "X") for n, r in zip(ca.residue_number, ca.residue_name)}


def main():
    d = pd.read_csv(SRC)
    rows = []
    for ab, g in d.groupby("antibody"):
        g = g.reset_index(drop=True)
        chids = ast.literal_eval(g.chids.iloc[0]); pdbid = str(g.pdbid.iloc[0])
        fp = pdb_path(ab, pdbid, chids)

        ref_i = g.kd_nM.idxmin()                       # highest affinity = lowest K_D
        ref = g.loc[ref_i]; ref_seq = ref.rbd_sequence; kd_ref = float(ref.kd_nM)
        alignable = g[g.rbd_sequence.str.len() == len(ref_seq)]
        xmap = xray_rbd_map(fp, chids[2]) if fp != "NA(missing)" else {}

        # COMPARISON union: ONLY positions that vary ACROSS this antibody's variants
        # (i.e. differ from the reference in some variant). Positions where all
        # variants agree are NOT part of the per-variant comparison.
        union = set()
        for _, v in alignable.iterrows():
            union |= {i for i in range(len(ref_seq)) if v.rbd_sequence[i] != ref_seq[i]}
        union = sorted(union); uset = set(union)
        ch = chids[2]; L = len(ref_seq)
        pos_list = ";".join(str(REF_START + i) for i in union)

        def apply_str(target_seq):
            """COMPLETE chain-prefixed real mutations to turn the crystal RBD into
            target_seq: every position where the crystal residue differs from the
            target (construct fixes + variant-specific), directly usable downstream."""
            return ",".join(f"{ch}:{xmap[REF_START + i]}{REF_START + i}{target_seq[i]}"
                            for i in range(L)
                            if (REF_START + i) in xmap and xmap[REF_START + i] != target_seq[i])

        # construct fixes: crystal deviations OUTSIDE the comparison union -> identical
        # for every variant (e.g. a stabilising V367F in the construct). Applied to
        # all, so excluded from the per-variant comparison: "don't touch it again".
        construct = [i for i in range(L) if i not in uset
                     and (REF_START + i) in xmap and xmap[REF_START + i] != ref_seq[i]]
        construct_fixes = ",".join(f"{ch}:{xmap[REF_START + i]}{REF_START + i}{ref_seq[i]}"
                                   for i in construct)

        # crystal -> reference state over the comparison positions only
        x2ref = ";".join(f"{xmap.get(REF_START + i, '?')}{REF_START + i}{ref_seq[i]}" for i in union)
        apply_ref = apply_str(ref_seq)
        n_missing = sum(1 for i in union if (REF_START + i) not in xmap)

        for _, v in g.iterrows():
            same_len = len(v.rbd_sequence) == len(ref_seq)
            if same_len:
                muts = ";".join(f"{ref_seq[i]}{REF_START + i}{v.rbd_sequence[i]}" for i in union)
                x2var = ";".join(f"{xmap.get(REF_START + i, '?')}{REF_START + i}{v.rbd_sequence[i]}"
                                 for i in union)
                apply_var = apply_str(v.rbd_sequence)
                n_real = sum(1 for i in union if v.rbd_sequence[i] != ref_seq[i])
            else:
                muts = x2var = apply_var = "cross-species(unaligned)"; n_real = -1
            rows.append({
                "antibody": ab, "pdbid": pdbid, "rbd_chain": chids[2],
                "chids": str(chids), "pdb_path": fp,
                "strain": v.strain, "kd_nM": float(v.kd_nM),
                "is_reference": bool(v.strain == ref.strain),
                "ref_strain": ref.strain,
                "dlog_kd_vs_ref": round(math.log10(float(v.kd_nM) / kd_ref), 4),
                "n_mut_vs_ref": n_real, "n_missing_in_xray": n_missing,
                "aligned_positions": pos_list,
                "xray_construct_fixes": construct_fixes,
                "mutations_ref_to_variant": muts,
                "mutations_relative_to_xray_ref": x2ref,
                "mutations_relative_to_xray_variant": x2var,
                "apply_to_xray_for_ref": apply_ref,
                "apply_to_xray_for_variant": apply_var,
            })

    out = pd.DataFrame(rows).sort_values(["antibody", "kd_nM"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(out)} rows, {out.antibody.nunique()} antibodies)")
    miss = out[out.pdb_path.str.startswith("NA")]
    print(f"pdb paths mapped: {(out.pdb_path != 'NA(missing)').sum()}/{len(out)}"
          + (f"  MISSING: {list(miss.antibody.unique())}" if len(miss) else ""))
    print(f"non-reference variants needing a real comparison: "
          f"{int((~out.is_reference).sum())}")
    print(f"rows with union positions absent in the crystal (n_missing_in_xray>0): "
          f"{int((out.n_missing_in_xray > 0).sum())}")
    print("\n--- CR3022 (construct fix shown once; comparison excludes it) ---")
    print("construct fixes (applied to ALL variants):",
          out[out.antibody == "CR3022"].xray_construct_fixes.iloc[0])
    cols = ["strain", "kd_nM", "is_reference", "n_mut_vs_ref", "apply_to_xray_for_variant"]
    cr = out[out.antibody == "CR3022"][cols]
    with pd.option_context("display.max_colwidth", 120):
        print(cr.to_string(index=False))


if __name__ == "__main__":
    main()
