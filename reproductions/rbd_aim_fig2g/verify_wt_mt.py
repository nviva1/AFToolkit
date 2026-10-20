#!/usr/bin/env python
"""Verify that for EVERY (antibody, reference, variant) the built structures carry
the CORRECT RBD sequence:
  * reference state  -> exactly the reference strain's RBD,
  * variant state    -> exactly the variant strain's RBD,
  * reference vs variant differ at EXACTLY the escape positions (no more/less).
Uses the same prep_state pipeline as the runs (load crystal -> mutate -> mask),
reads residue identity from CA atoms, compares to the canonical strain sequences
in kd_absolute_long.  Pure CPU, no inference.
"""
import ast, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_load as M
from aftoolkit.structure import AA_3_TO_1_DICT
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GAP = 25


def rbd_seq_map(df, rbd):
    ca = df[(df.chain_id_original == rbd) & (df.atom_name == "CA")].sort_values("residue_number_original")
    return {int(n): AA_3_TO_1_DICT.get(r, "X")
            for n, r in zip(ca.residue_number_original, ca.residue_name)}


def main():
    pairs = pd.read_csv(os.path.join(ROOT, "data/rbdaim/escape_pairs.csv"))
    long = pd.read_csv(os.path.join(ROOT, "data/rbdaim/kd_absolute_long.csv"))
    seqof = {(a, s): q for a, s, q in zip(long.antibody, long.strain, long.rbd_sequence)}
    bad = 0; ok = 0; skipped = 0
    for ab, g in pairs.groupby("antibody"):
        row0 = g.iloc[0]; chids = ast.literal_eval(row0.chids); rbd = row0.rbd_chain
        pdb = row0.pdb_path; ref_strain = row0.ref_strain
        mask_pos = M.positions_in(row0.apply_to_xray_for_ref)
        for _, r in g.iterrows():
            mask_pos |= M.positions_in(r.apply_to_xray_for_variant)

        ref_map = rbd_seq_map(M.prep_state(pdb, chids, rbd, row0.apply_to_xray_for_ref, mask_pos, GAP), rbd)
        exp_ref = seqof.get((ab, ref_strain))
        # check reference == reference strain at every resolved position
        rmis = [p for p, aa in ref_map.items() if exp_ref and aa != exp_ref[p - 334]]
        if rmis:
            print(f"  [REF MISMATCH] {ab}/{ref_strain}: positions {rmis[:6]}"); bad += 1
        for _, r in g.iterrows():
            if str(r.apply_to_xray_for_variant).startswith("cross-species"):
                skipped += 1; continue
            vmap = rbd_seq_map(M.prep_state(pdb, chids, rbd, r.apply_to_xray_for_variant, mask_pos, GAP), rbd)
            exp_v = seqof.get((ab, r.var_strain))
            vmis = [p for p, aa in vmap.items() if exp_v and aa != exp_v[p - 334]]
            # ref vs variant difference must equal the escape positions (within resolved set)
            esc = M.positions_in(r.escape_mutations)
            diff = {p for p in vmap if p in ref_map and vmap[p] != ref_map[p]}
            esc_present = {p for p in esc if p in vmap and p in ref_map}
            if vmis:
                print(f"  [VAR MISMATCH] {ab}/{r.var_strain}: {vmis[:6]} (built != strain)"); bad += 1
            elif diff != esc_present:
                print(f"  [DIFF MISMATCH] {ab}/{r.var_strain}: ref->var changes {sorted(diff)} "
                      f"!= escape {sorted(esc_present)}"); bad += 1
            else:
                ok += 1
    print(f"\nchecked: {ok} variant pairs OK, {bad} mismatches, {skipped} cross-species skipped")
    print("ALL WT/MT CORRECT ✓" if bad == 0 else f"*** {bad} PROBLEMS ***")


if __name__ == "__main__":
    main()
