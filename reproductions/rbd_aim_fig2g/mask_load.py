#!/usr/bin/env python
"""Loading helpers for the Fig 2g escape runner.

Core idea (no-leak masking): for any RBD position that is mutated relative to the
X-ray crystal in EITHER endpoint of a compared pair, keep ONLY backbone atoms
[N, CA, C, O] at that position -- in BOTH builds. This removes the crystal
sidechain rotamer symmetrically, so the reference can't leak a real sidechain at a
spot where the escape variant has to rebuild one (the E123E-in-WT / E123Q-in-escape
case). The backbone is left templated; only the sidechain is gone.

mutate_protein() already strips to backbone for residues it actually mutates, but
NOT for self-positions (E123E) -- this closes that gap.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aftoolkit.structure import load_protein, mutate_protein
from aftoolkit.mutations import parse_mutations

BACKBONE = ["N", "CA", "C", "O"]
_TOK = re.compile(r"^[A-Za-z0-9]+:[A-Z](\d+)[A-Z]$")


def positions_in(apply_str):
    """Canonical residue numbers referenced by a '<chain>:<WT><pos><MT>,...' string."""
    if not isinstance(apply_str, str) or not apply_str.strip() or ":" not in apply_str:
        return set()
    out = set()
    for tok in apply_str.split(","):
        m = _TOK.match(tok.strip())
        if m:
            out.add(int(m.group(1)))
    return out


def pair_mask_positions(apply_ref, apply_var):
    """Union of positions mutated-vs-crystal across the reference and the variant.
    These are the positions to reduce to backbone in BOTH builds."""
    return positions_in(apply_ref) | positions_in(apply_var)


def keep_backbone_at(pdb_df, rbd_chain, positions):
    """Drop every non-backbone atom at (rbd_chain, position) for position in `positions`.
    Operates on original numbering/chain (canonical RBD resnums, RBD chain id)."""
    pos = {int(p) for p in positions}
    if not pos:
        return pdb_df
    sel = (pdb_df["chain_id_original"] == rbd_chain) & \
          (pdb_df["residue_number_original"].isin(pos))
    drop = sel & (~pdb_df["atom_name"].isin(BACKBONE))
    return pdb_df[~drop].copy()


def keep_backbone_cb(pdb_df, rbd_chain, touched_positions):
    """Global backbone+CB template: every residue keeps only [N, CA, C, O, CB];
    the TOUCHED RBD positions keep only [N, CA, C, O] (CB dropped too). Strips all
    longer sidechain atoms everywhere -> AF2Rank-style coarse template, with the
    mutated sites one step more ambiguous than the rest."""
    pos = {int(p) for p in touched_positions}
    is_touched = (pdb_df["chain_id_original"] == rbd_chain) & \
                 (pdb_df["residue_number_original"].isin(pos))
    keep = pdb_df["atom_name"].isin(BACKBONE) | \
           ((pdb_df["atom_name"] == "CB") & (~is_touched))
    return pdb_df[keep].copy()


def prep_state(pdb_path, chains, rbd_chain, apply_str, mask_positions, chain_gap=25):
    """Load crystal -> apply this state's mutations -> strip the pair's mask positions
    to backbone-only. Returns a pdb_df ready for OpenFoldWrapper.evaluate()."""
    d = load_protein(pdb_path, chains=chains, chain_gap=chain_gap)
    if isinstance(apply_str, str) and apply_str.strip() and ":" in apply_str:
        d = mutate_protein(d, parse_mutations(apply_str))
    d = keep_backbone_at(d, rbd_chain, mask_positions)
    return d


def prep_state_cb(pdb_path, chains, rbd_chain, apply_str, mask_positions, chain_gap=25):
    """As prep_state, but the whole complex is reduced to backbone+CB, and the
    touched (mutated) RBD positions to backbone-only."""
    d = load_protein(pdb_path, chains=chains, chain_gap=chain_gap)
    if isinstance(apply_str, str) and apply_str.strip() and ":" in apply_str:
        d = mutate_protein(d, parse_mutations(apply_str))
    d = keep_backbone_cb(d, rbd_chain, mask_positions)
    return d
