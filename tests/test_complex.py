"""Tests for ProteinComplexTask mutation routing and chain splitting."""

import pytest

from aftoolkit import ProteinComplexTask, parse_mutations, parse_positions
from aftoolkit.structure import DEFAULT_CHAIN_GAP, RELPOS_K, load_protein


def _task(complex_pdb):
    return ProteinComplexTask().set_input(
        complex_pdb, protein_chains=["I"], complex_chains=["E", "I"]
    )


def test_chain_gap_saturates_relpos(complex_pdb):
    # Multi-chain -> single chain "A"; the inter-chain numbering gap must be
    # >= relpos_k (32) so AF2's relative-position encoding treats the chains as
    # distinct (the junction offset lands in the terminal bin).
    assert DEFAULT_CHAIN_GAP >= RELPOS_K
    ca = load_protein(complex_pdb, chains=["E", "I"])
    ca = ca[ca.atom_name == "CA"].reset_index(drop=True)
    boundary = ca.index[ca["chain_id_original"] != ca["chain_id_original"].shift(-1)][0]
    offset = int(ca.loc[boundary + 1, "residue_number"] - ca.loc[boundary, "residue_number"])
    assert offset >= RELPOS_K


def test_chain_gap_is_configurable(complex_pdb):
    # The RBD-AIM protocol pins this to 25 rather than the default. Note the
    # resulting numbering offset is chain_gap + 1: the next chain starts one
    # past the gap, so `chain_gap` is the count of *skipped* indices.
    ca = load_protein(complex_pdb, chains=["E", "I"], chain_gap=25)
    ca = ca[ca.atom_name == "CA"].reset_index(drop=True)
    boundary = ca.index[ca["chain_id_original"] != ca["chain_id_original"].shift(-1)][0]
    offset = int(ca.loc[boundary + 1, "residue_number"] - ca.loc[boundary, "residue_number"])
    assert offset == 26


def test_complex_chain_splitting(complex_pdb):
    task = _task(complex_pdb).set_mutations("I:L38S")
    task.build()

    protein = task.protein.wildtype.structure
    bound = task.complex.wildtype.structure
    assert not protein.empty and not bound.empty
    # the complex (E+I) is strictly larger than the isolated protein (I)
    assert bound.shape[0] > protein.shape[0]


def test_mutation_routed_to_protein_and_complex(complex_pdb):
    task = _task(complex_pdb).set_mutations("I:L38S")
    expected = parse_mutations("I:L38S")
    # a mutation on chain I belongs to both the isolated protein and the complex
    assert task.protein.mutations == expected
    assert task.complex.mutations == expected
    assert task.mutations == expected


def test_mutation_on_partner_chain_excluded_from_protein(complex_pdb):
    # chain E is part of the complex but NOT the isolated protein
    task = _task(complex_pdb).set_mutations("E:T1A")
    assert task.protein.mutations == ()
    assert task.complex.mutations == parse_mutations("E:T1A")


def test_observables_routed_by_chain(complex_pdb):
    task = _task(complex_pdb).set_observables(parse_positions("I:38, E:1"))
    assert [r.chain for r in task.protein.observables] == ["I"]
    assert [r.chain for r in task.complex.observables] == ["I", "E"]


def test_add_mutation_routes_too(complex_pdb):
    task = _task(complex_pdb)
    task.add_mutation("I:L38S")
    task.add_mutation("E:T1A")
    assert len(task.complex.mutations) == 2
    assert len(task.protein.mutations) == 1


def test_repr_mentions_both_sides(complex_pdb):
    task = _task(complex_pdb).set_mutations("I:L38S")
    text = repr(task)
    assert "I:L38S" in text and "protein_chains" in text


def test_deletion_on_the_mutated_chain_shortens_both_sides(complex_pdb):
    task = _task(complex_pdb).set_mutations("I:L38-")
    task.set_observables()
    task.build()

    for side in (task.protein, task.complex):
        wt = side._resolve_observables(side.wildtype.structure)
        mt = side._resolve_observables(side.mutant.structure)
        assert len(mt) == len(wt) - 1


def test_insertion_on_the_mutated_chain_lengthens_both_sides(complex_pdb):
    task = _task(complex_pdb).set_mutations("I:-38GG")
    task.set_observables()
    task.build()

    for side in (task.protein, task.complex):
        wt = side._resolve_observables(side.wildtype.structure)
        mt = side._resolve_observables(side.mutant.structure)
        assert len(mt) == len(wt) + 2


def test_partner_chain_deletion_leaves_the_unbound_side_alone(complex_pdb):
    # Chain E is in the complex but not the isolated protein, so a deletion
    # there must not touch the unbound form at all.
    task = _task(complex_pdb).set_mutations("E:A1-")
    task.set_observables()
    task.build()

    assert task.protein.mutations == ()
    assert task.protein.mutant.structure.empty  # nothing to mutate

    # Deleting a chain's *first* residue moves the region's anchor for both
    # structures, so the two end up covering exactly the same residues -- which
    # is what makes their embeddings comparable.
    bound_wt = task.complex._resolve_observables(task.complex.wildtype.structure)
    bound_mt = task.complex._resolve_observables(task.complex.mutant.structure)
    assert set(bound_wt) == set(bound_mt)
    assert "A_E_1" not in bound_wt  # the deleted residue is excluded from both


def test_interior_deletion_drops_only_that_residue_from_the_mutant(complex_pdb):
    task = _task(complex_pdb).set_mutations("I:L38-")
    task.set_observables()
    task.build()

    wt = task.complex._resolve_observables(task.complex.wildtype.structure)
    mt = task.complex._resolve_observables(task.complex.mutant.structure)
    assert set(wt) - set(mt) == {"L_I_38"}


def test_multiple_sites_must_all_be_observed(complex_pdb):
    """A mutated site that is not observed is reported by name, rather than
    silently dropping out of the embedding."""
    task = _task(complex_pdb).set_mutations("I:L38S,I:V45A")
    task.set_observables(parse_positions("I:38"))
    task.complex.wildtype.features = {"L_I_38": {"feat": [1.0]}}
    task.complex.mutant.features = {"L_I_38": {"feat": [2.0]}}
    task.complex.wildtype.observed = task.complex.mutant.observed = {"L_I_38": 0}

    with pytest.raises(KeyError, match="V_I_45"):
        task.complex.embeddings(["feat"])
