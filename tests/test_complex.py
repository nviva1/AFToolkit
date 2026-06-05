"""Tests for ProteinComplexTask mutation routing and chain splitting."""

from aftoolkit import ProteinComplexTask, parse_mutations
from aftoolkit.processing.utils import RELPOS_K, DEFAULT_CHAIN_GAP, load_protein


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


def test_complex_chain_splitting(complex_pdb):
    task = ProteinComplexTask()
    task.set_input_protein_task(
        complex_pdb, chains_for_protein=["I"], chains_for_complex=["E", "I"]
    )
    task.set_task_mutants(parse_mutations("I:L38S"))
    task.set_wildtype()
    task.set_mutate_protein()

    protein = task.get_wildtype_protein()
    complex_ = task.get_wildtype_complex()
    assert not protein.empty and not complex_.empty
    # the complex (E+I) is strictly larger than the isolated protein (I)
    assert complex_.shape[0] > protein.shape[0]


def test_mutation_routed_to_protein_and_complex(complex_pdb):
    task = ProteinComplexTask()
    task.set_input_protein_task(
        complex_pdb, chains_for_protein=["I"], chains_for_complex=["E", "I"]
    )
    task.set_task_mutants(parse_mutations("I:L38S"))
    # mutation on chain I belongs to both the isolated protein and the complex
    assert task.protein.get_mutations() == {("L", 38, "I"): "S"}
    assert task.complex.get_mutations() == {("L", 38, "I"): "S"}


def test_mutation_on_partner_chain_excluded_from_protein(complex_pdb):
    task = ProteinComplexTask()
    task.set_input_protein_task(
        complex_pdb, chains_for_protein=["I"], chains_for_complex=["E", "I"]
    )
    # mutate chain E, which is part of the complex but NOT the isolated protein
    task.set_task_mutants(parse_mutations("E:T1A"))
    assert task.protein.get_mutations() == {}
    assert task.complex.get_mutations() == {("T", 1, "E"): "A"}
