"""Tests for ProteinTask construction and mutation-type detection."""

import pytest

from aftoolkit import ProteinTask, parse_mutations


@pytest.mark.parametrize(
    "mut_str,expected_type",
    [
        ("A:S11A", "ss"),
        ("A:S11-", "del"),
        ("A:-11G", "ins"),
        ("A:S11A, A:T12V", "multi"),
    ],
)
def test_mut_type_detection(mut_str, expected_type):
    task = ProteinTask()
    task.set_task_mutants(parse_mutations(mut_str))
    assert task.get_mut_type() == expected_type


def test_wildtype_and_mutant_construction(monomer_pdb):
    task = ProteinTask()
    task.set_input_protein_task(monomer_pdb, chains=["A"])
    task.set_task_mutants(parse_mutations("A:S11A"))
    task.set_wildtype_protein()
    task.set_mutate_protein()

    assert not task.get_wildtype_protein().empty
    assert not task.get_mutate_protein().empty
    assert task.get_wt_aa(11, "A") == "S"


def test_observable_positions_default_covers_chain(monomer_pdb):
    task = ProteinTask()
    task.set_input_protein_task(monomer_pdb, chains=["A"])
    task.set_task_mutants(parse_mutations("A:S11A"))
    task.set_wildtype_protein()
    task.set_observable_positions()
    obs = task.task["obs_positions"]
    assert len(obs) == 1
    assert obs[0]["chain_id"] == "A"
    assert obs[0]["shift_from_resi"] >= 0


def test_add_mutation_roundtrip():
    task = ProteinTask()
    task.add_mutation(chain_id="A", resi=11, aa_mt="A", aa_wt="S")
    assert task.get_mutations() == {("S", 11, "A"): "A"}
