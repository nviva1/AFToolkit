"""Tests for PDB loading and structure mutation utilities."""

import pandas as pd
import pytest

from aftoolkit.processing.utils import (
    load_protein,
    mutate_protein,
    remove_sidechain_for_mutate_position,
    safe_filename_part,
)


def test_load_protein_renumbers_and_relabels(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    assert not df.empty
    # internal numbering collapses everything onto chain "A"
    assert set(df["chain_id"].unique()) == {"A"}
    # original numbering is preserved in dedicated columns
    assert "residue_number_original" in df.columns
    assert "chain_id_original" in df.columns
    # hydrogens are stripped
    assert (df["element_symbol"] != "H").all()


def test_substitution_keeps_backbone_only_and_renames(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    mutants = {("S", 11, "A"): "A"}  # S11A
    mt = mutate_protein(df, mutants)
    res11 = mt[mt["residue_number_original"] == 11]
    # mutated residue is renamed to ALA and reduced to the 4 backbone atoms
    assert list(res11["residue_name"].unique()) == ["ALA"]
    assert set(res11["atom_name"]) == {"N", "CA", "C", "O"}


def test_deletion_reduces_residue_count(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    n_res = df[df["atom_name"] == "CA"].shape[0]
    mt = mutate_protein(df, {("S", 11, "A"): "-"})
    n_res_mt = mt[mt["atom_name"] == "CA"].shape[0]
    assert n_res_mt == n_res - 1


def test_insertion_increases_residue_count(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    n_res = df[df["atom_name"] == "CA"].shape[0]
    mt = mutate_protein(df, {("-", 11, "A"): "G"})
    n_res_mt = mt[mt["atom_name"] == "CA"].shape[0]
    assert n_res_mt == n_res + 1


def test_remove_sidechain_noop_without_mutants(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    out = remove_sidechain_for_mutate_position(df, {})
    assert out.shape == df.shape


@pytest.mark.parametrize(
    "value,expected",
    [
        ("data/examples/PDB/1A7V.pdb", "1A7V.pdb"),
        ("A:A66H, B:R5Y", "A_A66H_B_R5Y"),
        (None, "none"),
        ("A", "A"),
        ("///", "none"),
    ],
)
def test_safe_filename_part(value, expected):
    assert safe_filename_part(value) == expected
