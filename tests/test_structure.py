"""Tests for PDB loading and structure mutation utilities."""

import pytest

from aftoolkit import Mutation, parse_mutations
from aftoolkit.structure import (
    load_protein,
    mutate_protein,
    safe_filename_part,
    strip_mutated_sidechains,
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
    mt = mutate_protein(df, parse_mutations("A:S11A"))
    res11 = mt[mt["residue_number_original"] == 11]
    # mutated residue is renamed to ALA and reduced to the 4 backbone atoms
    assert list(res11["residue_name"].unique()) == ["ALA"]
    assert set(res11["atom_name"]) == {"N", "CA", "C", "O"}


def test_mutate_protein_accepts_a_string(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    mt = mutate_protein(df, "A:S11A")
    res11 = mt[mt["residue_number_original"] == 11]
    assert list(res11["residue_name"].unique()) == ["ALA"]


def test_mutate_protein_leaves_input_untouched(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    before = df.copy()
    mutate_protein(df, parse_mutations("A:S11A"))
    assert df.equals(before)


def test_deletion_reduces_residue_count(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    n_res = df[df["atom_name"] == "CA"].shape[0]
    mt = mutate_protein(df, parse_mutations("A:S11-"))
    assert mt[mt["atom_name"] == "CA"].shape[0] == n_res - 1


def test_insertion_increases_residue_count(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    n_res = df[df["atom_name"] == "CA"].shape[0]
    mt = mutate_protein(df, parse_mutations("A:-11G"))
    assert mt[mt["atom_name"] == "CA"].shape[0] == n_res + 1


def test_multi_residue_insertion_adds_each_residue(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    n_res = df[df["atom_name"] == "CA"].shape[0]
    mt = mutate_protein(df, parse_mutations("A:-11GGS"))
    assert mt[mt["atom_name"] == "CA"].shape[0] == n_res + 3


def test_unknown_residue_code_raises(monomer_pdb):
    # Previously an unrecognised code was silently dropped from the conversion
    # table, so the mutation was never applied and no error was raised.
    df = load_protein(monomer_pdb, chains=["A"])
    with pytest.raises(ValueError, match="Unknown residue code"):
        mutate_protein(df, [Mutation(chain="A", position=11, wt="S", mt="Z")])


def test_strip_sidechains_noop_without_mutations(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    assert strip_mutated_sidechains(df, ()).shape == df.shape


def test_strip_sidechains_trims_substituted_residue(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    out = strip_mutated_sidechains(df, parse_mutations("A:S11A"))
    res11 = out[out["residue_number_original"] == 11]
    # side chain gone, but the residue keeps its wildtype identity
    assert set(res11["atom_name"]) == {"N", "CA", "C", "O"}
    assert list(res11["residue_name"].unique()) == ["SER"]


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


def test_wrong_wildtype_residue_is_rejected(monomer_pdb):
    """Regression: a mutation naming the wrong wildtype residue matched no
    residue group and was silently dropped, so the 'mutant' came back identical
    to the wildtype and any ddG computed from it was meaningless."""
    df = load_protein(monomer_pdb, chains=["A"])
    # Position 11 of 1A0F chain A is SER, not THR.
    with pytest.raises(ValueError, match="expected THR .* found SER"):
        mutate_protein(df, "A:T11A")


def test_wrong_wildtype_residue_rejected_for_deletion(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    with pytest.raises(ValueError, match="does not match the structure"):
        mutate_protein(df, "A:T11-")


def test_mutation_at_missing_position_is_rejected(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    with pytest.raises(ValueError, match="no residue at A:9999"):
        mutate_protein(df, "A:S9999A")


def test_all_mismatches_are_reported_together(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    with pytest.raises(ValueError) as excinfo:
        mutate_protein(df, "A:T11A,A:S9999A")
    message = str(excinfo.value)
    assert "A:T11A" in message and "A:S9999A" in message


def test_strip_sidechains_also_validates(monomer_pdb):
    df = load_protein(monomer_pdb, chains=["A"])
    with pytest.raises(ValueError, match="does not match the structure"):
        strip_mutated_sidechains(df, parse_mutations("A:T11A"))


def test_save_to_pdb_orders_columns_before_writing(tmp_path):
    """Regression: biopandas writes columns in the order it finds them, so a
    hand-built frame whose columns were ordered differently was written with its
    fields in the wrong fixed-width positions -- a corrupt file, written without
    any error."""
    import pandas as pd

    from aftoolkit.structure import save_to_pdb

    # Deliberately non-canonical column order, and several columns absent.
    rows = [{"residue_name": "ALA", "b_factor": -5.0, "atom_name": atom,
             "residue_number": 10, "chain_id": "A",
             "residue_number_original": 10, "chain_id_original": "A",
             "x_coord": 1.0, "y_coord": 2.0, "z_coord": 3.0}
            for atom in ("N", "CA", "C", "O")]
    out = tmp_path / "hand_built.pdb"
    save_to_pdb(pd.DataFrame(rows), out)

    reloaded = load_protein(out)
    ca = reloaded[reloaded.atom_name == "CA"]
    assert len(ca) == 1
    assert float(ca.b_factor.iloc[0]) == pytest.approx(-5.0)
    assert float(ca.x_coord.iloc[0]) == pytest.approx(1.0)
    assert ca.residue_name.iloc[0] == "ALA"
