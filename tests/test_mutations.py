"""Tests for mutation/position parsing and the Mutation value type."""

import pytest

from aftoolkit import Mutation, ObservableRegion, parse_mutations, parse_positions
from aftoolkit.mutations import parse_point_mutations, residue_key


def test_parse_single_mutation():
    assert parse_mutations("A:D123C") == (
        Mutation(chain="A", position=123, wt="D", mt="C"),
    )


def test_parse_multiple_mutations_with_whitespace():
    assert parse_mutations("A:D123C, B:R56Y") == (
        Mutation(chain="A", position=123, wt="D", mt="C"),
        Mutation(chain="B", position=56, wt="R", mt="Y"),
    )


def test_parse_deletion():
    mutation, = parse_mutations("A:D123-")
    assert mutation.is_deletion
    assert not mutation.is_insertion
    assert not mutation.is_substitution


def test_parse_insertion():
    mutation, = parse_mutations("A:-123G")
    assert mutation.is_insertion
    assert not mutation.is_deletion


def test_parse_multi_residue_insertion():
    mutation, = parse_mutations("A:-123GGS")
    assert mutation.is_insertion
    assert mutation.mt == "GGS"


def test_parse_substitution():
    mutation, = parse_mutations("A:S11A")
    assert mutation.is_substitution


def test_parse_mutations_none_and_empty():
    assert parse_mutations(None) == ()
    assert parse_mutations("") == ()


def test_parse_mutations_accepts_mutation_objects():
    mutation = Mutation(chain="A", position=11, wt="S", mt="A")
    assert parse_mutations([mutation]) == (mutation,)


def test_parse_mutations_rejects_malformed():
    with pytest.raises(ValueError, match="Invalid mutation"):
        parse_mutations("nonsense")


def test_duplicate_site_is_rejected():
    # Mutations used to live in a dict keyed by position, so a second mutation
    # at the same site silently replaced the first.
    with pytest.raises(ValueError, match="each site may be mutated once"):
        parse_mutations("A:D1C, A:D1Y")


def test_mutation_is_hashable_and_ordered():
    muts = parse_mutations("B:R56Y, A:D123C")
    assert sorted(muts)[0].chain == "A"
    assert len(set(muts)) == 2


def test_mutation_str_roundtrips():
    assert str(Mutation(chain="A", position=11, wt="S", mt="A")) == "A:S11A"
    assert Mutation.parse("A:S11A") == Mutation(chain="A", position=11, wt="S", mt="A")


def test_mutation_key_matches_residue_key():
    mutation = Mutation(chain="A", position=90, wt="K", mt="M")
    assert mutation.key == residue_key("K", "A", 90) == "K_A_90"


def test_residue_key_marks_inserted_residues():
    assert residue_key("G", "A", 11, inserted=True) == "G_A_11i"
    assert residue_key("G", "A", 11, insertion="B") == "G_A_11B"


def test_parse_point_mutations_attaches_chain():
    assert parse_point_mutations("K90M,N107S", "A") == (
        Mutation(chain="A", position=90, wt="K", mt="M"),
        Mutation(chain="A", position=107, wt="N", mt="S"),
    )


def test_parse_point_mutations_rejects_malformed():
    with pytest.raises(ValueError, match="Invalid mutation"):
        parse_point_mutations("A:K90M", "A")


def test_parse_positions():
    assert parse_positions("A:10, B:20") == (
        ObservableRegion(chain="A", start=10),
        ObservableRegion(chain="B", start=20),
    )


def test_parse_positions_none():
    assert parse_positions(None) == ()


def test_observable_region_defaults_its_name():
    assert ObservableRegion(chain="A", start=10).name == "10_A"


def test_observable_region_rejects_negative_span():
    with pytest.raises(ValueError, match="span must be >= 0"):
        ObservableRegion(chain="A", start=10, span=-1)
