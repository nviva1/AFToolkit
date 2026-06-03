"""Tests for mutation/position string parsing."""

from aftoolkit import parse_mutations, parse_positions


def test_parse_single_mutation():
    assert parse_mutations("A:D123C") == {("D", 123, "A"): "C"}


def test_parse_multiple_mutations_with_whitespace():
    assert parse_mutations("A:D123C, B:R56Y") == {
        ("D", 123, "A"): "C",
        ("R", 56, "B"): "Y",
    }


def test_parse_deletion():
    # aa_mt == "-" encodes a deletion
    assert parse_mutations("A:D123-") == {("D", 123, "A"): "-"}


def test_parse_insertion():
    # aa_wt == "-" encodes an insertion
    assert parse_mutations("A:-123G") == {("-", 123, "A"): "G"}


def test_parse_mutations_none():
    assert parse_mutations(None) == {}


def test_parse_positions():
    assert parse_positions("A:10, B:20") == [
        {"resi": 10, "shift_from_resi": 0, "chain_id": "A"},
        {"resi": 20, "shift_from_resi": 0, "chain_id": "B"},
    ]


def test_parse_positions_none():
    assert parse_positions(None) == []
