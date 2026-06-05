"""Unit tests for the torch-free interface-PAE computation."""

import numpy as np
import pytest

from aftoolkit.scoring import interface_pae


def _pae():
    # 4 residues: chains [A, A, B, B]; PAE[i,j] given.
    return np.array([
        [0.0, 1.0, 5.0, 6.0],
        [1.0, 0.0, 7.0, 8.0],
        [5.0, 7.0, 0.0, 1.0],
        [6.0, 8.0, 1.0, 0.0],
    ])


def test_interface_pae_between_two_chains():
    labels = ["A", "A", "B", "B"]
    # cross-pairs A<->B: 5,6,7,8 (A->B) and 5,7,6,8 (B->A) -> mean 6.5
    assert interface_pae(_pae(), labels, ["A"], ["B"]) == pytest.approx(6.5)


def test_interface_pae_symmetric():
    labels = ["A", "A", "B", "B"]
    assert interface_pae(_pae(), labels, ["A"], ["B"]) == \
           interface_pae(_pae(), labels, ["B"], ["A"])


def test_interface_pae_multichain_group():
    labels = ["H", "L", "E", "E"]
    # antibody = {H, L}, antigen = {E}; cross pairs only between {0,1} and {2,3}
    out = interface_pae(_pae(), labels, ["E"], ["H", "L"])
    assert out == pytest.approx(6.5)


def test_interface_pae_empty_group_raises():
    labels = ["A", "A", "A", "A"]
    with pytest.raises(ValueError):
        interface_pae(_pae(), labels, ["A"], ["B"])
