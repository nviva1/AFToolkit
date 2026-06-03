"""Tests for ProteinTask.get_protein_embeddings feature aggregation.

This exercises the core embedding-aggregation logic with synthetic OpenFold
feature dicts, so no torch / model inference is required.
"""

import numpy as np
import pytest

from aftoolkit import ProteinTask, parse_mutations


def _make_task(mutants):
    """Build a ProteinTask with two observable positions and synthetic feats."""
    task = ProteinTask()
    task.set_task_mutants(mutants)

    # Two positions: the mutated S@A11 and a neighbour T@A12.
    wt_features = {
        "S_A_11": {"feat": np.array([1.0, 1.0])},
        "T_A_12": {"feat": np.array([2.0, 2.0])},
    }
    mt_features = {
        "S_A_11": {"feat": np.array([10.0, 10.0])},
        "T_A_12": {"feat": np.array([20.0, 20.0])},
    }
    positions = {"S_A_11": 0, "T_A_12": 1}

    task.set_wildtype_of_protein(of_features=wt_features)
    task.set_mutate_of_protein(of_features=mt_features)
    task.set_wildtype_observable_position(positions)
    task.set_mutate_observable_position(positions)
    return task


def test_mutpos_picks_mutation_position():
    task = _make_task(parse_mutations("A:S11A"))
    wt, mt = task.get_protein_embeddings(["feat"], protein_aggregation="mutpos")
    np.testing.assert_array_equal(wt, [1.0, 1.0])
    np.testing.assert_array_equal(mt, [10.0, 10.0])


def test_sum_aggregation():
    task = _make_task(parse_mutations("A:S11A"))
    wt, mt = task.get_protein_embeddings(["feat"], protein_aggregation="sum")
    np.testing.assert_array_equal(wt, [3.0, 3.0])
    np.testing.assert_array_equal(mt, [30.0, 30.0])


def test_mean_aggregation():
    task = _make_task(parse_mutations("A:S11A"))
    wt, mt = task.get_protein_embeddings(["feat"], protein_aggregation="mean")
    np.testing.assert_array_equal(wt, [1.5, 1.5])
    np.testing.assert_array_equal(mt, [15.0, 15.0])


def test_invalid_protein_aggregation_raises():
    task = _make_task(parse_mutations("A:S11A"))
    with pytest.raises(ValueError):
        task.get_protein_embeddings(["feat"], protein_aggregation="bogus")


def test_invalid_multi_aggregation_raises():
    task = _make_task(parse_mutations("A:S11A"))
    with pytest.raises(ValueError):
        task.get_protein_embeddings(["feat"], multi_aggregation="bogus")
