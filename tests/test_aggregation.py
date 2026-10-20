"""Tests for feature aggregation (aftoolkit.embeddings).

Exercised with synthetic feature dicts, so no torch / model inference is needed.
"""

import numpy as np
import pytest

from aftoolkit import ProteinTask, parse_mutations
from aftoolkit.embeddings import aggregate

# Two observable residues: the mutated S@A11 and its neighbour T@A12.
WT_FEATURES = {
    "S_A_11": {"feat": np.array([1.0, 1.0])},
    "T_A_12": {"feat": np.array([2.0, 2.0])},
}
MT_FEATURES = {
    "S_A_11": {"feat": np.array([10.0, 10.0])},
    "T_A_12": {"feat": np.array([20.0, 20.0])},
}
POSITIONS = {"S_A_11": 0, "T_A_12": 1}


def _make_task(mutations):
    """A task carrying precomputed features, bypassing inference."""
    task = ProteinTask()
    task.set_mutations(mutations)
    task.wildtype.features = WT_FEATURES
    task.mutant.features = MT_FEATURES
    task.wildtype.observed = POSITIONS
    task.mutant.observed = POSITIONS
    return task


def test_mutpos_picks_mutation_position():
    task = _make_task(parse_mutations("A:S11A"))
    wt, mt = task.embeddings(["feat"], protein_aggregation="mutpos")
    np.testing.assert_array_equal(wt, [1.0, 1.0])
    np.testing.assert_array_equal(mt, [10.0, 10.0])


def test_sum_aggregation():
    task = _make_task(parse_mutations("A:S11A"))
    wt, mt = task.embeddings(["feat"], protein_aggregation="sum")
    np.testing.assert_array_equal(wt, [3.0, 3.0])
    np.testing.assert_array_equal(mt, [30.0, 30.0])


def test_mean_aggregation():
    task = _make_task(parse_mutations("A:S11A"))
    wt, mt = task.embeddings(["feat"], protein_aggregation="mean")
    np.testing.assert_array_equal(wt, [1.5, 1.5])
    np.testing.assert_array_equal(mt, [15.0, 15.0])


def test_multi_mutation_sums_each_site():
    task = _make_task(parse_mutations("A:S11A, A:T12V"))
    wt, mt = task.embeddings(["feat"], multi_aggregation="sum")
    np.testing.assert_array_equal(wt, [3.0, 3.0])
    np.testing.assert_array_equal(mt, [30.0, 30.0])


def test_multi_mutation_means_each_site():
    task = _make_task(parse_mutations("A:S11A, A:T12V"))
    wt, mt = task.embeddings(["feat"], multi_aggregation="mean")
    np.testing.assert_array_equal(wt, [1.5, 1.5])
    np.testing.assert_array_equal(mt, [15.0, 15.0])


def test_deletion_uses_adjacent_residues():
    # The deleted residue has no embedding of its own, so its neighbours stand
    # in: the wildtype averages over what flanks the gap.
    task = _make_task(parse_mutations("A:S11-"))
    wt, mt = task.embeddings(["feat"])
    np.testing.assert_array_equal(wt, [2.0, 2.0])   # only T_A_12 follows
    np.testing.assert_array_equal(mt, [10.0, 10.0])


def test_invalid_protein_aggregation_raises():
    task = _make_task(parse_mutations("A:S11A"))
    with pytest.raises(ValueError, match="protein_aggregation"):
        task.embeddings(["feat"], protein_aggregation="bogus")


def test_invalid_multi_aggregation_raises():
    task = _make_task(parse_mutations("A:S11A"))
    with pytest.raises(ValueError, match="multi_aggregation"):
        task.embeddings(["feat"], multi_aggregation="bogus")


def test_mutpos_without_mutations_explains_itself():
    task = _make_task(())
    with pytest.raises(ValueError, match="needs at least one mutation"):
        task.embeddings(["feat"], protein_aggregation="mutpos")


def test_missing_feature_names_the_residue():
    with pytest.raises(KeyError, match="missing feature"):
        aggregate(WT_FEATURES, POSITIONS, MT_FEATURES, POSITIONS,
                  feature_names=["absent"], mutations=parse_mutations("A:S11A"))


def test_mutation_outside_observed_positions_is_reported():
    task = _make_task(parse_mutations("A:Q99E"))
    with pytest.raises(KeyError, match="not among the observed positions"):
        task.embeddings(["feat"])
