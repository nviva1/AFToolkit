"""Tests for backwards-compatibility of legacy `AFToolKit` module paths."""

import importlib
import pickle

import pytest

import aftoolkit
from aftoolkit import Mutation, ProteinTask


def test_legacy_module_alias_is_same_object():
    new = importlib.import_module("aftoolkit.tasks.protein")
    old = importlib.import_module("AFToolKit.tasks.protein")
    assert old is new
    assert old.ProteinTask is new.ProteinTask


def test_legacy_models_alias():
    old = importlib.import_module("AFToolKit.models.adapter")
    new = importlib.import_module("aftoolkit.models.adapter")
    assert old is new


def test_legacy_pickle_roundtrip():
    """A pickle whose class path is the old `AFToolKit.*` name must load.

    The adapter models published for download were pickled under that name, so
    the alias has to keep resolving.
    """
    task = ProteinTask()
    task.set_mutations("A:S11A")

    # Simulate an artifact pickled in the capital-K era.
    original_module = ProteinTask.__module__
    try:
        ProteinTask.__module__ = "AFToolKit.tasks.protein"
        data = pickle.dumps(task)
    finally:
        ProteinTask.__module__ = original_module

    assert b"AFToolKit" in data
    loaded = pickle.loads(data)
    assert isinstance(loaded, ProteinTask)
    assert loaded.mutations == (Mutation(chain="A", position=11, wt="S", mt="A"),)


def test_version_exposed():
    assert isinstance(aftoolkit.__version__, str)


# --------------------------------------------------------------------------- #
# Pre-2.0 pickles: moved modules and the old attribute layout
# --------------------------------------------------------------------------- #
import pandas as pd  # noqa: E402

from aftoolkit._compat import (  # noqa: E402
    LEGACY_MODULES,
    migrate_protein_task_state,
)


def _legacy_task_state():
    """A ProteinTask.__dict__ as written by the pre-2.0 code."""
    return {
        "task": {
            "input_protein": {"path": "p.pdb", "chains": ["A"], "regions": None},
            "mutants": {("S", 11, "A"): "A"},
            "obs_positions": [{"resi": 11, "shift_from_resi": 2, "chain_id": "A"}],
        },
        "protein_job": {
            "protein_wt": {"protein": pd.DataFrame({"x": [1]}), "obs_positions": {"S_A_11": 10}},
            "protein_mt": {"protein": pd.DataFrame({"x": [2]}), "obs_positions": {"S_A_11": 10}},
        },
        "protein_of": {
            "protein_wt": {"protein": pd.DataFrame({"y": [1]}),
                           "features": {"S_A_11": {"plddt": [90.0]}}, "cycles": {}},
            "protein_mt": {"protein": pd.DataFrame({"y": [2]}),
                           "features": {"S_A_11": {"plddt": [85.0]}}, "cycles": {}},
        },
    }


@pytest.mark.parametrize("old,new", sorted(LEGACY_MODULES.items()))
def test_moved_modules_still_import(old, new):
    """Pickles name the module their class came from; the old paths must resolve."""
    assert importlib.import_module(old) is importlib.import_module(new)


def test_migrate_legacy_protein_task_state():
    migrated = migrate_protein_task_state(_legacy_task_state())

    assert migrated["mutations"] == (Mutation(chain="A", position=11, wt="S", mt="A"),)
    assert migrated["input"].path == "p.pdb"
    assert migrated["input"].chains == ["A"]
    region, = migrated["observables"]
    assert (region.chain, region.start, region.span) == ("A", 11, 2)
    assert migrated["wildtype"].observed == {"S_A_11": 10}
    assert migrated["wildtype"].features == {"S_A_11": {"plddt": [90.0]}}
    assert migrated["mutant"].features == {"S_A_11": {"plddt": [85.0]}}
    assert migrated["wildtype"].cycles == []        # {} became a list


def test_migrate_is_a_noop_on_current_state():
    current = {"mutations": (), "observables": ()}
    assert migrate_protein_task_state(current) is current


def test_unpickling_a_legacy_task_yields_a_working_object():
    """The whole point: an old feature file must load and still give embeddings."""
    task = ProteinTask.__new__(ProteinTask)
    task.__setstate__(_legacy_task_state())

    assert task.mutations == (Mutation(chain="A", position=11, wt="S", mt="A"),)
    wt, mt = task.embeddings(["plddt"])
    assert wt.tolist() == [90.0] and mt.tolist() == [85.0]
