"""Tests for backwards-compatibility of legacy `AFToolKit` pickle paths."""

import importlib
import pickle

import aftoolkit
from aftoolkit import ProteinTask, parse_mutations


def test_legacy_module_alias_is_same_object():
    new = importlib.import_module("aftoolkit.processing.protein_task")
    old = importlib.import_module("AFToolKit.processing.protein_task")
    assert old is new
    assert old.ProteinTask is new.ProteinTask


def test_legacy_models_alias():
    old = importlib.import_module("AFToolKit.models.adapter")
    new = importlib.import_module("aftoolkit.models.adapter")
    assert old is new


def test_legacy_pickle_roundtrip():
    """A pickle whose class path is the old `AFToolKit.*` name must load."""
    task = ProteinTask()
    task.set_task_mutants(parse_mutations("A:S11A"))

    # Simulate an artifact pickled in the capital-K era.
    original_module = ProteinTask.__module__
    try:
        ProteinTask.__module__ = "AFToolKit.processing.protein_task"
        data = pickle.dumps(task)
    finally:
        ProteinTask.__module__ = original_module

    assert b"AFToolKit" in data
    loaded = pickle.loads(data)
    assert isinstance(loaded, ProteinTask)
    assert loaded.get_mutations() == {("S", 11, "A"): "A"}


def test_version_exposed():
    assert isinstance(aftoolkit.__version__, str)
