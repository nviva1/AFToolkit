"""End-to-end OpenFold inference tests (monomer + complex).

These are marked ``inference`` and require the optional torch/OpenFold stack
(``pip install .[openfold]``) plus the AlphaFold weights
(``weights/params_model_2_ptm.npz``). They are skipped automatically when either
is unavailable, so the default/CI suite stays torch-free.

Run them explicitly with::

    pytest -m inference
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.inference

# Skip the whole module unless the inference stack is importable.
pytest.importorskip("torch")
pytest.importorskip("ml_collections")
pytest.importorskip("tree")

from aftoolkit import (  # noqa: E402  (after importorskip)
    OpenFoldWrapper,
    ProteinComplexTask,
    ProteinTask,
    parse_positions,
)
from aftoolkit.constants import MODEL_WEIGHTS_DIR, OF_WEIGHTS  # noqa: E402

_WEIGHTS = os.path.join(MODEL_WEIGHTS_DIR, OF_WEIGHTS)


@pytest.fixture(scope="module")
def wrapper():
    if not os.path.exists(_WEIGHTS):
        pytest.skip(f"AlphaFold weights not found at {_WEIGHTS}")
    return OpenFoldWrapper(device="mps", inference_n_recycle=1).init_model()


@pytest.fixture(scope="module")
def cycling_wrapper():
    if not os.path.exists(_WEIGHTS):
        pytest.skip(f"AlphaFold weights not found at {_WEIGHTS}")
    return OpenFoldWrapper(device="mps", inference_n_recycle=1,
                           return_all_cycles=True).init_model()


def _check_features(feats):
    assert feats, "no per-residue features extracted"
    first = next(iter(feats))
    assert np.asarray(feats[first]["single"]).shape == (384,)
    assert np.asarray(feats[first]["pair"]).shape == (128,)
    assert np.asarray(feats[first]["lddt_logits"]).shape == (50,)


def test_monomer_inference(wrapper, monomer_pdb):
    task = ProteinTask()
    task.set_input(monomer_pdb, chains=["A"])
    task.set_mutations("A:S11A")
    task.set_observables(parse_positions("A:11"))
    task.evaluate(wrapper, store_structure=True)

    assert not task.wildtype.folded.empty
    _check_features(task.wildtype.features)
    _check_features(task.mutant.features)

    wt, mt = task.embeddings(["pair", "lddt_logits", "plddt"])
    assert wt.shape == mt.shape == (179,)  # 128 + 50 + 1


def test_per_cycle_features_are_populated(cycling_wrapper, monomer_pdb):
    """Per-recycle features come back for both states against the real model.

    That each state uses its *own* observed indices -- the bug that the shared
    evaluate path removes -- is pinned down in ``test_protein_task.py``, which
    can assert on the exact indices without running inference.
    """
    task = ProteinTask()
    task.set_input(monomer_pdb, chains=["A"])
    task.set_mutations("A:S11-")
    task.set_observables(parse_positions("A:12"))
    task.evaluate(cycling_wrapper, store_structure=True)

    assert task.wildtype.cycles and task.mutant.cycles
    _check_features(task.mutant.cycles[0])
    assert set(task.mutant.cycles[0]) == set(task.mutant.observed)


def test_complex_inference(wrapper, complex_pdb):
    task = ProteinComplexTask()
    task.set_input(complex_pdb, protein_chains=["I"], complex_chains=["E", "I"])
    task.set_mutations("I:L38S")
    task.set_observables(parse_positions("I:38"))
    task.evaluate(wrapper, store_structure=True)

    _check_features(task.protein.wildtype.features)
    _check_features(task.complex.wildtype.features)
    _check_features(task.complex.mutant.features)
