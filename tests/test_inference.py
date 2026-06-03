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
    ProteinTask,
    ProteinComplexTask,
    parse_mutations,
    parse_positions,
)
from aftoolkit.constants import MODEL_WEIGHTS_DIR, OF_WEIGHTS  # noqa: E402

_WEIGHTS = os.path.join(MODEL_WEIGHTS_DIR, OF_WEIGHTS)
_FEATURE_KEYS = ["single", "pair", "lddt_logits", "plddt"]


@pytest.fixture(scope="module")
def wrapper():
    if not os.path.exists(_WEIGHTS):
        pytest.skip(f"AlphaFold weights not found at {_WEIGHTS}")
    w = OpenFoldWrapper(device="mps", inference_n_recycle=1, return_all_cycles=False)
    w.init_model()
    return w


def _check_features(feats):
    assert feats, "no per-residue features extracted"
    first = next(iter(feats))
    assert np.asarray(feats[first]["single"]).shape == (384,)
    assert np.asarray(feats[first]["pair"]).shape == (128,)
    assert np.asarray(feats[first]["lddt_logits"]).shape == (50,)


def test_monomer_inference(wrapper, monomer_pdb):
    task = ProteinTask()
    task.set_input_protein_task(monomer_pdb, chains=["A"])
    task.set_task_mutants(parse_mutations("A:S11A"))
    task.set_observable_positions(parse_positions("A:11"))
    task.evaluate(of_wrapper=wrapper, store_of_protein=True)

    pred = task.get_wildtype_protein_of()
    assert pred is not None and not pred.empty
    _check_features(task.get_wildtype_protein_of_features())
    _check_features(task.get_mutate_protein_of_features())

    wt, mt = task.get_protein_embeddings(["pair", "lddt_logits", "plddt"])
    assert wt.shape == mt.shape == (179,)  # 128 + 50 + 1


def test_complex_inference(wrapper, complex_pdb):
    task = ProteinComplexTask()
    task.set_input_protein_task(
        complex_pdb, chains_for_protein=["I"], chains_for_complex=["E", "I"]
    )
    task.set_task_mutants(parse_mutations("I:L38S"))
    task.set_observable_positions(parse_positions("I:38"))
    task.set_wildtype()
    task.set_mutate_protein()
    task.evaluate(of_wrapper=wrapper, store_of_protein=True)

    _check_features(task.get_wildtype_protein_of_features())
    _check_features(task.get_wildtype_complex_of_features())
    _check_features(task.get_mutate_complex_of_features())
