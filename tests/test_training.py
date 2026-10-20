"""Tests for adapters and the training pipeline, on synthetic features.

No inference and no PDB: tasks are built with their feature dicts filled in
directly, so the whole train -> save -> predict path runs in the fast suite.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from aftoolkit import AdapterMonomer, ProteinTask
from aftoolkit.training import build_estimator, train_adapter

FEATURES = ["feat"]


def _task(seed: int, mutations: str) -> ProteinTask:
    """A task carrying synthetic per-residue features for each mutated site."""
    task = ProteinTask().set_mutations(mutations)
    rng = np.random.default_rng(seed)

    wt, mt, observed = {}, {}, {}
    for index, mutation in enumerate(task.mutations):
        base = rng.normal(size=4)
        wt[mutation.key] = {"feat": base}
        mt[mutation.key] = {"feat": base + rng.normal(scale=0.5, size=4)}
        observed[mutation.key] = index

    task.wildtype.features, task.mutant.features = wt, mt
    task.wildtype.observed = task.mutant.observed = observed
    return task


@pytest.fixture
def dataset(tmp_path):
    """A small train/test split of pickled tasks plus its CSV."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    rng = np.random.default_rng(0)

    rows = []
    for i in range(24):
        ident = f"m{i}"
        _task(i, f"A:S{11 + i}A").save(features_dir / f"{ident}.pkl")
        rows.append({"id": ident, "ddg": float(rng.normal()),
                     "split": "train" if i < 18 else "test", "mut_type": "ss"})

    csv = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv)
    return csv, features_dir


@pytest.mark.parametrize("model", ["svm", "mlp", "catboost"])
def test_train_adapter_round_trip(dataset, tmp_path, model):
    csv, features_dir = dataset
    out = train_adapter(
        model=model, target="monomer", train_csv=str(csv),
        features_dir=str(features_dir), npy_dir=str(tmp_path / f"npy_{model}"),
        out_dir=str(tmp_path / "models"), features=FEATURES, mut_types=["ss"],
    )

    with open(out, "rb") as f:
        adapter = pickle.load(f)
    prediction = adapter.predict(_task(999, "A:S11A"))
    assert np.isfinite(np.ravel(prediction)).all()


def test_trained_filename_matches_published_convention(dataset, tmp_path):
    csv, features_dir = dataset
    out = train_adapter(
        model="svm", target="monomer", train_csv=str(csv),
        features_dir=str(features_dir), npy_dir=str(tmp_path / "npy"),
        out_dir=str(tmp_path / "models"), features=FEATURES, mut_types=["ss"],
    )
    assert out.endswith("trained_svm_concat_nomultitrain_aggmutpos_multisum.pkl")


def test_cached_datasets_are_reused(dataset, tmp_path):
    """The second run needs no features_dir, because the .npy cache is reused."""
    csv, features_dir = dataset
    npy = tmp_path / "npy"
    train_adapter(model="svm", target="monomer", train_csv=str(csv),
                  features_dir=str(features_dir), npy_dir=str(npy),
                  out_dir=str(tmp_path / "m1"), features=FEATURES, mut_types=["ss"])

    assert len(list(npy.glob("*.npy"))) == 4
    train_adapter(model="svm", target="monomer", train_csv=str(csv),
                  features_dir=None, npy_dir=str(npy),
                  out_dir=str(tmp_path / "m2"), features=FEATURES, mut_types=["ss"])


def test_missing_cache_without_features_dir_is_reported(dataset, tmp_path):
    csv, _ = dataset
    with pytest.raises(ValueError, match="no features_dir"):
        train_adapter(model="svm", target="monomer", train_csv=str(csv),
                      features_dir=None, npy_dir=str(tmp_path / "empty"),
                      out_dir=str(tmp_path / "m"), features=FEATURES, mut_types=["ss"])


def test_unknown_target_is_rejected(dataset, tmp_path):
    csv, features_dir = dataset
    with pytest.raises(ValueError, match="Unknown target"):
        train_adapter(model="svm", target="trimer", train_csv=str(csv),
                      features_dir=str(features_dir), npy_dir=str(tmp_path / "n"),
                      out_dir=str(tmp_path / "m"), features=FEATURES)


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="Unknown estimator"):
        build_estimator("random-forest")


def test_predict_as_singles_restores_the_tasks_mutations():
    """Regression: scoring a multi-mutant one site at a time overwrote the
    task's mutations with the last single mutation and never put them back, so
    the caller's task was silently left holding one mutation."""
    class _Echo:
        def predict(self, X):
            return np.array([float(np.sum(X))])

    adapter = AdapterMonomer(features_list=FEATURES, base_model=_Echo(),
                             multi_as_singlessum=True)
    task = _task(1, "A:S11A, A:T12V")
    before = task.mutations

    adapter.predict(task)
    assert task.mutations == before


def test_predict_as_singles_sums_the_sites():
    class _One:
        def predict(self, X):
            return np.array([1.0])

    adapter = AdapterMonomer(features_list=FEATURES, base_model=_One(),
                             multi_as_singlessum=True)
    total = adapter.predict(_task(1, "A:S11A, A:T12V, A:L13I"))
    assert float(np.ravel(total)[0]) == pytest.approx(3.0)
