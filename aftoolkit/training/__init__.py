"""Training adapters on precomputed AlphaFold features.

Previously two near-identical module-level scripts (``train_adapter_monomer.py``
and ``train_adapter_multimer.py``, the latter carrying its own verbatim copy of
``get_model_suffix``) plus four ``run_train_*.py`` runners that differed only in
which scikit-learn estimator they constructed. All of that is one function here,
parameterised by ``target`` and ``model``.
"""

from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Sequence

import numpy as np
import pandas as pd

from aftoolkit.models.adapter import AdapterMonomer, AdapterMultimer
from aftoolkit.models.utils import get_model_suffix

__all__ = ["AGGREGATION_GRID", "ESTIMATORS", "build_estimator", "train_adapter"]

logger = logging.getLogger(__name__)

ESTIMATORS = ("svm", "mlp", "catboost")

DEFAULT_FEATURES = ("pair", "lddt_logits", "plddt")
DEFAULT_MUT_TYPES = ("ss", "ins", "del")

#: The five aggregation settings the published models were swept over.
AGGREGATION_GRID = (
    {"mut_types": ("ss", "ins", "del"), "protein_aggregation": "mutpos", "multi_aggregation": "sum"},
    {"mut_types": ("ss", "multi", "ins", "del"), "protein_aggregation": "mutpos", "multi_aggregation": "sum"},
    {"mut_types": ("ss", "multi", "ins", "del"), "protein_aggregation": "mutpos", "multi_aggregation": "mean"},
    {"mut_types": ("ss", "multi", "ins", "del"), "protein_aggregation": "mean", "multi_aggregation": "sum"},
    {"mut_types": ("ss", "multi", "ins", "del"), "protein_aggregation": "sum", "multi_aggregation": "sum"},
)


def build_estimator(name: str, random_seed: int = 42):
    """Construct the scikit-learn estimator behind an adapter.

    Raises:
        ValueError: if ``name`` is not one of :data:`ESTIMATORS`.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "svm":
        from sklearn.svm import SVR

        return Pipeline([("scaler", StandardScaler()), ("svm", SVR(cache_size=2000))])
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor

        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=500,
                                 early_stopping=True, random_state=random_seed)),
        ])
    if name == "catboost":
        from catboost import CatBoostRegressor

        # allow_writing_files=False keeps CatBoost from dropping a
        # `catboost_info/` directory into whatever the working directory is.
        return CatBoostRegressor(verbose=0, random_seed=random_seed,
                                 allow_writing_files=False)

    raise ValueError(f"Unknown estimator {name!r}; choose from {list(ESTIMATORS)}.")


def train_adapter(
    model: str,
    train_csv: str,
    features_dir: str,
    target: str = "monomer",
    npy_dir: str = "./",
    out_dir: str = "data/models",
    features: Sequence[str] = DEFAULT_FEATURES,
    concat_features: bool = True,
    mut_types: Sequence[str] = DEFAULT_MUT_TYPES,
    protein_aggregation: str = "mutpos",
    multi_aggregation: str = "sum",
    add_reverse: bool = True,
    shuffle_train: bool = True,
    only_complex_features: bool = True,
    estimator=None,
) -> str:
    """Train one adapter and save it.

    Args:
        model: estimator name, one of :data:`ESTIMATORS`; ignored when
            ``estimator`` is given.
        train_csv: CSV of mutations with a ``split`` column of train/test, a
            ``mut_type`` column, an ``id`` matching the feature files, and ``ddg``.
        features_dir: directory of per-mutation ``ProteinTask`` ``.pkl`` files.
        target: ``"monomer"`` (stability) or ``"multimer"`` (affinity).
        npy_dir: where assembled ``.npy`` datasets are cached and reused.
        out_dir: root directory to write the trained adapter into.
        features: AlphaFold feature names forming each residue embedding.
        concat_features: concatenate the wildtype and mutant embeddings rather
            than taking their difference.
        mut_types: mutation types to train on.
        protein_aggregation / multi_aggregation: see :mod:`aftoolkit.embeddings`.
        add_reverse: augment with reverse mutations.
        shuffle_train: shuffle before fitting.
        only_complex_features: multimer only -- use the bound form alone.
        estimator: a prebuilt scikit-learn estimator, overriding ``model``.

    Returns:
        Path to the pickled adapter.

    Raises:
        ValueError: if ``target`` is unknown or ``train_csv`` lacks a column.
    """
    if target not in ("monomer", "multimer"):
        raise ValueError(f"Unknown target {target!r}; expected 'monomer' or 'multimer'.")

    frame = pd.read_csv(train_csv, index_col=0)
    for column in ("split", "mut_type"):
        if column not in frame.columns:
            raise ValueError(f"{train_csv} has no {column!r} column.")

    trains_on_multi = "multi" in mut_types
    adapter_class = AdapterMonomer if target == "monomer" else AdapterMultimer
    extra = {} if target == "monomer" else {"only_complex_features": only_complex_features}
    adapter = adapter_class(
        features_list=list(features),
        base_model=estimator if estimator is not None else build_estimator(model),
        concat_features=concat_features,
        protein_aggregation=protein_aggregation,
        multi_aggregation=multi_aggregation,
        multi_as_singlessum=not trains_on_multi,
        **extra,
    )

    suffix = get_model_suffix(adapter, trains_on_multi)
    logger.info("Training %s %s adapter (%s)", model, target, suffix)

    splits = _load_or_build_datasets(adapter, frame, mut_types, npy_dir, features_dir,
                                     suffix)
    (train_X, train_Y), (test_X, test_Y) = splits

    adapter.train(train_X, train_Y, add_reverse=add_reverse,
                  shuffle_train=shuffle_train)
    for split_name, (X, Y) in (("Train", (train_X, train_Y)),
                               ("Test", (test_X, test_Y))):
        _, correlation = adapter.test(X, Y)
        logger.info("%s set Spearman correlation: %.3f", split_name, correlation)

    out_path = os.path.join(out_dir, target, "+".join(features),
                            f"trained_{model}{suffix}.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(adapter, f)
    logger.info("Saved adapter -> %s", out_path)
    return out_path


def _load_or_build_datasets(adapter, frame, mut_types, npy_dir, features_dir, suffix):
    """Load cached ``.npy`` splits, or build them from the feature pickles.

    Building is the expensive step (it unpickles one task per mutation), so the
    result is cached under ``npy_dir`` keyed by the model suffix.
    """
    os.makedirs(npy_dir, exist_ok=True)
    paths = {
        name: os.path.join(npy_dir, f"{name}{suffix}.npy")
        for name in ("train_X", "train_y", "test_X", "test_y")
    }

    if all(os.path.exists(p) for p in paths.values()):
        logger.info("Loading cached datasets from %s", npy_dir)
        # Targets are stored as a column; estimators want a flat vector.
        return (
            (np.load(paths["train_X"]), np.load(paths["train_y"]).ravel()),
            (np.load(paths["test_X"]), np.load(paths["test_y"]).ravel()),
        )

    if features_dir is None:
        raise ValueError(
            f"No cached datasets in {npy_dir} and no features_dir to build them from."
        )

    built = {}
    for split in ("train", "test"):
        rows = frame[(frame["split"] == split) & frame["mut_type"].isin(mut_types)]
        logger.info("Building %s dataset from %d mutations", split, len(rows))
        X, Y = adapter.create_npy_dataset(rows, features_dir)
        np.save(paths[f"{split}_X"], X)
        np.save(paths[f"{split}_y"], Y)
        built[split] = (X, Y.ravel())

    return built["train"], built["test"]
