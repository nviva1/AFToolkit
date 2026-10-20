"""Aggregating per-residue AlphaFold features into a protein embedding.

An adapter consumes one fixed-length vector per structure, but AlphaFold gives
per-residue features. This module turns the latter into the former.

Two choices stack:

``protein_aggregation``
    ``"sum"`` / ``"mean"`` reduce over every observed residue. ``"mutpos"``
    instead looks only where the mutation is -- which residues count depends on
    the mutation kind, since a deletion or insertion has no single "its own"
    residue and is represented by its neighbours.

``multi_aggregation``
    How several mutated sites combine, when ``protein_aggregation="mutpos"``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from aftoolkit.mutations import Mutation, residue_key

__all__ = [
    "MULTI_AGGREGATIONS",
    "PROTEIN_AGGREGATIONS",
    "aggregate",
    "feature_matrix",
]

PROTEIN_AGGREGATIONS = ("mutpos", "sum", "mean")
MULTI_AGGREGATIONS = ("sum", "mean")

_REDUCERS = {
    "sum": lambda a: a.sum(0),
    "mean": lambda a: a.mean(0),
}


def feature_matrix(features: dict, positions: dict,
                   feature_names: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Stack selected per-residue features into one ``(n_residues, n_features)`` array.

    Args:
        features: ``{residue_key: {feature_name: vector}}``.
        positions: ``{residue_key: internal_index}``; its values order the rows.
        feature_names: which features to concatenate per residue.

    Returns:
        ``(ordered_keys, matrix)`` with rows in increasing internal index, so the
        row order matches the residue order of the structure.

    Raises:
        KeyError: if a residue or feature is missing, naming what was missing.
    """
    ordered = sorted(positions, key=lambda key: positions[key])

    rows = []
    for key in ordered:
        if key not in features:
            raise KeyError(
                f"No AlphaFold features for residue {key!r}; "
                f"has the structure been evaluated?"
            )
        residue = features[key]
        missing = [name for name in feature_names if name not in residue]
        if missing:
            raise KeyError(
                f"Residue {key!r} is missing feature(s) {missing}; "
                f"available: {sorted(residue)}."
            )
        rows.append(np.concatenate([residue[name] for name in feature_names]))

    return ordered, np.array(rows)


def _index_of(key: str, ordered: Sequence[str], mutation: Mutation) -> int:
    try:
        return ordered.index(key)
    except ValueError:
        raise KeyError(
            f"Mutation {mutation} refers to residue {key!r}, which is not among "
            f"the observed positions. Observed: {list(ordered)[:8]}"
            f"{' ...' if len(ordered) > 8 else ''}"
        ) from None


def _substitution_indices(mutation, ordered, _resolve_wt_aa):
    """A substitution is represented by its own residue, in both structures."""
    idx = _index_of(mutation.key, ordered, mutation)
    return [idx], [idx]


def _deletion_indices(mutation, ordered, _resolve_wt_aa):
    """A deleted residue is represented by the residues flanking the gap.

    The mutant is one residue shorter, so after the deletion point its indices
    are shifted back by one relative to the wildtype.
    """
    idx = _index_of(mutation.key, ordered, mutation)
    wt_idxs, mt_idxs = [], []
    if idx != 0:
        wt_idxs.append(idx - 1)
        mt_idxs.append(idx - 1)
    if idx != len(ordered) - 1:
        wt_idxs.append(idx + 1)
        mt_idxs.append(idx)
    if not wt_idxs:
        raise ValueError(
            f"Cannot represent deletion {mutation}: the structure has no other residue."
        )
    return wt_idxs, mt_idxs


def _insertion_indices(mutation, ordered, resolve_wt_aa):
    """An insertion is represented by the residues flanking the inserted stretch.

    ``mutation.wt`` is the gap marker, so the anchor residue's identity has to be
    read back from the structure to build its key.
    """
    anchor = residue_key(resolve_wt_aa(mutation.chain, mutation.position),
                         mutation.chain, mutation.position)
    idx = _index_of(anchor, ordered, mutation)
    wt_idxs, mt_idxs = [idx], [idx]
    if idx != len(ordered) - 1:
        wt_idxs.append(idx + 1)
        mt_idxs.append(idx + 2)
    return wt_idxs, mt_idxs


def _mutation_indices(mutation: Mutation, ordered: Sequence[str],
                      resolve_wt_aa: Callable[[str, int], str]):
    if mutation.is_deletion:
        return _deletion_indices(mutation, ordered, resolve_wt_aa)
    if mutation.is_insertion:
        return _insertion_indices(mutation, ordered, resolve_wt_aa)
    return _substitution_indices(mutation, ordered, resolve_wt_aa)


def _at_mutations(wt_matrix, mt_matrix, ordered, mutations, multi_aggregation,
                  resolve_wt_aa):
    """Reduce to the mutated sites only."""
    reduce_sites = _REDUCERS[multi_aggregation]

    if len(mutations) == 1:
        mutation = mutations[0]
        wt_idxs, mt_idxs = _mutation_indices(mutation, ordered, resolve_wt_aa)
        # A lone substitution picks a single residue outright; a deletion or
        # insertion always averages the residues standing in for it.
        reduce_one = _REDUCERS["mean"]
        if mutation.is_substitution:
            return wt_matrix[wt_idxs[0]], mt_matrix[mt_idxs[0]]
        return reduce_one(wt_matrix[wt_idxs]), reduce_one(mt_matrix[mt_idxs])

    # Several mutations: each contributes its own residue, then they combine.
    # TODO: indels among multiple mutations shift the mutant indices; only
    # substitutions are handled correctly here.
    wt_idxs, mt_idxs = [], []
    for mutation in mutations:
        wt_i, mt_i = _mutation_indices(mutation, ordered, resolve_wt_aa)
        wt_idxs.extend(wt_i)
        mt_idxs.extend(mt_i)
    return reduce_sites(wt_matrix[wt_idxs]), reduce_sites(mt_matrix[mt_idxs])


def aggregate(
    wt_features: dict,
    wt_positions: dict,
    mt_features: dict,
    mt_positions: dict,
    feature_names: Sequence[str],
    mutations: Sequence[Mutation],
    protein_aggregation: str = "mutpos",
    multi_aggregation: str = "sum",
    resolve_wt_aa: Callable[[str, int], str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate wildtype and mutant per-residue features into two vectors.

    Args:
        wt_features / mt_features: ``{residue_key: {feature_name: vector}}``.
        wt_positions / mt_positions: ``{residue_key: internal_index}``.
        feature_names: features to concatenate per residue.
        mutations: the mutations relating the two structures.
        protein_aggregation: one of :data:`PROTEIN_AGGREGATIONS`.
        multi_aggregation: one of :data:`MULTI_AGGREGATIONS`; only consulted for
            ``protein_aggregation="mutpos"``.
        resolve_wt_aa: ``(chain, position) -> one-letter code``, required only
            when an insertion is present.

    Returns:
        ``(wildtype_vector, mutant_vector)``.

    Raises:
        ValueError: on an unknown aggregation, or ``"mutpos"`` without mutations.
    """
    if protein_aggregation not in PROTEIN_AGGREGATIONS:
        raise ValueError(
            f"Expected 'protein_aggregation' to be one of {list(PROTEIN_AGGREGATIONS)}, "
            f"got {protein_aggregation!r} instead."
        )
    if multi_aggregation not in MULTI_AGGREGATIONS:
        raise ValueError(
            f"Expected 'multi_aggregation' to be one of {list(MULTI_AGGREGATIONS)}, "
            f"got {multi_aggregation!r} instead."
        )

    ordered, wt_matrix = feature_matrix(wt_features, wt_positions, feature_names)
    _, mt_matrix = feature_matrix(mt_features, mt_positions, feature_names)

    if protein_aggregation != "mutpos":
        reduce_all = _REDUCERS[protein_aggregation]
        return reduce_all(wt_matrix), reduce_all(mt_matrix)

    if not mutations:
        raise ValueError(
            "protein_aggregation='mutpos' needs at least one mutation to locate; "
            "use 'mean' or 'sum' for an unmutated structure."
        )

    def _unavailable(chain, position):
        raise ValueError(
            f"An insertion at {chain}:{position} needs the wildtype residue, but "
            f"no 'resolve_wt_aa' was supplied."
        )

    return _at_mutations(
        wt_matrix, mt_matrix, ordered, list(mutations), multi_aggregation,
        resolve_wt_aa or _unavailable,
    )
