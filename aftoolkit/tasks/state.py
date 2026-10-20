"""State held by a task.

The wildtype and the mutant go through exactly the same steps -- build a
structure, locate the residues of interest, fold, extract features. Giving them
one shared :class:`ProteinState` type is what lets the task run that pipeline
once with a parameter instead of twice in parallel branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from aftoolkit.structure import DEFAULT_CHAIN_GAP

__all__ = ["InputProtein", "ProteinState"]


@dataclass
class InputProtein:
    """Where a task's structure comes from, and how to read it.

    Attributes:
        path: path to the input PDB file.
        chains: chain ids to keep; ``None`` keeps all of them.
        regions: ``{chain: [(start, end), ...]}`` residue ranges to keep.
        chain_gap: residue-index gap inserted between chains when they are
            collapsed onto one chain for AlphaFold.
    """

    path: str | None = None
    chains: list | None = None
    regions: dict | None = None
    chain_gap: int = DEFAULT_CHAIN_GAP


@dataclass
class ProteinState:
    """One structure at every stage of the pipeline.

    Attributes:
        structure: the input structure as a protein dataframe, with mutations
            applied for the mutant.
        observed: ``{residue_key: internal_index}`` for the residues features are
            extracted at. Built from the task's observable regions, resolved
            against *this* structure -- which is why the wildtype and mutant need
            separate copies: a deletion shifts the mutant's indices.
        folded: the structure AlphaFold predicted.
        features: ``{residue_key: {feature_name: vector}}``.
        cycles: per-recycle copies of ``features``, when the wrapper was asked
            for them.
    """

    structure: pd.DataFrame = field(default_factory=pd.DataFrame)
    observed: dict = field(default_factory=dict)
    folded: pd.DataFrame = field(default_factory=pd.DataFrame)
    features: dict = field(default_factory=dict)
    cycles: list = field(default_factory=list)

    @property
    def is_built(self) -> bool:
        """True once the input structure has been loaded."""
        return not self.structure.empty

    @property
    def is_folded(self) -> bool:
        """True once AlphaFold features have been extracted."""
        return bool(self.features)
