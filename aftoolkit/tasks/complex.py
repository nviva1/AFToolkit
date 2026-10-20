"""A two-sided complex task: the same mutations scored bound and unbound.

Binding affinity is a difference between two contexts, so this holds two
:class:`~aftoolkit.tasks.protein.ProteinTask`\\ s over one structure:

``protein``
    only the mutated side's chains -- the unbound form.
``complex``
    every chain -- the bound form.

Both are ordinary tasks, so callers reach straight through (``task.complex.folded``,
``task.protein.embeddings(...)``) rather than through forwarding methods. This
class exists only to keep the pair in step: routing each mutation and observable
to the side it belongs on, and driving them together.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Sequence

from aftoolkit.mutations import Mutation, ObservableRegion, parse_mutations
from aftoolkit.tasks.protein import ProteinTask

__all__ = ["ProteinComplexTask"]


class ProteinComplexTask:
    """The unbound and bound forms of one mutated structure.

    Attributes:
        protein: the unbound task (mutated chains only).
        complex: the bound task (all chains).
        path: the shared input structure.
        protein_chains / complex_chains: the chain selection for each side.
    """

    def __init__(self, path=None, protein_chains=None, complex_chains=None,
                 chain_gap: int | None = None):
        self.protein = ProteinTask()
        self.complex = ProteinTask()
        self.path = None
        self.protein_chains: list = []
        self.complex_chains: list | None = None
        if path is not None or protein_chains is not None:
            self.set_input(path, protein_chains, complex_chains, chain_gap)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(path={self.path!r}, "
                f"protein_chains={self.protein_chains}, "
                f"complex_chains={self.complex_chains}, "
                f"mutations={','.join(str(m) for m in self.mutations) or 'none'})")

    @property
    def mutations(self) -> tuple[Mutation, ...]:
        """Every mutation, across both sides."""
        return self.complex.mutations

    # -- setup ------------------------------------------------------------- #

    def set_input(self, path=None, protein_chains=None, complex_chains=None,
                  chain_gap: int | None = None) -> ProteinComplexTask:
        """Point both sides at one structure.

        Args:
            path: the complex PDB file.
            protein_chains: chains of the side carrying the mutations.
            complex_chains: all chains to fold as the bound form; ``None`` keeps
                every chain in the file.
        """
        self.path = path
        self.protein_chains = list(protein_chains) if protein_chains else []
        self.complex_chains = list(complex_chains) if complex_chains else None

        self.protein.set_input(path, chains=self.protein_chains,
                               chain_gap=chain_gap)
        self.complex.set_input(path, chains=self.complex_chains,
                               chain_gap=chain_gap)
        return self

    def _on_protein_side(self, chain) -> bool:
        return chain in self.protein_chains

    def set_mutations(self, mutations) -> ProteinComplexTask:
        """Set the mutations, giving the unbound side only the ones it contains."""
        mutations = parse_mutations(mutations)
        self.complex.set_mutations(mutations)
        self.protein.set_mutations(
            [m for m in mutations if self._on_protein_side(m.chain)]
        )
        return self

    def add_mutation(self, mutation=None, **fields) -> ProteinComplexTask:
        """Add one mutation to the complex, and to the protein if it applies."""
        if mutation is None:
            mutation = Mutation(**fields)
        return self.set_mutations([*self.mutations, mutation])

    def set_observables(self, regions: Iterable[ObservableRegion] | None = None
                        ) -> ProteinComplexTask:
        """Set observable regions, routing each to the sides that contain it."""
        if regions is None:
            self.protein.set_observables(None)
            self.complex.set_observables(None)
            return self

        regions = tuple(regions)
        self.complex.set_observables(regions)
        self.protein.set_observables(
            [r for r in regions if self._on_protein_side(r.chain)]
        )
        return self

    def add_observable(self, chain, start, span=0, name=None) -> ProteinComplexTask:
        """Add one observable region to the sides that contain its chain."""
        self.complex.add_observable(chain, start, span, name)
        if self._on_protein_side(chain):
            self.protein.add_observable(chain, start, span, name)
        return self

    # -- structures and inference ------------------------------------------ #

    def build(self) -> ProteinComplexTask:
        """Build the wildtype and mutant structures for both sides."""
        for side in (self.protein, self.complex):
            side.build_wildtype()
            if side.mutations:
                side.build_mutant()
        return self

    def evaluate(self, wrapper, store_structure: bool = True) -> None:
        """Fold and extract features for both sides."""
        self.protein.evaluate(wrapper, store_structure=store_structure)
        self.complex.evaluate(wrapper, store_structure=store_structure)

    def embeddings(self, feature_names: Sequence[str],
                   protein_aggregation: str = "mutpos",
                   multi_aggregation: str = "sum",
                   include_unbound: bool = False):
        """Aggregated features for the bound form, optionally with the unbound.

        Returns:
            ``(wt, mt)`` vectors for the complex, or, with
            ``include_unbound=True``, ``(wt_unbound, mt_unbound, wt, mt)``.
        """
        bound = self.complex.embeddings(feature_names, protein_aggregation,
                                        multi_aggregation)
        if not include_unbound:
            return bound
        unbound = self.protein.embeddings(feature_names, protein_aggregation,
                                          multi_aggregation)
        return (*unbound, *bound)

    # -- persistence ------------------------------------------------------- #

    def __setstate__(self, state):
        """Restore from a pickle, migrating the pre-2.0 attribute layout."""
        from aftoolkit._compat import migrate_protein_complex_task_state

        self.__dict__.update(migrate_protein_complex_task_state(state))

    def save(self, path) -> None:
        """Pickle the whole task to ``path``."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> ProteinComplexTask:
        """Load a task pickled by :meth:`save`."""
        with open(path, "rb") as f:
            return pickle.load(f)
