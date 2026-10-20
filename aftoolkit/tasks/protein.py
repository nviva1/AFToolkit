"""A single-protein mutation task."""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Sequence

import pandas as pd

from aftoolkit import embeddings
from aftoolkit.mutations import Mutation, ObservableRegion, parse_mutations, residue_key
from aftoolkit.structure import (
    AA_3_TO_1_DICT,
    load_protein,
    mutate_protein,
    strip_mutated_sidechains,
)
from aftoolkit.tasks.state import InputProtein, ProteinState

__all__ = ["ProteinTask"]


class ProteinTask:
    """A protein, a set of mutations, and the features extracted for both forms.

    Typical use::

        task = ProteinTask()
        task.set_input("protein.pdb", chains=["A"])
        task.set_mutations("A:S11A")
        task.set_observables()          # whole chain, by default
        task.evaluate(wrapper)
        wt_vector, mt_vector = task.embeddings(["pair", "plddt"])

    Attributes:
        input: :class:`~aftoolkit.tasks.state.InputProtein` describing the source.
        mutations: tuple of :class:`~aftoolkit.mutations.Mutation`.
        observables: regions to extract features for.
        wildtype / mutant: the two :class:`~aftoolkit.tasks.state.ProteinState`.
    """

    def __init__(self, input: InputProtein | None = None,
                 mutations: Iterable[Mutation] = (),
                 observables: Iterable[ObservableRegion] = ()):
        self.input = input if input is not None else InputProtein()
        self.mutations = parse_mutations(mutations)
        self.observables = tuple(observables)
        self.wildtype = ProteinState()
        self.mutant = ProteinState()

    def __repr__(self) -> str:
        muts = ",".join(str(m) for m in self.mutations) or "none"
        return (f"{type(self).__name__}(path={self.input.path!r}, "
                f"chains={self.input.chains}, mutations={muts})")

    # -- setup ------------------------------------------------------------- #

    def set_input(self, path=None, chains=None, regions=None,
                  chain_gap: int | None = None) -> ProteinTask:
        """Point the task at a structure. Returns ``self`` so calls can chain."""
        self.input = InputProtein(
            path=path,
            chains=list(chains) if chains is not None else None,
            regions=regions,
            chain_gap=self.input.chain_gap if chain_gap is None else chain_gap,
        )
        return self

    def set_mutations(self, mutations) -> ProteinTask:
        """Replace the task's mutations. Accepts a string, or Mutation objects."""
        self.mutations = parse_mutations(mutations)
        return self

    def add_mutation(self, mutation=None, *, chain=None, position=None,
                     wt=None, mt=None) -> ProteinTask:
        """Add one mutation, as an object, a string, or explicit fields."""
        if mutation is None:
            mutation = Mutation(chain=chain, position=position, wt=wt, mt=mt)
        self.mutations = parse_mutations([*self.mutations, mutation])
        return self

    def set_observables(self, regions=None) -> ProteinTask:
        """Set the regions to extract features for.

        With ``regions=None`` every residue of every selected chain is observed,
        which requires the wildtype structure and so builds it if needed.
        """
        if regions is not None:
            self.observables = tuple(regions)
            return self

        self.observables = tuple(self._whole_chain_regions())
        return self

    def add_observable(self, chain, start, span=0, name=None) -> ProteinTask:
        """Add one observable region."""
        self.observables = (*self.observables,
                            ObservableRegion(chain=chain, start=start,
                                             span=span, name=name))
        return self

    def _whole_chain_regions(self) -> list[ObservableRegion]:
        """One region per chain, spanning all of its residues."""
        structure = self.build_wildtype()
        chains = self.input.chains
        if chains is None:
            chains = structure.chain_id_original.unique()

        regions = []
        for chain in chains:
            in_chain = structure["chain_id_original"] == chain
            start = structure.loc[in_chain, "residue_number_original"].min()
            # span=None: every residue of the chain, however long this
            # particular structure's copy of it turns out to be.
            regions.append(ObservableRegion(chain=chain, start=start, span=None))
        return regions

    # -- structures -------------------------------------------------------- #

    def build_wildtype(self, structure=None):
        """Build (and cache) the wildtype structure.

        Side chains at substituted positions are stripped, so the wildtype and
        mutant are threaded through AlphaFold on equal footing.
        """
        if structure is None and self.wildtype.is_built:
            return self.wildtype.structure
        if structure is None:
            if self.input.path is None:
                raise ValueError(
                    "No input structure: call set_input(path=...) first."
                )
            structure = load_protein(self.input.path, self.input.chains,
                                     self.input.regions, self.input.chain_gap)
        self.wildtype.structure = strip_mutated_sidechains(structure, self.mutations)
        return self.wildtype.structure

    def build_mutant(self, structure=None):
        """Build (and cache) the mutant structure."""
        if structure is None and self.mutant.is_built:
            return self.mutant.structure
        if structure is None:
            structure = mutate_protein(self.build_wildtype(), self.mutations)
        self.mutant.structure = structure
        return self.mutant.structure

    def wt_residue(self, position, chain, insertion="") -> str:
        """One-letter wildtype residue at a position, read from the structure."""
        structure = self.build_wildtype()
        match = structure.loc[
            (structure["residue_number_original"] == position)
            & (structure["chain_id_original"] == chain)
            & (structure["insertion"] == insertion),
            "residue_name",
        ].unique()
        if len(match) == 0:
            raise KeyError(f"No residue at {chain}:{position}{insertion}.")
        return AA_3_TO_1_DICT[match[0]]

    # -- observable positions ---------------------------------------------- #

    def _resolve_observables(self, structure) -> dict:
        """Map every observable region onto ``{residue_key: index}`` in ``structure``.

        Indices are into the CA-only view, which is the residue order AlphaFold's
        outputs use.
        """
        if not self.observables:
            self.set_observables()

        ca = structure[structure["atom_name"] == "CA"].reset_index(drop=True)
        deleted = {(m.chain, m.position) for m in self.mutations if m.is_deletion}

        observed = {}
        for region in self.observables:
            observed.update(self._resolve_region(region, ca, deleted))
        return observed

    def _resolve_region(self, region: ObservableRegion, ca, deleted) -> dict:
        start = region.start
        if (region.chain, start) in deleted:
            # The region starts on a residue this task deletes, so it is absent
            # from the mutant; anchor on a neighbour instead.
            start = 2 if start == 1 else start - 1

        anchors = ca[(ca["residue_number_original"] == start)
                     & (ca["chain_id_original"] == region.chain)
                     & (ca["insertion"] == "")].index.values
        if len(anchors) == 0:
            raise KeyError(
                f"Observable residue {region.chain}:{start} is not in the structure."
            )
        if len(anchors) > 1:
            raise KeyError(
                f"{len(anchors)} residues match observable {region.chain}:{start}; "
                f"expected exactly one."
            )

        # A fixed span is measured on the wildtype, but this structure may be
        # shorter (a deletion) or longer (an insertion), so never index past the
        # end of this chain. ``span=None`` means the whole chain outright.
        chain_end = ca.index[ca["chain_id_original"] == region.chain].max()
        last = chain_end if region.span is None else min(anchors[0] + region.span,
                                                         chain_end)

        observed = {}
        for idx in range(anchors[0], last + 1):
            row = ca.loc[idx]
            observed[self._residue_key_for(row, region.chain)] = idx
        return observed

    def _residue_key_for(self, row, chain) -> str:
        """Residue key for one CA row of a structure.

        Normal residues are keyed by their *wildtype* identity so that a
        substitution's wildtype and mutant features land under the same key and
        can be compared. An inserted residue has no wildtype counterpart, so it
        is keyed by its own identity instead.
        """
        position = row["residue_number_original"]
        if row["mask"]:
            return residue_key(
                aa=AA_3_TO_1_DICT[row["residue_name"]],
                chain=chain,
                position=position,
                insertion=row["insertion"],
                inserted=True,
            )
        return residue_key(
            aa=self.wt_residue(position, chain, row["insertion"]),
            chain=chain,
            position=position,
            insertion=row["insertion"],
        )

    # -- inference --------------------------------------------------------- #

    def _evaluate_state(self, state: ProteinState, wrapper,
                        store_structure: bool = True) -> None:
        """Fold one state and extract its features.

        The wildtype and the mutant differ only in which state is passed here --
        which is the point. Running them through separate near-identical methods
        is what previously let the mutant silently reuse the wildtype's observed
        positions when extracting per-cycle features.
        """
        state.observed = self._resolve_observables(state.structure)

        wrapper.evaluate(protein_df=state.structure)
        state.features = wrapper.get_of_features_by_positions(state.observed)
        state.folded = wrapper.get_of_protein() if store_structure else pd.DataFrame()
        state.cycles = (
            wrapper.get_of_features_by_positions_per_cycle(state.observed)
            if wrapper.of_output_per_cycle is not None else []
        )

    def evaluate_wildtype(self, wrapper, store_structure: bool = True) -> None:
        """Fold the wildtype and extract its features."""
        self.build_wildtype()
        self._evaluate_state(self.wildtype, wrapper, store_structure)

    def evaluate_mutant(self, wrapper, store_structure: bool = True) -> None:
        """Fold the mutant and extract its features."""
        self.build_mutant()
        self._evaluate_state(self.mutant, wrapper, store_structure)

    def evaluate(self, wrapper, store_structure: bool = True) -> None:
        """Fold the wildtype, and the mutant too when there are mutations."""
        self.evaluate_wildtype(wrapper, store_structure)
        if self.mutations:
            self.evaluate_mutant(wrapper, store_structure)

    # -- features ---------------------------------------------------------- #

    def embeddings(self, feature_names: Sequence[str],
                   protein_aggregation: str = "mutpos",
                   multi_aggregation: str = "sum"):
        """Aggregate per-residue features into a wildtype and a mutant vector.

        See :mod:`aftoolkit.embeddings` for what the two aggregation
        settings mean.
        """
        return embeddings.aggregate(
            wt_features=self.wildtype.features,
            wt_positions=self.wildtype.observed,
            mt_features=self.mutant.features,
            mt_positions=self.mutant.observed,
            feature_names=feature_names,
            mutations=self.mutations,
            protein_aggregation=protein_aggregation,
            multi_aggregation=multi_aggregation,
            resolve_wt_aa=lambda chain, position: self.wt_residue(position, chain),
        )

    # -- persistence ------------------------------------------------------- #

    def __setstate__(self, state):
        """Restore from a pickle, migrating the pre-2.0 attribute layout.

        Feature files are two AlphaFold passes each, so old ones are worth
        keeping readable rather than regenerating.
        """
        from aftoolkit._compat import migrate_protein_task_state

        self.__dict__.update(migrate_protein_task_state(state))

    def save(self, path) -> None:
        """Pickle the whole task to ``path``."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> ProteinTask:
        """Load a task pickled by :meth:`save`."""
        with open(path, "rb") as f:
            return pickle.load(f)
