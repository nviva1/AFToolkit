"""Mutations and residue identity.

A mutation used to be a dict entry keyed by a bare tuple::

    {("S", 11, "A"): "A"}    # (wt_aa, position, chain) -> mt_aa

which forced every consumer to index the key positionally. That spelling drifted:
the chain was read as ``key[2]`` in one module and ``key[-1]`` in another, and the
residue name ``f"{key[0]}_{key[-1]}_{key[1]}"`` was rebuilt by hand in four places.

:class:`Mutation` replaces it with named fields and owns the residue-key format,
so the layout is stated once.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "GAP",
    "Mutation",
    "ObservableRegion",
    "parse_mutations",
    "parse_point_mutations",
    "parse_positions",
    "residue_key",
]

#: Marker used for a deleted residue (as ``mt``) or an insertion point (as ``wt``).
GAP = "-"

# "<chain>:<wt><pos><mt>", e.g. "A:D123C". The mutant part may be several
# residues (an insertion) or the gap marker (a deletion).
_MUTATION_RE = re.compile(
    r"^(?P<chain>[^:\s]+)\s*:\s*(?P<wt>[A-Z-])(?P<pos>-?\d+)(?P<mt>[A-Z-]+)$"
)
# "<chain>:<pos>", e.g. "A:123".
_POSITION_RE = re.compile(r"^(?P<chain>[^:\s]+)\s*:\s*(?P<pos>-?\d+)$")
# "<wt><pos><mt>" with the chain supplied separately, e.g. "E285K".
_BARE_MUTATION_RE = re.compile(r"^(?P<wt>[A-Z])(?P<pos>\d+)(?P<mt>[A-Z])$")


def residue_key(aa: str, chain: str, position: int, insertion: str = "",
                inserted: bool = False) -> str:
    """Canonical name for one residue, e.g. ``"K_A_90"``.

    This is the key under which per-residue AlphaFold features are stored, so it
    must stay stable across the wildtype and mutant of the same structure.

    Args:
        aa: one-letter residue code.
        chain: original chain id.
        position: original residue number.
        insertion: PDB insertion code, or the ordinal within an inserted stretch.
        inserted: True when this residue was *added* by an insertion mutation. It
            is then suffixed ``i`` plus ``insertion``, to tell it both from the
            residue it was threaded next to and from the other residues of the
            same inserted stretch -- which all share a position number, and so
            would otherwise collapse onto one key.
    """
    if inserted:
        return f"{aa}_{chain}_{position}i{insertion}"
    return f"{aa}_{chain}_{position}{insertion}"


@dataclass(frozen=True, order=True)
class Mutation:
    """A single point mutation, deletion or insertion.

    Ordered by ``(chain, position)`` so a set of mutations has a stable,
    human-sensible sort order.

    Attributes:
        chain: original chain id the mutation applies to.
        position: original residue number.
        wt: wildtype one-letter code, or ``"-"`` for an insertion.
        mt: mutant one-letter code, ``"-"`` for a deletion, or a multi-residue
            sequence for an insertion.
    """

    chain: str
    position: int
    wt: str
    mt: str

    def __post_init__(self) -> None:
        if not self.wt:
            raise ValueError(f"{self!r}: wildtype residue must not be empty.")
        if not self.mt:
            raise ValueError(f"{self!r}: mutant residue must not be empty.")
        if self.is_insertion and self.is_deletion:
            raise ValueError(f"{self!r}: cannot be both an insertion and a deletion.")

    @property
    def is_deletion(self) -> bool:
        """True when the residue is removed."""
        return self.mt == GAP

    @property
    def is_insertion(self) -> bool:
        """True when ``mt`` is a sequence threaded in at ``position``."""
        return self.wt == GAP

    @property
    def is_substitution(self) -> bool:
        """True for an ordinary one-for-one residue swap."""
        return not self.is_deletion and not self.is_insertion

    @property
    def key(self) -> str:
        """Residue key of the *wildtype* residue this mutation sits on."""
        return residue_key(self.wt, self.chain, self.position)

    def __str__(self) -> str:
        return f"{self.chain}:{self.wt}{self.position}{self.mt}"

    @classmethod
    def parse(cls, text: str) -> Mutation:
        """Parse one ``"<chain>:<wt><pos><mt>"`` string, e.g. ``"A:D123C"``."""
        m = _MUTATION_RE.match(text.strip())
        if not m:
            raise ValueError(
                f"Invalid mutation {text!r}; expected '<chain>:<wt><pos><mt>', "
                f"e.g. 'A:D123C' (deletion 'A:D123-', insertion 'A:-123GGS')."
            )
        return cls(chain=m["chain"], position=int(m["pos"]), wt=m["wt"], mt=m["mt"])


@dataclass(frozen=True)
class ObservableRegion:
    """A stretch of residues to extract AlphaFold features for.

    Replaces the untyped ``{"resi": .., "shift_from_resi": .., "chain_id": ..}``
    dict. ``span`` is the number of *additional* residues after ``start``, so 0
    means a single residue.

    ``span=None`` means "to the end of the chain", resolved against whichever
    structure it is applied to. That matters for indels: a fixed span measured
    on the wildtype would overrun a shortened mutant, and would miss the extra
    residues of a lengthened one.
    """

    chain: str
    start: int
    span: int | None = 0
    name: str | None = None

    def __post_init__(self) -> None:
        if self.span is not None and self.span < 0:
            raise ValueError(f"{self!r}: span must be >= 0 or None.")
        if self.name is None:
            object.__setattr__(self, "name", f"{self.start}_{self.chain}")


def parse_mutations(mutations: str | Iterable[str] | None) -> tuple[Mutation, ...]:
    """Parse comma-separated mutations into a tuple of :class:`Mutation`.

    Accepts ``"A:D123C, A:R56Y"``, an iterable of such strings, an iterable of
    :class:`Mutation`, or ``None`` (an empty result).

    Raises:
        ValueError: if any entry is malformed, or if two entries target the same
            chain and position -- previously such a collision silently discarded
            one of them, because mutations were stored in a dict keyed by position.
    """
    if mutations is None:
        return ()
    if isinstance(mutations, str):
        items: Sequence[str] = [m for m in mutations.split(",") if m.strip()]
    else:
        items = list(mutations)

    parsed = tuple(
        m if isinstance(m, Mutation) else Mutation.parse(m) for m in items
    )
    _reject_duplicate_sites(parsed)
    return parsed


def _reject_duplicate_sites(mutations: Sequence[Mutation]) -> None:
    seen: dict[tuple[str, int], Mutation] = {}
    for mut in mutations:
        site = (mut.chain, mut.position)
        if site in seen:
            raise ValueError(
                f"Two mutations target {mut.chain}:{mut.position} "
                f"({seen[site]} and {mut}); each site may be mutated once."
            )
        seen[site] = mut


def parse_positions(positions: str | Iterable[str] | None) -> tuple[ObservableRegion, ...]:
    """Parse comma-separated ``"<chain>:<pos>"`` strings into observable regions."""
    if positions is None:
        return ()
    if isinstance(positions, str):
        items: Sequence[str] = [p for p in positions.split(",") if p.strip()]
    else:
        items = list(positions)

    regions = []
    for item in items:
        if isinstance(item, ObservableRegion):
            regions.append(item)
            continue
        m = _POSITION_RE.match(str(item).strip())
        if not m:
            raise ValueError(
                f"Invalid position {item!r}; expected '<chain>:<pos>', e.g. 'A:123'."
            )
        regions.append(ObservableRegion(chain=m["chain"], start=int(m["pos"])))
    return tuple(regions)


def parse_point_mutations(mutations: str | Iterable[str] | None,
                          chain: str) -> tuple[Mutation, ...]:
    """Parse chain-less point mutations and attach them to ``chain``.

    The command-line form ``"K90M,N107S,W103F"`` names a chain once, separately,
    rather than repeating it on every mutation. Only plain substitutions are
    accepted in this spelling.

    Raises:
        ValueError: if an entry is malformed or two target the same position.
    """
    if mutations is None:
        return ()
    if isinstance(mutations, str):
        items: Sequence[str] = [m for m in mutations.split(",") if m.strip()]
    else:
        items = list(mutations)

    parsed = []
    for item in items:
        if isinstance(item, Mutation):
            parsed.append(item)
            continue
        m = _BARE_MUTATION_RE.match(str(item).strip())
        if not m:
            raise ValueError(
                f"Invalid mutation {item!r}; expected <WT><pos><MT>, e.g. 'E285K'."
            )
        parsed.append(
            Mutation(chain=chain, position=int(m["pos"]), wt=m["wt"], mt=m["mt"])
        )

    parsed = tuple(parsed)
    _reject_duplicate_sites(parsed)
    return parsed

