"""The ``task`` and ``complex`` subcommands: extract features for mutations."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

import pandas as pd

from aftoolkit.cli.common import build_wrapper, run_tasks
from aftoolkit.mutations import parse_mutations, parse_positions
from aftoolkit.tasks import ProteinComplexTask, ProteinTask

__all__ = ["run_complex", "run_protein", "tasks_from_csv"]

logger = logging.getLogger(__name__)


def _as_text(value) -> str | None:
    """Normalise an optional CLI value or CSV cell to a string, or ``None``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value).strip() or None


def _split(value) -> list[str] | None:
    """Split a comma-separated value, treating blanks as unset."""
    text = _as_text(value)
    if text is None:
        return None
    return [part for part in text.split(",") if part] or None


def tasks_from_csv(path: str, is_complex: bool = False
                   ) -> Iterator[tuple[str, Callable[[], object]]]:
    """Read a task CSV into ``(name, factory)`` pairs.

    Expected columns: ``pdb_id``, ``path``, ``mut_info``, and either ``chain``
    (protein) or ``chains_for_protein`` plus optional ``chains_for_complex``
    (complex). ``mutations`` and ``positions`` are optional.

    Raises:
        ValueError: if a required column is missing.
    """
    table = pd.read_csv(path)
    chain_column = "chains_for_protein" if is_complex else "chain"
    required = {"pdb_id", "path", "mut_info", chain_column}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s) {sorted(missing)}; "
            f"found {sorted(table.columns)}."
        )

    for row in table.to_dict("records"):
        name = f"{row['pdb_id']}_{row[chain_column]}_{row['mut_info']}"
        yield name, _factory(row, chain_column, is_complex)


def _factory(row: dict, chain_column: str, is_complex: bool) -> Callable[[], object]:
    """Build a task from one CSV row, deferred so failures are per-task."""
    def build():
        mutations = parse_mutations(_as_text(row.get("mutations")))
        positions = parse_positions(_as_text(row.get("positions"))) or None
        if is_complex:
            task = ProteinComplexTask()
            task.set_input(row["path"],
                           protein_chains=_split(row[chain_column]),
                           complex_chains=_split(row.get("chains_for_complex")))
        else:
            task = ProteinTask()
            task.set_input(row["path"], chains=_split(row[chain_column]))
        task.set_mutations(mutations)
        task.set_observables(positions)
        return task

    return build


def run_protein(args) -> None:
    """``aftoolkit task``: features for a mutated protein."""
    if args.input:
        tasks = list(tasks_from_csv(args.input, is_complex=False))
    else:
        def build():
            task = ProteinTask()
            task.set_input(args.pdb, chains=_split(args.chain),
                           chain_gap=args.chain_gap)
            task.set_mutations(parse_mutations(args.mutations))
            task.set_observables(parse_positions(args.positions) or None)
            return task

        name = "_".join(str(p) for p in (args.pdb, args.chain, args.mutations))
        tasks = [(name, build)]

    run_tasks(tasks, build_wrapper(args), args.output_dir,
              args.store_protein, keep_going=bool(args.input))


def run_complex(args) -> None:
    """``aftoolkit complex``: features for a mutated complex, bound and unbound."""
    if args.input:
        tasks = list(tasks_from_csv(args.input, is_complex=True))
    else:
        def build():
            task = ProteinComplexTask()
            task.set_input(args.pdb,
                           protein_chains=_split(args.chains_for_protein),
                           complex_chains=_split(args.chains_for_complex),
                           chain_gap=args.chain_gap)
            task.set_mutations(parse_mutations(args.mutations))
            task.set_observables(parse_positions(args.positions) or None)
            return task

        name = "_".join(str(p) for p in
                        (args.pdb, args.chains_for_protein, args.mutations))
        tasks = [(name, build)]

    run_tasks(tasks, build_wrapper(args), args.output_dir,
              args.store_protein, keep_going=bool(args.input))
