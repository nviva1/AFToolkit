"""Backwards-compatibility shims for pickles written by older versions.

Two things changed that a stored pickle records and therefore trips over.

**The package name.** This was once imported as ``AFToolKit`` (capital ``K``),
which only resolved on case-insensitive filesystems, and has since been renamed
to the canonical lowercase ``aftoolkit``.

**The module layout.** In 2.0 the first-party modules moved out of
``aftoolkit.processing`` (now only the vendored OpenFold distribution) and
``ProteinTask``/``ProteinComplexTask`` were rebuilt around dataclasses. A pickle
stores the class's *module path* and the instance's ``__dict__``, so both the
path and the attribute layout need translating.

:func:`install_legacy_aliases` installs an import hook covering the renames;
:func:`migrate_protein_task_state` converts the old attribute layout, and is
called from ``ProteinTask.__setstate__``. Together they let feature files
computed by any earlier version load unchanged -- which matters, because those
files are expensive: each one is two AlphaFold passes.
"""

import importlib
import importlib.util
import sys
from importlib.abc import Loader, MetaPathFinder

__all__ = [
    "install_legacy_aliases",
    "migrate_protein_task_state",
    "migrate_protein_complex_task_state",
    "LEGACY_MODULES",
]

_OLD_NAME = "AFToolKit"
_NEW_NAME = "aftoolkit"

#: Modules that moved in 2.0, old path -> new path.
LEGACY_MODULES = {
    "aftoolkit.processing.protein_task": "aftoolkit.tasks.protein",
    "aftoolkit.processing.protein_complex_task": "aftoolkit.tasks.complex",
    "aftoolkit.processing.utils": "aftoolkit.structure",
    "aftoolkit.processing.arg_parser": "aftoolkit.mutations",
    "aftoolkit.processing.openfold_wrapper": "aftoolkit.folding.wrapper",
    "aftoolkit.processing.run_protein_task": "aftoolkit.cli.tasks",
    "aftoolkit.processing.run_protein_complex_task": "aftoolkit.cli.tasks",
}


def _canonical(fullname: str) -> str | None:
    """The module a legacy import should resolve to, or ``None`` if it is not one."""
    if fullname == _OLD_NAME:
        return _NEW_NAME
    if fullname.startswith(_OLD_NAME + "."):
        fullname = _NEW_NAME + fullname[len(_OLD_NAME):]
        if fullname in LEGACY_MODULES:
            return LEGACY_MODULES[fullname]
        return fullname
    return LEGACY_MODULES.get(fullname)


class _LegacyNameFinder(MetaPathFinder, Loader):
    """Redirect legacy module paths to where the code lives now."""

    def find_spec(self, fullname, path=None, target=None):
        if _canonical(fullname) is not None:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(_canonical(spec.name))

    def exec_module(self, module):
        # The module object is the already-imported canonical module; nothing
        # further to execute.
        return None


def install_legacy_aliases():
    """Idempotently install the legacy-name import hook.

    Inserted at the front of ``sys.meta_path`` so it pre-empts the default
    path-based finder. This matters on case-insensitive filesystems (macOS),
    where ``import AFToolKit.*`` would otherwise be resolved to the
    ``aftoolkit/`` directory as a *separate* module object, breaking pickle's
    class-identity check.
    """
    if any(isinstance(f, _LegacyNameFinder) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _LegacyNameFinder())


# --------------------------------------------------------------------------- #
# State migration
# --------------------------------------------------------------------------- #
def _legacy_state(state: dict) -> bool:
    """True when ``state`` is the pre-2.0 three-nested-dicts layout."""
    return "task" in state and ("protein_job" in state or "protein_of" in state)


def _migrate_side(job: dict, of: dict, side: str):
    """Build one :class:`~aftoolkit.tasks.state.ProteinState` from the old dicts."""
    import pandas as pd

    from aftoolkit.tasks.state import ProteinState

    job_side = (job or {}).get(side) or {}
    of_side = (of or {}).get(side) or {}
    cycles = of_side.get("cycles") or []
    if isinstance(cycles, dict):           # older runs stored {} rather than []
        cycles = list(cycles.values())

    return ProteinState(
        structure=job_side.get("protein", pd.DataFrame()),
        observed=job_side.get("obs_positions") or {},
        folded=of_side.get("protein", pd.DataFrame()),
        features=of_side.get("features") or {},
        cycles=cycles,
    )


def migrate_protein_task_state(state: dict) -> dict:
    """Translate a pre-2.0 ``ProteinTask.__dict__`` into the current layout.

    Returns ``state`` unchanged when it is already current, so this is safe to
    call from ``__setstate__`` for every unpickle.

    The old ``chain_gap`` is not recorded in the pickle -- it was a module-level
    constant at the time -- so the current default is assumed. That only affects
    rebuilding structures from the input PDB; the features already stored in the
    file are untouched.
    """
    if not _legacy_state(state):
        return state

    from aftoolkit.mutations import Mutation, ObservableRegion
    from aftoolkit.tasks.state import InputProtein

    task = state.get("task") or {}
    source = task.get("input_protein") or {}

    mutations = []
    for key, mt in (task.get("mutants") or {}).items():
        wt, position, chain = key[0], key[1], key[-1]
        mutations.append(Mutation(chain=chain, position=position, wt=wt, mt=mt))
    mutations.sort(key=lambda m: (m.chain, m.position))

    observables = []
    for region in task.get("obs_positions") or []:
        observables.append(ObservableRegion(
            chain=region["chain_id"],
            start=region["resi"],
            span=region.get("shift_from_resi", 0),
            name=region.get("name"),
        ))

    return {
        "input": InputProtein(
            path=source.get("path"),
            chains=source.get("chains"),
            regions=source.get("regions"),
        ),
        "mutations": tuple(mutations),
        "observables": tuple(observables),
        "wildtype": _migrate_side(state.get("protein_job"),
                                  state.get("protein_of"), "protein_wt"),
        "mutant": _migrate_side(state.get("protein_job"),
                                state.get("protein_of"), "protein_mt"),
    }


def migrate_protein_complex_task_state(state: dict) -> dict:
    """Translate a pre-2.0 ``ProteinComplexTask.__dict__`` into the current layout.

    The two sides were already separate ``ProteinTask`` objects, so they migrate
    themselves; only the wrapper's own fields need rebuilding.
    """
    if "task" not in state or "protein" not in state:
        return state

    task = state.get("task") or {}
    source = task.get("input_protein") or {}
    return {
        "protein": state["protein"],
        "complex": state["complex"],
        "path": source.get("path"),
        "protein_chains": list(source.get("chains_for_protein") or []),
        "complex_chains": source.get("chains_for_complex"),
    }
