"""AFToolkit — protein engineering with AlphaFold2-derived representations.

Typical usage::

    from aftoolkit import ProteinTask, OpenFoldWrapper, parse_mutations

    wrapper = OpenFoldWrapper(device="cuda:0", inference_n_recycle=3)
    wrapper.init_model()

    task = ProteinTask()
    task.set_input_protein_task("protein.pdb", chains=["A"])
    task.set_task_mutants(parse_mutations("A:S11A"))
    task.set_observable_positions()
    task.evaluate(of_wrapper=wrapper)

The lightweight task/adapter APIs do not require PyTorch. ``OpenFoldWrapper``
(and anything that runs AlphaFold inference) does; it is imported lazily so the
rest of the package is usable without the optional ``[openfold]`` extra.
"""

from aftoolkit._compat import install_legacy_aliases

# Lightweight, torch-free public API (safe to import eagerly).
from aftoolkit.processing.protein_task import ProteinTask
from aftoolkit.processing.protein_complex_task import ProteinComplexTask
from aftoolkit.processing.arg_parser import parse_mutations, parse_positions
from aftoolkit.models.adapter import (
    AdapterMonomer,
    AdapterMultimer,
    AdapterMonomerPerResiduePredictor,
)
from aftoolkit.models.utils import calculate_metrics

# Allow old `AFToolKit.*` pickles to resolve to the renamed package.
install_legacy_aliases()

__version__ = "1.0.0"

__all__ = [
    "ProteinTask",
    "ProteinComplexTask",
    "parse_mutations",
    "parse_positions",
    "AdapterMonomer",
    "AdapterMultimer",
    "AdapterMonomerPerResiduePredictor",
    "calculate_metrics",
    "OpenFoldWrapper",
    "resolve_device",
    "__version__",
]


def __getattr__(name):
    """Lazily expose torch-dependent objects from the top-level package."""
    if name in ("OpenFoldWrapper", "resolve_device"):
        from aftoolkit.processing import openfold_wrapper

        return getattr(openfold_wrapper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
