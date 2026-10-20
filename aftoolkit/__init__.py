"""AFToolkit -- protein engineering with AlphaFold2-derived representations.

Typical usage::

    from aftoolkit import ProteinTask, OpenFoldWrapper

    wrapper = OpenFoldWrapper(device="cuda:0", inference_n_recycle=3).init_model()

    task = ProteinTask()
    task.set_input("protein.pdb", chains=["A"])
    task.set_mutations("A:S11A")
    task.set_observables()
    task.evaluate(wrapper)

    wt_vector, mt_vector = task.embeddings(["pair", "lddt_logits", "plddt"])

The task, mutation, structure and adapter APIs are pure NumPy/pandas and do not
require PyTorch. ``OpenFoldWrapper`` (and anything that runs AlphaFold inference)
does; it is imported lazily so the rest of the package stays usable without the
optional ``[openfold]`` extra.
"""

from aftoolkit._compat import install_legacy_aliases
from aftoolkit.models.adapter import (
    AdapterMonomer,
    AdapterMonomerPerResiduePredictor,
    AdapterMultimer,
)
from aftoolkit.models.utils import calculate_metrics
from aftoolkit.mutations import (
    Mutation,
    ObservableRegion,
    parse_mutations,
    parse_point_mutations,
    parse_positions,
)
from aftoolkit.structure import (
    DEFAULT_CHAIN_GAP,
    load_protein,
    mutate_protein,
    save_to_pdb,
)
from aftoolkit.tasks import InputProtein, ProteinComplexTask, ProteinState, ProteinTask

# Lets artifacts pickled under the old `AFToolKit.*` module path still resolve.
install_legacy_aliases()

__version__ = "2.0.0"

__all__ = [
    "DEFAULT_CHAIN_GAP",
    "AdapterMonomer",
    "AdapterMonomerPerResiduePredictor",
    "AdapterMultimer",
    "InputProtein",
    "Mutation",
    "ObservableRegion",
    "OpenFoldWrapper",
    "ProteinComplexTask",
    "ProteinState",
    "ProteinTask",
    "__version__",
    "calculate_metrics",
    "load_protein",
    "mutate_protein",
    "parse_mutations",
    "parse_point_mutations",
    "parse_positions",
    "resolve_device",
    "save_to_pdb",
]

_LAZY = {"OpenFoldWrapper", "resolve_device"}


def __getattr__(name):
    """Expose the torch-dependent folding objects without importing torch eagerly."""
    if name in _LAZY:
        from aftoolkit import folding

        return getattr(folding, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
