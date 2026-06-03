from aftoolkit.processing.protein_task import ProteinTask
from aftoolkit.processing.protein_complex_task import ProteinComplexTask

__all__ = ["ProteinTask", "ProteinComplexTask", "OpenFoldWrapper"]


def __getattr__(name):
    """Lazily expose torch-dependent objects.

    ``OpenFoldWrapper`` pulls in the full PyTorch/OpenFold stack, which is not
    needed for the lightweight task-construction APIs (``ProteinTask`` /
    ``ProteinComplexTask``). Importing it lazily keeps those usable without a
    GPU/torch install.
    """
    if name == "OpenFoldWrapper":
        from aftoolkit.processing.openfold_wrapper import OpenFoldWrapper
        return OpenFoldWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
