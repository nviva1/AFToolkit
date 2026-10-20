"""AlphaFold2/OpenFold inference.

Everything here needs the optional ``[openfold]`` extra (torch and the vendored
OpenFold stack), so it is imported lazily from the top-level package.
"""

from aftoolkit.folding.weights import WeightsUnavailable, ensure_weights
from aftoolkit.folding.wrapper import OpenFoldWrapper, resolve_device

__all__ = ["OpenFoldWrapper", "WeightsUnavailable", "ensure_weights", "resolve_device"]
