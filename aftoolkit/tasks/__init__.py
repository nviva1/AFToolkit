"""Task objects: a structure, its mutations, and the features extracted for both."""

from aftoolkit.tasks.complex import ProteinComplexTask
from aftoolkit.tasks.protein import ProteinTask
from aftoolkit.tasks.state import InputProtein, ProteinState

__all__ = ["InputProtein", "ProteinComplexTask", "ProteinState", "ProteinTask"]
