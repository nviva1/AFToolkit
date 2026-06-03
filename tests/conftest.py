import os

import pytest

# Example PDBs shipped in the repo.
PDB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "examples",
    "PDB",
)


@pytest.fixture
def monomer_pdb():
    """Single-chain example structure (1A0F, chain A)."""
    path = os.path.join(PDB_DIR, "1A0F.pdb")
    if not os.path.exists(path):
        pytest.skip(f"example PDB missing: {path}")
    return path


@pytest.fixture
def complex_pdb():
    """Protein-protein complex example (1CSE, chains E + I)."""
    path = os.path.join(PDB_DIR, "1CSE.pdb")
    if not os.path.exists(path):
        pytest.skip(f"example PDB missing: {path}")
    return path
