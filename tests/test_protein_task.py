"""Tests for ProteinTask construction, observables and mutation handling."""

import pytest

from aftoolkit import Mutation, ProteinTask, parse_mutations, parse_positions


def test_wildtype_and_mutant_construction(monomer_pdb):
    task = ProteinTask()
    task.set_input(monomer_pdb, chains=["A"])
    task.set_mutations("A:S11A")
    task.build_wildtype()
    task.build_mutant()

    assert task.wildtype.is_built
    assert task.mutant.is_built
    assert task.wt_residue(11, "A") == "S"


def test_setters_chain(monomer_pdb):
    task = (ProteinTask()
            .set_input(monomer_pdb, chains=["A"])
            .set_mutations("A:S11A")
            .set_observables(parse_positions("A:11")))
    assert task.mutations == parse_mutations("A:S11A")
    assert len(task.observables) == 1


def test_observables_default_covers_chain(monomer_pdb):
    task = ProteinTask()
    task.set_input(monomer_pdb, chains=["A"])
    task.set_mutations("A:S11A")
    task.build_wildtype()
    task.set_observables()

    assert len(task.observables) == 1
    region, = task.observables
    assert region.chain == "A"
    # span=None means "to the end of the chain", resolved per structure, so an
    # indel cannot make the region overrun or under-cover the mutant.
    assert region.span is None


def test_observables_default_without_explicit_wildtype(monomer_pdb):
    # set_observables() must lazily build the wildtype structure rather than
    # failing when it has not been built yet.
    task = ProteinTask()
    task.set_input(monomer_pdb, chains=["A"])
    task.set_mutations("A:S11A")
    task.set_observables()

    region, = task.observables
    assert region.chain == "A"


def test_add_mutation_roundtrip():
    task = ProteinTask()
    task.add_mutation(chain="A", position=11, wt="S", mt="A")
    assert task.mutations == (Mutation(chain="A", position=11, wt="S", mt="A"),)


def test_add_mutation_accepts_a_string():
    task = ProteinTask()
    task.add_mutation("A:S11A")
    assert task.mutations == (Mutation(chain="A", position=11, wt="S", mt="A"),)


def test_add_duplicate_mutation_is_rejected():
    task = ProteinTask()
    task.add_mutation("A:S11A")
    with pytest.raises(ValueError, match="each site may be mutated once"):
        task.add_mutation("A:S11G")


def test_build_without_input_explains_itself():
    with pytest.raises(ValueError, match="call set_input"):
        ProteinTask().build_wildtype()


def test_wt_residue_missing_position_raises(monomer_pdb):
    task = ProteinTask().set_input(monomer_pdb, chains=["A"])
    with pytest.raises(KeyError, match="No residue at"):
        task.wt_residue(99999, "A")


def test_observed_positions_resolve_to_residue_keys(monomer_pdb):
    task = ProteinTask()
    task.set_input(monomer_pdb, chains=["A"])
    task.set_mutations("A:S11A")
    task.set_observables(parse_positions("A:11"))

    observed = task._resolve_observables(task.build_wildtype())
    assert "S_A_11" in observed


def test_unknown_observable_position_is_reported(monomer_pdb):
    task = ProteinTask()
    task.set_input(monomer_pdb, chains=["A"])
    task.set_observables(parse_positions("A:99999"))
    with pytest.raises(KeyError, match="not in the structure"):
        task._resolve_observables(task.build_wildtype())


def test_save_and_load_roundtrip(monomer_pdb, tmp_path):
    task = ProteinTask()
    task.set_input(monomer_pdb, chains=["A"])
    task.set_mutations("A:S11A")

    path = tmp_path / "task.pkl"
    task.save(path)
    loaded = ProteinTask.load(path)
    assert loaded.mutations == task.mutations
    assert loaded.input.path == task.input.path


def test_repr_mentions_mutations(monomer_pdb):
    task = ProteinTask().set_input(monomer_pdb, chains=["A"]).set_mutations("A:S11A")
    assert "A:S11A" in repr(task)


class _RecordingWrapper:
    """A stand-in for OpenFoldWrapper that records what it was asked for."""

    of_output_per_cycle = [{"cycle": 0}]

    def __init__(self):
        self.feature_calls = []
        self.per_cycle_calls = []
        self.folded_structures = []

    def evaluate(self, protein_df):
        self.folded_structures.append(protein_df)

    def get_of_features_by_positions(self, positions):
        self.feature_calls.append(dict(positions))
        return {name: {"feat": idx} for name, idx in positions.items()}

    def get_of_features_by_positions_per_cycle(self, positions):
        self.per_cycle_calls.append(dict(positions))
        return [{name: {"feat": idx} for name, idx in positions.items()}]

    def get_of_protein(self):
        import pandas as pd

        return pd.DataFrame({"atom_name": ["CA"]})


def test_each_state_is_evaluated_at_its_own_positions(monomer_pdb):
    """Regression: the mutant's per-cycle features were extracted at the
    *wildtype's* observed indices.

    The two evaluate methods were copy-pasted and one substitution was missed,
    so with an indel -- which shifts the mutant's indices -- the mutant's
    per-cycle features silently described the wrong residues.
    """
    task = ProteinTask()
    task.set_input(monomer_pdb, chains=["A"])
    task.set_mutations("A:S11-")           # a deletion shifts later indices
    task.set_observables(parse_positions("A:12"))

    wrapper = _RecordingWrapper()
    task.evaluate(wrapper)

    wt_positions, mt_positions = wrapper.per_cycle_calls
    assert wt_positions == task.wildtype.observed
    assert mt_positions == task.mutant.observed
    # The deletion really does move the residue, so the two differ: that
    # difference is exactly what the old code lost.
    assert wt_positions != mt_positions


def test_whole_chain_observables_track_a_deletion(monomer_pdb):
    """Regression: the observable span was measured on the wildtype and then
    indexed into the mutant, so a deletion ran one past the end (KeyError)."""
    task = ProteinTask().set_input(monomer_pdb, chains=["A"])
    task.set_mutations("A:S11-")
    task.set_observables()

    wt = task._resolve_observables(task.build_wildtype())
    mt = task._resolve_observables(task.build_mutant())
    assert len(mt) == len(wt) - 1


def test_whole_chain_observables_track_an_insertion(monomer_pdb):
    """Regression: inserted residues have no wildtype counterpart, so building
    their key from the wildtype raised; and a fixed span missed the extra
    residues entirely."""
    task = ProteinTask().set_input(monomer_pdb, chains=["A"])
    task.set_mutations("A:-11GGS")
    task.set_observables()

    wt = task._resolve_observables(task.build_wildtype())
    mt = task._resolve_observables(task.build_mutant())
    assert len(mt) == len(wt) + 3


def test_inserted_residues_get_distinct_keys(monomer_pdb):
    """Regression: every residue of an inserted stretch shares a position
    number, so a single ``i`` suffix collapsed them onto one key."""
    task = ProteinTask().set_input(monomer_pdb, chains=["A"])
    task.set_mutations("A:-11GGS")
    task.set_observables()

    mt = task._resolve_observables(task.build_mutant())
    inserted = sorted(k for k in mt if "i" in k.rsplit("_", 1)[-1])
    assert inserted == ["G_A_11i0", "G_A_11i1", "S_A_11i2"]
