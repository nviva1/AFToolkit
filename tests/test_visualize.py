"""Tests for the confidence-difference map and PyMOL output.

Synthetic predicted structures, so no inference and no PyMOL needed.
"""

import os
import shutil

import pandas as pd
import pytest

from aftoolkit.visualize import (
    delta_bfactor_structure,
    pymol_available,
    write_pymol_session,
)


def _structure(plddt_by_residue, chain="A"):
    """A minimal predicted-structure frame: one N/CA/C/O set per residue."""
    rows = []
    for resnum, plddt in plddt_by_residue.items():
        for atom in ("N", "CA", "C", "O"):
            rows.append({
                "record_name": "ATOM", "atom_name": atom, "residue_name": "ALA",
                "chain_id": "A", "residue_number": resnum,
                "chain_id_original": chain, "residue_number_original": resnum,
                "insertion": "", "b_factor": plddt, "occupancy": 1.0,
                "x_coord": float(resnum), "y_coord": 0.0, "z_coord": 0.0,
                "element_symbol": atom[0], "charge": 0, "alt_loc": "",
                "segment_id": "", "blank_1": "", "blank_2": "", "blank_3": "",
                "blank_4": "", "atom_number": len(rows) + 1, "line_idx": len(rows),
            })
    return pd.DataFrame(rows)


def test_delta_is_mutant_minus_wildtype():
    wt = _structure({10: 90.0, 11: 80.0, 12: 70.0})
    mt = _structure({10: 85.0, 11: 80.0, 12: 75.0})

    delta, unmatched = delta_bfactor_structure(wt, mt)
    by_residue = delta.groupby("residue_number_original")["b_factor"].first()

    assert by_residue[10] == pytest.approx(-5.0)   # confidence lost
    assert by_residue[11] == pytest.approx(0.0)    # unchanged
    assert by_residue[12] == pytest.approx(+5.0)   # confidence gained
    assert unmatched == []


def test_delta_can_be_written_onto_the_wildtype_frame():
    wt = _structure({10: 90.0, 11: 80.0})
    mt = _structure({10: 85.0, 11: 80.0})

    on_wt, _ = delta_bfactor_structure(wt, mt, frame="wildtype")
    assert len(on_wt) == len(wt)
    assert on_wt.groupby("residue_number_original")["b_factor"].first()[10] == \
        pytest.approx(-5.0)


def test_unmatched_residues_are_zeroed_and_reported():
    """An insertion adds residues with no wildtype counterpart to difference."""
    wt = _structure({10: 90.0, 11: 80.0})
    mt = _structure({10: 85.0, 11: 80.0, 12: 60.0})   # residue 12 is new

    delta, unmatched = delta_bfactor_structure(wt, mt)
    assert delta.groupby("residue_number_original")["b_factor"].first()[12] == 0.0
    assert unmatched == [("A", 12, "")]


def test_invalid_frame_is_rejected():
    s = _structure({10: 90.0})
    with pytest.raises(ValueError, match="frame must be"):
        delta_bfactor_structure(s, s, frame="both")


def test_session_writes_pdb_and_script(tmp_path):
    wt = _structure({10: 90.0, 11: 80.0, 12: 70.0})
    mt = _structure({10: 85.0, 11: 80.0, 12: 75.0})

    result = write_pymol_session(wt, mt, str(tmp_path / "demo"), sites=[10],
                                 label="demo A E10K", run_pymol=False)

    assert os.path.exists(result["delta_pdb"])
    assert os.path.exists(result["pml"])
    assert result["pse"] is None          # run_pymol=False
    assert result["max_abs_delta"] == pytest.approx(5.0)

    script = open(result["pml"]).read()
    assert "spectrum b, red_white_blue" in script
    assert "minimum=-5.000, maximum=5.000" in script   # symmetric about zero
    assert "resi 10" in script
    assert script.rstrip().endswith("save demo_delta.pse")
    # the .pml references files by basename, so the directory stays relocatable
    assert "demo_delta.pdb" in script and str(tmp_path) not in script


def test_session_scale_limit_can_be_pinned(tmp_path):
    wt = _structure({10: 90.0})
    mt = _structure({10: 85.0})
    result = write_pymol_session(wt, mt, str(tmp_path / "demo"), sites=[10],
                                 run_pymol=False, limit=20.0)
    assert "minimum=-20.000, maximum=20.000" in open(result["pml"]).read()


def test_written_pdb_carries_the_delta(tmp_path):
    from aftoolkit.structure import load_protein

    wt = _structure({10: 90.0, 11: 80.0})
    mt = _structure({10: 85.0, 11: 80.0})
    result = write_pymol_session(wt, mt, str(tmp_path / "demo"), sites=[10],
                                 run_pymol=False)

    reloaded = load_protein(result["delta_pdb"])
    ca = reloaded[reloaded.atom_name == "CA"]
    assert float(ca[ca.residue_number_original == 10].b_factor.iloc[0]) == \
        pytest.approx(-5.0)


@pytest.mark.skipif(pymol_available() is None, reason="PyMOL not installed")
def test_pymol_actually_produces_a_session(tmp_path):
    wt = _structure({i: 90.0 for i in range(10, 25)})
    mt = _structure({i: 90.0 - (i == 17) * 8.0 for i in range(10, 25)})

    result = write_pymol_session(wt, mt, str(tmp_path / "demo"), sites=[17],
                                 run_pymol=True)
    assert result["pse"] is not None and os.path.exists(result["pse"])
    assert os.path.getsize(result["pse"]) > 0


def test_missing_pymol_is_not_fatal(tmp_path, monkeypatch):
    """Without PyMOL the .pml is still written, so it can be run later."""
    monkeypatch.setattr(shutil, "which", lambda _: None)
    wt = _structure({10: 90.0})
    mt = _structure({10: 85.0})

    result = write_pymol_session(wt, mt, str(tmp_path / "demo"), sites=[10],
                                 run_pymol=True)
    assert result["pse"] is None
    assert os.path.exists(result["pml"])
