"""Unit tests for the torch-free inference logic (parsing, confidence, report).

These import only ``aftoolkit.inference``'s pure helpers, which carry no
PyTorch/OpenFold dependency, so they run in the default (CI) suite.
"""

import json

import numpy as np
import pytest

from aftoolkit.inference import (
    Confidence,
    PredictionReport,
    confidence_change,
    confidence_from_output,
    parse_point_mutation,
)


def test_parse_point_mutation_valid():
    assert parse_point_mutation("E285K") == ("E", 285, "K")
    assert parse_point_mutation(" A11S ") == ("A", 11, "S")


@pytest.mark.parametrize("bad", ["E285", "285K", "e285k", "EK", "E285KK", ""])
def test_parse_point_mutation_invalid(bad):
    with pytest.raises(ValueError):
        parse_point_mutation(bad)


def test_confidence_from_output():
    of = {
        "plddt": np.array([90.0, 80.0, 70.0]),
        "predicted_tm_score": np.array([0.85]),
        "predicted_aligned_error": np.array([[1.0, 2.0, 3.0],
                                             [2.0, 1.0, 2.0],
                                             [3.0, 2.0, 1.0]]),
    }
    conf = confidence_from_output(of, site_idx=0)
    assert conf.mean_plddt == pytest.approx(80.0)
    assert conf.plddt_at_site == pytest.approx(90.0)
    assert conf.ptm == pytest.approx(0.85)
    assert conf.mean_pae_at_site == pytest.approx(2.0)  # mean of row 0


def test_confidence_from_output_no_site():
    conf = confidence_from_output({"plddt": np.array([50.0, 60.0])}, site_idx=None)
    assert conf.plddt_at_site is None
    assert conf.ptm is None


def test_confidence_change():
    wt = Confidence(mean_plddt=88.0, plddt_at_site=95.0, ptm=0.86, mean_pae_at_site=3.0)
    mt = Confidence(mean_plddt=88.0, plddt_at_site=92.0, ptm=0.86, mean_pae_at_site=4.0)
    change = confidence_change(mt, wt)
    assert change["delta_plddt_at_site"] == pytest.approx(-3.0)
    assert change["delta_mean_pae_at_site"] == pytest.approx(1.0)
    assert change["delta_ptm"] == pytest.approx(0.0)


def test_prediction_report_serialization(tmp_path):
    report = PredictionReport(
        input={"mutation": "E285K"},
        predicted_ddg={"per_model": {"svm": -0.5}, "ensemble_mean": -0.5},
        confidence={"wildtype": {}, "mutant": {}, "change": {}},
        structures={"wildtype_pdb": "wt.pdb", "mutant_pdb": "mt.pdb"},
    )
    out = tmp_path / "report.json"
    report.to_json(str(out))
    loaded = json.loads(out.read_text())
    assert loaded["schema_version"] == "1.0"
    assert loaded["predicted_ddg"]["ensemble_mean"] == -0.5
    assert loaded["input"]["mutation"] == "E285K"
