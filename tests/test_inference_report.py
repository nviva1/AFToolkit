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
    site_index_map,
)


def test_site_index_map_reads_residue_keys():
    assert site_index_map({"K_A_90": 0, "N_A_107": 12}) == {90: 0, 107: 12}


def test_site_index_map_skips_unnumbered_keys():
    assert site_index_map({"weird": 3}) == {}


def test_confidence_from_output():
    of = {
        "plddt": np.array([90.0, 80.0, 70.0]),
        "predicted_tm_score": np.array([0.85]),
        "predicted_aligned_error": np.array([[1.0, 2.0, 3.0],
                                             [2.0, 1.0, 2.0],
                                             [3.0, 2.0, 1.0]]),
    }
    conf = confidence_from_output(of, site_indices={90: 0})
    assert conf.mean_plddt == pytest.approx(80.0)
    assert conf.per_site_plddt["90"] == pytest.approx(90.0)
    assert conf.ptm == pytest.approx(0.85)
    assert conf.per_site_pae["90"] == pytest.approx(2.0)  # mean of row 0


def test_confidence_from_output_multi_site():
    of = {"plddt": np.array([90.0, 80.0, 70.0])}
    conf = confidence_from_output(of, site_indices={90: 0, 107: 2})
    assert conf.per_site_plddt == {"90": 90.0, "107": 70.0}
    assert conf.ptm is None
    assert conf.per_site_pae == {}


def test_confidence_change():
    wt = Confidence(mean_plddt=88.0, ptm=0.86,
                    per_site_plddt={"90": 95.0}, per_site_pae={"90": 3.0})
    mt = Confidence(mean_plddt=88.0, ptm=0.86,
                    per_site_plddt={"90": 92.0}, per_site_pae={"90": 4.0})
    change = confidence_change(mt, wt)
    assert change["delta_per_site_plddt"]["90"] == pytest.approx(-3.0)
    assert change["delta_per_site_pae"]["90"] == pytest.approx(1.0)
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
    assert loaded["schema_version"] == "1.2"
    assert loaded["predicted_ddg"]["ensemble_mean"] == -0.5
    assert loaded["input"]["mutation"] == "E285K"
