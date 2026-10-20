"""Mutation-impact prediction with trained AFToolkit adapters.

Given a structure and one or more point mutations on a single chain, this runs
the modified AlphaFold2 inference on the wildtype and the mutant, predicts the
ddG impact with one or more trained adapters, gathers AlphaFold confidence per
mutated site, and emits:

* a versioned ``<name>.json`` report (predictions, per-site confidence, change); and
* the predicted structures as ``<name>_wt.pdb`` / ``<name>_mt.pdb``.

The report-building logic is free of any PyTorch/OpenFold import, so it is unit
tested without the heavy stack (see ``tests/test_inference_report.py``).
"""

from __future__ import annotations

import dataclasses
import glob
import json
import logging
import os
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from aftoolkit.structure import DEFAULT_CHAIN_GAP

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.2"
DEFAULT_FEATURES = ("pair", "lddt_logits", "plddt")

#: Inference formats, as ``OpenFoldWrapper`` flags. See that class for what each
#: one shows the model.
PIPELINES = {
    "mask":    dict(always_use_template=False, side_chain_mask=False),
    "vanilla": dict(always_use_template=True,  side_chain_mask=False),
    "af2rank": dict(always_use_template=False, side_chain_mask=True),
}

__all__ = [
    "DEFAULT_FEATURES",
    "PIPELINES",
    "SCHEMA_VERSION",
    "Confidence",
    "PredictionReport",
    "confidence_change",
    "confidence_from_output",
    "predict_mutation",
]


# --------------------------------------------------------------------------- #
# Report building (torch-free, unit-tested)
# --------------------------------------------------------------------------- #
@dataclass
class Confidence:
    """AlphaFold confidence for one predicted structure (per mutated site)."""

    mean_plddt: float
    ptm: float | None = None
    per_site_plddt: dict = field(default_factory=dict)   # {"90": 96.3}
    per_site_pae: dict = field(default_factory=dict)     # {"90": 3.4}


def confidence_from_output(of_output: dict, site_indices: dict) -> Confidence:
    """Build :class:`Confidence` from an OpenFold output dict.

    Args:
        of_output: raw model output (``plddt``, optional ``predicted_tm_score``
            and ``predicted_aligned_error``).
        site_indices: ``{residue_number: internal_index}`` for the sites of
            interest.
    """
    plddt = np.asarray(of_output["plddt"]).reshape(-1)
    conf = Confidence(mean_plddt=round(float(plddt.mean()), 3))
    if "predicted_tm_score" in of_output:
        conf.ptm = round(float(np.asarray(of_output["predicted_tm_score"]).reshape(-1)[0]), 4)

    pae = (np.asarray(of_output["predicted_aligned_error"])
           if "predicted_aligned_error" in of_output else None)
    for resnum, idx in site_indices.items():
        conf.per_site_plddt[str(resnum)] = round(float(plddt[idx]), 3)
        if pae is not None:
            conf.per_site_pae[str(resnum)] = round(float(pae[idx].mean()), 3)
    return conf


def confidence_change(mutant: Confidence, wildtype: Confidence) -> dict:
    """Mutant-minus-wildtype change for global and per-site confidence."""
    change = {
        "delta_mean_plddt": round(mutant.mean_plddt - wildtype.mean_plddt, 3),
        "delta_per_site_plddt": {
            k: round(mutant.per_site_plddt[k] - wildtype.per_site_plddt[k], 3)
            for k in wildtype.per_site_plddt if k in mutant.per_site_plddt
        },
        "delta_per_site_pae": {
            k: round(mutant.per_site_pae[k] - wildtype.per_site_pae[k], 3)
            for k in wildtype.per_site_pae if k in mutant.per_site_pae
        },
    }
    if wildtype.ptm is not None and mutant.ptm is not None:
        change["delta_ptm"] = round(mutant.ptm - wildtype.ptm, 4)
    return change


@dataclass
class PredictionReport:
    """The full, serialisable result of a mutation prediction."""

    input: dict
    predicted_ddg: dict
    confidence: dict
    structures: dict
    schema_version: str = SCHEMA_VERSION
    report_json: str | None = None

    def to_json(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)
        self.report_json = path
        return path


def site_index_map(observed: dict) -> dict:
    """Map residue number -> internal index from a ``{residue_key: index}`` dict.

    Residue keys look like ``"K_A_90"`` (``<aa>_<chain>_<resnum>``).
    """
    sites = {}
    for key, idx in observed.items():
        digits = "".join(ch for ch in key.split("_")[-1] if ch.isdigit())
        if digits:
            sites[int(digits)] = idx
    return sites


# --------------------------------------------------------------------------- #
# Orchestration (requires the OpenFold stack)
# --------------------------------------------------------------------------- #
def predict_mutation(
    pdb: str,
    chain: str,
    mutations,
    model_paths: Sequence[str] = (),
    device: str | None = None,
    n_recycle: int = 1,
    out_dir: str = ".",
    features: Sequence[str] = DEFAULT_FEATURES,
    pipeline: str = "mask",
    chain_gap: int = DEFAULT_CHAIN_GAP,
    save_structures: bool = True,
    pymol_session: bool = False,
    wrapper=None,
) -> PredictionReport:
    """Predict the impact of one or more mutations on ``chain``.

    Args:
        pdb: path to the input PDB structure.
        chain: chain identifier the mutations apply to.
        mutations: a single mutation (``"E285K"``) or several, comma-separated
            (``"K90M,N107S,W103F"``) or as a list.
        model_paths: trained adapter ``.pkl`` paths/globs (empty -> confidence only).
        device: ``cuda:0``/``mps``/``cpu``; auto-detected when ``None``.
        n_recycle: AlphaFold recycling iterations (match the model's training).
        out_dir: directory for the JSON report and predicted PDB structures.
        features: feature names the adapter consumes.
        pipeline: one of :data:`PIPELINES`.
        chain_gap: residue-index gap inserted between chains.
        save_structures: write the predicted wildtype and mutant PDB files.
        pymol_session: also write a structure whose B-factor column is the
            mutant-minus-wildtype pLDDT, a PyMOL script colouring it on a
            diverging ramp, and (if PyMOL is installed) a ``.pse`` session.
        wrapper: optional pre-initialised ``OpenFoldWrapper``.

    Returns:
        The :class:`PredictionReport` (also written to ``<out_dir>/<name>.json``).

    Raises:
        FileNotFoundError: if ``pdb`` is missing.
        ValueError: if a mutation is malformed, the pipeline is unknown, or a
            mutation's wildtype residue does not match the structure.
    """
    from aftoolkit.mutations import ObservableRegion, parse_point_mutations
    from aftoolkit.structure import safe_filename_part, save_to_pdb
    from aftoolkit.tasks import ProteinTask

    pdb = os.path.expanduser(pdb)
    if not os.path.exists(pdb):
        raise FileNotFoundError(f"PDB not found: {pdb}")
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown pipeline {pipeline!r}; choose from {list(PIPELINES)}.")

    parsed = parse_point_mutations(mutations, chain)
    if not parsed:
        raise ValueError("No mutations given.")
    label = "_".join(f"{m.wt}{m.position}{m.mt}" for m in parsed)
    name = f"{safe_filename_part(os.path.basename(pdb))}_{chain}_{label}"
    os.makedirs(out_dir, exist_ok=True)

    # Build the task and validate the wildtype residues *before* loading the
    # model, so a typo fails in a second rather than after inference.
    task = ProteinTask()
    task.set_input(pdb, chains=[chain], chain_gap=chain_gap)
    task.set_mutations(parsed)
    for mutation in parsed:
        found = task.wt_residue(mutation.position, chain)
        if found != mutation.wt:
            raise ValueError(
                f"Mutation {mutation} expects wildtype {mutation.wt} at "
                f"{chain}:{mutation.position}, but the structure has {found}."
            )
    task.set_observables(
        [ObservableRegion(chain=chain, start=m.position) for m in parsed]
    )

    if wrapper is None:
        from aftoolkit.folding import OpenFoldWrapper

        wrapper = OpenFoldWrapper(device=device, inference_n_recycle=n_recycle,
                                  **PIPELINES[pipeline])
        wrapper.init_model()

    logger.info("Running wildtype inference for %s", name)
    # The structures are always kept: the confidence-difference map needs both,
    # even when the caller does not want them written out.
    task.evaluate_wildtype(wrapper, store_structure=True)
    wt_conf = confidence_from_output(wrapper.of_output,
                                     site_index_map(task.wildtype.observed))

    logger.info("Running mutant inference for %s", name)
    task.evaluate_mutant(wrapper, store_structure=True)
    mt_conf = confidence_from_output(wrapper.of_output,
                                     site_index_map(task.mutant.observed))

    structures = {}
    if save_structures:
        structures["wildtype_pdb"] = os.path.join(out_dir, f"{name}_wt.pdb")
        structures["mutant_pdb"] = os.path.join(out_dir, f"{name}_mt.pdb")
        save_to_pdb(task.wildtype.folded, structures["wildtype_pdb"])
        save_to_pdb(task.mutant.folded, structures["mutant_pdb"])

    if pymol_session:
        from aftoolkit.visualize import write_pymol_session

        session = write_pymol_session(
            task.wildtype.folded, task.mutant.folded,
            out_prefix=os.path.join(out_dir, name),
            sites=[m.position for m in parsed],
            label=f"{os.path.basename(pdb)} {chain} {label}",
        )
        structures["delta_pdb"] = session["delta_pdb"]
        structures["pymol_script"] = session["pml"]
        structures["pymol_session"] = session["pse"]
        structures["max_abs_delta_plddt"] = round(session["max_abs_delta"], 3)
        logger.info("Confidence-difference map -> %s (max |delta pLDDT| %.1f)",
                    session["pse"] or session["pml"], session["max_abs_delta"])

    predictions = _run_adapters(model_paths, task)
    ensemble = (round(float(np.mean(list(predictions.values()))), 4)
                if predictions else None)

    report = PredictionReport(
        input={
            "pdb": pdb, "chain": chain,
            "mutations": [f"{m.wt}{m.position}{m.mt}" for m in parsed],
            "positions": [m.position for m in parsed],
            "device": wrapper.device, "n_recycle": n_recycle, "pipeline": pipeline,
            "chain_gap": chain_gap, "features": list(features),
        },
        predicted_ddg={
            "per_model": predictions,
            "ensemble_mean": ensemble,
            "units": "kcal/mol (sign per the adapter's training convention; "
                     "multi-mutants = sum of single-mutation effects)",
        },
        confidence={
            "wildtype": dataclasses.asdict(wt_conf),
            "mutant": dataclasses.asdict(mt_conf),
            "change": confidence_change(mt_conf, wt_conf),
        },
        structures=structures,
    )
    path = report.to_json(os.path.join(out_dir, f"{name}.json"))
    logger.info("Wrote report -> %s", path)
    return report


def _run_adapters(model_paths: Sequence[str], task) -> dict:
    """Predict with every adapter matching ``model_paths``, keyed by filename."""
    resolved = [m for pattern in model_paths
                for m in sorted(glob.glob(os.path.expanduser(pattern)))]
    if model_paths and not resolved:
        logger.warning("No adapter models matched %s", list(model_paths))

    predictions = {}
    for path in resolved:
        with open(path, "rb") as f:
            adapter = pickle.load(f)
        predictions[os.path.basename(path)] = round(
            float(np.ravel(adapter.predict(task))[0]), 4
        )
    return predictions
