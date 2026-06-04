"""Mutation-impact inference for trained AFToolkit adapters.

Given a structure and a point mutation, this runs the modified AlphaFold2
inference on the wildtype and mutant, predicts the ddG impact with one or more
trained adapter models, gathers AlphaFold confidence scores (pLDDT, pTM, PAE),
and emits:

* a versioned, structured ``<name>.json`` report (predictions, confidence,
  change); and
* the predicted wildtype and mutant structures as ``<name>_wt.pdb`` /
  ``<name>_mt.pdb``.

The numeric/serialisation logic is kept free of any PyTorch/OpenFold imports so
it can be unit-tested without the heavy stack (see ``tests/test_inference.py``).

CLI::

    aftoolkit-predict --pdb p53.pdb --chain A --mutation E285K \\
        --models "data/models/.../trained_*.pkl" --out-dir results --device mps
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import logging
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
DEFAULT_FEATURES = ("pair", "lddt_logits", "plddt")
_MUTATION_RE = re.compile(r"^(?P<wt>[A-Z])(?P<pos>\d+)(?P<mt>[A-Z])$")


# --------------------------------------------------------------------------- #
# Pure, torch-free logic (unit-tested)
# --------------------------------------------------------------------------- #
def parse_point_mutation(mutation: str) -> tuple[str, int, str]:
    """Parse a point mutation string ``<WT><pos><MT>`` (e.g. ``E285K``).

    Returns:
        ``(wt_aa, position, mt_aa)``.

    Raises:
        ValueError: if the string is not a valid single-residue substitution.
    """
    m = _MUTATION_RE.match(mutation.strip())
    if not m:
        raise ValueError(
            f"Invalid mutation {mutation!r}; expected <WT><pos><MT>, e.g. 'E285K'."
        )
    return m["wt"], int(m["pos"]), m["mt"]


@dataclass
class Confidence:
    """AlphaFold confidence scores for a single predicted structure."""

    mean_plddt: float
    plddt_at_site: Optional[float]
    ptm: Optional[float] = None
    mean_pae_at_site: Optional[float] = None


def confidence_from_output(of_output: dict, site_idx: Optional[int]) -> Confidence:
    """Build a :class:`Confidence` from a raw OpenFold output dict."""
    plddt = np.asarray(of_output["plddt"]).reshape(-1)
    conf = Confidence(
        mean_plddt=round(float(plddt.mean()), 3),
        plddt_at_site=round(float(plddt[site_idx]), 3) if site_idx is not None else None,
    )
    if "predicted_tm_score" in of_output:
        conf.ptm = round(float(np.asarray(of_output["predicted_tm_score"]).reshape(-1)[0]), 4)
    if "predicted_aligned_error" in of_output and site_idx is not None:
        pae = np.asarray(of_output["predicted_aligned_error"])
        conf.mean_pae_at_site = round(float(pae[site_idx].mean()), 3)
    return conf


def confidence_change(mutant: Confidence, wildtype: Confidence) -> dict:
    """Mutant-minus-wildtype change for each shared numeric confidence field."""
    out = {}
    for f in dataclasses.fields(Confidence):
        wt_v, mt_v = getattr(wildtype, f.name), getattr(mutant, f.name)
        if isinstance(wt_v, (int, float)) and isinstance(mt_v, (int, float)):
            out[f"delta_{f.name}"] = round(mt_v - wt_v, 3)
    return out


@dataclass
class PredictionReport:
    """The full, serialisable result of a single mutation prediction."""

    input: dict
    predicted_ddg: dict
    confidence: dict
    structures: dict
    schema_version: str = SCHEMA_VERSION
    report_json: Optional[str] = field(default=None)

    def to_json(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)
        self.report_json = path
        return path


# --------------------------------------------------------------------------- #
# Orchestration (requires the OpenFold stack)
# --------------------------------------------------------------------------- #
def predict_mutation(
    pdb: str,
    chain: str,
    mutation: str,
    model_paths: Sequence[str] = (),
    device: Optional[str] = None,
    n_recycle: int = 1,
    out_dir: str = ".",
    features: Sequence[str] = DEFAULT_FEATURES,
    wrapper=None,
) -> PredictionReport:
    """Predict the impact of ``chain:mutation`` and write a JSON report + PDBs.

    Args:
        pdb: path to the input PDB structure.
        chain: chain identifier the mutation applies to.
        mutation: point mutation as ``<WT><pos><MT>`` (e.g. ``E285K``).
        model_paths: trained adapter ``.pkl`` paths/globs (empty -> confidence only).
        device: ``cuda:0``/``mps``/``cpu``; auto-detected when ``None``.
        n_recycle: number of AlphaFold recycling iterations (match the model's training).
        out_dir: directory for the JSON report and predicted PDB structures.
        features: feature names the adapter consumes.
        wrapper: optional pre-initialised ``OpenFoldWrapper`` (else one is created).

    Returns:
        The :class:`PredictionReport` (also written to ``<out_dir>/<name>.json``).

    Raises:
        FileNotFoundError: if ``pdb`` does not exist.
        ValueError: if ``mutation`` is malformed or its WT residue does not match
            the residue found at that position in the structure.
    """
    from aftoolkit.processing.protein_task import ProteinTask
    from aftoolkit.processing.arg_parser import parse_mutations, parse_positions
    from aftoolkit.processing.utils import save_to_pdb, safe_filename_part

    pdb = os.path.expanduser(pdb)
    if not os.path.exists(pdb):
        raise FileNotFoundError(f"PDB not found: {pdb}")
    wt_aa, pos, mt_aa = parse_point_mutation(mutation)
    os.makedirs(out_dir, exist_ok=True)
    mut_spec = f"{chain}:{mutation}"
    name = f"{safe_filename_part(os.path.basename(pdb))}_{chain}_{mutation}"

    # Build the task and validate the declared WT residue against the structure
    # *before* loading the (expensive) model, so bad input fails fast.
    task = ProteinTask()
    task.set_input_protein_task(pdb, chains=[chain])
    task.set_task_mutants(parse_mutations(mut_spec))
    task.set_wildtype_protein()
    found = task.get_wt_aa(pos, chain)
    if found != wt_aa:
        raise ValueError(
            f"Mutation {mutation} expects wildtype {wt_aa} at {chain}:{pos}, "
            f"but the structure has {found}."
        )
    task.set_observable_positions(parse_positions(f"{chain}:{pos}"))

    if wrapper is None:
        from aftoolkit.processing.openfold_wrapper import OpenFoldWrapper
        wrapper = OpenFoldWrapper(device=device, inference_n_recycle=n_recycle)
        wrapper.init_model()

    # WT and mutant evaluated separately so each structure's confidence
    # (the wrapper retains only the latest output) is captured.
    logger.info("Running wildtype inference for %s", name)
    task.evaluate_wildtype_protein(wrapper, store_of_protein=True)
    wt_idx = next(iter(task.get_wildtype_observable_positions().values()), None)
    wt_conf = confidence_from_output(wrapper.of_output, wt_idx)
    wt_pdb = os.path.join(out_dir, f"{name}_wt.pdb")
    save_to_pdb(task.get_wildtype_protein_of(), wt_pdb)

    logger.info("Running mutant inference for %s", name)
    task.evaluate_mutate_protein(wrapper, store_of_protein=True)
    mt_idx = next(iter(task.get_mutate_observable_positions().values()), None)
    mt_conf = confidence_from_output(wrapper.of_output, mt_idx)
    mt_pdb = os.path.join(out_dir, f"{name}_mt.pdb")
    save_to_pdb(task.get_mutate_protein_of(), mt_pdb)

    predictions = {}
    resolved = [m for pat in model_paths for m in sorted(glob.glob(os.path.expanduser(pat)))]
    if model_paths and not resolved:
        logger.warning("No adapter models matched %s", list(model_paths))
    for mp in resolved:
        with open(mp, "rb") as f:
            adapter = pickle.load(f)
        predictions[os.path.basename(mp)] = round(float(np.ravel(adapter.predict(task))[0]), 4)
    ensemble = round(float(np.mean(list(predictions.values()))), 4) if predictions else None

    report = PredictionReport(
        input={
            "pdb": pdb, "chain": chain, "mutation": mutation, "position": pos,
            "device": wrapper.device, "n_recycle": n_recycle, "features": list(features),
        },
        predicted_ddg={
            "per_model": predictions,
            "ensemble_mean": ensemble,
            "units": "kcal/mol (sign per the adapter's training convention)",
        },
        confidence={
            "wildtype": dataclasses.asdict(wt_conf),
            "mutant": dataclasses.asdict(mt_conf),
            "change": confidence_change(mt_conf, wt_conf),
        },
        structures={"wildtype_pdb": wt_pdb, "mutant_pdb": mt_pdb},
    )
    path = report.to_json(os.path.join(out_dir, f"{name}.json"))
    logger.info("Wrote report -> %s", path)
    return report


def main() -> None:
    from aftoolkit.logging_config import configure_logging

    p = argparse.ArgumentParser(
        description="Predict mutation impact (ddG + confidence), writing a JSON "
                    "report and predicted WT/mutant PDB structures.",
    )
    p.add_argument("--pdb", required=True, help="Input PDB structure.")
    p.add_argument("--chain", required=True, help="Chain the mutation applies to.")
    p.add_argument("--mutation", required=True, help="Point mutation, e.g. E285K.")
    p.add_argument("--models", nargs="*", default=[],
                   help="Trained adapter .pkl paths/globs (omit for confidence only).")
    p.add_argument("--out-dir", default="results", help="Output directory.")
    p.add_argument("--device", default=None, help="cuda:0 / mps / cpu (auto if unset).")
    p.add_argument("--n-recycle", type=int, default=1)
    args = p.parse_args()

    configure_logging()
    try:
        report = predict_mutation(
            pdb=args.pdb, chain=args.chain, mutation=args.mutation,
            model_paths=args.models, device=args.device, n_recycle=args.n_recycle,
            out_dir=args.out_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        # Expected user-input errors: report cleanly, no stack trace.
        raise SystemExit(f"error: {exc}")
    print(json.dumps(dataclasses.asdict(report), indent=2))


if __name__ == "__main__":
    main()
