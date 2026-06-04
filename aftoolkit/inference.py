"""Mutation-impact inference for trained AFToolkit adapters.

Given a structure and one or more point mutations (on a single chain), this runs
the modified AlphaFold2 inference on the wildtype and mutant, predicts the ddG
impact with one or more trained adapter models, gathers AlphaFold confidence
scores (pLDDT, pTM, PAE) per mutated site, and emits:

* a versioned, structured ``<name>.json`` report (predictions, per-site
  confidence, change); and
* the predicted wildtype and mutant structures as ``<name>_wt.pdb`` /
  ``<name>_mt.pdb``.

The numeric/serialisation logic is free of any PyTorch/OpenFold imports so it can
be unit-tested without the heavy stack (see ``tests/test_inference_report.py``).

CLI::

    aftoolkit-predict --pdb 5BPJ.pdb --chain A --mutations K90M,N107S,W103F \\
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

SCHEMA_VERSION = "1.1"
DEFAULT_FEATURES = ("pair", "lddt_logits", "plddt")
_MUTATION_RE = re.compile(r"^(?P<wt>[A-Z])(?P<pos>\d+)(?P<mt>[A-Z])$")


# --------------------------------------------------------------------------- #
# Pure, torch-free logic (unit-tested)
# --------------------------------------------------------------------------- #
def parse_point_mutation(mutation: str) -> tuple[str, int, str]:
    """Parse a single point mutation ``<WT><pos><MT>`` (e.g. ``E285K``).

    Returns ``(wt_aa, position, mt_aa)``; raises ``ValueError`` if malformed.
    """
    m = _MUTATION_RE.match(mutation.strip())
    if not m:
        raise ValueError(
            f"Invalid mutation {mutation!r}; expected <WT><pos><MT>, e.g. 'E285K'."
        )
    return m["wt"], int(m["pos"]), m["mt"]


def parse_mutations_list(mutations) -> list[tuple[str, int, str]]:
    """Parse one or more comma-separated point mutations into ``(wt, pos, mt)``."""
    if isinstance(mutations, str):
        mutations = [m for m in mutations.split(",") if m.strip()]
    return [parse_point_mutation(m) for m in mutations]


@dataclass
class Confidence:
    """AlphaFold confidence for one predicted structure (per mutated site)."""

    mean_plddt: float
    ptm: Optional[float] = None
    per_site_plddt: dict = field(default_factory=dict)   # {"90": 96.3}
    per_site_pae: dict = field(default_factory=dict)     # {"90": 3.4}


def confidence_from_output(of_output: dict, site_indices: dict) -> Confidence:
    """Build :class:`Confidence` from an OpenFold output dict.

    Args:
        of_output: raw model output (``plddt``, optional ``predicted_tm_score``
            and ``predicted_aligned_error``).
        site_indices: mapping ``{residue_number: internal_index}`` for the sites
            of interest.
    """
    plddt = np.asarray(of_output["plddt"]).reshape(-1)
    conf = Confidence(mean_plddt=round(float(plddt.mean()), 3))
    if "predicted_tm_score" in of_output:
        conf.ptm = round(float(np.asarray(of_output["predicted_tm_score"]).reshape(-1)[0]), 4)
    pae = np.asarray(of_output["predicted_aligned_error"]) if "predicted_aligned_error" in of_output else None
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
    report_json: Optional[str] = None

    def to_json(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)
        self.report_json = path
        return path


def _site_index_map(obs_positions: dict) -> dict:
    """Map residue number -> internal index from a ``{pos_name: idx}`` dict.

    Position names look like ``"K_A_90"`` (``<aa>_<chain>_<resnum>``).
    """
    out = {}
    for name, idx in obs_positions.items():
        digits = "".join(ch for ch in name.split("_")[-1] if ch.isdigit())
        if digits:
            out[int(digits)] = idx
    return out


# --------------------------------------------------------------------------- #
# Orchestration (requires the OpenFold stack)
# --------------------------------------------------------------------------- #
def predict_mutation(
    pdb: str,
    chain: str,
    mutations,
    model_paths: Sequence[str] = (),
    device: Optional[str] = None,
    n_recycle: int = 1,
    out_dir: str = ".",
    features: Sequence[str] = DEFAULT_FEATURES,
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
        wrapper: optional pre-initialised ``OpenFoldWrapper``.

    Returns:
        The :class:`PredictionReport` (also written to ``<out_dir>/<name>.json``).

    Raises:
        FileNotFoundError: if ``pdb`` is missing.
        ValueError: if any mutation is malformed or its WT residue does not match
            the structure.
    """
    from aftoolkit.processing.protein_task import ProteinTask
    from aftoolkit.processing.arg_parser import parse_mutations, parse_positions
    from aftoolkit.processing.utils import save_to_pdb, safe_filename_part

    pdb = os.path.expanduser(pdb)
    if not os.path.exists(pdb):
        raise FileNotFoundError(f"PDB not found: {pdb}")
    parsed = parse_mutations_list(mutations)
    mut_strs = [f"{wt}{pos}{mt}" for wt, pos, mt in parsed]
    os.makedirs(out_dir, exist_ok=True)
    mut_spec = ",".join(f"{chain}:{m}" for m in mut_strs)
    pos_spec = ",".join(f"{chain}:{pos}" for _, pos, _ in parsed)
    label = "_".join(mut_strs)
    name = f"{safe_filename_part(os.path.basename(pdb))}_{chain}_{label}"

    # Build task and validate WT residues *before* loading the model (fail fast).
    task = ProteinTask()
    task.set_input_protein_task(pdb, chains=[chain])
    task.set_task_mutants(parse_mutations(mut_spec))
    task.set_wildtype_protein()
    for wt, pos, _ in parsed:
        found = task.get_wt_aa(pos, chain)
        if found != wt:
            raise ValueError(
                f"Mutation {wt}{pos}? expects wildtype {wt} at {chain}:{pos}, "
                f"but the structure has {found}."
            )
    task.set_observable_positions(parse_positions(pos_spec))

    if wrapper is None:
        from aftoolkit.processing.openfold_wrapper import OpenFoldWrapper
        wrapper = OpenFoldWrapper(device=device, inference_n_recycle=n_recycle)
        wrapper.init_model()

    logger.info("Running wildtype inference for %s", name)
    task.evaluate_wildtype_protein(wrapper, store_of_protein=True)
    wt_sites = _site_index_map(task.get_wildtype_observable_positions())
    wt_conf = confidence_from_output(wrapper.of_output, wt_sites)
    wt_pdb = os.path.join(out_dir, f"{name}_wt.pdb")
    save_to_pdb(task.get_wildtype_protein_of(), wt_pdb)

    logger.info("Running mutant inference for %s", name)
    task.evaluate_mutate_protein(wrapper, store_of_protein=True)
    mt_sites = _site_index_map(task.get_mutate_observable_positions())
    mt_conf = confidence_from_output(wrapper.of_output, mt_sites)
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
            "pdb": pdb, "chain": chain, "mutations": mut_strs, "positions": [p for _, p, _ in parsed],
            "device": wrapper.device, "n_recycle": n_recycle, "features": list(features),
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
    p.add_argument("--chain", required=True, help="Chain the mutation(s) apply to.")
    p.add_argument("--mutations", "--mutation", dest="mutations", required=True,
                   help="One mutation (E285K) or several, comma-separated "
                        "(K90M,N107S,W103F).")
    p.add_argument("--models", nargs="*", default=[],
                   help="Trained adapter .pkl paths/globs (omit for confidence only).")
    p.add_argument("--out-dir", default="results", help="Output directory.")
    p.add_argument("--device", default=None, help="cuda:0 / mps / cpu (auto if unset).")
    p.add_argument("--n-recycle", type=int, default=1)
    args = p.parse_args()

    configure_logging()
    try:
        report = predict_mutation(
            pdb=args.pdb, chain=args.chain, mutations=args.mutations,
            model_paths=args.models, device=args.device, n_recycle=args.n_recycle,
            out_dir=args.out_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")
    print(json.dumps(dataclasses.asdict(report), indent=2))


if __name__ == "__main__":
    main()
