"""Recompute AFToolkit structure-confidence scores for protein complexes.

Runs the (masked-template, MSA-free) OpenFold pipeline on a multi-chain complex
and reports:

* ``ipae``  -- interface PAE: mean predicted-aligned-error over residue pairs
               spanning two chain groups (e.g. antibody vs antigen). This is the
               score used for antibody-RBD interface quality (Fig. 2g of the
               RBD-AIM paper).
* ``mean_plddt`` / ``ptm`` -- global confidence.

The numeric ``interface_pae`` / ``interface_iptm`` helpers are free of any
torch/OpenFold import, so they are unit-tested without the heavy stack.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence

import numpy as np

from aftoolkit.inference import PIPELINES
from aftoolkit.structure import DEFAULT_CHAIN_GAP

logger = logging.getLogger(__name__)

__all__ = ["interface_iptm", "interface_pae", "score_complex"]


def interface_pae(pae, chain_labels, group_a, group_b, coords=None, cutoff=None) -> float:
    """Interface predicted-aligned-error between two groups of chains.

    Args:
        pae: (N, N) predicted-aligned-error matrix (residue order matches
            ``chain_labels``).
        chain_labels: length-N sequence of each residue's original chain id.
        group_a, group_b: iterables of chain ids defining the two interface sides.
        coords: optional (N, 3) CA coordinates. When given with ``cutoff``, the
            mean is restricted to cross-group residue pairs in **contact** (CA-CA
            distance < ``cutoff``) -- the true interface PAE. Without it, the mean
            spans all cross-group pairs.
        cutoff: contact distance in Angstrom (e.g. 12.0).

    Returns:
        Mean PAE over the selected cross-group residue pairs (both directions).

    Raises:
        ValueError: if either group is empty in ``chain_labels``.
    """
    pae = np.asarray(pae, dtype=float)
    labels = np.asarray(chain_labels)
    a = np.isin(labels, list(group_a))
    b = np.isin(labels, list(group_b))
    if not a.any() or not b.any():
        raise ValueError(
            f"Empty interface side: |group_a|={int(a.sum())}, |group_b|={int(b.sum())}."
        )
    pab, pba = pae[np.ix_(a, b)], pae[np.ix_(b, a)]
    if coords is not None and cutoff is not None:
        coords = np.asarray(coords, dtype=float)
        d = np.linalg.norm(coords[a][:, None, :] - coords[b][None, :, :], axis=-1)
        mask = d < cutoff
        if mask.any():
            return float(np.concatenate([pab[mask], pba.T[mask]]).mean())
        return float(min(pab.min(), pba.min()))  # no contacts -> best-case PAE
    return float(np.concatenate([pab.ravel(), pba.ravel()]).mean())


def interface_iptm(pae, chain_labels, group_a, group_b) -> float:
    """Interface pTM (ipTM) between two chain groups, derived from the PAE.

    pTM is computed from the predicted aligned error via the TM-score transform
    ``f(e) = 1 / (1 + (e/d0)^2)`` with ``d0 = 1.24*(N-15)^(1/3) - 1.8``. For the
    *interface* version, each residue is scored only against residues of the
    other group, and ipTM is the best (max) such per-residue score -- higher is
    a more confident interface (range ~0-1).
    """
    pae = np.asarray(pae, dtype=float)
    labels = np.asarray(chain_labels)
    a = np.isin(labels, list(group_a))
    b = np.isin(labels, list(group_b))
    if not a.any() or not b.any():
        raise ValueError("interface_iptm: an interface side is empty.")
    n = len(labels)
    d0 = max(1.0, 1.24 * (n - 15) ** (1.0 / 3.0) - 1.8)
    tm = 1.0 / (1.0 + (pae / d0) ** 2)
    best = 0.0
    for gi, gj in [(a, b), (b, a)]:
        per_i = tm[np.ix_(gi, gj)].mean(axis=1)  # each i averaged over other-group j
        best = max(best, float(per_i.max()))
    return best


def score_complex(
    pdb: str,
    antigen_chains: Sequence[str],
    antibody_chains: Sequence[str],
    device: str | None = None,
    n_recycle: int = 1,
    out_dir: str | None = None,
    chain_gap: int = DEFAULT_CHAIN_GAP,
    pipeline: str = "mask",
    contact_cutoff: float | None = None,
    wrapper=None,
) -> dict:
    """Score a complex structure: interface PAE + global confidence.

    Args:
        pdb: complex structure (multi-chain).
        antigen_chains / antibody_chains: chain ids for each interface side.
        device, n_recycle: OpenFold inference settings.
        out_dir: if given, the predicted structure + JSON report are written here.
        chain_gap: residue-index gap inserted between chains. The default
            saturates AlphaFold's relative-position window; published protocols
            may specify another value (the RBD-AIM paper uses 25).
        pipeline: one of :data:`aftoolkit.inference.PIPELINES`.
        contact_cutoff: when given, iPAE is restricted to cross-group residue
            pairs whose CA atoms are within this distance, in Angstrom.
        wrapper: optional pre-initialised ``OpenFoldWrapper``.

    Returns:
        dict with ``ipae``, ``iptm``, ``mean_plddt``, ``ptm``, ``n_residues``
        and the inputs used.

    Raises:
        FileNotFoundError: if ``pdb`` is missing.
        ValueError: if ``pipeline`` is unknown.
    """
    from aftoolkit.folding import OpenFoldWrapper
    from aftoolkit.structure import load_protein, save_to_pdb

    pdb = os.path.expanduser(pdb)
    if not os.path.exists(pdb):
        raise FileNotFoundError(f"PDB not found: {pdb}")
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown pipeline {pipeline!r}; choose from {list(PIPELINES)}.")
    chains = list(antigen_chains) + list(antibody_chains)

    protein_df = load_protein(pdb, chains=chains, chain_gap=chain_gap)
    if wrapper is None:
        wrapper = OpenFoldWrapper(device=device, inference_n_recycle=n_recycle,
                                  **PIPELINES[pipeline])
        wrapper.init_model()
    wrapper.evaluate(protein_df=protein_df)
    of_output = wrapper.of_output

    # Per-residue original chain id, in the CA order matching the PAE matrix.
    ca = wrapper.get_of_protein()
    ca = ca[ca["atom_name"] == "CA"].reset_index(drop=True)
    chain_labels = ca["chain_id_original"].tolist()

    pae = of_output["predicted_aligned_error"]
    plddt = np.asarray(of_output["plddt"]).reshape(-1)
    coords = ca[["x_coord", "y_coord", "z_coord"]].to_numpy() if contact_cutoff else None
    result = {
        "input": {"pdb": pdb, "antigen_chains": list(antigen_chains),
                  "antibody_chains": list(antibody_chains),
                  "device": wrapper.device, "n_recycle": n_recycle,
                  "chain_gap": chain_gap, "pipeline": pipeline,
                  "contact_cutoff": contact_cutoff},
        "ipae": round(interface_pae(pae, chain_labels, antigen_chains, antibody_chains,
                                    coords=coords, cutoff=contact_cutoff), 4),
        "iptm": round(interface_iptm(pae, chain_labels, antigen_chains, antibody_chains), 4),
        "mean_plddt": round(float(plddt.mean()), 3),
        "ptm": round(float(np.asarray(of_output["predicted_tm_score"]).reshape(-1)[0]), 4)
               if "predicted_tm_score" in of_output else None,
        "n_residues": len(chain_labels),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(pdb))[0]
        save_to_pdb(wrapper.get_of_protein(), os.path.join(out_dir, f"{base}_pred.pdb"))
        with open(os.path.join(out_dir, f"{base}_score.json"), "w") as f:
            json.dump(result, f, indent=2)
    logger.info("iPAE=%.3f  mean_pLDDT=%.1f  pTM=%s  (%s)",
                result["ipae"], result["mean_plddt"], result["ptm"], os.path.basename(pdb))
    return result
