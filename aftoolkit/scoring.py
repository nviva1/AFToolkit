"""Recompute AFToolkit structure-confidence scores for protein complexes.

Runs the (masked-template, MSA-free) OpenFold pipeline on a multi-chain complex
and reports:

* ``ipae``  -- interface PAE: mean predicted-aligned-error over residue pairs
               spanning two chain groups (e.g. antibody vs antigen). This is the
               score used for antibody-RBD interface quality (Fig. 2g of the
               RBD-AIM paper).
* ``mean_plddt`` / ``ptm`` -- global confidence.

The numeric ``interface_pae`` helper is free of any torch/OpenFold import so it
is unit-tested without the heavy stack.

CLI::

    aftoolkit-score-complex --pdb complex.pdb --antigen E --antibody H,L \\
        --device mps --out-dir results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def interface_pae(pae, chain_labels, group_a, group_b) -> float:
    """Mean predicted-aligned-error between two groups of chains.

    Args:
        pae: (N, N) predicted-aligned-error matrix (residue order matches
            ``chain_labels``).
        chain_labels: length-N sequence of each residue's original chain id.
        group_a, group_b: iterables of chain ids defining the two interface sides.

    Returns:
        Mean PAE over all cross-group residue pairs (both directions).

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
    cross = np.concatenate([pae[np.ix_(a, b)].ravel(), pae[np.ix_(b, a)].ravel()])
    return float(cross.mean())


def score_complex(
    pdb: str,
    antigen_chains: Sequence[str],
    antibody_chains: Sequence[str],
    device: Optional[str] = None,
    n_recycle: int = 1,
    out_dir: Optional[str] = None,
    wrapper=None,
) -> dict:
    """Score a complex structure: interface PAE + global confidence.

    Args:
        pdb: complex structure (multi-chain).
        antigen_chains / antibody_chains: chain ids for each interface side.
        device, n_recycle: OpenFold inference settings.
        out_dir: if given, the predicted structure + JSON report are written here.
        wrapper: optional pre-initialised ``OpenFoldWrapper``.

    Returns:
        dict with ``ipae``, ``mean_plddt``, ``ptm``, ``n_residues`` and inputs.
    """
    from aftoolkit.processing.utils import load_protein, save_to_pdb
    from aftoolkit.processing.openfold_wrapper import OpenFoldWrapper

    pdb = os.path.expanduser(pdb)
    if not os.path.exists(pdb):
        raise FileNotFoundError(f"PDB not found: {pdb}")
    chains = list(antigen_chains) + list(antibody_chains)

    protein_df = load_protein(pdb, chains=chains)
    if wrapper is None:
        wrapper = OpenFoldWrapper(device=device, inference_n_recycle=n_recycle)
        wrapper.init_model()
    wrapper.evaluate(protein_df=protein_df)
    of_output = wrapper.of_output

    # Per-residue original chain id, in the CA order matching the PAE matrix.
    ca = wrapper.get_of_protein()
    ca = ca[ca["atom_name"] == "CA"].reset_index(drop=True)
    chain_labels = ca["chain_id_original"].tolist()

    pae = of_output["predicted_aligned_error"]
    plddt = np.asarray(of_output["plddt"]).reshape(-1)
    result = {
        "input": {"pdb": pdb, "antigen_chains": list(antigen_chains),
                  "antibody_chains": list(antibody_chains),
                  "device": wrapper.device, "n_recycle": n_recycle},
        "ipae": round(interface_pae(pae, chain_labels, antigen_chains, antibody_chains), 4),
        "mean_plddt": round(float(plddt.mean()), 3),
        "ptm": round(float(np.asarray(of_output["predicted_tm_score"]).reshape(-1)[0]), 4)
               if "predicted_tm_score" in of_output else None,
        "n_residues": int(len(chain_labels)),
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


def main() -> None:
    from aftoolkit.logging_config import configure_logging

    p = argparse.ArgumentParser(
        description="Recompute interface PAE (iPAE) + confidence for a complex.")
    p.add_argument("--pdb", required=True)
    p.add_argument("--antigen", required=True, help="Antigen chain id(s), comma-separated.")
    p.add_argument("--antibody", required=True, help="Antibody chain id(s), comma-separated.")
    p.add_argument("--device", default=None)
    p.add_argument("--n-recycle", type=int, default=1)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    configure_logging()
    try:
        result = score_complex(
            pdb=args.pdb,
            antigen_chains=args.antigen.split(","),
            antibody_chains=args.antibody.split(","),
            device=args.device, n_recycle=args.n_recycle, out_dir=args.out_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
