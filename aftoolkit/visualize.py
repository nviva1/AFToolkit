"""Visualising a mutation's effect as a structure coloured by confidence change.

AlphaFold writes per-residue pLDDT into the B-factor column of the structures it
predicts. Subtracting the wildtype's from the mutant's therefore gives, for free,
a per-residue map of *where the model became less sure* when the mutation was
introduced -- which is the spatial read-out of a mutation's effect that a single
ddG number cannot give.

:func:`delta_bfactor_structure` builds that map, and :func:`write_pymol_session`
writes it out as a PDB plus a PyMOL script (and a ``.pse`` session when PyMOL is
installed), coloured on a diverging ramp centred at zero: red where confidence
was lost, blue where it was gained.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Iterable, Sequence

__all__ = [
    "delta_bfactor_structure",
    "pymol_available",
    "write_pymol_session",
]

logger = logging.getLogger(__name__)

#: Column triple identifying a residue across the wildtype/mutant pair.
_RESIDUE_KEY = ["chain_id_original", "residue_number_original", "insertion"]


def pymol_available() -> str | None:
    """Path to the ``pymol`` executable, or ``None`` if it is not installed."""
    return shutil.which("pymol")


def _residue_bfactors(df) -> dict:
    """``{(chain, resnum, insertion): pLDDT}`` from a predicted structure."""
    ca = df[df["atom_name"] == "CA"]
    return {
        (row.chain_id_original, row.residue_number_original, row.insertion): row.b_factor
        for row in ca.itertuples()
    }


def delta_bfactor_structure(wildtype_df, mutant_df, frame: str = "mutant"):
    """Structure whose B-factor column holds the mutant-minus-wildtype pLDDT.

    Residues are matched on their *original* numbering, so the map survives the
    renumbering AlphaFold inference goes through. Residues present in only one of
    the two structures -- the residues an insertion adds, or a deletion removes --
    have no counterpart to difference against and are set to 0.

    Args:
        wildtype_df / mutant_df: predicted protein dataframes (as produced by
            ``ProteinTask.wildtype.folded`` / ``.mutant.folded``).
        frame: which structure to write the values onto, ``"mutant"`` or
            ``"wildtype"``.

    Returns:
        ``(structure, unmatched)`` -- a copy of the chosen frame with ``b_factor``
        replaced by the delta, and the list of residue keys that had no
        counterpart.

    Raises:
        ValueError: if ``frame`` is not one of the two structures.
    """
    if frame not in ("mutant", "wildtype"):
        raise ValueError(f"frame must be 'mutant' or 'wildtype', got {frame!r}.")

    wt_plddt = _residue_bfactors(wildtype_df)
    mt_plddt = _residue_bfactors(mutant_df)

    target = (mutant_df if frame == "mutant" else wildtype_df).copy()
    deltas, unmatched = [], []
    for row in target.itertuples():
        key = (row.chain_id_original, row.residue_number_original, row.insertion)
        if key in wt_plddt and key in mt_plddt:
            deltas.append(mt_plddt[key] - wt_plddt[key])
        else:
            deltas.append(0.0)
            if key not in unmatched:
                unmatched.append(key)

    target["b_factor"] = deltas
    return target, unmatched


def _selection_for(sites: Iterable) -> str:
    """A PyMOL residue selection for the mutated positions."""
    resi = "+".join(str(s) for s in sites)
    return f"resi {resi}" if resi else "none"


def write_pymol_session(
    wildtype_df,
    mutant_df,
    out_prefix: str,
    sites: Sequence = (),
    label: str = "",
    frame: str = "mutant",
    run_pymol: bool = True,
    limit: float | None = None,
) -> dict:
    """Write the delta-coloured structure, a PyMOL script, and a ``.pse`` session.

    Args:
        wildtype_df / mutant_df: predicted structures.
        out_prefix: path prefix; ``<prefix>_delta.pdb`` / ``.pml`` / ``.pse``.
        sites: mutated residue numbers, highlighted as sticks.
        label: short text for the session's title.
        frame: write the deltas onto the mutant or the wildtype structure.
        run_pymol: also run PyMOL to produce the ``.pse`` (skipped when PyMOL is
            not installed -- the ``.pml`` is still written and can be run later).
        limit: symmetric colour-scale limit; the largest absolute delta when
            ``None``.

    Returns:
        ``{"delta_pdb", "pml", "pse", "max_abs_delta", "unmatched"}``; ``pse`` is
        ``None`` when PyMOL did not run.
    """
    from aftoolkit.structure import save_to_pdb

    delta, unmatched = delta_bfactor_structure(wildtype_df, mutant_df, frame=frame)

    delta_pdb = f"{out_prefix}_delta.pdb"
    save_to_pdb(delta, delta_pdb)

    max_abs = float(delta["b_factor"].abs().max())
    scale = limit if limit is not None else max(max_abs, 1e-6)

    pml_path = f"{out_prefix}_delta.pml"
    pse_path = f"{out_prefix}_delta.pse"
    selection = _selection_for(sites)
    title = label or os.path.basename(out_prefix)

    pml = f"""\
# AFToolkit: {title}
# B-factor column holds mutant-minus-wildtype pLDDT, written onto the
# {frame} structure. Red = confidence lost on mutation, blue = gained.
# Residues with no counterpart across the pair (indels) are 0.

load {os.path.basename(delta_pdb)}, delta
hide everything
bg_color white
show cartoon, delta
set cartoon_transparency, 0.15, delta

# diverging ramp, symmetric about zero so the midpoint is "no change"
spectrum b, red_white_blue, delta and polymer, minimum=-{scale:.3f}, maximum={scale:.3f}

# the mutated site(s)
show sticks, delta and ({selection}) and not hydrogens
color yellow, delta and ({selection}) and elem C
set stick_radius, 0.18
label delta and ({selection}) and name CA, "%s%s" % (resn, resi)
set label_size, 16
set label_color, black

# Whole domain by default: the point is to see how far the effect spreads
# from the mutated site. Zoom in with the line below.
orient delta
deselect
# zoom delta and ({selection}), 8
set ray_opaque_background, 0

# colour scale: -{scale:.2f} (red) .. 0 (white) .. +{scale:.2f} (blue) pLDDT
save {os.path.basename(pse_path)}
"""

    with open(pml_path, "w") as f:
        f.write(pml)

    produced_pse = None
    if run_pymol:
        binary = pymol_available()
        if binary is None:
            logger.warning("PyMOL not found on PATH; wrote %s but no .pse. "
                           "Run it later with: pymol -cq %s", pml_path, pml_path)
        else:
            try:
                subprocess.run([binary, "-cq", os.path.basename(pml_path)],
                               cwd=os.path.dirname(os.path.abspath(pml_path)) or ".",
                               check=True, capture_output=True, text=True, timeout=300)
                produced_pse = pse_path if os.path.exists(pse_path) else None
                if produced_pse is None:
                    logger.warning("PyMOL ran but produced no session at %s", pse_path)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                stderr = getattr(exc, "stderr", "") or ""
                logger.warning("PyMOL failed (%s); the .pml is still usable: %s",
                               type(exc).__name__, stderr.strip()[:300])

    if unmatched:
        logger.info("%d residue(s) had no wildtype/mutant counterpart (indel), "
                    "set to 0 in the delta map", len(unmatched))

    return {
        "delta_pdb": delta_pdb,
        "pml": pml_path,
        "pse": produced_pse,
        "max_abs_delta": max_abs,
        "unmatched": unmatched,
    }
