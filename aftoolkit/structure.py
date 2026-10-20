"""Protein structures as pandas dataframes.

Loading, renumbering, mutating and writing PDB structures. A "protein dataframe"
here is a biopandas ``ATOM`` frame with four added columns:

``residue_number_original`` / ``chain_id_original``
    The numbering as it appeared in the input file. All downstream naming keys
    off these, so they survive renumbering.
``residue_number`` / ``chain_id``
    The *internal* AlphaFold numbering: every chain collapsed onto chain ``"A"``
    with a gap between chains (see :data:`DEFAULT_CHAIN_GAP`).
``mask``
    True for atoms that were inserted rather than observed, so the template
    features for them can be zeroed.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from functools import reduce

import pandas as pd
from biopandas.pdb import PandasPdb

from aftoolkit.mutations import GAP, Mutation

__all__ = [
    "AA_1_TO_3_DICT",
    "AA_3_TO_1_DICT",
    "BACKBONE_ATOMS",
    "DEFAULT_CHAIN_GAP",
    "DF_COLUMNS",
    "RELPOS_K",
    "get_internal_residue_numbering",
    "get_regions_mask",
    "insert_protein_region",
    "load_protein",
    "mutate_protein",
    "pdb_str_to_dataframe",
    "safe_filename_part",
    "save_to_pdb",
    "strip_mutated_sidechains",
]

AA_3_TO_1_DICT = {"CYS": "C", "ASP": "D", "SER": "S", "GLN": "Q", "LYS": "K", "ILE": "I",
                  "PRO": "P", "THR": "T", "PHE": "F", "ASN": "N", "GLY": "G", "HIS": "H",
                  "LEU": "L", "ARG": "R", "TRP": "W", "ALA": "A", "VAL": "V", "GLU": "E",
                  "TYR": "Y", "MET": "M", "UNK": "U", GAP: GAP}

AA_1_TO_3_DICT = {v: k for k, v in AA_3_TO_1_DICT.items()}

DF_COLUMNS = ["record_name", "atom_number", "blank_1", "atom_name", "alt_loc",
              "residue_name", "blank_2", "chain_id", "residue_number", "insertion",
              "blank_3", "x_coord", "y_coord", "z_coord", "occupancy", "b_factor",
              "blank_4", "segment_id", "element_symbol", "charge", "line_idx",
              ]

#: Backbone atoms kept when a side chain is removed for threading.
BACKBONE_ATOMS = ("N", "CA", "C", "O")

# AlphaFold2's relative-position encoding clips residue-index offsets to
# +/- relpos_k (config: model.input_embedder.relpos_k = 32). When several chains
# are collapsed onto a single chain, the inter-chain gap in the residue numbering
# must be >= relpos_k so the closest cross-chain residue pair saturates into the
# terminal "maximally separated" bin -- i.e. the chains read as distinct. Anything
# >= 32 is equivalent; 32 is the minimal optimal value. (The previous default of
# 25 was below the window and leaked a spurious sequential-adjacency signal.)
#
# ``chain_gap`` is exposed as an inference parameter so callers may override this
# default -- e.g. the RBD-AIM reproduction passes 25 to match that paper's
# published setting (doi:10.1038/s41467-026-70553-6).
RELPOS_K = 32
DEFAULT_CHAIN_GAP = RELPOS_K


def get_internal_residue_numbering(pdb_df, chain_gap: int = DEFAULT_CHAIN_GAP):
    """Collapse a (possibly multichain) structure onto a single chain's numbering.

    All chains are renumbered into one continuous index, with ``chain_gap`` empty
    residue positions inserted between chains so AlphaFold treats them as separate
    (see :data:`DEFAULT_CHAIN_GAP`).

    Args:
        pdb_df: protein dataframe for the original structure.
        chain_gap: residue-index gap inserted between chains.

    Returns:
        List of new residue numbers, parallel to ``pdb_df``'s rows.
    """
    residue_number_af = []
    n_chain = 0
    bias = 0
    for _chain, df_ in pdb_df.groupby(["chain_id_original"], sort=False):
        residue_number_af_ = []
        for ids, df__ in df_.groupby(["residue_number_original", "insertion"],
                                     sort=False):
            if ids[1] != '':
                bias += 1
            residue_number_af__ = df__["residue_number_original"].to_numpy() + bias
            residue_number_af_ += list(residue_number_af__)

        residue_number_af_ -= residue_number_af_[0] - 1 - n_chain
        residue_number_af += list(residue_number_af_)
        n_chain = chain_gap + residue_number_af[-1]  # add gap between chains
    return residue_number_af


def get_regions_mask(regions, protein):
    """Boolean mask selecting ``{chain: [(start, end), ...]}`` residue ranges."""
    conditions = []
    for chain in regions:
        for r in regions[chain]:
            mask = (protein["chain_id_original"] == chain) & \
                   (protein["residue_number_original"].isin(range(r[0], r[1] + 1)))
            conditions.append(mask)
    return reduce(lambda x, y: x | y, conditions)


def load_protein(path, chains=None, regions=None, chain_gap: int = DEFAULT_CHAIN_GAP):
    """Load a PDB file into a protein dataframe renumbered for AlphaFold.

    Keeps the first model, drops hydrogens/deuteriums and alternate locations
    other than ``A``. Multiple chains are collapsed onto a single chain ``"A"``
    with a ``chain_gap`` residue-index gap between them.

    Args:
        path: path to the PDB file.
        chains: chain ids to keep (all of them when ``None``).
        regions: ``{chain: [(start, end), ...]}`` residue ranges to keep.
        chain_gap: residue-index gap inserted between chains.

    Returns:
        Protein dataframe prepared for AlphaFold inference.
    """
    pdb_df = PandasPdb().read_pdb(path).label_models().df["ATOM"]
    pdb_df = pdb_df[pdb_df.model_id == 1]
    pdb_df = pdb_df[pdb_df["alt_loc"].isin(["A", ""])]
    pdb_df = pdb_df[~pdb_df["element_symbol"].isin(["H", "D"])]

    pdb_df["residue_number_original"] = pdb_df["residue_number"]
    pdb_df["chain_id_original"] = pdb_df["chain_id"]
    if chains is not None:
        pdb_df = pdb_df[pdb_df["chain_id_original"].isin(chains)]

    pdb_df["residue_number"] = get_internal_residue_numbering(pdb_df, chain_gap)
    pdb_df["chain_id"] = "A"
    pdb_df["mask"] = False

    if regions is not None:
        pdb_df = pdb_df[get_regions_mask(regions, pdb_df)]
        pdb_df["residue_number"] = get_internal_residue_numbering(pdb_df, chain_gap)

    return pdb_df


def _three_letter_substitutions(mutations: Iterable[Mutation]) -> dict:
    """Index substitutions/deletions by their three-letter residue group key.

    The key matches how :func:`mutate_protein` groups the dataframe:
    ``(residue_name, residue_number_original, chain_id_original)``.

    Insertions are excluded -- they do not correspond to an existing residue.

    Raises:
        ValueError: if a residue code is not a known amino acid.
    """
    converted = {}
    for mut in mutations:
        if mut.is_insertion:
            continue
        if mut.wt not in AA_1_TO_3_DICT or mut.mt not in AA_1_TO_3_DICT:
            raise ValueError(
                f"Unknown residue code in mutation {mut}; "
                f"expected one-letter amino acid codes or '{GAP}'."
            )
        key = (AA_1_TO_3_DICT[mut.wt], mut.position, mut.chain)
        converted[key] = AA_1_TO_3_DICT[mut.mt]
    return converted


def _validate_against_structure(pdb_df, mutations: Iterable[Mutation]) -> None:
    """Check each mutation's wildtype residue is really in the structure.

    Mutations are matched by ``(residue_name, number, chain)``, so a wrong
    wildtype code simply matched nothing and the mutation was silently dropped --
    yielding an unmutated structure, and a ddG computed against it, with no
    error anywhere.

    Insertions are skipped: their ``wt`` is the gap marker.

    Raises:
        ValueError: listing every mutation that does not match.
    """
    problems = []
    for mut in mutations:
        if mut.is_insertion:
            continue
        residues = pdb_df[(pdb_df["residue_number_original"] == mut.position)
                          & (pdb_df["chain_id_original"] == mut.chain)]
        if residues.empty:
            problems.append(f"{mut}: no residue at {mut.chain}:{mut.position}")
            continue
        found = residues["residue_name"].iloc[0]
        expected = AA_1_TO_3_DICT[mut.wt]
        if found != expected:
            problems.append(
                f"{mut}: expected {expected} ({mut.wt}) at {mut.chain}:{mut.position}, "
                f"found {found} ({AA_3_TO_1_DICT.get(found, '?')})"
            )
    if problems:
        raise ValueError("Mutation does not match the structure -- " + "; ".join(problems))


def _keep_backbone(residue_df):
    """Drop side-chain atoms, keeping only the backbone.

    Raises:
        ValueError: if the residue does not have a complete backbone, which would
            silently corrupt the template fed to AlphaFold.
    """
    backbone = residue_df[residue_df["atom_name"].isin(BACKBONE_ATOMS)]
    if backbone.shape[0] != len(BACKBONE_ATOMS):
        raise ValueError(
            f"Expected {len(BACKBONE_ATOMS)} backbone atoms "
            f"{list(BACKBONE_ATOMS)}, found {backbone.shape[0]}."
        )
    return backbone


def strip_mutated_sidechains(protein_wt, mutations: Sequence[Mutation]):
    """Remove side chains at substituted positions, keeping residue identity.

    Applied to the *wildtype* so it is threaded through AlphaFold on the same
    footing as the mutant: both see backbone-only templates at mutated sites, so
    a feature difference between them reflects the substitution rather than the
    presence or absence of side-chain coordinates.
    """
    if not mutations:
        return protein_wt

    _validate_against_structure(protein_wt, mutations)
    substitutions = _three_letter_substitutions(mutations)
    protein_wt = protein_wt.copy()
    protein_wt["residue_name_wt"] = protein_wt["residue_name"]

    residues = []
    for key, residue_df in protein_wt.groupby(
            ["residue_name", "residue_number_original", "chain_id_original"], sort=False
    ):
        if substitutions.get(key, GAP) != GAP:
            residue_df = _keep_backbone(residue_df)
        residues.append(residue_df.copy())

    return pd.concat(residues)


def _apply_substitutions_and_deletions(protein_wt, mutations: Sequence[Mutation]):
    """Apply substitutions and deletions, renumbering around removed residues.

    For a substitution only the backbone is kept and the residue is renamed, so
    AlphaFold rebuilds the side chain. A deletion drops the residue and shifts
    the internal numbering of everything after it.
    """
    _validate_against_structure(protein_wt, mutations)
    substitutions = _three_letter_substitutions(mutations)

    protein_wt = protein_wt.copy()
    protein_wt["residue_name_wt"] = protein_wt["residue_name"]

    residues = []
    n_shift = 0
    for key, residue_df in protein_wt.groupby(
            ["residue_name", "residue_number_original", "chain_id_original"], sort=False
    ):
        residue_df = residue_df.copy()
        residue_df["residue_number"] += n_shift

        mutant_name = substitutions.get(key)
        if mutant_name == GAP:
            n_shift -= 1  # deletion: drop it and pull the rest forward
            continue
        if mutant_name is not None:
            residue_df = _keep_backbone(residue_df).copy()
            residue_df["residue_name"] = mutant_name

        residues.append(residue_df)

    return pd.concat(residues)


def insert_protein_region(pdb_df, insertion_residue_position, sequence):
    """Thread ``sequence`` into the structure as masked dummy residues.

    The inserted residues carry the neighbouring CA's coordinates and are flagged
    ``mask=True``, so their template features are zeroed and AlphaFold has to
    place them itself.

    Args:
        pdb_df: protein dataframe.
        insertion_residue_position: ``(residue_number, chain_id)``, or a string
            like ``"123A"``.
        sequence: one-letter residue sequence to insert.
    """
    if isinstance(insertion_residue_position, str):
        ins_residue_number = int(insertion_residue_position[:-1])
        ins_residue_chain = insertion_residue_position[-1]
    else:
        ins_residue_number, ins_residue_chain = insertion_residue_position

    pdb_df = pdb_df.copy()

    def residue_at(number):
        return pdb_df[(pdb_df["residue_number_original"] == number)
                      & (pdb_df["chain_id_original"] == ins_residue_chain)]

    before_or_after = "after"
    anchor = residue_at(ins_residue_number)
    if anchor.shape[0] == 0:
        # Nothing at the requested position: anchor on the next residue instead
        # and insert ahead of it.
        ins_residue_number += 1
        anchor = residue_at(ins_residue_number)
        before_or_after = "before"
    if anchor.shape[0] == 0:
        raise ValueError(
            f"No residue to anchor the insertion at "
            f"{ins_residue_chain}:{ins_residue_number}."
        )
    anchor = anchor.copy()

    resi = anchor["residue_number"].iloc[0]
    pdb_df_before = pdb_df[pdb_df["residue_number"] < resi].copy()
    pdb_df_after = pdb_df[pdb_df["residue_number"] > resi].copy()

    dummy_atoms = []
    for i, residue in enumerate(sequence):
        dummy_atom = anchor[anchor["atom_name"] == "CA"].copy()
        dummy_atom["residue_number_original"] = ins_residue_number
        dummy_atom["insertion"] = i
        dummy_atom["chain_id_original"] = ins_residue_chain
        dummy_atom["residue_number"] = i
        dummy_atom["mask"] = True
        dummy_atom["residue_name"] = AA_1_TO_3_DICT[residue]
        dummy_atoms.append(dummy_atom)
    dummy_atoms = pd.concat(dummy_atoms)

    if before_or_after == "before":
        dummy_atoms["residue_number"] += resi
        anchor["residue_number"] += len(sequence)
        pdb_df_after["residue_number"] += len(sequence)
        return pd.concat([pdb_df_before, dummy_atoms, anchor, pdb_df_after])

    dummy_atoms["residue_number"] += resi + 1
    pdb_df_after["residue_number"] += len(sequence)
    return pd.concat([pdb_df_before, anchor, dummy_atoms, pdb_df_after])


def mutate_protein(pdb_df, mutations, ignore_not_found: bool = False):
    """Apply mutations to a protein dataframe.

    Substitutions keep only backbone atoms (so AlphaFold rebuilds the side chain),
    deletions drop the residue and renumber, and insertions thread in masked
    dummy residues.

    Args:
        pdb_df: protein dataframe.
        mutations: a sequence of :class:`~aftoolkit.mutations.Mutation`, or a
            comma-separated string such as ``"A:D123C,A:R56Y"``.
        ignore_not_found: accepted for backwards compatibility and ignored; an
            unknown residue code now always raises.

    Returns:
        A new protein dataframe with the mutations applied.
    """
    from aftoolkit.mutations import parse_mutations

    mutations = parse_mutations(mutations)
    substitutions = [m for m in mutations if not m.is_insertion]
    insertions = [m for m in mutations if m.is_insertion]

    if substitutions:
        pdb_df = _apply_substitutions_and_deletions(pdb_df, substitutions)
    for mut in insertions:
        pdb_df = insert_protein_region(pdb_df, (mut.position, mut.chain), mut.mt)
    return pdb_df


def pdbline_to_dict(line: str) -> dict:
    """Parse one fixed-column ``ATOM`` record into a dataframe row."""
    atom_name = line[13:15]
    if line[15] != " ":
        atom_name += line[15]
    residue_name = line[17:20]
    chain_name = line[21]
    residue_number = int(line[22:26])

    # Coordinate fields abut each other, so a wide value can run into the
    # preceding column; drop the leading character when that happens.
    coords = []
    for field in (line[30:38], line[38:46], line[46:54]):
        coords.append(float(field[1:] if field[0] != " " else field))
    x, y, z = coords

    atom_type = line[77] if len(line) >= 77 else atom_name[0]

    row = {c: "" for c in DF_COLUMNS}
    row.update({
        "record_name": "ATOM",
        "atom_number": int(line[4:11]),
        "atom_name": atom_name,
        "residue_name": residue_name,
        "aa": AA_3_TO_1_DICT[residue_name],
        "chain_id": chain_name,
        "residue_number": residue_number,
        "x_coord": x,
        "y_coord": y,
        "z_coord": z,
        "occupancy": 1.0,
        "b_factor": float(line[61:66]),
        "charge": 0,
        "element_symbol": atom_type,
    })
    return row


def pdb_str_to_dataframe(pdb_lines, pdb_df_prev=None):
    """Parse AlphaFold's predicted PDB text into a protein dataframe.

    Args:
        pdb_lines: predicted structure, as a string or list of lines.
        pdb_df_prev: the input dataframe, used to restore the original numbering
            columns that AlphaFold's output does not carry.
    """
    if isinstance(pdb_lines, str):
        pdb_lines = pdb_lines.split("\n")

    rows = [pdbline_to_dict(line) for line in pdb_lines if line.startswith("ATOM")]
    pdb_df = pd.DataFrame(rows).reindex(columns=DF_COLUMNS)
    pdb_df["line_idx"] = pdb_df.index
    if pdb_df_prev is None:
        return pdb_df

    original = {
        key: (
            residue.iloc[0]["residue_number_original"],
            residue.iloc[0]["chain_id_original"],
            residue.iloc[0]["insertion"],
        )
        for key, residue in pdb_df_prev.groupby(["residue_number", "chain_id"], sort=False)
    }
    restored = [original[key] for key in zip(pdb_df["residue_number"], pdb_df["chain_id"])]

    pdb_df["residue_number_original"] = [r[0] for r in restored]
    pdb_df["chain_id_original"] = [r[1] for r in restored]
    pdb_df["insertion"] = [r[2] for r in restored]
    return pdb_df


#: Defaults for canonical PDB columns a caller's frame may not carry.
_PDB_COLUMN_DEFAULTS = {
    "record_name": "ATOM", "alt_loc": "", "insertion": "", "occupancy": 1.0,
    "b_factor": 0.0, "segment_id": "", "charge": 0,
    "blank_1": "", "blank_2": "", "blank_3": "", "blank_4": "",
}


def save_to_pdb(pdb_df, output_name, original_numbering: bool = True):
    """Write a protein dataframe to a PDB file.

    The frame is put into canonical PDB column order first. biopandas writes
    columns in the order it finds them, so a frame whose columns are ordered
    differently -- any frame built by hand rather than read from a file -- would
    otherwise be written with its fields in the wrong fixed-width positions, and
    the result is a corrupt file that fails silently.

    Args:
        pdb_df: protein dataframe.
        output_name: output path.
        original_numbering: restore the input file's residue numbers and chain
            ids, rather than the internal AlphaFold numbering.
    """
    pdb_df = pdb_df.copy()
    if original_numbering:
        pdb_df["residue_number"] = pdb_df["residue_number_original"]
        pdb_df["chain_id"] = pdb_df["chain_id_original"]

    ordered = pdb_df.reindex(columns=DF_COLUMNS)
    for column, default in _PDB_COLUMN_DEFAULTS.items():
        if ordered[column].isna().all():
            ordered[column] = default
        else:
            ordered[column] = ordered[column].fillna(default)
    if ordered["atom_number"].isna().all():
        ordered["atom_number"] = range(1, len(ordered) + 1)
    if ordered["line_idx"].isna().all():
        ordered["line_idx"] = range(len(ordered))
    if ordered["element_symbol"].isna().all():
        ordered["element_symbol"] = [str(a).strip()[:1] for a in ordered["atom_name"]]

    prot = PandasPdb()
    prot.df["ATOM"] = ordered
    prot.to_pdb(output_name)


def safe_filename_part(value) -> str:
    """Sanitize an arbitrary string into a single safe filename component.

    Strips any directory components and replaces characters that are awkward in
    filenames (path separators, whitespace, colons, ...) with underscores.
    ``None`` becomes the literal ``"none"``.
    """
    if value is None:
        return "none"
    value = os.path.basename(str(value))
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value or "none"
