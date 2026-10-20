"""Running AlphaFold2/OpenFold over a protein dataframe.

This is the only first-party module that imports the vendored OpenFold stack, so
it is also the only one that needs the optional ``[openfold]`` extra.

The pipeline is MSA-free: the input structure is supplied as a *template* and the
sequence as a single-row MSA. Which template the model sees on each recycle is
what distinguishes the three inference formats -- see :class:`OpenFoldWrapper`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial

import numpy as np
import pandas as pd
import torch

import aftoolkit.processing.openfold.np.protein as of_protein
from aftoolkit.folding.weights import ensure_weights
from aftoolkit.processing.openfold.config import model_config
from aftoolkit.processing.openfold.data import feature_pipeline
from aftoolkit.processing.openfold.np import residue_constants
from aftoolkit.processing.openfold.utils.script_utils import (
    load_models_from_command_line,
    prep_output,
)
from aftoolkit.processing.openfold.utils.tensor_utils import dict_multimap, tensor_tree_map
from aftoolkit.structure import pdb_str_to_dataframe

__all__ = ["RESIDUE_FEATURES", "OpenFoldWrapper", "resolve_device"]

logger = logging.getLogger(__name__)

_MODEL_NAME = "model_2_ptm"

#: Per-residue features extracted from the model output. Pair-indexed features
#: are read on the diagonal, i.e. a residue against itself.
RESIDUE_FEATURES = (
    "msa", "pair", "lddt_logits", "distogram_logits", "aligned_confidence_probs",
    "predicted_aligned_error", "plddt", "single", "tm_logits",
)


def resolve_device(device=None) -> str:
    """Resolve the compute device.

    If ``device`` is given it is honored as-is. Otherwise the best available
    backend is auto-detected, preferring CUDA, then Apple MPS, then CPU.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OpenFoldWrapper:
    """Folds protein dataframes and extracts per-residue representations.

    Three inference formats differ only in the template the model is shown:

    ``mask`` (the default: ``always_use_template=False``)
        The template is supplied on the first pass and masked out for every
        recycle after it, so later recycles refine from the representation
        rather than re-reading the structure.
    ``vanilla`` (``always_use_template=True``)
        The template is kept on every recycle, as in standard AlphaFold.
    ``af2rank`` (``side_chain_mask=True``)
        Side chains are stripped and residues renamed to ``UNK``, so only the
        backbone is shown -- the AF2Rank "gap" trick.

    Attributes:
        of_output: the model's output for the last :meth:`evaluate` call.
        of_output_per_cycle: per-recycle outputs, or ``None`` unless
            ``return_all_cycles`` was set.
        of_protein: the predicted structure as a protein dataframe.
    """

    def __init__(
        self,
        feature_list: Sequence[str] | None = None,
        device: str | None = None,
        inference_n_recycle: int = 1,
        always_use_template: bool = False,
        side_chain_mask: bool = False,
        return_all_cycles: bool = False,
    ):
        """
        Args:
            feature_list: unused; retained for call-compatibility.
            device: e.g. ``"cuda:0"`` or ``"mps"``; auto-detected when ``None``.
            inference_n_recycle: number of recycling iterations.
            always_use_template: keep the template on every recycle.
            side_chain_mask: show only backbone atoms.
            return_all_cycles: also keep each recycle's output.
        """
        self.device = resolve_device(device)
        self.feature_list = feature_list
        self.config = model_config(_MODEL_NAME, low_prec=True)
        self.feature_processor = feature_pipeline.FeaturePipeline(self.config.data)
        self.output_directory = "./"
        self.model = None

        self.protein = pd.DataFrame()
        self.of_output = None
        self.of_protein = None
        self.of_output_per_cycle = None

        self.n_recycle = inference_n_recycle
        self.always_use_template = always_use_template
        self.side_chain_mask = side_chain_mask
        self.return_all_cycles = return_all_cycles

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(device={self.device!r}, "
                f"n_recycle={self.n_recycle}, "
                f"always_use_template={self.always_use_template}, "
                f"side_chain_mask={self.side_chain_mask})")

    # -- model ------------------------------------------------------------- #

    def init_model(self, weights_dir=None, download: bool = True) -> OpenFoldWrapper:
        """Load the AlphaFold parameters, downloading them if necessary.

        Args:
            weights_dir: directory holding the parameter file; the packaged
                default when ``None``.
            download: fetch the parameters when absent.

        Raises:
            WeightsUnavailable: if the parameters cannot be found or fetched.
        """
        from aftoolkit.constants import MODEL_WEIGHTS_DIR

        weights_path = ensure_weights(
            weights_dir if weights_dir is not None else MODEL_WEIGHTS_DIR,
            download=download,
        )
        logger.info("Loading %s onto %s", _MODEL_NAME, self.device)
        for model, output_directory in load_models_from_command_line(
            self.config, self.device, None, weights_path, None
        ):
            self.model = model
            self.output_directory = output_directory
        self.model.to(self.device)
        return self

    # -- feature preparation ------------------------------------------------ #

    @staticmethod
    def prepare_features(target_protein) -> dict:
        """Build the AlphaFold input features for one structure.

        The structure becomes a template and its sequence a single-row MSA, so
        the model folds from the structure alone with no evolutionary input.
        """
        sequence = residue_constants.aatype_to_str_sequence(target_protein.aatype)
        num_res = len(sequence)

        template_features = {
            "template_all_atom_positions": target_protein.atom_positions[None, ...],
            "template_all_atom_mask": target_protein.atom_mask[None, ...],
            "template_sequence": [sequence],
            "template_aatype": target_protein.aatype[None, ...],
            "template_domain_names": [None],
        }

        sequence_features = {
            "aatype": residue_constants.sequence_to_onehot(
                sequence=sequence,
                mapping=residue_constants.restype_order_with_x,
                map_unknown_to_x=True,
            ),
            "between_segment_residues": np.zeros((num_res,), dtype=np.int32),
            "domain_name": np.array([b"input"], dtype=np.object_),
            "residue_index": np.array(target_protein.residue_index, dtype=np.int32),
            "seq_length": np.array([num_res] * num_res, dtype=np.int32),
            "sequence": np.array([sequence.encode("utf-8")], dtype=np.object_),
            "deletion_matrix_int": np.zeros((1, num_res), dtype=np.int32),
            "msa": np.array(
                [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence],
                dtype=np.int32,
            )[None, ...],
            "num_alignments": np.array(np.ones(num_res), dtype=np.int32),
            "msa_species_identifiers": np.array([b""]),
        }
        return {**sequence_features, **template_features}

    def _build_input_features(self, pdb_df) -> dict:
        """Assemble the batched, recycled feature dict for one structure."""
        structure = of_protein.from_pdb_df(pdb_df)
        features = self.prepare_features(structure)

        if self.side_chain_mask:
            backbone = pdb_df[pdb_df["atom_name"].isin(["N", "C", "CA", "O", "CB"])].copy()
            backbone["residue_name"] = "UNK"
            backbone_features = self.prepare_features(of_protein.from_pdb_df(backbone))
            features.update({
                name: value for name, value in backbone_features.items()
                if name.startswith("template")
            })

        # Zero the template for residues that were inserted rather than observed.
        ca = pdb_df[pdb_df["atom_name"] == "CA"].reset_index()
        features["template_all_atom_mask"][:, ca[ca["mask"]].index, :] *= 0

        processed = self.feature_processor.process_features(features, mode="predict")

        # One copy per recycle; all but the first may have the template masked.
        recycles = [processed]
        for _ in range(self.n_recycle):
            recycle = {k: v.detach().clone() for k, v in processed.items()}
            if not self.always_use_template:
                recycle["template_mask"] *= 0
            recycles.append(recycle)
        batched = dict_multimap(partial(torch.cat, dim=-1), recycles)

        for name, value in batched.items():
            dtype = torch.float32 if value.dtype == torch.float64 else None
            batched[name] = torch.as_tensor(
                value, dtype=dtype, device=self.device
            )[None, ...]
        return batched

    # -- output extraction -------------------------------------------------- #

    @staticmethod
    def get_batch_features(af_output_batch, af_input_features, batch_id):
        """Select one batch element from the model output and input features."""
        out = {"sm": {}}
        for key, value in af_output_batch.items():
            if key == "sm":
                for name, tensor in value.items():
                    if tensor.shape[0] == 1:
                        out["sm"][name] = tensor[batch_id, ...]
                    if tensor.shape[0] == 8:
                        out["sm"][name] = tensor[:, batch_id, ...]
                continue
            out[key] = value if len(value.shape) == 0 else value[batch_id, ...]

        features = tensor_tree_map(
            lambda x: np.array(x[batch_id, ..., -1].cpu()), af_input_features
        )
        return out, features

    @staticmethod
    def extract_residue_features(idx: int, of_output: dict) -> dict:
        """Per-residue features at internal index ``idx``.

        Pair-indexed outputs are taken on the diagonal, so every entry describes
        the single residue rather than a pair.
        """
        return {
            "msa": of_output["msa"][0, idx, :],
            "pair": of_output["pair"][idx, idx, :],
            "lddt_logits": of_output["lddt_logits"][idx, :],
            "distogram_logits": of_output["distogram_logits"][idx, idx, :],
            "aligned_confidence_probs": of_output["aligned_confidence_probs"][idx, idx, :],
            "predicted_aligned_error": of_output["predicted_aligned_error"][idx, idx].reshape(-1),
            "plddt": of_output["plddt"][idx].reshape(-1),
            "single": of_output["single"][idx, :],
            "tm_logits": of_output["tm_logits"][idx, idx, :],
        }

    def _check_alignment(self, of_output) -> None:
        """Verify the model output and the predicted structure agree in length."""
        ca = self.of_protein[self.of_protein["atom_name"] == "CA"]
        n_residues, n_output = ca.shape[0], of_output["single"].shape[0]
        if n_residues != n_output:
            raise RuntimeError(
                f"AlphaFold output covers {n_output} residues but the predicted "
                f"structure has {n_residues}; feature indices would be misaligned."
            )

    def get_of_features_by_positions(self, positions: dict) -> dict:
        """Extract features for ``{residue_key: internal_index}``."""
        self._check_alignment(self.of_output)
        return {
            name: self.extract_residue_features(idx, self.of_output)
            for name, idx in positions.items()
        }

    def get_of_features_by_positions_per_cycle(self, positions: dict) -> list:
        """Extract features for each recycle, in order."""
        per_cycle = []
        for cycle_output in self.of_output_per_cycle or []:
            self._check_alignment(cycle_output)
            per_cycle.append({
                name: self.extract_residue_features(idx, cycle_output)
                for name, idx in positions.items()
            })
        return per_cycle

    def get_of_protein(self):
        """The predicted structure of the last :meth:`evaluate` call."""
        return self.of_protein

    # -- inference ---------------------------------------------------------- #

    def inference_monomer(self, pdb_df):
        """Fold one structure.

        Returns:
            ``(output, per_cycle_outputs, predicted_structure)``; the middle
            element is ``None`` unless ``return_all_cycles`` was set.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded: call init_model() first.")

        batched = self._build_input_features(pdb_df)
        with torch.no_grad():
            out_batch, out_per_cycle = self.model(batched)

        def to_numpy(x):
            return np.array(x[...].detach().cpu())

        per_cycle = []
        for cycle in out_per_cycle:
            cycle_out, _ = self.get_batch_features(tensor_tree_map(to_numpy, cycle), {}, 0)
            per_cycle.append(cycle_out)

        out, input_features = self.get_batch_features(
            tensor_tree_map(to_numpy, out_batch), batched, 0
        )
        unrelaxed = prep_output(
            out, input_features, input_features, self.feature_processor,
            _MODEL_NAME, 200, False,
        )
        predicted = pdb_str_to_dataframe(of_protein.to_pdb(unrelaxed), pdb_df)

        return out, (per_cycle if self.return_all_cycles else None), predicted

    def set_protein(self, protein_df) -> None:
        """Record the structure about to be folded."""
        self.protein = protein_df

    def evaluate(self, protein_df) -> None:
        """Fold ``protein_df`` and store the output on this wrapper."""
        self.set_protein(protein_df)
        self.of_output, self.of_output_per_cycle, self.of_protein = \
            self.inference_monomer(pdb_df=protein_df)
