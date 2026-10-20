#!/usr/bin/env python
"""Run the PAPER'S OWN code end-to-end (their OpenFoldWraper + their homology_model
builder) for a couple of antibodies, and compare the iPAE to their stored PAE_NN.

openmm/simtk/pdbfixer are stubbed (used only by their AMBER relaxation, not by
inference) so their wrapper imports without those heavy deps.
"""
import sys, types, json, os, subprocess
from unittest.mock import MagicMock

RBDAIM = "/Users/nikitaivanisenko/projects/AFToolkit/rbd-aim"
sys.path.insert(0, RBDAIM)
# Stub the AMBER-relaxation subtree (openmm/pdbfixer deps) -- never used by inference.
for m in ["pdbfixer", "openmm", "simtk",
          "utils.openfold.np.relax", "utils.openfold.np.relax.relax",
          "utils.openfold.np.relax.amber_minimize", "utils.openfold.np.relax.cleanup",
          "utils.openfold.np.relax.utils"]:
    sys.modules[m] = MagicMock()

import numpy as np
import pandas as pd
from utils.openfold_wrapper import OpenFoldWraper                     # THEIR wrapper
from utils.protein_utils import load_protein, save_pdb, make_dummy_protein, aa_3_to_1, aa_1_to_3

DS = os.path.join(RBDAIM, "extra_downloads/figures/datasets")
REF = os.path.join(DS, "ref.pdb"); TPLD = os.path.join(DS, "pdb_clean_ab_only")
WEIGHTS = "/Users/nikitaivanisenko/projects/AFToolkit/AFToolkit/weights/params_model_2_ptm.npz"


def kalign(seq1, seq2):                                              # THEIR kalign (verbatim)
    fasta = f">protein1\n{seq1}\n>protein2\n{seq2}\n"
    p = subprocess.Popen(["kalign"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    out, _ = p.communicate(input=fasta)
    al = []
    for r in out.split("\n"):
        if len(r) == 0: continue
        if r[0] == ">": al.append([]); continue
        if len(al) == 0: continue
        al[-1].append(r.rstrip())
    return ["".join(al[0]), "".join(al[1])]


def homology_model(input_sequence, protein):                        # THEIR homology_model (verbatim)
    p_ca = protein[protein["atom_name"] == "CA"].reset_index(drop=True)
    seq = "".join(aa_3_to_1[r] for r in list(p_ca["residue_name"]))
    seq_target, seq_template = kalign(input_sequence, seq)
    dummy = make_dummy_protein(input_sequence)
    seq_len = len(seq_target.replace("-", ""))
    model = [None] * seq_len
    n_target = n_template = -1
    for i in range(len(seq_target)):
        resi = None
        if seq_template[i] != "-":
            n_template += 1
            resi = p_ca.iloc[n_template]["residue_number"]
            residue = protein[protein["residue_number"] == resi]
        if seq_target[i] != "-":
            n_target += 1
        if seq_target[i] == "-":
            continue
        if resi is None:
            residue = dummy[dummy["residue_number"] == n_target + 1]
        if seq_target[i] != aa_3_to_1[residue["residue_name"].iloc[0]]:
            residue = residue.copy(); residue["residue_name"] = aa_1_to_3[seq_target[i]]
            residue = residue[residue["atom_name"].isin(["N", "C", "CA", "O"])]
        residue = residue.copy(); residue["residue_number"] = n_target
        model[n_target] = residue
    model = pd.concat(model, axis=0).reset_index(drop=True)
    model["residue_number_original"] = model["residue_number"]
    return model


def model_rbd(template_id, inp, ofr):                               # THEIR model_rbd, their wrapper
    parts = template_id.split("_"); hch, lch = parts[1], parts[2]
    rbd = load_protein(REF); rbd["chain_id_original"] = "1"
    template = load_protein(os.path.join(TPLD, template_id + ".pdb"))
    template["residue_number"] += rbd.iloc[-1]["residue_number"] + 25
    protein = pd.concat([rbd, template]).reset_index(drop=True)
    models = []
    for ch in ["1", hch, lch]:
        tp = protein[protein["chain_id_original"] == ch]
        nres = (models[-1].iloc[-1]["residue_number"] + 25) if models else 1
        m = homology_model(inp[ch], tp); m["residue_number"] += nres
        m["chain_id_original"] = ch; m["chain_id"] = ch
        models.append(m)
    pm = pd.concat(models).reset_index(drop=True)
    pm["line_idx"] = pm.index; pm["atom_number"] = pm.index
    of_out, pdb_pred = ofr.inference_monomer(pm, n_recycle=2)        # THEIR inference
    return of_out, pm


def main():
    abs_ = json.load(open("/tmp/two_ab.json"))
    wuhan = ("NLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQ"
             "IAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEG"
             "FNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGP")
    ofr = OpenFoldWraper(device="mps", weights_path=WEIGHTS)
    for a in abs_:
        parts = a["tpl"].split("_"); hch, lch = parts[1], parts[2]
        inp = {"1": wuhan, hch: a["H"], lch: a["L"]}
        of_out, pm = model_rbd(a["tpl"], inp, ofr)
        pae = np.asarray(of_out["predicted_aligned_error"])
        ca = pm[pm["atom_name"] == "CA"].reset_index(drop=True)
        labels = np.array(ca["chain_id_original"].tolist())
        ids1 = np.where(labels == "1")[0]; ids2 = np.where(labels != "1")[0]
        ipae = float(np.average(pae[np.ix_(ids1, ids2)]))
        print(f"[{a['Name']:<13}] THEIR-code iPAE={ipae:.3f}  paper PAE_NN={a['PAE_NN']:.3f}  "
              f"diff={ipae-a['PAE_NN']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
