#!/usr/bin/env python
"""Faithful reproduction: build the complex with the PAPER'S OWN homology_model
(kalign threading of the target antibody onto its NN-selected Fab template + the
strain RBD onto ref.pdb, keeping [N,C,CA,O] at mismatches), then run inference and
score one-direction iPAE = np.average(pae[RBD_rows][:, Ab_cols]).

Validation: WT iPAE should match the paper's stored PAE_NN for that antibody.
"""
import json, os, subprocess, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

RBDAIM = "/Users/nikitaivanisenko/projects/AFToolkit/rbd-aim"
sys.path.insert(0, RBDAIM)
from biopandas.pdb import PandasPdb
# Use THEIR actual loader (insertion-code-aware renumbering) + builders.
from utils.protein_utils import (load_protein as tload, save_pdb, make_dummy_protein,
                                 aa_3_to_1, aa_1_to_3)

DS = os.path.join(RBDAIM, "extra_downloads/figures/datasets")
REF = os.path.join(DS, "ref.pdb")
TPLD = os.path.join(DS, "pdb_clean_ab_only")
HERE = os.path.dirname(os.path.abspath(__file__))


def kalign(seq1, seq2):                                   # verbatim from reward_test.py
    fasta = f">protein1\n{seq1}\n>protein2\n{seq2}\n"
    p = subprocess.Popen(["kalign"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    out, _ = p.communicate(input=fasta)
    al = []
    for r in out.split("\n"):
        if len(r) == 0:
            continue
        if r[0] == ">":
            al.append([]); continue
        if len(al) == 0:
            continue
        al[-1].append(r.rstrip())
    return ["".join(al[0]), "".join(al[1])]


def homology_model(input_sequence, protein):              # verbatim logic from reward_test.py
    p_ca = protein[protein["atom_name"] == "CA"].reset_index(drop=True)
    seq = "".join(aa_3_to_1[r] for r in list(p_ca["residue_name"]))
    seq_target, seq_template = kalign(input_sequence, seq)
    dummy = make_dummy_protein(input_sequence)
    seq_len = len(seq_target.replace("-", ""))
    model = [None for _ in range(seq_len)]
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
            residue = residue.copy()
            residue["residue_name"] = aa_1_to_3[seq_target[i]]
            residue = residue[residue["atom_name"].isin(["N", "C", "CA", "O"])]
        residue = residue.copy(); residue["residue_number"] = n_target
        model[n_target] = residue
    model = pd.concat(model, axis=0).reset_index(drop=True)
    model["residue_number_original"] = model["residue_number"]
    return model


def build(template_id, heavy, light, rbd_seq, out_pdb):
    parts = template_id.split("_")
    hch, lch = parts[1], parts[2]
    rbd = tload(REF); rbd["chain_id_original"] = "1"
    template = tload(os.path.join(TPLD, template_id + ".pdb"))
    template["residue_number"] += rbd.iloc[-1]["residue_number"] + 25
    protein = pd.concat([rbd, template]).reset_index(drop=True)
    inp = {"1": rbd_seq, hch: heavy, lch: light}
    models = []
    for ch in ["1", hch, lch]:
        tp = protein[protein["chain_id_original"] == ch]
        nres = (models[-1].iloc[-1]["residue_number"] + 25) if models else 1
        m = homology_model(inp[ch], tp)
        m["residue_number"] += nres
        m["chain_id_original"] = ch
        m["chain_id"] = ch
        models.append(m)
    pm = pd.concat(models).reset_index(drop=True)
    pm["line_idx"] = pm.index; pm["atom_number"] = pm.index
    save_pdb(pm, out_pdb)
    return out_pdb, ["1", hch, lch]


def main():
    from aftoolkit.structure import load_protein
    from aftoolkit.folding import OpenFoldWrapper
    abs_ = json.load(open("/tmp/two_ab.json"))
    wuhan = json.load(open(os.path.join(os.path.dirname(os.path.dirname(HERE)),
                      "data/rbdaim/datasets/rbd_strains.json")))["Wuhan"]
    wrapper = OpenFoldWrapper(device="mps", inference_n_recycle=2,
                              always_use_template=False, return_all_cycles=True)
    wrapper.init_model()
    tdir = os.path.join(HERE, "faithful_tmp"); os.makedirs(tdir, exist_ok=True)

    for a in abs_:
        try:
            out, chains = build(a["tpl"], a["H"], a["L"], wuhan,
                                os.path.join(tdir, f"{a['Name']}_wt.pdb"))
            df = load_protein(out, chains=chains, chain_gap=25)
            wrapper.evaluate(df)
            ca = wrapper.get_of_protein(); ca = ca[ca.atom_name == "CA"].reset_index(drop=True)
            labels = np.array(ca["chain_id_original"].tolist())
            pae = np.asarray(wrapper.of_output["predicted_aligned_error"])
            ids1 = np.where(labels == "1")[0]; ids2 = np.where(labels != "1")[0]
            ipae = float(pae[np.ix_(ids1, ids2)].mean())     # one-direction, paper's def
            print(f"[{a['Name']:<13}] tpl={a['tpl']:<10}  my WT iPAE={ipae:.3f}   "
                  f"paper PAE_NN={a['PAE_NN']:.3f}   diff={ipae-a['PAE_NN']:+.3f}", flush=True)
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"[{a['Name']}] FAILED: {exc}", flush=True)


if __name__ == "__main__":
    main()
