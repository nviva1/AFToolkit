# How RBD-AIM Fig 2f was produced (iPAE vs RMSD — model-quality / predictive accuracy)

Fig 2f shows that the **interface PAE (iPAE)** of a *modelled* antibody–RBD complex
predicts the **quality of that model** (its RMSD to the true crystal): low iPAE ⇒
low RMSD ⇒ good model. It is the validation that iPAE is a usable confidence score.
**Fig 2f and Fig 2g use DIFFERENT protocols** (see note at the end).

Source/verification: `rbd-aim` repo — `torch_serve_class.py` (`RBDClassifierHandler.model_rbd`,
`postprocess`, `find_homologs`), `rbdaim.py`, `utils/openfold_wrapper.py`, and the
released data `dataset_production_all_with_rms.pkl` + MOESM9 sheet `Fig 2f`.

## Protocol (per antibody, WT RBD)

1. **Homology modelling onto an NN-selected template** (this is the key Fig-2f step):
   - The antibody (heavy+light) has **no assumed structure**. A Fab **template** is
     chosen from a library (`datasets/pdb_clean_ab_only/`, 764 cleaned Fab structures)
     by `find_homologs`: candidates ranked by **Levenshtein** CDR similarity and by an
     **AntiBERTy-based NN classifier**; the template giving the **lowest WT iPAE** is kept
     (`np.argmin` over candidates).  → stored template id = `template_pred_levin_NN`.
   - RBD = the single WT `datasets/ref.pdb` (Wuhan).
   - The target antibody sequence is **threaded** onto the template Fab and the WT
     sequence onto `ref.pdb` via `homology_model` (kalign / AbNumber alignment),
     **keeping only [N, C, CA, O] backbone atoms at mismatched positions** and full
     sidechains where target == template; insertions get dummy residues.

2. **Merged input**: `[RBD chain "1"]  +  [heavy]  +  [light]`, concatenated with a
   **25-residue gap** between chains (positional-embedding separation).

3. **OpenFold inference** (`model_2_ptm`, MSA = single sequence):
   - **Masked template**: template used in the 1st recycle only, masked in cycles 2–3.
   - **3 rounds of recycling** (`n_recycle=2` → cycles 0,1,2); read the **final** cycle.

4. **iPAE** = `np.average(pae[ids1][:, ids2])` where `ids1` = RBD residues (chain "1"),
   `ids2` = antibody residues — i.e. the **all-pairs mean PAE, RBD→antibody (one
   direction)**. Stored as **`PAE_NN`** (NN template) / `PAE_no_NN` (Levenshtein template).

5. **RMSD** (`RMS_NN`): RMSD of the predicted complex vs the antibody's true deposited
   crystal — the model-quality ground truth. (High RMSD = bad model.)

6. **Fig 2f** = scatter of `iPAE (PAE_NN)` vs `RMSD (RMS_NN)` over the production
   antibodies (n≈379 with RMSD), plus a **ROC** for iPAE classifying good vs bad models.

## Numbers (paper) and our reproduction

| quantity | paper | reproduced here |
|---|---|---|
| iPAE values (`PAE_NN`) | source data MOESM9 'Fig 2f' (`iPAE_score`, n=379) | **my iPAE vs `PAE_NN`: Pearson 0.99** (exact ≤0.05 when the model is good; drift only on high-RMS bad models) |
| ROC AUC (iPAE → low RMSD) | ~0.9 | **0.91 (RMSD<2 Å) / 0.92 (RMSD<4 Å)** from `PAE_NN`/`RMS_NN`; `PAE_no_NN` 0.87–0.91 |

Reproduction code: `reproductions/rbd_aim_fig2g/paper_faithful.py` (builds the complex
with the paper's `homology_model` + NN template, AFToolkit wrapper, one-direction iPAE);
`run_their_code.py` runs the paper's own wrapper. Validated antibody-by-antibody:
CR3022 8.045 vs paper 8.033, TAU-2303 7.350 vs 7.338, etc.

## Why Fig 2f ≠ Fig 2g (do not mix them)

- **Fig 2f** (this file): the antibody is **homology-modelled onto an NN-selected
  template** (it has no known structure); RBD is WT `ref.pdb`. Metric vs **RMSD**.
- **Fig 2g**: uses each antibody's **own deposited X-ray complex** as ground truth and
  **mutates the RBD** to each strain; metric is **ΔiPAE vs Δlog₁₀K_D** (escape).
  No homology template.

The iPAE *engine* (gap=25, masked template, 3 recycles, one-direction all-pairs iPAE,
`model_2_ptm`) is shared; the **structure source differs** (NN-template homology for 2f,
own-crystal + RBD mutation for 2g), so their PAE values are not interchangeable.
