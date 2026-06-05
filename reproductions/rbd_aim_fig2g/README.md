# Reproducing RBD-AIM Fig 2g with AFToolkit

Reproduces **Fig 2g** of *"Mining antibody functionality via AI-guided structural
landscape profiling"* (RBD-AIM), Nature Communications 2026,
[doi:10.1038/s41467-026-70553-6](https://doi.org/10.1038/s41467-026-70553-6).

Fig 2g shows that the **interface PAE (iPAE)** of an OpenFold-modelled
antibody–RBD complex tracks the antibody's **binding selectivity across RBD
strains**: Δ iPAE vs Δ log₁₀ K_D, Pearson |R| ≈ 0.52.

The method (template-based, MSA-free OpenFold) is exactly AFToolkit's complex
pipeline, so we recompute iPAE with `aftoolkit.scoring`.

## What's here
| File | Purpose |
|---|---|
| `recompute_from_structures.py` | **Main reproduction:** input structures → run AFToolkit → iPAE/pLDDT/pTM scores → (if affinities given) the Fig 2g Δ iPAE vs Δ log K_D plot |
| `replot_from_source_data.py` | Sanity check: re-plot the journal's published Fig 2g points (gives R ≈ 0.52) |
| `example_manifest.csv` | 3 real Ab–RBD crystal complexes (8WFH, 8WFM, 9JEB) for a score-only demo |
| `fetch_example_structures.sh` | Download those structures from RCSB into `inputs/` |
| `fig2g_reproduced.png` | The source-data re-plot (committed as reference) |

## Quickstart

Prerequisite: the package + OpenFold extra installed and AlphaFold weights present
(see the repo README), then run from the repository root.

**1. Recompute AFToolkit scores from structures (the reproduction):**
```bash
bash reproductions/rbd_aim_fig2g/fetch_example_structures.sh
python reproductions/rbd_aim_fig2g/recompute_from_structures.py \
    --manifest reproductions/rbd_aim_fig2g/example_manifest.csv \
    --device mps --out reproductions/rbd_aim_fig2g/recomputed
```
Output: `recomputed/scores.csv` with `ipae, mean_plddt, ptm` per input complex.

**2. Sanity re-plot from the journal Source Data:**
```bash
# Download "Source Data" (MOESM9) from the article page first.
python reproductions/rbd_aim_fig2g/replot_from_source_data.py \
    --source-data ~/Downloads/41467_2026_70553_MOESM9_ESM.xlsx
```

## Manifest format
`recompute_from_structures.py` reads a CSV of complexes:

| column | required | meaning |
|---|---|---|
| `pdb` | yes | path to the Ab–RBD complex structure |
| `antigen` | yes | antigen (RBD) chain id(s), comma-separated |
| `antibody` | yes | antibody chain id(s), comma-separated |
| `ab_id` | for Fig 2g | antibody id (groups its RBD-strain complexes) |
| `delta_log_kd` | for Fig 2g | experimental Δ log₁₀ K_D (0 for the reference strain) |
| `is_reference` | for Fig 2g | 1 for the antibody's reference-strain complex |

With only the first three columns it runs in **score-only** mode (input → scores).
Add the affinity columns to reproduce the **full Fig 2g correlation** (the
Δ log₁₀ K_D values come from the Source Data; the Ab×strain complex structures
come from the RBD-AIM modelling step or PDB).

## Data sources
- **Structures:** RCSB PDB (8WFH, 8WFM, 9JEB are deposited with the paper; the
  full Ab×strain set is modelled by RBD-AIM / available on request).
- **Affinities (Δ log₁₀ K_D):** the journal **Source Data** file (MOESM9, sheet `Fig 2g`).
- **RBD strain sequences & Fab templates:** the RBD-AIM dataset bundle
  (`github.com/tanishAI/rbd-aim`).

## Verified
- `replot_from_source_data.py` → Pearson **−0.55** on the affinity-changing subset
  (n=43), matching the paper's reported **0.52**.
- `recompute_from_structures.py` on the real crystal complex **8WFH** →
  **iPAE 8.59** (< 13.5 confidence threshold), mean pLDDT 90.7, pTM 0.85.
