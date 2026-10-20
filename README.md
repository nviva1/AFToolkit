# AFToolkit

> A refactored version of [AIRI-Institute/AFToolkit](https://github.com/AIRI-Institute/AFToolkit).
> Same method and the same vendored OpenFold; the package layout, public API and
> command line were rewritten in 2.0, and several correctness bugs fixed along
> the way (see the commit history). Original authors retain credit — this is not
> a replacement for upstream.

**AFToolkit** is a framework for molecular modeling of proteins with
AlphaFold2-derived representations. It runs a protein structure through a
(vendored, modified) OpenFold model, extracts per-residue embeddings and logits
at positions of interest, and trains lightweight "adapter" models (SVM / MLP /
CatBoost) on top to predict properties such as:

- **protein stability** (ΔΔG of single/multiple substitutions, insertions, deletions) — *monomer* tasks;
- **protein–protein binding affinity** — *multimer* (complex) tasks.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Extracting AlphaFold features](#extracting-alphafold-features)
- [Using the Python API](#using-the-python-api)
- [Visualising a mutation's effect](#visualising-a-mutations-effect)
- [Worked examples](#worked-examples)
- [Pretrained models & data](#pretrained-models--data)
- [Rebuilding the adapters](#rebuilding-the-adapters)
- [Training your own adapter](#training-your-own-adapter)
- [Development](#development)
- [License](#license)

## Installation

The lightweight task-construction and adapter APIs need only standard scientific
Python and install with plain `pip`:

```bash
git clone https://github.com/nviva1/AFToolkit.git
cd AFToolkit
pip install .
```

Running AlphaFold/OpenFold inference (the `OpenFoldWrapper` and the
`aftoolkit task` / `aftoolkit complex` commands) additionally requires
PyTorch and several heavier dependencies. The pip-installable subset is exposed
as an optional extra:

```bash
pip install ".[openfold]"
```

A number of dependencies (OpenMM, PDBFixer, flash-attention, HMMER, HHsuite,
Kalign) are difficult to install via pip. The reproducible way to get the full
inference environment is conda:

```bash
conda env create --name aftoolkit_env -f environment.yml
conda activate aftoolkit_env
pip install ".[openfold]"
```

> **Note on devices.** The wrapper auto-detects the compute device
> (CUDA → Apple MPS → CPU). You can force one with `--device cuda:0` or
> `--device mps` on the CLI, or `OpenFoldWrapper(device=...)` in Python.

## Quick start

```python
import aftoolkit
print(aftoolkit.__version__)
```

The lightweight API works without PyTorch:

```python
from aftoolkit import ProteinTask

task = ProteinTask()
task.set_input("data/examples/PDB/1A0F.pdb", chains=["A"])
task.set_mutations("A:S11A")
task.build_wildtype()
task.build_mutant()
```

The setters return the task, so they can also be chained:

```python
task = (ProteinTask()
        .set_input("data/examples/PDB/1A0F.pdb", chains=["A"])
        .set_mutations("A:S11A"))
```

## Extracting AlphaFold features

Calculate AF2 features for a single protein:

```bash
aftoolkit task \
    --pdb data/examples/PDB/1A7V.pdb \
    --chain A \
    --mutations A:A66H \
    -n 3 \
    --return-all-cycles \
    -o ./output
```

…or for a protein complex (specify which chains form the isolated protein and
which form the complex):

```bash
aftoolkit complex \
    --pdb data/examples/PDB/1CSE.pdb \
    --chains-for-protein I \
    --chains-for-complex E,I \
    --mutations I:L38S \
    -n 1 \
    --return-all-cycles \
    -o ./output
```

Both commands also accept a task table via `-i / --input` (see the example
CSVs in `data/examples/`):

```bash
aftoolkit task    -i data/examples/monomer_task.csv -o ./output -n 3
aftoolkit complex -i data/examples/ppi_task.csv     -o ./output -n 3
```

Run with `-h` for the full list of options:

```bash
aftoolkit --help          # all subcommands
aftoolkit task --help
aftoolkit complex --help
```

### Commands

| Command | Purpose |
|---|---|
| `aftoolkit task` | AF2 features for a protein and its mutants |
| `aftoolkit complex` | the same, bound and unbound, for a complex |
| `aftoolkit predict` | ΔΔG + confidence for a mutation, via trained adapters |
| `aftoolkit score` | interface PAE (iPAE) and confidence for a complex |
| `aftoolkit train` | train a stability or affinity adapter |

## Using the Python API

You can drive the pipeline directly and, for example, save the recycled
structure to PDB:

```python
from aftoolkit import ProteinTask, OpenFoldWrapper, save_to_pdb

of_wrapper = OpenFoldWrapper(
    device="cuda:0",          # or "mps", or None to auto-detect
    inference_n_recycle=3,
    always_use_template=False,
    side_chain_mask=False,
    return_all_cycles=False,
)
of_wrapper.init_model()

task = ProteinTask()
task.set_input("data/examples/PDB/1A0F.pdb", chains=["A"])
task.set_mutations("A:S11A")
task.set_observables()                 # every residue, by default
task.evaluate(of_wrapper, store_structure=True)

# aggregated embeddings for the wildtype and the mutant
wt_vector, mt_vector = task.embeddings(["pair", "lddt_logits", "plddt"])

# save the recycled wildtype structure
save_to_pdb(task.wildtype.folded, "test_save.pdb")
```

Each task holds two `ProteinState`s, `task.wildtype` and `task.mutant`, with the
same fields on both:

| Field | Contents |
|---|---|
| `.structure` | the input structure (mutations applied, for the mutant) |
| `.observed` | `{residue_key: index}` for the residues features are taken at |
| `.folded` | the structure AlphaFold predicted |
| `.features` | `{residue_key: {feature_name: vector}}` |
| `.cycles` | per-recycle features, with `return_all_cycles=True` |

A `ProteinComplexTask` is a pair of such tasks, `task.protein` (unbound) and
`task.complex` (bound), kept in step with each other.

### Mutation syntax

Mutations are given as `<chain>:<WT><pos><MT>`, comma-separated:

| Kind         | Example     | Meaning                              |
|--------------|-------------|--------------------------------------|
| substitution | `A:S11A`    | Ser11 → Ala on chain A               |
| deletion     | `A:S11-`    | delete Ser11                         |
| insertion    | `A:-11G`    | insert Gly after position 11         |
| insertion    | `A:-11GGS`  | insert a sequence after position 11  |
| multiple     | `A:S11A,A:T12V` | combined                         |

Parsing yields `Mutation` objects, so the fields are named rather than
positional:

```python
>>> from aftoolkit import parse_mutations
>>> mutation, = parse_mutations("A:S11A")
>>> mutation.chain, mutation.position, mutation.wt, mutation.mt
('A', 11, 'S', 'A')
>>> mutation.is_substitution
True
```

## Visualising a mutation's effect

AlphaFold stores per-residue pLDDT in the B-factor column, so differencing the
wildtype and mutant predictions gives a per-residue map of *where the model
became less sure* — the spatial read-out a single ΔΔG number cannot provide.

```bash
aftoolkit predict --pdb 2OCJ.pdb --chain A --mutations E285K \
    --models "data/models/monomer/pair+lddt_logits+plddt/trained_*.pkl" \
    -n 4 --pymol --out-dir results
```

| Flag | Effect |
|---|---|
| `--save-structures` / `--no-save-structures` | write the predicted wildtype and mutant PDBs (default: on) |
| `--pymol` | write `<name>_delta.pdb` (B-factor = mutant − wildtype pLDDT), `<name>_delta.pml`, and a `<name>_delta.pse` session if PyMOL is on `PATH` |

The session colours the structure on a diverging ramp centred at zero — red
where confidence was lost, blue where it was gained — and shows the mutated
residues as sticks. Residues with no counterpart across the pair (those an indel
adds or removes) are set to 0. Without PyMOL installed the `.pml` is still
written, so the session can be built later with `pymol -cq <name>_delta.pml`.

From Python:

```python
from aftoolkit.visualize import write_pymol_session

write_pymol_session(task.wildtype.folded, task.mutant.folded,
                    out_prefix="results/E285K", sites=[285])
```

## Worked examples

| Example | What it shows |
|---|---|
| [`reproductions/p53_e285k/`](reproductions/p53_e285k/) | p53 E285K (Dunsche et al. 2023): 62 mutations calibrated against 42 measured ΔΔG values, a substitution series, salt-bridge analysis, and the confidence-difference map |
| [`reproductions/rbd_aim_fig2g/`](reproductions/rbd_aim_fig2g/) | antibody–RBD interface PAE (RBD-AIM Fig. 2g) |

### Example: scoring the p53 variants of Dunsche et al. 2023

The five constructs tested in
[Cell Death & Disease **14**:831](https://doi.org/10.1038/s41419-023-06360-4),
scored on PDB 2OCJ chain A at 4 recycles:

```bash
for mut in E285K E285R K132E R175H R248W; do
  aftoolkit predict --pdb 2OCJ.pdb --chain A --mutations $mut \
      --models "data/models/monomer/pair+lddt_logits+plddt/trained_*.pkl" \
      -n 4 --pymol --out-dir results/p53
done
```

| Variant | Predicted ΔΔG | Δ pLDDT at site | Δ PAE | Measured ΔΔG | Reported in the paper |
|---|---|---|---|---|---|
| **E285K** | −0.66 *(19th pct)* | **−5.17** *(14th)* | +1.38 | — | cytosolic, transcriptionally dead, chemoresistant |
| **E285R** | −0.76 *(12th pct)* | −3.49 *(17th)* | +1.50 | — | charge control; also failed to restore cisplatin sensitivity |
| **K132E** | −0.67 *(19th pct)* | −0.35 *(62nd)* | +0.11 | — | phenocopies E285K for chemoresistance |
| **R175H** | −0.52 *(33rd pct)* | **−5.92** *(10th)* | +3.41 | −3.52 | GOF comparator |
| **R248W** | +0.27 *(86th pct)* | −0.65 *(52nd)* | −0.20 | — | GOF comparator |

Percentiles are against the 42 p53 mutations with experimental ΔΔG in
`data/prostata/p53_task.csv`, so "14th" means more destabilising than 86% of
measured variants. Negative ΔΔG = destabilising.

Two things the table shows:

- **E285K, E285R and R175H all register as structural perturbations**, while
  **R248W does not** (86th percentile, i.e. benign for the fold). That is the
  expected split: R248W is a DNA-*contact* mutant that leaves the fold intact,
  so a stability predictor should not flag it — and does not.
- **K132E disagrees between the two metrics** — destabilising by ΔΔG (19th pct)
  but almost unchanged locally by Δ pLDDT (62nd). Its partner E285 sits across
  the salt bridge, so the local confidence read-out at residue 132 misses an
  effect the whole-structure ΔΔG picks up.

These adapters predict **ΔΔG of folding stability only**. Localisation,
transcriptional activity and chemoresistance are experimental results from the
paper, not predictions. See the
[full example](reproductions/p53_e285k/) for the calibration, the E285
substitution series, and the charge-swap rescue prediction.

### Example: a compensatory second mutation

E285 sits at the centre of a salt-bridge chain measured from the crystal:

```
K164 —2.91Å— E271 —2.91Å— K132 —3.00Å— E285 —3.80Å— R273
```

If E285K is disruptive because a reversed charge now repels K132 and R273, then
reversing a *partner's* charge should restore the pairing. Scoring each
candidate second site and reading Δ pLDDT at residue 285 only:

```bash
aftoolkit predict --pdb 2OCJ.pdb --chain A --mutations E285K,K132E \
    -n 4 --pymol --out-dir results/p53
```

| Construct | Δ pLDDT at 285 | Recovered | Second site is… |
|---|---|---|---|
| E285K | −5.17 | — | the variant alone |
| **E285K + K132E** | **−1.62** | **69%** | direct partner, charge reversed |
| **E285K + R273E** | **−2.27** | **56%** | the other direct partner, charge reversed |
| E285K + K132D | −4.04 | 22% | same charge, one CH₂ shorter |
| E285K + E271K | −4.22 | 18% | second shell, not in contact with 285 |
| E285K + R273H | −4.41 | 15% | same site as R273E, but *not* charge-reversed |
| E285K + D281K | −5.72 | none (worse) | nearby, outside the chain |

The pattern is specific rather than a single hit: **both direct partners rescue
when their charge is reversed**, while the same two sites mutated the wrong way
(R273H, K132D) do not, and neither does a second-shell or off-chain site. Chain
geometry shows up too — K132D carries the same charge as K132E but is one
methylene shorter, and recovers a third as much.

**Read this as a hypothesis, not a result.** Three limits matter:

- It comes from the AlphaFold confidence of the double-mutant structure, **not**
  from the adapter ΔΔG. The adapters are additive by construction
  (`multi_as_singlessum=True`) and cannot represent epistasis at all, so no
  ΔΔG-based rescue signal is even possible here.
- Each number is one deterministic prediction with no error bars, and pLDDT is a
  confidence measure, not a free energy.
- Every double strips one extra side chain from the template, which on its own
  perturbs the local environment — the D281K and R273H arms are the controls for
  that, and they show no rescue.

## Pretrained models & data

> **The external download host currently refuses anonymous reads.** Every object
> under `bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru`, and the bucket
> root, returns `AccessDenied`, so the `wget` commands below do not work as
> written. Until it is re-opened or re-hosted, the monomer adapters can be
> rebuilt locally — see [Rebuilding the adapters](#rebuilding-the-adapters).

Pretrained adapter weights and benchmark feature files are hosted externally.
Download them with `wget` (each URL already encodes the target filename):

Stability (monomer) models:

```bash
mkdir -p data/models/monomer/pair+lddt_logits+plddt
cd data/models/monomer/pair+lddt_logits+plddt
wget https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/models/monomer/trained_catboost_concat_nomultitrain_aggmutpos_multisum.pkl
wget https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/models/monomer/trained_mlp_concat_nomultitrain_aggmutpos_multisum.pkl
wget https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/models/monomer/trained_svm_concat_nomultitrain_aggmutpos_multisum.pkl
```

Affinity (multimer) model:

```bash
mkdir -p data/models/multimer/pair+lddt_logits+plddt
cd data/models/multimer/pair+lddt_logits+plddt
wget https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/models/multimer/trained_svm_concat_nomultitrain_aggmutpos_multisum.pkl
```

Benchmark task files and precomputed feature `.pkl`s (S669, Ssym, ProTherm,
cDNA de-novo / indels, S4169, C380) are available under the same host, e.g.:

```bash
cd data
wget https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/data/stability_task_files.zip
wget https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit/data/ssym_pkls.zip
# ... see the OBS bucket for the full list
```

> Feature files for the largest training set are not distributed; regenerate
> them with `aftoolkit task` from the corresponding task CSV.

### Evaluating a pretrained model

```bash
python scripts/evaluate.py \
    --model "data/models/monomer/pair+lddt_logits+plddt/trained_svm_*.pkl" \
    --test-csv data/prostata/s669_task.csv \
    --features-dir data/prostata/features
```

## Training your own adapter

Build a dataset with `aftoolkit task` / `aftoolkit complex`, then train an
adapter:

```bash
aftoolkit train \
    --model svm \
    --target monomer \
    --train-csv data/cdna/cdna_task_train.csv \
    --features-dir data/cdna/features \
    --npy-dir data/cdna/npy
```

`--model` selects the estimator (`svm`, `mlp`, `catboost`) and `--target`
switches between stability (`monomer`) and affinity (`multimer`). From Python,
`aftoolkit.training.train_adapter` takes an `estimator=` argument, so any object
with the scikit-learn `fit`/`predict` interface can back the adapter.

## Rebuilding the adapters

The monomer adapters can be retrained from the cached feature files in `data/`,
which makes them independent of the external host:

```bash
for m in svm mlp catboost; do
  aftoolkit train --model $m --target monomer \
      --train-csv data/cdna/mini2k_grouped_split.csv \
      --features-dir data/cdna/features_n4 \
      --out-dir data/models_retrained
done
```

`mini2k_grouped_split.csv` splits by parent structure rather than by mutation,
so no fold appears in both train and test. On the p53 benchmark — held out from
both — the rebuilt ensemble matches the published one:

| Ensemble | Spearman ρ | Pearson r |
|---|---|---|
| published (`data/models`) | +0.734 | +0.784 |
| rebuilt (`data/models_retrained`) | +0.743 | +0.766 |

The two agree with each other at ρ = 0.96.

`--target multimer` works the same way, but this repository carries no complex
dataset with measured binding ΔΔG to train it on.

## Reproducing the paper

The `scripts/` directory provides a standalone, idempotent reproduction of the
monomer stability experiment (data → AF2 features → SVM adapter → benchmark):

| Script | Purpose |
|---|---|
| `scripts/build_cdna_task.py` | MegaScale (Tsuboyama 2023) → task CSV + structures |
| `scripts/build_prostata_task.py` | PROSTATA dataset CSV → task CSV (uses local PDBs) |
| `scripts/evaluate.py` | benchmark metrics (Spearman/Pearson/RMSE/AUC/MCC) |
| `scripts/reproduce.py` | **standalone** end-to-end orchestrator (all of the above) |

```bash
# one-time environment (uv)
uv venv --python 3.11 .venv
uv pip install --python .venv -e ".[openfold]" pyarrow

# quick smoke (subset) or full run (PROSTATA S2648 train ≈ overnight on one GPU/MPS)
.venv/bin/python scripts/reproduce.py --max-train 400 --datasets p53
.venv/bin/python scripts/reproduce.py            # full; compares to paper Table 3
```

Feature generation is the only heavy step (one OpenFold pass per mutation); it
scales linearly and is embarrassingly parallel across mutations.

## Development

```bash
pip install ".[dev]"   # pytest + ruff
pytest                 # run the (torch-free) test suite
ruff check aftoolkit   # lint
```

CI runs the test suite on Linux across Python 3.10–3.12.

The `aftoolkit/processing/openfold/` directory is a **vendored, modified** copy
of [OpenFold](https://github.com/aqlaboratory/openfold); see
[`aftoolkit/processing/openfold/MODIFICATIONS.md`](aftoolkit/processing/openfold/MODIFICATIONS.md)
for the divergences from upstream and why it is vendored.

## License

This project is released under the MIT License — see [LICENSE](LICENSE).
