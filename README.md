# AFToolkit

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
- [Pretrained models & data](#pretrained-models--data)
- [Training your own adapter](#training-your-own-adapter)
- [Development](#development)
- [License](#license)

## Installation

The lightweight task-construction and adapter APIs need only standard scientific
Python and install with plain `pip`:

```bash
git clone https://github.com/AIRI-Institute/AFToolkit.git
cd AFToolkit
pip install .
```

Running AlphaFold/OpenFold inference (the `OpenFoldWrapper` and the
`run_protein_task` / `run_protein_complex_task` CLIs) additionally requires
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
from aftoolkit import ProteinTask, parse_mutations

task = ProteinTask()
task.set_input_protein_task("data/examples/PDB/1A0F.pdb", chains=["A"])
task.set_task_mutants(parse_mutations("A:S11A"))
task.set_wildtype_protein()
task.set_mutate_protein()
```

## Extracting AlphaFold features

Calculate AF2 features for a single protein:

```bash
run_protein_task \
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
run_protein_complex_task \
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
run_protein_task         -i data/examples/monomer_task.csv -o ./output -n 3
run_protein_complex_task -i data/examples/ppi_task.csv     -o ./output -n 3
```

Run with `-h` for the full list of options:

```bash
run_protein_task -h
run_protein_complex_task -h
```

## Using the Python API

You can drive the pipeline directly and, for example, save the recycled
structure to PDB:

```python
from aftoolkit import ProteinTask, OpenFoldWrapper, parse_mutations
from aftoolkit.processing.utils import save_to_pdb

of_wrapper = OpenFoldWrapper(
    device="cuda:0",          # or "mps", or None to auto-detect
    inference_n_recycle=3,
    always_use_template=False,
    side_chain_mask=False,
    return_all_cycles=False,
)
of_wrapper.init_model()

task = ProteinTask()
task.set_input_protein_task("data/examples/PDB/1A0F.pdb", chains=["A"])
task.set_task_mutants(parse_mutations("A:S11A"))
task.set_observable_positions()
task.evaluate(of_wrapper=of_wrapper, store_of_protein=True)

# save the recycled wildtype structure
save_to_pdb(task.get_wildtype_protein_of(), "test_save.pdb")
```

### Mutation syntax

Mutations are given as `<chain>:<WT><pos><MT>`, comma-separated:

| Kind         | Example     | Meaning                              |
|--------------|-------------|--------------------------------------|
| substitution | `A:S11A`    | Ser11 → Ala on chain A               |
| deletion     | `A:S11-`    | delete Ser11                         |
| insertion    | `A:-11G`    | insert Gly after position 11         |
| multiple     | `A:S11A,A:T12V` | combined                         |

## Pretrained models & data

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
> them with `run_protein_task` from the corresponding task CSV.

### Evaluating a pretrained model

See `run_test_monomer_example.py` (stability) and
`run_test_multimer_example.py` (affinity). Set the dataset / model paths at the
top of the script, then run it, e.g.:

```bash
python run_test_monomer_example.py
```

## Training your own adapter

Build a dataset with `run_protein_task` / `run_protein_complex_task`, then train
an adapter. Worked examples (edit the paths at the top, then run):

- `run_train_svm_monomer.py`, `run_train_mlp_monomer.py`, `run_train_catboost_monomer.py` — stability;
- `run_train_svm_multimer.py` — affinity.

These wrap `train_adapter_monomer.py` / `train_adapter_multimer.py`, which you
can also call directly. Any estimator with the scikit-learn `fit`/`predict`
interface can be used as the adapter's base model.

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
