# Vendored OpenFold — modifications

This directory is a **vendored, modified copy** of [OpenFold](https://github.com/aqlaboratory/openfold),
an open-source reimplementation of AlphaFold2. It is intentionally not a pip/git
dependency on upstream, because AFToolkit relies on changes to the model's
forward pass that are not present upstream.

It is imported by absolute path (`aftoolkit.processing.openfold.*`), including
from within the vendored tree itself, so the package must stay at this location.

## Why it is vendored (cannot use upstream as-is)

AFToolkit's feature-extraction method needs **per-recycle** model outputs and a
custom energy head. Both require editing OpenFold internals, so a clean
dependency on the upstream package is not possible without carrying these
patches.

## Divergences from upstream

### 1. `AlphaFold.forward` returns per-recycle outputs
`model/model.py` — `AlphaFold.forward(batch)`

Upstream returns a single `outputs` dict (the final recycle only). The vendored
version additionally collects the output of **every** recycling iteration,
runs the auxiliary heads on each, and returns a tuple:

```python
return outputs, cycle_output   # cycle_output: list, one dict per recycle
```

This powers AFToolkit's `--return-all-cycles` feature and the masked-template
recycling pipeline in `OpenFoldWrapper.inference_monomer`
(`aftoolkit/processing/openfold_wrapper.py`), which expects the call
`self.model(processed_feature_dict)` to yield `(out_batch, out_per_cycle)`.

### 2. `AlphaFoldRosettaHead`
`model/model.py` — class `AlphaFoldRosettaHead`

A custom head added on top of `AlphaFold` that projects the pair and single
representations to per-residue / per-pair energy terms (used for the
Rosetta-energy prediction experiments). Not part of upstream OpenFold.

### 3. Lazy `relax` import in `utils/script_utils.py`
`utils/script_utils.py`

Upstream imports the Amber/OpenMM relaxation module (`np.relax.relax`) at module
load time. AFToolkit's feature-extraction pipeline only uses
`load_models_from_command_line` and `prep_output` from this module and never
relaxes structures, so the import was moved *inside* `relax_protein`. This keeps
the heavy `openmm`/`pdbfixer` stack an optional dependency (the `[relax]` extra)
rather than a hard requirement for running inference.

## Maintenance note

If you ever re-sync with upstream OpenFold, re-apply the two changes above.
Search for `cycle_output` and `AlphaFoldRosettaHead` to locate them.
