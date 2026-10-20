#!/usr/bin/env python
"""Explicit assert that the masked-template protocol does what we think:
the structural template is present ONLY at cycle 0 and MASKED for every recycle
after it (always_use_template=False => template_mask *= 0 for cycles >= 1).

We spy on the EXACT feature dict handed to the model in a real inference and assert
on its `template_mask` (the recycling dim is the last axis after the cat).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_load as M

N_RECYCLE = 2                      # -> cycles c0,c1,c2  (template only at c0)
PDB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "x_ray_pdbs/COVOX-278_7OR9_HLE.pdb")


def main():
    from aftoolkit.folding import OpenFoldWrapper
    wr = OpenFoldWrapper(device="mps", inference_n_recycle=N_RECYCLE,
                         always_use_template=False, return_all_cycles=True)
    wr.init_model()
    df = M.prep_state(PDB, ["H", "L", "E"], "E", "", set(), chain_gap=25)

    # spy: capture the processed_feature_dict actually passed to the model
    cap = {}
    orig = wr.model
    def spy(fd):
        cap["template_mask"] = np.asarray(fd["template_mask"].detach().cpu())
        return orig(fd)
    wr.model = spy
    wr.evaluate(df)
    wr.model = orig

    tm = cap["template_mask"]
    print(f"template_mask shape = {tm.shape}  (last axis = recycling dim)")
    ncyc = tm.shape[-1]

    # --- ASSERTS ---
    assert ncyc == N_RECYCLE + 1, f"expected {N_RECYCLE+1} cycles, got {ncyc}"
    c0 = float(np.abs(tm[..., 0]).sum())
    assert c0 > 0, f"cycle 0 template_mask is zero ({c0}) -- template NOT provided at c0!"
    print(f"  c0: |template_mask| sum = {c0:.3f}  -> template PRESENT ✓")
    for k in range(1, ncyc):
        ck = float(np.abs(tm[..., k]).sum())
        assert ck == 0, f"cycle {k} template_mask = {ck} (expected 0) -- template NOT masked after c0!"
        print(f"  c{k}: |template_mask| sum = {ck:.3f}  -> template MASKED ✓")

    print("\nPASS: template present at cycle 0, masked for every cycle >= 1.")

    # sanity counter-check: with always_use_template=True the template should persist
    wr2 = OpenFoldWrapper(device="mps", inference_n_recycle=N_RECYCLE,
                          always_use_template=True, return_all_cycles=True)
    wr2.init_model()
    cap2 = {}; o2 = wr2.model
    def spy2(fd):
        cap2["tm"] = np.asarray(fd["template_mask"].detach().cpu()); return o2(fd)
    wr2.model = spy2; wr2.evaluate(df); wr2.model = o2
    tm2 = cap2["tm"]
    persists = all(float(np.abs(tm2[..., k]).sum()) > 0 for k in range(tm2.shape[-1]))
    print(f"counter-check (always_use_template=True): template present every cycle = {persists} "
          f"{'✓' if persists else '✗'}")


if __name__ == "__main__":
    main()
