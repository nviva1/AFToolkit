#!/usr/bin/env python
"""Parse MOESM4_affinity.csv -> clean CSV(s) of ABSOLUTE K_D values (nM) per
(antibody, strain). Relative values (ΔlogKD) are NOT computed here -- done later.

Outputs:
  kd_absolute_long.csv  : one row per measurement  (antibody, pdbid, chids, strain, kd_nM)
  kd_absolute_wide.csv  : antibody x strain matrix of K_D (nM)
Each kept K_D is asserted finite; non-positive K_D (0 / below-detection placeholders)
are reported and flagged, not silently dropped.
"""
import os, re, sys, json
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_full as R   # STRAIN_MAP, rbd_mutations

SRC = "/Users/nikitaivanisenko/projects/AFToolkit/rbd-aim/extra_downloads/MOESM4_affinity.csv"
OUTDIR = "/Users/nikitaivanisenko/projects/AFToolkit/AFToolkit/data/rbdaim"
STRAINS = json.load(open(os.path.join(OUTDIR, "datasets/rbd_strains.json")))
WUHAN = STRAINS["Wuhan"]; REFN = list(range(334, 334 + len(WUHAN)))

# ---- strains absent from rbd_strains.json, resolved from public databases ----
# SARS-CoV-1 RBD (UniProt P59594 spike, residues 318-513). Different virus: a
# standalone sequence, NOT expressible as point-mutations of the Wuhan RBD.
SARSCOV1_RBD = ("ITNLCPFGEVFNATKFPSVYAWERKKISNCVADYSVLYNSTFFSTFKCYGVSATKLNDLCFSNVYADSFVVKG"
                "DDVRQIAPGQTGVIADYNYKLPDDFMGCVLAWNTRNIDATSTGNYNYKYRYLRHGKLRPFERDISNVPFSPDG"
                "KPCTPPALNCYWPLNDYGFYTTTGIGYQPYRVVVLSFELLNAPATVCGPK")
# BA.3 RBD mutation set vs Wuhan (covariants.org / Public Health Ontario brief).
BA3_MUTS = ["G339D", "S371F", "S373P", "S375F", "D405N", "K417N", "N440K", "G446S",
            "S477N", "T478K", "E484A", "Q493R", "Q498R", "N501Y", "Y505H"]


def _apply(seq, muts):
    """Apply WT<pos>MT mutations (canonical numbering, pos>=334) to an RBD sequence."""
    s = list(seq)
    for m in muts:
        wt, mt, pos = m[0], m[-1], int(m[1:-1]); i = pos - 334
        assert s[i] == wt, f"{m}: have {s[i]} at {pos}, expected {wt}"
        s[i] = mt
    return "".join(s)


def full_sequence(strain_name):
    """Full RBD sequence for a strain. Resolves the 4 strains missing from
    rbd_strains.json (BA1.1, BA2.13, BA3 by composition; SARS-CoV standalone)."""
    if strain_name == "WT":
        return WUHAN
    key = R.STRAIN_MAP.get(strain_name)
    if key is not None and key in STRAINS:
        return STRAINS[key]
    if strain_name == "BA1.1":                       # BA.1 + R346K
        return _apply(STRAINS["21K_(BA.1)"], ["R346K"])
    if strain_name == "BA2.13":                      # BA.2 + L452M
        return _apply(STRAINS["21L_(BA.2)"], ["L452M"])
    if strain_name == "BA3":                         # composed onto Wuhan
        return _apply(WUHAN, BA3_MUTS)
    if strain_name == "SARS-CoV":                    # SARS-CoV-1, standalone
        return SARSCOV1_RBD
    return None


def strain_mutations(strain_name):
    """Mutations of a strain vs Wuhan RBD (canonical positions). Returns (str, n)."""
    if strain_name == "WT":
        return "", 0
    seq = full_sequence(strain_name)
    if seq is None:
        return "NA(no_seq)", -1
    if len(seq) != len(WUHAN):                       # cross-species (SARS-CoV-1)
        return "cross-species(see rbd_sequence)", -1
    ms = R.rbd_mutations(WUHAN, REFN, seq)
    return ";".join(f"{w}{p}{m}" for w, p, m in ms), len(ms)


def assert_kd_value(v, tag):
    """A K_D value is a finite, positive number (nM). Non-positive => flag (not valid)."""
    assert v == v and np.isfinite(v), f"[{tag}] non-finite K_D = {v}"
    return v > 0


def parse_kd_cell(raw):
    """Parse a K_D cell -> (kd_nM, censored). Handles plain numbers and CENSORED
    bounds ('>1000.0' -> 1000.0 with censored='>'; '<0.1' -> 0.1 with '<').
    Returns (None, None) for empty / unparseable cells."""
    if raw is None:
        return None, None
    s = str(raw).strip()
    if s in ("", "nan", "NaN", "None"):
        return None, None
    m = re.match(r"^([<>]?)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", s)
    if not m:
        return None, None
    return float(m.group(2)), m.group(1)


def main():
    d = pd.read_csv(SRC)
    kdcols = [c for c in d.columns if "AB KD" in str(c)]
    strain_of = lambda c: c.replace(" AB KD, nM", "").strip()
    meta = ["Name", "pdbid", "chids"]
    long_rows = []
    nonpos = []
    for _, r in d.iterrows():
        for c in kdcols:
            kd, cen = parse_kd_cell(r[c])
            if kd is None:
                continue                       # no measurement for this strain
            tag = f"{r['Name']}/{strain_of(c)}"
            ok = assert_kd_value(kd, tag)         # asserts finite
            if not ok:
                nonpos.append(tag)
            st = strain_of(c); muts, nmut = strain_mutations(st)
            seq = full_sequence(st)
            long_rows.append({"antibody": r["Name"], "pdbid": r.get("pdbid"),
                              "chids": r.get("chids"), "strain": st,
                              "kd_nM": kd, "valid": bool(ok), "censored": cen,
                              "mutations_vs_WT": muts, "n_mut": nmut,
                              "rbd_sequence": seq if seq is not None else "NA"})
    long = pd.DataFrame(long_rows)
    long.to_csv(os.path.join(OUTDIR, "kd_absolute_long.csv"), index=False)

    # wide matrix (antibody x strain), only valid (>0) K_D
    wide = (long[long.valid].pivot_table(index=["antibody", "pdbid", "chids"],
                                         columns="strain", values="kd_nM", aggfunc="first")
            .reset_index())
    wide.to_csv(os.path.join(OUTDIR, "kd_absolute_wide.csv"), index=False)

    # --- report ---
    print(f"parsed {SRC}")
    print(f"total K_D measurements (non-null): {len(long)}")
    print(f"  valid (finite & > 0):  {int(long.valid.sum())}")
    print(f"  non-positive (0 / below-detection placeholders, flagged valid=False): {len(nonpos)}")
    if nonpos:
        print("   ", nonpos)
    print(f"antibodies with >=1 K_D: {long.antibody.nunique()}")
    print(f"\nwrote:\n  {OUTDIR}/kd_absolute_long.csv  ({len(long)} rows)")
    print(f"  {OUTDIR}/kd_absolute_wide.csv  ({len(wide)} antibodies x {len([c for c in wide.columns if c not in meta])} strains)")
    print("\nsample (long):")
    print(long.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
