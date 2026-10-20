#!/usr/bin/env python
"""Run the p53 DBD mutation panel: E285K, a substitution series, and controls.

Calibrates against the 42 p53 mutations that have experimental ddG in
``data/prostata/p53_task.csv``, then scores the panel of interest on the same
structure, so the predictions can be read as percentiles of measured variants
rather than as bare numbers.

Examples
--------
    # full panel (62 mutations, ~30 min on an M-series GPU)
    python reproductions/p53_e285k/run_panel.py --pdb path/to/2OCJ.pdb

    # just the variants of interest, with a PyMOL session for each
    python reproductions/p53_e285k/run_panel.py --pdb 2OCJ.pdb \
        --groups E285_series --pymol

    # one mutation, structures and session, no adapters
    python reproductions/p53_e285k/run_panel.py --pdb 2OCJ.pdb \
        --only E285K --pymol --no-models
"""

import argparse
import glob
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from aftoolkit.folding import OpenFoldWrapper          # noqa: E402
from aftoolkit.inference import predict_mutation       # noqa: E402

DEFAULT_MODELS = "data/models/monomer/pair+lddt_logits+plddt/trained_*.pkl"
#: Calibration labels ship with the example, so a fresh clone can run it.
BENCHMARK_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "p53_benchmark.csv")
#: The example fetches its own structure rather than depending on a local copy.
PDB_URL = "https://files.rcsb.org/download/2OCJ.pdb"

#: The panel, beyond the experimentally measured calibration set.
PANEL = {
    # charge reversal (the Dunsche et al. variant) vs conservative vs neutral
    "E285_series": ["E285K", "E285R", "E285D", "E285Q", "E285A", "E285G", "E285W"],
    # the salt-bridge chain K164-E271-K132-E285-R273 and its packing neighbours
    "salt_bridge_network": ["K132E", "K132A", "R273C", "R273L", "D281K", "F134A"],
    # classic hotspots absent from the calibration set
    "other_hotspots": ["R248W", "R175C", "Y220S", "G245D", "R282Q"],
    # NOTE: the adapters are additive by construction, so these cannot show
    # epistasis in ddG; the compensatory signal appears in the confidence map.
    "double_additive": ["E285K,K132E", "E285K,R273H"],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", default=None,
                   help="p53 core-domain structure; downloaded from RCSB "
                        "(2OCJ) into --out-dir when omitted.")
    p.add_argument("--chain", default="A")
    p.add_argument("--out-dir", default="reproductions/p53_e285k/results")
    p.add_argument("--device", default=None, help="cuda:0 / mps / cpu (auto if unset).")
    p.add_argument("-n", "--n-recycle", type=int, default=4,
                   help="Recycles; 4 matches the published adapter setting.")
    p.add_argument("--models", default=DEFAULT_MODELS,
                   help="Glob of trained adapter .pkl files.")
    p.add_argument("--no-models", action="store_true",
                   help="Skip ddG prediction; report confidence only.")
    p.add_argument("--benchmark-csv", default=BENCHMARK_CSV,
                   help="CSV of p53 mutations with experimental ddG.")
    p.add_argument("--groups", nargs="*", default=None,
                   choices=["benchmark", *PANEL],
                   help="Subset of groups to run (default: all).")
    p.add_argument("--only", nargs="*", default=None,
                   help="Run just these mutations, e.g. --only E285K E285D.")
    p.add_argument("--pymol", action="store_true",
                   help="Write a confidence-difference structure, a PyMOL "
                        "script and a .pse session for each mutation.")
    p.add_argument("--save-structures", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Write the predicted wildtype/mutant PDBs "
                        "(default: off for a 62-mutation run).")
    return p.parse_args()


def ensure_structure(path, out_dir):
    """Return a local 2OCJ, fetching it from RCSB the first time."""
    import urllib.request

    if path:
        if not os.path.exists(path):
            raise SystemExit(f"error: --pdb not found: {path}")
        return path

    local = os.path.join(out_dir, "2OCJ.pdb")
    if not os.path.exists(local):
        print(f"fetching {PDB_URL} -> {local}", flush=True)
        urllib.request.urlretrieve(PDB_URL, local)  # noqa: S310 - fixed https URL
    return local


def build_jobs(args):
    """``(mutation, group, experimental_ddg)`` for everything to be run."""
    jobs = []
    wanted = set(args.groups) if args.groups else None

    if wanted is None or "benchmark" in wanted:
        path = args.benchmark_csv
        if os.path.exists(path):
            bench = pd.read_csv(path, index_col=0)
            jobs += [(m, "benchmark", float(d))
                     for m, d in zip(bench.mut_info, bench.ddg)]
        else:
            print(f"  ! no benchmark CSV at {path}; skipping calibration set")

    for group, muts in PANEL.items():
        if wanted is None or group in wanted:
            jobs += [(m, group, None) for m in muts]

    if args.only:
        keep = set(args.only)
        jobs = [j for j in jobs if j[0] in keep]
        missing = keep - {j[0] for j in jobs}
        jobs += [(m, "custom", None) for m in sorted(missing)]

    seen, ordered = set(), []
    for mut, group, exp in jobs:
        if mut not in seen:
            seen.add(mut)
            ordered.append((mut, group, exp))
    return ordered


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    reports = os.path.join(args.out_dir, "reports")
    os.makedirs(reports, exist_ok=True)

    jobs = build_jobs(args)
    args.pdb = ensure_structure(args.pdb, args.out_dir)

    models = [] if args.no_models else [os.path.join(REPO, args.models)]
    if models and not glob.glob(models[0]):
        print(f"  ! no adapters matched {models[0]}\n"
              f"    running confidence-only; pass --no-models to silence this.",
              flush=True)
        models = []
    print(f"{len(jobs)} mutations on {args.pdb} chain {args.chain}, "
          f"{args.n_recycle} recycles", flush=True)

    wrapper = OpenFoldWrapper(device=args.device,
                              inference_n_recycle=args.n_recycle).init_model()

    rows, t0 = [], time.time()
    for i, (mut, group, exp) in enumerate(jobs, 1):
        try:
            report = predict_mutation(
                pdb=args.pdb, chain=args.chain, mutations=mut,
                model_paths=models, n_recycle=args.n_recycle, out_dir=reports,
                save_structures=args.save_structures, pymol_session=args.pymol,
                wrapper=wrapper,
            )
            conf, change = report.confidence, report.confidence["change"]
            n = len(conf["wildtype"]["per_site_plddt"]) or 1
            row = {
                "mutation": mut, "group": group, "experimental_ddg": exp,
                "pred_ddg_ensemble": report.predicted_ddg["ensemble_mean"],
                "delta_plddt_site": sum(change["delta_per_site_plddt"].values()) / n,
                "delta_pae_site": sum(change["delta_per_site_pae"].values()) / n,
                "delta_mean_plddt": change["delta_mean_plddt"],
                "delta_ptm": change.get("delta_ptm"),
                "pymol_session": report.structures.get("pymol_session"),
            }
            for name, value in report.predicted_ddg["per_model"].items():
                row["ddg_" + name.split("_")[1]] = value
            rows.append(row)
            pd.DataFrame(rows).to_csv(
                os.path.join(args.out_dir, "p53_results.csv"), index=False)

            ddg = row["pred_ddg_ensemble"]
            print(f"[{i:2d}/{len(jobs)}] {mut:14s} {group:20s} "
                  f"ddG={'   n/a' if ddg is None else f'{ddg:+.3f}'} "
                  f"dpLDDT={row['delta_plddt_site']:+6.2f} "
                  f"dPAE={row['delta_pae_site']:+5.2f}  "
                  f"({(time.time()-t0)/i:.0f}s/mut)", flush=True)
        except Exception as exc:
            print(f"[{i:2d}/{len(jobs)}] {mut:14s} FAILED "
                  f"{type(exc).__name__}: {exc}", flush=True)

    out = os.path.join(args.out_dir, "p53_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n{len(rows)}/{len(jobs)} in {(time.time()-t0)/60:.1f} min -> {out}")
    if args.pymol:
        print(f"PyMOL sessions in {reports}/  (open a *_delta.pse)")


if __name__ == "__main__":
    main()
