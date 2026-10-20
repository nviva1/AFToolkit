#!/usr/bin/env python
"""Standalone reproduction of the AFToolkit monomer stability experiment.

Pipeline (mirrors the paper, Sindeeva et al., Brief. Bioinform. 2025):

    PROSTATA data  ->  OpenFold features (N=4 recycles, no-MSA + masked template)
                   ->  SVM adapter (StandardScaler + SVR, C=1)
                   ->  evaluate on s669 / ssym / p53   (compare to Table 3)

Prerequisites (one-time):
    uv venv --python 3.11 .venv
    uv pip install --python .venv -e ".[openfold]" pyarrow
    # AlphaFold weights: auto-downloaded on first model init, or place
    # weights/params_model_2_ptm.npz yourself.
    # PROSTATA data + structures expected at ../PROSTATA (DATASETS/, PDB/).

Run (inside the venv):
    .venv/bin/python scripts/reproduce.py                       # full (S2648 train ~ overnight)
    .venv/bin/python scripts/reproduce.py --max-train 400       # quick smoke
    .venv/bin/python scripts/reproduce.py --datasets p53        # eval one benchmark

Every step is idempotent: existing task CSVs, feature .pkl files and trained
models are reused, so the run resumes where it left off.
"""

import argparse
import os
import sys

from aftoolkit.training import train_adapter

# The sibling helper scripts are plain modules, not a package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_prostata_task                       # noqa: E402
from evaluate import evaluate as eval_adapter    # noqa: E402

PROSTATA_DS = "../PROSTATA/DATASETS"
WORK = "data/prostata"
TEST_CSVS = {"s669": "s669.csv", "ssym": "ssym.csv", "p53": "p53.csv"}

# Paper Table 3 (AFToolkit SVM) for reference.
PAPER_SPEARMAN = {"s669": 0.51, "ssym": 0.77, "p53": None}


def generate_features(task_csv, out_dir, wrapper):
    """Run OpenFold for every mutation in ``task_csv`` (idempotent)."""
    from aftoolkit.cli.tasks import tasks_from_csv

    tasks = list(tasks_from_csv(task_csv))
    os.makedirs(out_dir, exist_ok=True)
    done = 0
    for task_id, build in tasks:
        out = os.path.join(out_dir, f"{task_id}.pkl")
        if os.path.exists(out):
            done += 1
            continue
        try:
            task = build()
            task.evaluate(wrapper, store_structure=False)
            task.save(out)
            done += 1
        except Exception as exc:               # keep going on a bad structure
            print(f"  ! {task_id}: {exc}")
    print(f"  features ready: {done}/{len(tasks)} in {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="mps")
    p.add_argument("--n-recycle", type=int, default=4, help="Paper uses 4.")
    p.add_argument("--train-csv", default="S2648.csv",
                   help="PROSTATA training CSV under ../PROSTATA/DATASETS.")
    p.add_argument("--max-train", type=int, default=None,
                   help="Subset the training set (omit for the full set).")
    p.add_argument("--datasets", nargs="+", default=list(TEST_CSVS),
                   choices=list(TEST_CSVS))
    p.add_argument("--features-dir", default=os.path.join(WORK, "features_n4"))
    args = p.parse_args()

    os.makedirs(WORK, exist_ok=True)

    # 1. Task CSVs ----------------------------------------------------------
    train_task = os.path.join(WORK, "train_task.csv")
    build_prostata_task.build(os.path.join(PROSTATA_DS, args.train_csv), train_task,
                              split="train", max_mutations=args.max_train)
    test_tasks = {}
    for name in args.datasets:
        out = os.path.join(WORK, f"{name}_task.csv")
        build_prostata_task.build(os.path.join(PROSTATA_DS, TEST_CSVS[name]), out, split="test")
        test_tasks[name] = out

    # 2. Features (single model load) --------------------------------------
    from aftoolkit import OpenFoldWrapper
    wrapper = OpenFoldWrapper(device=args.device, inference_n_recycle=args.n_recycle)
    wrapper.init_model()
    print("Generating features (this is the slow phase)…")
    generate_features(train_task, args.features_dir, wrapper)
    for name, csv in test_tasks.items():
        generate_features(csv, args.features_dir, wrapper)

    # 3. Train SVM adapter --------------------------------------------------
    npy_dir = os.path.join(WORK, "npy", "pair+lddt_logits+plddt")
    os.makedirs(npy_dir, exist_ok=True)
    model = train_adapter(
        model="svm", target="monomer", train_csv=train_task,
        features_dir=args.features_dir, npy_dir=npy_dir,
        features=["pair", "lddt_logits", "plddt"], concat_features=True,
        mut_types=["ss"], protein_aggregation="mutpos", multi_aggregation="sum",
    )

    # 4. Evaluate vs the paper ---------------------------------------------
    print("\n================ RESULTS vs paper (Table 3) ================")
    print(f"{'dataset':<8} {'ours_Rs':>8} {'paper_Rs':>9}")
    for name, csv in test_tasks.items():
        metrics, n = eval_adapter(model, csv, args.features_dir)
        paper = PAPER_SPEARMAN.get(name)
        print(f"{name:<8} {metrics['SpearmanR']:>8.3f} "
              f"{('%.2f' % paper) if paper else '   n/a':>9}  (n={n})")


if __name__ == "__main__":
    main()
