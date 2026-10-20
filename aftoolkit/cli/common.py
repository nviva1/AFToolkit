"""Argument groups and the batch runner shared by the subcommands.

``run_protein_task`` and ``run_protein_complex_task`` were previously two
near-identical ``main()`` functions: the same validation, the same wrapper
construction, the same single-structure/CSV-batch branch and the same
error-handling loop, differing only in which task class they built. Both of
those behaviours live here once, parameterised by a factory.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Iterable

from aftoolkit.inference import PIPELINES
from aftoolkit.structure import DEFAULT_CHAIN_GAP, safe_filename_part

__all__ = [
    "add_inference_args",
    "add_output_args",
    "add_structure_args",
    "build_wrapper",
    "run_tasks",
]

logger = logging.getLogger(__name__)


def add_structure_args(parser: argparse.ArgumentParser) -> None:
    """Input structure, mutations and observable positions."""
    parser.add_argument("--pdb", help="Path to a protein structure in PDB format.")
    parser.add_argument("--input", "-i", help="CSV of tasks to run as a batch.")
    parser.add_argument(
        "--mutations",
        help="Mutations as <chain>:<wt><pos><mt>, comma-separated. "
             "Example: A:D123C,A:R56Y",
    )
    parser.add_argument(
        "--positions",
        help="Positions to extract features for, as <chain>:<pos>, "
             "comma-separated. Example: A:123,A:56. Default: every residue.",
    )


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    """AlphaFold inference settings."""
    parser.add_argument(
        "-d", "--device",
        help='Compute device, e.g. "cuda:0" or "mps". '
             "Auto-detects (CUDA, then MPS, then CPU) when omitted.",
    )
    parser.add_argument("-n", "--n-recycle", type=int, default=1,
                        help="Number of AlphaFold recycling iterations.")
    parser.add_argument(
        "--format", dest="pipeline", default="mask", choices=list(PIPELINES),
        help="Inference format: mask (AFToolkit, default), vanilla (standard "
             "AF2), or af2rank (the 'gap' trick).",
    )
    parser.add_argument(
        "--chain-gap", type=int, default=DEFAULT_CHAIN_GAP,
        help=f"Residue-index gap inserted between chains (default: "
             f"{DEFAULT_CHAIN_GAP}, which saturates AlphaFold's "
             f"relative-position window).",
    )
    parser.add_argument("--return-all-cycles", action=argparse.BooleanOptionalAction,
                        default=False, help="Keep each recycle's features.")


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """Where results go."""
    parser.add_argument("-o", "--output-dir", default="./",
                        help="Output directory.")
    parser.add_argument("--store-protein", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Store the predicted structure in the result.")


def build_wrapper(args):
    """Construct and initialise an ``OpenFoldWrapper`` from parsed arguments.

    Imported here rather than at module scope so ``--help`` and argument
    validation work without the optional torch/OpenFold stack installed.
    """
    from aftoolkit.folding import OpenFoldWrapper

    logger.info("Initializing OpenFold wrapper (device=%s, format=%s)",
                args.device or "auto", args.pipeline)
    wrapper = OpenFoldWrapper(
        device=args.device,
        inference_n_recycle=args.n_recycle,
        return_all_cycles=args.return_all_cycles,
        **PIPELINES[args.pipeline],
    )
    wrapper.init_model()
    return wrapper


def run_tasks(tasks: Iterable[tuple[str, Callable[[], object]]], wrapper,
              output_dir: str, store_protein: bool, keep_going: bool) -> int:
    """Evaluate each task and pickle the result into ``output_dir``.

    Args:
        tasks: ``(name, factory)`` pairs; the factory builds the task object.
        wrapper: an initialised ``OpenFoldWrapper``.
        output_dir: directory for the ``<name>.pkl`` results.
        store_protein: keep predicted structures in the saved task.
        keep_going: log and skip a failing task rather than aborting. A single
            structure run always aborts, so a typo is not silently swallowed.

    Returns:
        The number of tasks that succeeded.
    """
    os.makedirs(output_dir, exist_ok=True)
    tasks = list(tasks)
    n_ok = 0

    for name, factory in tasks:
        try:
            logger.info("Processing task %s", name)
            task = factory()
            task.evaluate(wrapper, store_structure=store_protein)
            out_path = os.path.join(output_dir, f"{safe_filename_part(name)}.pkl")
            task.save(out_path)
            logger.info("Saved results to %s", out_path)
            n_ok += 1
        except Exception:
            if not keep_going:
                raise
            logger.exception("Failed to process task %s; skipping", name)

    if len(tasks) > 1:
        logger.info("Processed %d/%d tasks successfully", n_ok, len(tasks))
    return n_ok
